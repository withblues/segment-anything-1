import numpy as np
import torch

from segment_anything.modeling import Sam

from typing import Optional, Tuple, List

from .utils.transforms import ResizeLongestSide

class BatchedSamPredictor:
    def __init__(
        self,
        sam_model: Sam,
    ) -> None:
        """
        Uses SAM to calculate the image embedding for an image, and then
        allow repeated, efficient mask prediction given prompts.

        Arguments:
          sam_model (Sam): The model to use for mask prediction.
        """
        super().__init__()
        self.model = sam_model
        self.transform = ResizeLongestSide(sam_model.image_encoder.img_size)
        self.reset_image()

    def set_images(
        self,
        images: List[np.ndarray],
        image_format: str = "RGB",
    ) -> None:
        """
        Calculates the image embeddings for the provided image, allowing
        masks to be predicted with the 'predict' method.

        Arguments:
          image (np.ndarray): The image for calculating masks. Expects an
            image in HWC uint8 format, with pixel values in [0, 255].
          image_format (str): The color format of the image, in ['RGB', 'BGR'].
        """
        if not images:
            raise ValueError("Input 'images' list cannot be empty.")

        input_images_torch_list = []
        original_size_list = []
        input_sizes_list = []

        for image in images:
            assert image_format in [
                "RGB",
                "BGR",
            ], f"image_format must be in ['RGB', 'BGR'], is {image_format}."
            if image_format != self.model.image_format:
                image = image[..., ::-1]

            # transform the iamge to the form expected by the model

            # Transform the image to the form expected by the model
            input_image = self.transform.apply_image(image)
            input_image_torch = torch.as_tensor(input_image, device=self.device)
            input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]

            input_images_torch_list.append(input_image_torch)
            original_size_list.append(image.shape[:2])
            input_sizes_list.append(tuple(input_image_torch.shape[-2:]))


        self.set_torch_images(input_images_torch_list, original_size_list, input_sizes_list)
    
    @torch.no_grad()
    def set_torch_images(
            self,
            transformed_images: List[torch.Tensor],
            original_image_sizes: List[Tuple[int, ...]],
            input_sizes: List[Tuple[int, ...]],
    ) -> None:
        """
        Calculates the image embeddings for the provided images, allowing
        masks to be predicted with the 'predict' method. Expects the input
        images to be already transformed to the format expected by the model.

        Arguments:
        transformed_images (list(torch.Tensor)): The input images, with shape
            Bx3xHxW, which has been transformed with ResizeLongestSide.
        original_image_size (list(tuple(int, int))): The size of the images
            before transformation, in (H, W) format.
        """
        self.reset_image()

        self.original_sizes = original_image_sizes
        self.input_sizes = input_sizes

        # preprocess each image 
        preprocessed_images = []
        for img in transformed_images:
            preprocessed_images.append(self.model.preprocess(img))

        batched_preprocessed_images = torch.cat(preprocessed_images, dim=0)
        self.features = self.model.image_encoder(batched_preprocessed_images)
        self.is_batch_set = True

    def predict(
        self,
        point_coords: Optional[List[np.ndarray]] = None,
        point_labels: Optional[List[np.ndarray]] = None,
        boxes: Optional[List[np.ndarray]] = None,
        masks_input: Optional[List[np.ndarray]] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
        return_mask_embeddings: bool = False,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """
        Predict masks for the given input prompts, using the currently set of images.

        Arguments:
          point_coords (np.ndarray or None): A Nx2 array of point prompts to the
            model. Each point is in (X,Y) in pixels.
          point_labels (np.ndarray or None): A length N array of labels for the
            point prompts. 1 indicates a foreground point and 0 indicates a
            background point.
          box (np.ndarray or None): A length 4 array given a box prompt to the
            model, in XYXY format.
          mask_input (np.ndarray): A low resolution mask input to the model, typically
            coming from a previous prediction iteration. Has form 1xHxW, where
            for SAM, H=W=256.
          multimask_output (bool): If true, the model will return three masks.
            For ambiguous input prompts (such as a single click), this will often
            produce better masks than a single prediction. If only a single
            mask is needed, the model's predicted quality score can be used
            to select the best mask. For non-ambiguous prompts, such as multiple
            input prompts, multimask_output=False can give better results.
          return_logits (bool): If true, returns un-thresholded masks logits
            instead of a binary mask.

        Returns:
          (np.ndarray): The output masks in CxHxW format, where C is the
            number of masks, and (H, W) is the original image size.
          (np.ndarray): An array of length C containing the model's
            predictions for the quality of each mask.
          (np.ndarray): An array of shape CxHxW, where C is the number
            of masks and H=W=256. These low resolution logits can be passed to
            a subsequent iteration as mask input.
        """
        if not self.is_batched_set:
            raise RuntimeError("A batch must be set with .set_images(...) before mask prediction.")
        
        batch_size = len(self.original_sizes)
        batch_coords, batch_labels, batch_boxes, batch_masks, batch_prompt_image_indices = [], [], [], [], []

        for i in range(batch_size):
            num_prompts_for_current_image = 0

            if point_coords is not None and point_coords[i] is not None:
                assert (
                    point_labels is not None and point_labels[i] is not None
                ), "point_labels must be supplied if point_cords is supplied "
                point_coords = self.transform.apply_coords(point_coords, self.original_sizes[i])
                coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=self.device)
                labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=self.device)
                coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
                num_prompts_for_current_image += 1
            else:
                coords_torch = labels_torch = None

            if boxes is not None and boxes[i] is not None:
                box = self.transform.apply_boxes(boxes[i], self.original_sizes[i])
                box_torch = torch.as_tensor(box, dtype=torch.float, device=self.device)
                box_torch = box_torch[None, :]
            else:
                box_torch = None

            if masks_input is not None and masks_input[i] is not None:
                mask_input_torch = torch.as_tensor(masks_input[i], dtype=torch.float, device=self.device)
                mask_input_torch = mask_input_torch[None, :, :, :]
            else:
                mask_input_torch = None

        batch_coords.append(coords_torch)
        batch_labels.append(labels_torch)
        batch_boxes.append(box_torch)
        batch_masks.append(mask_input_torch)

        if num_prompts_for_current_image > 0:
            batch_prompt_image_indices.extend([i] * num_prompts_for_current_image)
        
        # filter out Nones
        coords_for_cat = [t for t in batch_coords if t is not None]
        labels_for_cat = [t for t in batch_labels if t is not None]
        boxes_for_cat = [t for t in batch_boxes if t is not None]
        masks_for_cat = [t for t in batch_masks if t is not None]

        # apply torch.cat
        final_coords_torch = torch.cat(coords_for_cat, dim=0) if coords_for_cat else None
        final_labels_torch = torch.cat(labels_for_cat, dim=0) if labels_for_cat else None
        final_boxes_torch = torch.cat(boxes_for_cat, dim=0) if boxes_for_cat else None
        final_masks_torch = torch.cat(masks_for_cat, dim=0) if masks_for_cat else None

        final_prompt_image_indices_torch = torch.as_tensor(batch_prompt_image_indices, dtype=torch.long, device=self.device)

        masks_t, iou_predictions_t, low_res_masks_t = self.predict_torch_batch(
            batch_idx=final_prompt_image_indices_torch,
            point_coords=final_coords_torch,
            point_labels=final_labels_torch,
            boxes=final_boxes_torch,
            mask_inputs=final_masks_torch,
            multimask_output=multimask_output,
            return_logits=return_logits,
        )

        return (
            masks_t.detach().cpu(),
            iou_predictions_t.detach().cpu(),
            low_res_masks_t.detach().cpu(),
            final_prompt_image_indices_torch.detach().cpu()
        )

    @torch.no_grad()
    def predict_torch_batch(
        self,
        batch_idx: torch.Tensor,  # (Total_Prompts,) indices to original image batch
        point_coords: Optional[torch.Tensor],  # (Total_Prompts, N_points, 2)
        point_labels: Optional[torch.Tensor],  # (Total_Prompts, N_points)
        boxes: Optional[torch.Tensor] = None,  # (Total_Prompts, 4)
        mask_inputs: Optional[torch.Tensor] = None,  # (Total_Prompts, 1, H_lowres, W_lowres)
        multimask_output: bool = True,
        return_logits: bool = False,
        return_mask_embeddings: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # Assuming no mask embeddings for now
        """
        Predicts masks for a batch of prompts across potentially multiple images.
        The image embeddings must be set first using `set_images`.

        Arguments:
          batch_idx (torch.Tensor): A 1D tensor of shape (total_prompts,)
                                    indicating which image in the batch each prompt corresponds to.
                                    Values should be from 0 to len(batched_images) - 1.
          point_coords (torch.Tensor or None): A Total_Prompts x N_points x 2 array of point prompts.
          point_labels (torch.Tensor or None): A Total_Prompts x N_points array of labels for the point prompts.
          boxes (torch.Tensor or None): A Total_Prompts x 4 array given box prompts.
          mask_inputs (torch.Tensor): A Total_Prompts x 1 x H_lowres x W_lowres low resolution mask input.
          multimask_output (bool): If true, the model will return three masks per prompt.
          return_logits (bool): If true, returns un-thresholded masks logits.
          return_mask_embeddings (bool): If true, returns mask embeddings.

        Returns:
          (torch.Tensor): The output masks, shape (Total_Prompts, C_masks, H_orig, W_orig) or
                          (Total_Prompts * C_masks, H_orig, W_orig) depending on how you flatten.
                          Let's return (Total_Prompts, C_masks, H_orig, W_orig) for now.
          (torch.Tensor): IoU predictions of shape (Total_Prompts, C_masks).
          (torch.Tensor): Low resolution masks of shape (Total_Prompts, C_masks, H_lowres, W_lowres).
        """
        if not self.is_batch_set or self.features is None:
            raise RuntimeError("A batch of images must be set with .set_images(...) before mask prediction.")

        # Handle case with no prompts
        if (point_coords is None or point_coords.numel() == 0) and \
           (boxes is None or boxes.numel() == 0) and \
           (mask_inputs is None or mask_inputs.numel() == 0):
            # Return empty tensors that match expected output shapes
            return (
                torch.empty(0, 0, 0, 0, device=self.device), # Masks
                torch.empty(0, 0, device=self.device),     # IoU predictions
                torch.empty(0, 0, 0, 0, device=self.device)  # Low res masks
            )

        total_prompts = batch_idx.shape[0] # Number of individual prompts/inputs in this combined batch

        # Select image embeddings for the prompts based on batch_idx
        selected_image_embeddings = self.features[batch_idx] # (Total_Prompts, C, H_emb, W_emb)

        # Prepare points tuple for prompt_encoder (coords, labels)
        points_for_encoder = (point_coords, point_labels) if point_coords is not None else None

        # Embed prompts
        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
            points=points_for_encoder,
            boxes=boxes,
            masks=mask_inputs,
        )
        # sparse_embeddings: (Total_Prompts, N_sparse, embed_dim)
        # dense_embeddings: (Total_Prompts, embed_dim, H_dense, W_dense)

        # Prepare image_pe for mask_decoder (it needs to be broadcastable or repeated)
        # model.prompt_encoder.get_dense_pe() is (1, C, H_pe, W_pe)
        # We need to expand it to match the batch size of selected_image_embeddings (Total_Prompts)
        image_pe = self.model.prompt_encoder.get_dense_pe().repeat(total_prompts, 1, 1, 1)

        # Predict masks
        if return_mask_embeddings:
            low_res_masks, iou_predictions, mask_tokens_out = self.model.mask_decoder(
                image_embeddings=selected_image_embeddings,
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
                return_mask_embeddings=True,
            )
        else:
            low_res_masks, iou_predictions = self.model.mask_decoder(
                image_embeddings=selected_image_embeddings,
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )
        # low_res_masks: (Total_Prompts, C_masks, H_lowres, W_lowres) where C_masks is 1 or 3
        # iou_predictions: (Total_Prompts, C_masks)

        # Upscale the masks to the original image resolution
        # This requires iterating per prompt in the batch, as original_size and input_size vary.
        upscaled_masks_list = []
        for i in range(total_prompts):
            original_img_idx = batch_idx[i].item() # Get the original image index for this specific prompt
            
            original_size = self.batched_original_sizes[original_img_idx]
            input_size_after_resize = self.batched_input_sizes[original_img_idx] # Size after ResizeLongestSide

            # Select the masks for the current prompt.
            # low_res_masks[i] is (C_masks, H_lowres, W_lowres)
            masks_for_current_prompt = low_res_masks[i]
            
            # Postprocess these masks: Sam's `postprocess_masks` expects (B, C, H, W) for `masks`.
            # We pass it with a dummy batch dimension of 1.
            upscaled_mask = self.model.postprocess_masks(
                masks_for_current_prompt.unsqueeze(0), # Add a dummy batch dim for the internal call
                input_size_after_resize, # Input size to the image encoder (after ResizeLongestSide)
                original_size,           # Original image size
            )
            upscaled_masks_list.append(upscaled_mask)
        
        # Concatenate all upscaled masks back into a single batch tensor
        # Result will be (Total_Prompts, C_masks, H_orig, W_orig)
        final_upscaled_masks = torch.cat(upscaled_masks_list, dim=0)

        # Apply threshold if not returning logits
        if not return_logits:
            final_upscaled_masks = final_upscaled_masks > self.model.mask_threshold

        if return_mask_embeddings:
            # You might need to flatten mask_tokens_out if it's (Total_Prompts, C_masks, Embed_dim)
            # or handle it based on how SamAutomaticMaskGenerator expects it.
            # For now, let's assume it's also (Total_Prompts, C_masks, Embed_dim)
            return final_upscaled_masks, iou_predictions, low_res_masks, mask_tokens_out
        else:
            return final_upscaled_masks, iou_predictions, low_res_masks

    
    @property
    def device(self) -> torch.device:
        return self.model.device

    def reset_image(self) -> None:
        """Resets the currently set image."""
        self.is_image_set = False
        self.features = None
        self.orig_h = None
        self.orig_w = None
        self.input_h = None
        self.input_w = None