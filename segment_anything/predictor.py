# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch

from segment_anything.modeling import Sam

from typing import Optional, Tuple, List, Dict, Union

from .utils.transforms import ResizeLongestSide


class SamPredictor:
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

    def set_image(
        self,
        image: np.ndarray,
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
        assert image_format in [
            "RGB",
            "BGR",
        ], f"image_format must be in ['RGB', 'BGR'], is {image_format}."
        if image_format != self.model.image_format:
            image = image[..., ::-1]

        # Transform the image to the form expected by the model
        input_image = self.transform.apply_image(image)
        input_image_torch = torch.as_tensor(input_image, device=self.device)
        input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]

        self.set_torch_image(input_image_torch, image.shape[:2])

    @torch.no_grad()
    def set_torch_image(
        self,
        transformed_image: torch.Tensor,
        original_image_size: Tuple[int, ...],
    ) -> None:
        """
        Calculates the image embeddings for the provided image, allowing
        masks to be predicted with the 'predict' method. Expects the input
        image to be already transformed to the format expected by the model.

        Arguments:
          transformed_image (torch.Tensor): The input image, with shape
            1x3xHxW, which has been transformed with ResizeLongestSide.
          original_image_size (tuple(int, int)): The size of the image
            before transformation, in (H, W) format.
        """
        assert (
            len(transformed_image.shape) == 4
            and transformed_image.shape[1] == 3
            and max(*transformed_image.shape[2:]) == self.model.image_encoder.img_size
        ), f"set_torch_image input must be BCHW with long side {self.model.image_encoder.img_size}."
        self.reset_image()

        self.original_size = original_image_size
        self.input_size = tuple(transformed_image.shape[-2:])
        input_image = self.model.preprocess(transformed_image)
        self.features = self.model.image_encoder(input_image)
        self.is_image_set = True

    def predict(
        self,
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None,
        mask_input: Optional[np.ndarray] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict masks for the given input prompts, using the currently set image.

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
        if not self.is_image_set:
            raise RuntimeError("An image must be set with .set_image(...) before mask prediction.")

        # Transform input prompts
        coords_torch, labels_torch, box_torch, mask_input_torch = None, None, None, None
        if point_coords is not None:
            assert (
                point_labels is not None
            ), "point_labels must be supplied if point_coords is supplied."
            point_coords = self.transform.apply_coords(point_coords, self.original_size)
            coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=self.device)
            labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=self.device)
            coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
        if box is not None:
            box = self.transform.apply_boxes(box, self.original_size)
            box_torch = torch.as_tensor(box, dtype=torch.float, device=self.device)
            box_torch = box_torch[None, :]
        if mask_input is not None:
            mask_input_torch = torch.as_tensor(mask_input, dtype=torch.float, device=self.device)
            mask_input_torch = mask_input_torch[None, :, :, :]

        masks, iou_predictions, low_res_masks = self.predict_torch(
            coords_torch,
            labels_torch,
            box_torch,
            mask_input_torch,
            multimask_output,
            return_logits=return_logits,
        )

        masks_np = masks[0].detach().cpu().numpy()
        iou_predictions_np = iou_predictions[0].detach().cpu().numpy()
        low_res_masks_np = low_res_masks[0].detach().cpu().numpy()
        return masks_np, iou_predictions_np, low_res_masks_np

    @torch.no_grad()
    def predict_torch(
        self,
        point_coords: Optional[torch.Tensor],
        point_labels: Optional[torch.Tensor],
        boxes: Optional[torch.Tensor] = None,
        mask_input: Optional[torch.Tensor] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
        return_mask_embeddings: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict masks for the given input prompts, using the currently set image.
        Input prompts are batched torch tensors and are expected to already be
        transformed to the input frame using ResizeLongestSide.

        Arguments:
          point_coords (torch.Tensor or None): A BxNx2 array of point prompts to the
            model. Each point is in (X,Y) in pixels.
          point_labels (torch.Tensor or None): A BxN array of labels for the
            point prompts. 1 indicates a foreground point and 0 indicates a
            background point.
          boxes (np.ndarray or None): A Bx4 array given a box prompt to the
            model, in XYXY format.
          mask_input (np.ndarray): A low resolution mask input to the model, typically
            coming from a previous prediction iteration. Has form Bx1xHxW, where
            for SAM, H=W=256. Masks returned by a previous iteration of the
            predict method do not need further transformation.
          multimask_output (bool): If true, the model will return three masks.
            For ambiguous input prompts (such as a single click), this will often
            produce better masks than a single prediction. If only a single
            mask is needed, the model's predicted quality score can be used
            to select the best mask. For non-ambiguous prompts, such as multiple
            input prompts, multimask_output=False can give better results.
          return_logits (bool): If true, returns un-thresholded masks logits
            instead of a binary mask.

        Returns:
          (torch.Tensor): The output masks in BxCxHxW format, where C is the
            number of masks, and (H, W) is the original image size.
          (torch.Tensor): An array of shape BxC containing the model's
            predictions for the quality of each mask.
          (torch.Tensor): An array of shape BxCxHxW, where C is the number
            of masks and H=W=256. These low res logits can be passed to
            a subsequent iteration as mask input.
        """
        if not self.is_image_set:
            raise RuntimeError("An image must be set with .set_image(...) before mask prediction.")

        if point_coords is not None:
            points = (point_coords, point_labels)
        else:
            points = None

        # Embed prompts
        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
            points=points,
            boxes=boxes,
            masks=mask_input,
        )

        # Predict masks
        if return_mask_embeddings:
          low_res_masks, iou_predictions, mask_tokens_out = self.model.mask_decoder(
              image_embeddings=self.features,
              image_pe=self.model.prompt_encoder.get_dense_pe(),
              sparse_prompt_embeddings=sparse_embeddings,
              dense_prompt_embeddings=dense_embeddings,
              multimask_output=multimask_output,
              return_mask_embeddings=True,
          )
        else:
          low_res_masks, iou_predictions = self.model.mask_decoder(
              image_embeddings=self.features,
              image_pe=self.model.prompt_encoder.get_dense_pe(),
              sparse_prompt_embeddings=sparse_embeddings,
              dense_prompt_embeddings=dense_embeddings,
              multimask_output=multimask_output,
          )

        # Upscale the masks to the original image resolution
        masks = self.model.postprocess_masks(low_res_masks, self.input_size, self.original_size)

        if not return_logits:
            masks = masks > self.model.mask_threshold
        if return_mask_embeddings:
          return masks, iou_predictions, low_res_masks, mask_tokens_out
        else:
          return masks, iou_predictions, low_res_masks

    def get_image_embedding(self) -> torch.Tensor:
        """
        Returns the image embeddings for the currently set image, with
        shape 1xCxHxW, where C is the embedding dimension and (H,W) are
        the embedding spatial dimension of SAM (typically C=256, H=W=64).
        """
        if not self.is_image_set:
            raise RuntimeError(
                "An image must be set with .set_image(...) to generate an embedding."
            )
        assert self.features is not None, "Features must exist if an image has been set."
        return self.features

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

    def set_images_batch(
        self,
        images: List[np.ndarray],
        image_format: str = "RGB",
        return_transformer_only: bool = False,
    ) -> List[Dict[str, Union[Tuple[int, int], torch.Tensor]]]:
        assert image_format in [
            "RGB",
            "BGR",
        ], f"image_format must be in ['RGB', 'BGR'], is {image_format}."

        original_sizes: List[Tuple[int, ...]] = []
        input_sizes: List[Tuple[int, ...]] = []
        transformed_images_list: List[torch.Tensor] = []

        # Determine the target size for padding based on the model's expected image size
        # This assumes your model.image_encoder.img_size is the target square dimension (e.g., 1024)
        model_img_size = self.model.image_encoder.img_size # e.g., 1024
        target_h, target_w = model_img_size, model_img_size # Pad to a square

        for image in images:
            current_image = image
            if image_format != self.model.image_format:
                current_image = current_image[..., ::-1]

            original_sizes.append(current_image.shape[:2])

            # Apply transformation (e.g., ResizeLongestSide)
            # This step resizes the longest side to `model_img_size`
            # The output `input_image_np` will have varying H, W based on aspect ratio
            input_image_np = self.transform.apply_image(current_image)
            input_sizes.append(input_image_np.shape[:2])

            # Convert to PyTorch tensor and permute dimensions (HWC -> CHW)
            input_image_torch = torch.as_tensor(input_image_np, device=self.device)
            input_image_torch = input_image_torch.permute(2, 0, 1).contiguous() # Shape 3xHxW

            # --- Add Padding Here ---
            # Calculate padding needed to reach target_h x target_w
            # PyTorch's F.pad expects (left, right, top, bottom)
            # This pads the image to a square shape, typically 1024x1024 for SAM
            pad_h = max(0, target_h - input_image_torch.shape[1])
            pad_w = max(0, target_w - input_image_torch.shape[2])
            
            # Apply padding. SAM typically pads with 0s.
            # (left, right, top, bottom) -> (pad_left, pad_right, pad_top, pad_bottom)
            # For simplicity, we can pad symmetrically or just to the bottom/right
            # The original SAM's `model.preprocess` uses F.pad which handles the full padding logic.
            # You might want to replicate SAM's exact padding if it's crucial for the model.
            # A common approach is to pad to the bottom-right.
            padded_image_torch = torch.nn.functional.pad(
                input_image_torch,
                (0, pad_w, 0, pad_h), # (left, right, top, bottom)
                mode='constant',
                value=0
            )
            # --- End Padding ---

            transformed_images_list.append(padded_image_torch)

        # Stack all transformed and padded images into a single batch tensor (NxCxHxW)
        # Now all tensors in transformed_images_list should be of the same HxW (e.g., 1024x1024)
        transformed_images_batch = torch.stack(transformed_images_list, dim=0)

        # Process the batched images through the model
        results = self._process_torch_images_batch(transformed_images_batch, original_sizes, input_sizes, return_transformer_only)
        return results

    @torch.no_grad()
    def _process_torch_images_batch(
        self,
        transformed_images: torch.Tensor,
        original_image_sizes: List[Tuple[int, ...]],
        input_image_sizes: List[Tuple[int, ...]],
        return_transformer_only: bool = False,
    ) -> List[Dict[str, Union[Tuple[int, int], Tuple[int, int], torch.Tensor]]]:
        """
        Internal helper function to calculate image embeddings for a batch of
        already transformed images.

        Arguments:
          transformed_images (torch.Tensor): A batch of input images, with shape
            Nx3xHxW, which have been transformed (e.g., with ResizeLongestSide).
          original_image_sizes (List[tuple(int, int)]): A list of the original sizes
            of the images before transformation, in (H, W) format, corresponding
            to the order in `transformed_images`.

        Returns:
          List[Dict[str, Union[Tuple[int, int], Tuple[int, int], torch.Tensor]]]:
            A list of dictionaries, where each dictionary corresponds to an input image
            and contains:
            - 'original_size': The original (H, W) size of the image before transformation.
            - 'input_size': The (H, W) size of the image after transformation, as input to the model.
            - 'features': The torch.Tensor containing the image embeddings for that specific image.
        """
        num_images_in_batch = transformed_images.shape[0]
        assert (
            len(transformed_images.shape) == 4
            and transformed_images.shape[1] == 3
            and max(*transformed_images.shape[2:]) == self.model.image_encoder.img_size
        ), (
            f"'_process_torch_images_batch' input must be Nx3xHxW where long side "
            f"is {self.model.image_encoder.img_size}. Got shape {transformed_images.shape}."
        )
        assert len(original_image_sizes) == num_images_in_batch, (
            f"Number of original image sizes ({len(original_image_sizes)}) must match "
            f"the batch size of transformed images ({num_images_in_batch})."
        )

        # Preprocess the batched images
        # This typically involves normalization and moving to model's expected input range
        input_images_preprocessed = self.model.preprocess(transformed_images)

        # Pass the batched images through the image encoder
        # This will return features with shape (N, C, H_feat, W_feat)
        x = self.model.image_encoder.patch_embed(input_images_preprocessed)
        if self.model.image_encoder.pos_embed is not None:
          x = x + self.model.image_encoder.pos_embed

        for blk in self.model.image_encoder.blocks:
          x = blk(x)

        # mimic image encoder forward pass until last transformer block
        if return_transformer_only:
            return x

        last_transformer_block = x.to('cpu').numpy()
        features_batch = self.model.image_encoder.neck(x.permute(0, 3, 1, 2))


        results: List[Dict[str, Union[Tuple[int, int], Tuple[int, int], torch.Tensor]]] = []
        for i in range(num_images_in_batch):
            results.append({
                'original_size': original_image_sizes[i],
                'input_size': input_image_sizes[i],
                'features': features_batch[i], # Extract features for each individual image
                'last_transformer_block': last_transformer_block[i]
            })
        return results
    
    def set_calculated_image(self, input_size, original_image_size, features):
        self.reset_image()

        self.original_size = original_image_size
        self.input_size = input_size
        self.features = features
        self.is_image_set = True