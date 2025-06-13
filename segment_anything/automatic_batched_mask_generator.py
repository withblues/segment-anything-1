# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from torchvision.ops.boxes import batched_nms, box_area  # type: ignore

from typing import Any, Dict, List, Optional, Tuple

from .modeling import Sam
from .batched_predictor import BatchedSamPredictor
from .utils.amg import (
    MaskData,
    area_from_rle,
    batch_iterator,
    batched_mask_to_box,
    box_xyxy_to_xywh,
    build_all_layer_point_grids,
    calculate_stability_score,
    coco_encode_rle,
    generate_crop_boxes,
    is_box_near_crop_edge,
    mask_to_rle_pytorch,
    remove_small_regions,
    rle_to_mask,
    uncrop_boxes_xyxy,
    uncrop_masks,
    uncrop_points,
)


class SamAutomaticMaskGenerator:
    def __init__(
        self,
        model: Sam,
        points_per_side: Optional[int] = 32,
        points_per_batch: int = 64,
        pred_iou_thresh: float = 0.88,
        stability_score_thresh: float = 0.95,
        stability_score_offset: float = 1.0,
        box_nms_thresh: float = 0.7,
        crop_n_layers: int = 0,
        crop_nms_thresh: float = 0.7,
        crop_overlap_ratio: float = 512 / 1500,
        crop_n_points_downscale_factor: int = 1,
        point_grids: Optional[List[np.ndarray]] = None,
        min_mask_region_area: int = 0,
        output_mode: str = "binary_mask",
    ) -> None:
        """
        Using a SAM model, generates masks for the entire image.
        Generates a grid of point prompts over the image, then filters
        low quality and duplicate masks. The default settings are chosen
        for SAM with a ViT-H backbone.

        Arguments:
          model (Sam): The SAM model to use for mask prediction.
          points_per_side (int or None): The number of points to be sampled
            along one side of the image. The total number of points is
            points_per_side**2. If None, 'point_grids' must provide explicit
            point sampling.
          points_per_batch (int): Sets the number of points run simultaneously
            by the model. Higher numbers may be faster but use more GPU memory.
          pred_iou_thresh (float): A filtering threshold in [0,1], using the
            model's predicted mask quality.
          stability_score_thresh (float): A filtering threshold in [0,1], using
            the stability of the mask under changes to the cutoff used to binarize
            the model's mask predictions.
          stability_score_offset (float): The amount to shift the cutoff when
            calculated the stability score.
          box_nms_thresh (float): The box IoU cutoff used by non-maximal
            suppression to filter duplicate masks.
          crop_n_layers (int): If >0, mask prediction will be run again on
            crops of the image. Sets the number of layers to run, where each
            layer has 2**i_layer number of image crops.
          crop_nms_thresh (float): The box IoU cutoff used by non-maximal
            suppression to filter duplicate masks between different crops.
          crop_overlap_ratio (float): Sets the degree to which crops overlap.
            In the first crop layer, crops will overlap by this fraction of
            the image length. Later layers with more crops scale down this overlap.
          crop_n_points_downscale_factor (int): The number of points-per-side
            sampled in layer n is scaled down by crop_n_points_downscale_factor**n.
          point_grids (list(np.ndarray) or None): A list over explicit grids
            of points used for sampling, normalized to [0,1]. The nth grid in the
            list is used in the nth crop layer. Exclusive with points_per_side.
          min_mask_region_area (int): If >0, postprocessing will be applied
            to remove disconnected regions and holes in masks with area smaller
            than min_mask_region_area. Requires opencv.
          output_mode (str): The form masks are returned in. Can be 'binary_mask',
            'uncompressed_rle', or 'coco_rle'. 'coco_rle' requires pycocotools.
            For large resolutions, 'binary_mask' may consume large amounts of
            memory.
        """

        assert (points_per_side is None) != (
            point_grids is None
        ), "Exactly one of points_per_side or point_grid must be provided."
        if points_per_side is not None:
            self.point_grids = build_all_layer_point_grids(
                points_per_side,
                crop_n_layers,
                crop_n_points_downscale_factor,
            )
        elif point_grids is not None:
            self.point_grids = point_grids
        else:
            raise ValueError("Can't have both points_per_side and point_grid be None.")

        assert output_mode in [
            "binary_mask",
            "uncompressed_rle",
            "coco_rle",
        ], f"Unknown output_mode {output_mode}."
        if output_mode == "coco_rle":
            from pycocotools import mask as mask_utils  # type: ignore # noqa: F401

        # if min_mask_region_area > 0:
        #     import cv2  # type: ignore # noqa: F401

        self.predictor = BatchedSamPredictor(model)
        self.points_per_batch = points_per_batch
        self.pred_iou_thresh = pred_iou_thresh
        self.stability_score_thresh = stability_score_thresh
        self.stability_score_offset = stability_score_offset
        self.box_nms_thresh = box_nms_thresh
        self.crop_n_layers = crop_n_layers
        self.crop_nms_thresh = crop_nms_thresh
        self.crop_overlap_ratio = crop_overlap_ratio
        self.crop_n_points_downscale_factor = crop_n_points_downscale_factor
        self.min_mask_region_area = min_mask_region_area
        self.output_mode = output_mode

    @torch.no_grad()
    def generate(self, images: List[np.ndarray]) -> List[List[Dict[str, Any]]]:
        """
        Generates masks for the given image.

        Arguments:
          image (np.ndarray): The image to generate masks for, in HWC uint8 format.

        Returns:
           list(dict(str, any)): A list over records for masks. Each record is
             a dict containing the following keys:
               segmentation (dict(str, any) or np.ndarray): The mask. If
                 output_mode='binary_mask', is an array of shape HW. Otherwise,
                 is a dictionary containing the RLE.
               bbox (list(float)): The box around the mask, in XYWH format.
               area (int): The area in pixels of the mask.
               predicted_iou (float): The model's own prediction of the mask's
                 quality. This is filtered by the pred_iou_thresh parameter.
               point_coords (list(list(float))): The point coordinates input
                 to the model to generate this mask.
               stability_score (float): A measure of the mask's quality. This
                 is filtered on using the stability_score_thresh parameter.
               crop_box (list(float)): The crop of the image used to generate
                 the mask, given in XYWH format.
        """

        # Generate masks
        all_image_mask_data = self._generate_batched_masks(images)

        # Post-processing and formatting for each image
        final_results_per_image = []
        for mask_data_for_single_image in all_image_mask_data:
            # Re-use your existing generate_curr_anns logic for each image's MaskData
            # This handles min_mask_region_area, encoding, etc.
            curr_anns = self.generate_curr_anns(mask_data_for_single_image)
            final_results_per_image.append(curr_anns)
        
        return final_results_per_image


    def generate_curr_anns(
            self,
            mask_data,

    ) -> List[Dict[str, Any]]:

        # Filter small disconnected regions and holes in masks
        if self.min_mask_region_area > 0:
            mask_data = self.postprocess_small_regions(
                mask_data,
                self.min_mask_region_area,
                max(self.box_nms_thresh, self.crop_nms_thresh),
            )

        # Encode masks
        if self.output_mode == "coco_rle":
            mask_data["segmentations"] = [coco_encode_rle(rle) for rle in mask_data["rles"]]
        elif self.output_mode == "binary_mask":
            mask_data["segmentations"] = [rle_to_mask(rle) for rle in mask_data["rles"]]
        else:
            mask_data["segmentations"] = mask_data["rles"]

        # Write mask records
        curr_anns = []
        has_embeddings = 'embeddings' in mask_data._stats
        for idx in range(len(mask_data["segmentations"])):
            ann = {
                "segmentation": mask_data["segmentations"][idx],
                "area": area_from_rle(mask_data["rles"][idx]),
                "bbox": box_xyxy_to_xywh(mask_data["boxes"][idx]).tolist(),
                "predicted_iou": mask_data["iou_preds"][idx].item(),
                "point_coords": [mask_data["points"][idx].tolist()],
                "stability_score": mask_data["stability_score"][idx].item(),
                "crop_box": box_xyxy_to_xywh(mask_data["crop_boxes"][idx]).tolist(),
            }
            
            if has_embeddings:
                ann["embeddings"] = mask_data["embeddings"][idx]
  
            curr_anns.append(ann)

        return curr_anns

    def _generate_batched_masks(self, images: List[np.ndarray]) -> List[MaskData]:
        all_crop_info_for_predictor = []
        all_cropped_images_np = []

        for original_image_idx, img_np in enumerate(images):
            orig_size = img_np.shape[:2]
            crop_boxes, layer_idxs = generate_crop_boxes(
                orig_size, self.crop_n_layers, self.crop_overlap_ratio
            )

            for crop_box, layer_idx in zip(crop_boxes, layer_idxs):
                x0, y0, x1, y1 = crop_box
                cropped_im_np = img_np[y0:y1, x0:x1, :]
                cropped_im_size = cropped_im_np.shape[:2]

                all_cropped_images_np.append(cropped_im_np)
                all_crop_info_for_predictor.append((original_image_idx, crop_box, layer_idx, orig_size, cropped_im_size))


        # set images
        self.predictor.set_images(all_cropped_images_np)

        # collect all prompts for each image
        all_prompts_coords_transformed = []
        all_prompts_labels = []

        # for mapping
        prompt_to_original_image_idx = []
        prompt_to_predictor_crop_idx = []

        for predictor_crop_idx, (original_image_idx, crop_box, layer_idx, orig_size, cropped_im_size) in enumerate(all_crop_info_for_predictor):
            points_scale = np.array(cropped_im_np)[None, ::-1]
            points_for_image = self.point_grids[layer_idx] * points_scale
            
            # just like in _process_batch()
            for (points_batch_np,) in batch_iterator(self.points_per_batch, points_for_image):
                transformed_points_batch_np = self.predictor.transform.apply_coords(points_batch_np, cropped_im_size)

                all_prompts_coords_transformed.append(torch.as_tensor(transformed_points_batch_np, device=self.predictor.device))
                all_prompts_labels.append(torch.ones(transformed_points_batch_np.shape[0], dtype=torch.int,  device=self.predictor.device))

                # store mapping indices
                num_points_in_batch = transformed_points_batch_np.shape[0]
                prompt_to_original_image_idx.extend([original_image_idx] * num_points_in_batch)
                prompt_to_predictor_crop_idx.extend([predictor_crop_idx] * num_points_in_batch)

        # create MaskData
        #TODO

        # create batch of prompts
        final_prompts_coords = torch.cat([p[None, :, :] for p in all_prompts_coords_transformed], dim=0)
        final_prompts_labels = torch.cat([l[None, :] for l in all_prompts_labels], dim=0)

        final_prompt_original_image_indices_t = torch.as_tensor(prompt_to_original_image_idx, dtype=torch.long, device=self.predictor.device)
        final_prompt_predictor_crop_indices_t = torch.as_tensor(prompt_to_predictor_crop_idx, dtype=torch.long, device=self.predictor.device)

        masks_t, iou_preds_t, low_res_masks_t, mask_tokens_out_t = self.predictor.predict_torch_batch(
            batch_idx=final_prompt_predictor_crop_indices_t,
            point_coords=final_prompts_coords,
            point_labels=final_prompts_labels,
            multimask_output=True,
            return_logits=True,
            return_mask_embeddings=True
        )


        # flat out prediction
        num_multimasks = masks_t.shape[1]
        flat_masks = masks_t.flatten(0, 1)
        flat_iou_preds = iou_preds_t.flatten(0, 1)
        flat_points_coords = torch.repeat_interleave(final_prompts_coords, num_multimasks, dim=0)

        flat_original_image_idx = torch.repeat_interleave(final_prompt_original_image_indices_t, num_multimasks, dim=0)
        flat_predictor_crop_idx = torch.repeat_interleave(final_prompt_predictor_crop_indices_t, num_multimasks, dim=0)

        # get embeddings 
        flat_embeddings = mask_tokens_out_t.flatten(0, 1) if 'mask_tokens_out_t' in locals() else None

        temp_all_mask_data = MaskData(
            masks=flat_masks,
            iou_preds=flat_iou_preds,
            points=flat_points_coords.squeeze(1).cpu().numpy(),
            original_image_idx=flat_original_image_idx,
            predictor_crop_idx=flat_predictor_crop_idx,
            embeddings=flat_embeddings,
        )
        
        # process mask like _process_batch_data
        temp_all_mask_data = self._apply_initial_filters_and_boxes_batched(temp_all_mask_data, all_crop_info_for_predictor)

        final_image_mask_data_list = [MaskData() for _ in range(len(images))]
        unique_image_indices = torch.unique(temp_all_mask_data["original_image_idx"])

        for img_idx in unique_image_indices:
            img_idx_mask = (temp_all_mask_data["original_image_idx"] == img_idx)

            current_image_data = MaskData()
            current_image_data["masks"] = temp_all_mask_data["masks"][img_idx_mask]
            current_image_data["iou_preds"] = temp_all_mask_data["iou_preds"][img_idx_mask]
            current_image_data["points"] = temp_all_mask_data["points"][img_idx_mask]
            current_image_data["boxes"] = temp_all_mask_data["boxes"][img_idx_mask] 
            current_image_data["stability_score"] = temp_all_mask_data["stability_score"][img_idx_mask]
            current_image_data["rles"] = temp_all_mask_data["rles"][img_idx_mask]

            # add crop boxes
            processed_mask_data = MaskData(
                masks=flat_masks,
                iou_preds=flat_iou_preds,
                points=flat_points_coords.squeeze(1).cpu().numpy(),
                original_image_idx=flat_original_image_idx,
                predictor_crop_idx=flat_predictor_crop_idx,
                embeddings=flat_embeddings,
            )

            processed_mask_data = self._process_all_batched_prompts_and_masks(
                processed_mask_data, all_crop_info_for_predictor
            )

            # filter through NMS
            final_image_mask_data_list = [MaskData() for _ in range(len(images))]
            unique_original_image_indices = torch.unique(processed_mask_data["original_image_idx"])

            for img_idx in unique_original_image_indices:
                img_mask = (processed_mask_data["original_image_idx"] == img_idx)

                single_image_mask_data = MaskData()
                single_image_mask_data["masks"] = processed_mask_data["masks"][img_mask]
                single_image_mask_data["iou_preds"] = processed_mask_data["iou_preds"][img_mask]
                single_image_mask_data["points"] = processed_mask_data["points"][img_mask]
                single_image_mask_data["boxes"] = processed_mask_data["boxes"][img_mask]
                single_image_mask_data["stability_score"] = processed_mask_data["stability_score"][img_mask]
                single_image_mask_data["rles"] = processed_mask_data["rles"][img_mask]
                single_image_mask_data["crop_boxes"] = processed_mask_data["crop_boxes"][img_mask]

                if len(crop_boxes_for_image) > 1:
                    scores = 1 / box_area(single_image_mask_data["crop_boxes"])
                    scores = scores.to(self.predictor.device)

                    keep_by_nms = batched_nms(
                        single_image_mask_data["boxes"].float(),
                        scores,
                        torch.zeros_like(single_image_mask_data["boxes"][:, 0]), # categories
                        iou_threshold=self.crop_nms_thresh,
                    )
                    single_image_mask_data.filter(keep_by_nms)

                final_image_mask_data_list[img_idx] = single_image_mask_data

            return final_image_mask_data_list
        
    def _process_all_batched_prompts_and_masks(
        self,
        batched_mask_data: MaskData, # Contains flat_masks, flat_iou_preds, flat_points, flat_original_image_idx, flat_predictor_crop_idx
        all_crop_info_for_predictor: List[Tuple[int, List[int], int, Tuple[int,int], Tuple[int,int]]],
    ) -> MaskData:
        """
        Applies initial filtering, calculates stability score, thresholds masks,
        calculates boxes, filters by crop edge, and converts to RLEs for
        the entire batch of mask candidates. Uncrops masks/boxes.
        """
        # Ensure all data is on the correct device for processing
        masks = batched_mask_data["masks"]
        iou_preds = batched_mask_data["iou_preds"]
        points = batched_mask_data["points"] # np array
        original_image_idx = batched_mask_data["original_image_idx"]
        predictor_crop_idx = batched_mask_data["predictor_crop_idx"]
        embeddings = batched_mask_data["embeddings"] if "embeddings" in batched_mask_data._stats else None


        # 1. Filter by predicted IoU
        if self.pred_iou_thresh > 0.0:
            keep_mask_iou = iou_preds > self.pred_iou_thresh
            masks = masks[keep_mask_iou]
            iou_preds = iou_preds[keep_mask_iou]
            points = points[keep_mask_iou.cpu().numpy()] # Filter numpy array
            original_image_idx = original_image_idx[keep_mask_iou]
            predictor_crop_idx = predictor_crop_idx[keep_mask_iou]
            if embeddings is not None: embeddings = embeddings[keep_mask_iou]


        # 2. Calculate stability score
        stability_score = calculate_stability_score(
            masks, self.predictor.model.mask_threshold, self.stability_score_offset
        )
        if self.stability_score_thresh > 0.0:
            keep_mask_stability = stability_score >= self.stability_score_thresh
            masks = masks[keep_mask_stability]
            iou_preds = iou_preds[keep_mask_stability]
            points = points[keep_mask_stability.cpu().numpy()]
            original_image_idx = original_image_idx[keep_mask_stability]
            predictor_crop_idx = predictor_crop_idx[keep_mask_stability]
            stability_score = stability_score[keep_mask_stability]
            if embeddings is not None: embeddings = embeddings[keep_mask_stability]


        # 3. Threshold masks and calculate boxes
        # This also generates boxes in the original image coordinate system
        thresholded_masks = masks > self.predictor.model.mask_threshold
        boxes = batched_mask_to_box(thresholded_masks) # boxes are in original image coordinates
        
        # 4. Filter boxes that touch crop boundaries (this needs individual crop_box and orig_size)
        # This is where a loop or a more complex batched operation is needed.
        keep_mask_crop_edge = torch.ones(len(boxes), dtype=torch.bool, device=self.predictor.device)
        
        # Collect relevant crop_boxes and original_sizes for each remaining mask
        relevant_crop_boxes_for_masks = []
        relevant_orig_sizes_for_masks = []
        
        # This mapping is crucial:
        # map `predictor_crop_idx` (which refers to index in all_cropped_images_np)
        # to the corresponding `all_crop_info_for_predictor` entry
        for i, p_crop_idx in enumerate(predictor_crop_idx):
            _, crop_box, _, orig_size, _ = all_crop_info_for_predictor[p_crop_idx]
            relevant_crop_boxes_for_masks.append(crop_box)
            relevant_orig_sizes_for_masks.append(orig_size) # Not directly used for `is_box_near_crop_edge` but good to have

        # Convert list of lists (crop_box) to a tensor if all have same inner dim, or iterate.
        # `is_box_near_crop_edge` takes a single box and single target_box.
        # We need to iterate this check.
        
        # Loop through the masks and apply crop edge filter individually
        # This could be optimized if `is_box_near_crop_edge` was batched.
        # For now, let's do it sequentially and build a boolean tensor.
        
        final_keep_indices = []
        final_uncropped_crop_boxes = [] # To store the uncropped crop_box for each mask
        
        for i in range(len(boxes)):
            box = boxes[i]
            crop_b = relevant_crop_boxes_for_masks[i]
            orig_s = relevant_orig_sizes_for_masks[i] # orig_h, orig_w
            
            # Check if box is near crop edge (using original image boundaries as target)
            if not is_box_near_crop_edge(box.unsqueeze(0), crop_b, [0, 0, orig_s[1], orig_s[0]]): # target_box is [x0,y0,x1,y1] -> [0,0,W,H]
                final_keep_indices.append(i)
                final_uncropped_crop_boxes.append(crop_b) # Store the original crop box for this mask

        # Filter all tensors by `final_keep_indices`
        masks = masks[final_keep_indices]
        iou_preds = iou_preds[final_keep_indices]
        points = points[np.array(final_keep_indices)] # Filter numpy array with numpy index
        original_image_idx = original_image_idx[final_keep_indices]
        predictor_crop_idx = predictor_crop_idx[final_keep_indices]
        stability_score = stability_score[final_keep_indices]
        boxes = boxes[final_keep_indices]
        if embeddings is not None: embeddings = embeddings[final_keep_indices]
        final_uncropped_crop_boxes_t = torch.as_tensor(final_uncropped_crop_boxes, device=self.predictor.device)


        # 5. Compress to RLE (masks are already uncropped by BatchedSamPredictor)
        rles = mask_to_rle_pytorch(masks) # masks should be (N, H, W) now

        # Create the output MaskData for this stage
        output_mask_data = MaskData(
            masks=masks, # Keep for debugging if needed, but will be removed
            iou_preds=iou_preds,
            points=points, # Already numpy
            stability_score=stability_score,
            boxes=boxes,
            rles=rles,
            original_image_idx=original_image_idx,
            crop_boxes=final_uncropped_crop_boxes_t, # These are the original crop boxes for each final mask
        )
        if embeddings is not None:
            output_mask_data["embeddings"] = embeddings

        del output_mask_data["masks"] # Remove large masks tensor if not needed later

        return output_mask_data

    def _generate_masks_data(
            self,
            data,
            crop_boxes,

    ) -> MaskData:

        # Remove duplicate masks between crops
        if len(crop_boxes) > 1:
            # Prefer masks from smaller crops
            scores = 1 / box_area(data["crop_boxes"])
            scores = scores.to(data["boxes"].device)
            keep_by_nms = batched_nms(
                data["boxes"].float(),
                scores,
                torch.zeros_like(data["boxes"][:, 0]),  # categories
                iou_threshold=self.crop_nms_thresh,
            )
            data.filter(keep_by_nms)

        data.to_numpy()
        return data

    def _process_crop(
        self,
        image: np.ndarray,
        crop_box: List[int],
        crop_layer_idx: int,
        orig_size: Tuple[int, ...],
    ) -> MaskData:
        # Crop the image and calculate embeddings
        x0, y0, x1, y1 = crop_box
        cropped_im = image[y0:y1, x0:x1, :]
        cropped_im_size = cropped_im.shape[:2]
        self.predictor.set_image(cropped_im)

        # Get points for this crop
        points_scale = np.array(cropped_im_size)[None, ::-1]
        points_for_image = self.point_grids[crop_layer_idx] * points_scale

        # Generate masks for this crop in batches
        data = MaskData()
        data_s, data_m, data_l = MaskData(), MaskData(), MaskData()
        for (points,) in batch_iterator(self.points_per_batch, points_for_image):
            batch_data, batch_data_s, batch_data_m, batch_data_l = self._process_batch(points, cropped_im_size, crop_box, orig_size)
            data.cat(batch_data)
            data_s.cat(batch_data_s)
            data_m.cat(batch_data_m)
            data_l.cat(batch_data_l)
            del batch_data, batch_data_s, batch_data_m, batch_data_l
        self.predictor.reset_image()
        data = self._process_crop_data(data, crop_box)
        data_s = self._process_crop_data(data_s, crop_box)
        data_m = self._process_crop_data(data_m, crop_box)
        data_l = self._process_crop_data(data_l, crop_box)
        return data, data_s, data_m, data_l


    def _process_crop_data(
            self,
            data,
            crop_box: List[int],
    ) -> MaskData:
        # Remove duplicates within this crop.
        keep_by_nms = batched_nms(
            data["boxes"].float(),
            data["iou_preds"],
            torch.zeros_like(data["boxes"][:, 0]),  # categories
            iou_threshold=self.box_nms_thresh,
        )
        data.filter(keep_by_nms)

        # Return to the original image frame
        data["boxes"] = uncrop_boxes_xyxy(data["boxes"], crop_box)
        data["points"] = uncrop_points(data["points"], crop_box)
        data["crop_boxes"] = torch.tensor([crop_box for _ in range(len(data["rles"]))])

        return data
    

    def _process_batch(
        self,
        points: np.ndarray,
        im_size: Tuple[int, ...],
        crop_box: List[int],
        orig_size: Tuple[int, ...],
    ) -> MaskData:
        orig_h, orig_w = orig_size

        # Run model on this batch
        transformed_points = self.predictor.transform.apply_coords(points, im_size)
        in_points = torch.as_tensor(transformed_points, device=self.predictor.device)
        in_labels = torch.ones(in_points.shape[0], dtype=torch.int, device=in_points.device)
        masks, iou_preds, low_res_masks, mask_tokens_out = self.predictor.predict_torch(
            in_points[:, None, :],
            in_labels[:, None],
            multimask_output=True,
            return_logits=True,
            return_mask_embeddings=True,
        )

        # print('...................points...........................')
        # print(in_points.shape)
        # print('...................embeddings...........................')
        # print(mask_tokens_out.shape)
        # print('...................low_res_mask...........................')
        # print(low_res_masks.shape)
        # print('...................mask...........................')
        # print(masks.shape)

        # print(masks.shape)torch.Size([64, 3, 738, 994])
        # print(masks.flatten(0, 1).shape)torch.Size([192, 738, 994])
        # print(masks[:,0,:,:].shape)torch.Size([64, 738, 994]
        # print(points.shape)(64, 2)
        # print(torch.as_tensor(points.repeat(masks.shape[1], axis=0)).shape)torch.Size([192, 2])
        # print(iou_preds.shape)torch.Size([64, 3])
        # Serialize predictions and store in MaskData
        # mask type: s m l
        data_s = MaskData(
            masks=masks[:,0,:,:],
            iou_preds=iou_preds[:,0],
            points=torch.as_tensor(points),
            embeddings=mask_tokens_out[:, 0, :],
        )
        data_m = MaskData(
            masks=masks[:,1,:,:],
            iou_preds=iou_preds[:,1],
            points=torch.as_tensor(points),
            embeddings=mask_tokens_out[:, 1, :],
        )
        data_l = MaskData(
            masks=masks[:,2,:,:],
            iou_preds=iou_preds[:,2],
            points=torch.as_tensor(points),
            embeddings=mask_tokens_out[:, 2, :],
        )
        data = MaskData(
            masks=masks.flatten(0, 1),
            iou_preds=iou_preds.flatten(0, 1),
            points=torch.as_tensor(points.repeat(masks.shape[1], axis=0)),
            embeddings=mask_tokens_out.reshape(-1, mask_tokens_out.shape[-1])
        )
        del masks

        data = self._process_batch_data(data, crop_box, orig_w, orig_h)
        data_s = self._process_batch_data(data_s, crop_box, orig_w, orig_h)
        data_m = self._process_batch_data(data_m, crop_box, orig_w, orig_h)
        data_l = self._process_batch_data(data_l, crop_box, orig_w, orig_h)
        return data, data_s, data_m, data_l
    
    def _process_batch_data(
            self,
            data,
            crop_box,
            orig_w,
            orig_h,

    ):
        # Filter by predicted IoU
        if self.pred_iou_thresh > 0.0:
            keep_mask = data["iou_preds"] > self.pred_iou_thresh
            data.filter(keep_mask)

        # Calculate stability score
        data["stability_score"] = calculate_stability_score(
            data["masks"], self.predictor.model.mask_threshold, self.stability_score_offset
        )
        if self.stability_score_thresh > 0.0:
            keep_mask = data["stability_score"] >= self.stability_score_thresh
            data.filter(keep_mask)

        # Threshold masks and calculate boxes
        data["masks"] = data["masks"] > self.predictor.model.mask_threshold
        data["boxes"] = batched_mask_to_box(data["masks"])

        # Filter boxes that touch crop boundaries
        keep_mask = ~is_box_near_crop_edge(data["boxes"], crop_box, [0, 0, orig_w, orig_h])
        if not torch.all(keep_mask):
            data.filter(keep_mask)

        # Compress to RLE
        data["masks"] = uncrop_masks(data["masks"], crop_box, orig_h, orig_w)
        data["rles"] = mask_to_rle_pytorch(data["masks"])
        del data["masks"]

        return data

    @staticmethod
    def postprocess_small_regions(
        mask_data: MaskData, min_area: int, nms_thresh: float
    ) -> MaskData:
        """
        Removes small disconnected regions and holes in masks, then reruns
        box NMS to remove any new duplicates.

        Edits mask_data in place.

        Requires open-cv as a dependency.
        """
        if len(mask_data["rles"]) == 0:
            return mask_data

        # Filter small disconnected regions and holes
        new_masks = []
        scores = []
        for rle in mask_data["rles"]:
            mask = rle_to_mask(rle)

            mask, changed = remove_small_regions(mask, min_area, mode="holes")
            unchanged = not changed
            mask, changed = remove_small_regions(mask, min_area, mode="islands")
            unchanged = unchanged and not changed

            new_masks.append(torch.as_tensor(mask).unsqueeze(0))
            # Give score=0 to changed masks and score=1 to unchanged masks
            # so NMS will prefer ones that didn't need postprocessing
            scores.append(float(unchanged))

        # Recalculate boxes and remove any new duplicates
        masks = torch.cat(new_masks, dim=0)
        boxes = batched_mask_to_box(masks)
        keep_by_nms = batched_nms(
            boxes.float(),
            torch.as_tensor(scores),
            torch.zeros_like(boxes[:, 0]),  # categories
            iou_threshold=nms_thresh,
        )

        # Only recalculate RLEs for masks that have changed
        for i_mask in keep_by_nms:
            if scores[i_mask] == 0.0:
                mask_torch = masks[i_mask].unsqueeze(0)
                mask_data["rles"][i_mask] = mask_to_rle_pytorch(mask_torch)[0]
                mask_data["boxes"][i_mask] = boxes[i_mask]  # update res directly
        mask_data.filter(keep_by_nms)

        return mask_data
