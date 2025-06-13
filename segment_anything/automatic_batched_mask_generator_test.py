# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from torchvision.ops.boxes import batched_nms, box_area  # type: ignore
from torch.nn import functional as F # Needed for padding if not in preprocess

from typing import Any, Dict, List, Optional, Tuple

# Assuming these are in the same directory or your PYTHONPATH is set up
# from .modeling import Sam # Not strictly needed for this file if SamPredictor handles Sam instance
# from .predictor import SamPredictor # Will be used
# from .utils.amg import (...) # All these utils are used

# STUBS - Replace with actual imports from segment_anything
class Sam: # Stub for type hinting
    mask_threshold: float = 0.0
    image_format: str = "RGB"
    def __init__(self):
        from types import SimpleNamespace
        self.image_encoder = SimpleNamespace(img_size=1024) # Mock attribute
        self.pixel_mean = torch.tensor([0.,0.,0.]).view(3,1,1)
        self.pixel_std = torch.tensor([1.,1.,1.]).view(3,1,1)
    @property
    def device(self): return torch.device("cpu")
    def preprocess(self, x: torch.Tensor) -> torch.Tensor: # Simplified
        x = (x - self.pixel_mean.to(x.device)) / self.pixel_std.to(x.device)
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        return F.pad(x, (0, padw, 0, padh))
    def postprocess_masks(self, masks, input_size, original_size): # Simplified
        masks = F.interpolate(masks, (self.image_encoder.img_size, self.image_encoder.img_size), mode="bilinear", align_corners=False)
        masks = masks[..., :input_size[0], :input_size[1]]
        return F.interpolate(masks, original_size, mode="bilinear", align_corners=False)


class SamPredictor: # Stub
    def __init__(self, model: Sam):
        self.model = model
        from segment_anything.utils.transforms import ResizeLongestSide # Use actual if available
        # Mock ResizeLongestSide if actual not available for stub
        class MockResizeLongestSide:
            def __init__(self, target_length): self.target_length = target_length
            def apply_coords(self, coords, original_size): return coords # Mock
            def get_input_size(self, h, w): # Mock: calculate new H, W after resizing longest to target_length
                scale = self.target_length / max(h,w)
                new_h, new_w = int(h * scale + 0.5), int(w * scale + 0.5)
                return (new_h, new_w)

        try:
            from segment_anything.utils.transforms import ResizeLongestSide
            self.transform = ResizeLongestSide(model.image_encoder.img_size)
        except ImportError:
            print("Warning: ResizeLongestSide not found, using mock.")
            self.transform = MockResizeLongestSide(model.image_encoder.img_size)

        self.device = model.device
    # set_image, reset_image, predict_torch are not directly used in the new flow's core path
    # but transform object and model access are.

# Assuming amg utils are available (using stubs from previous response if not)
from segment_anything.utils.amg import (
    MaskData, area_from_rle, batch_iterator, batched_mask_to_box, box_xyxy_to_xywh,
    build_all_layer_point_grids, calculate_stability_score, coco_encode_rle,
    generate_crop_boxes, is_box_near_crop_edge, mask_to_rle_pytorch,
    remove_small_regions, rle_to_mask, uncrop_boxes_xyxy, uncrop_masks, uncrop_points,
)
# End STUBS


class SamAutomaticMaskGenerator:
    def __init__(
        self,
        model: Sam,
        points_per_side: Optional[int] = 32,
        points_per_batch: int = 64,
        # ... (rest of __init__ arguments are the same)
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
            "binary_mask", "uncompressed_rle", "coco_rle",
        ], f"Unknown output_mode {output_mode}."
        if output_mode == "coco_rle":
            from pycocotools import mask as mask_utils
        # if min_mask_region_area > 0: import cv2

        self.predictor = SamPredictor(model) # Used for transform and model access
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
        self.model = model # Direct access to SAM model

    @torch.no_grad()
    def generate(
        self, images: List[np.ndarray]
    ) -> Tuple[List[List[Dict[str, Any]]], List[List[Dict[str, Any]]], List[List[Dict[str, Any]]], List[List[Dict[str, Any]]]]:
        
        num_original_images = len(images)
        all_crop_tasks = [] 

        # 1. Collect all crop tasks
        for img_idx, original_image_np in enumerate(images):
            original_size_hw = original_image_np.shape[:2]
            crop_boxes_xyxy, layer_indices = generate_crop_boxes(
                original_size_hw, self.crop_n_layers, self.crop_overlap_ratio
            )
            for crop_idx_in_image, (crop_box_xyxy, layer_idx) in enumerate(zip(crop_boxes_xyxy, layer_indices)):
                x0, y0, x1, y1 = crop_box_xyxy
                cropped_im_np = original_image_np[y0:y1, x0:x1, :]
                # H,W of the current crop before any resizing
                cropped_im_original_size_hw = cropped_im_np.shape[:2] 
                
                # Points are generated relative to the crop's original size
                points_scale = np.array(cropped_im_original_size_hw)[None, ::-1] 
                points_for_crop_np = self.point_grids[layer_idx] * points_scale
                
                # Apply the ResizeLongestSide transform to the numpy crop
                # self.predictor.transform is an instance of ResizeLongestSide
                # Its apply_image method takes HWC numpy and returns HWC numpy
                resized_cropped_im_np = self.predictor.transform.apply_image(cropped_im_np)
                
                # Convert the resized numpy crop to a CHW tensor for model.preprocess
                resized_cropped_im_tensor_chw = torch.as_tensor(
                    resized_cropped_im_np.astype(np.float32), device=self.model.device
                )
                resized_cropped_im_tensor_chw = resized_cropped_im_tensor_chw.permute(2, 0, 1)

                # This is the H,W of the crop AFTER resizing by transform, but BEFORE padding by model.preprocess
                # This is the 'input_size' needed for model.postprocess_masks
                input_size_transformed_hw = tuple(resized_cropped_im_tensor_chw.shape[-2:])

                # Now, assemble the complete task dictionary
                all_crop_tasks.append({
                    "original_image_idx": img_idx,
                    "crop_box_xyxy": crop_box_xyxy,
                    "layer_idx": layer_idx,
                    "resized_cropped_im_tensor_chw": resized_cropped_im_tensor_chw, # For model.preprocess
                    "cropped_im_original_size_hw": cropped_im_original_size_hw, # For transform.apply_coords & postprocess_masks original_size
                    "points_for_crop_np": points_for_crop_np, # Relative to cropped_im_original_size_hw
                    "input_size_transformed_hw": input_size_transformed_hw, # For postprocess_masks input_size
                    "original_image_size_hw": original_size_hw, # For final uncropping
                })

        if not all_crop_tasks:
            empty_results = tuple([[] for _ in range(num_original_images)] for _ in range(4))
            return empty_results

        # 2. Batch preprocess and encode all crops
        # model.preprocess will normalize and pad each tensor in the batch
        preprocessed_crop_tensors_for_encoder = torch.stack(
            [self.model.preprocess(task["resized_cropped_im_tensor_chw"]) for task in all_crop_tasks]
        )
        
        all_crop_embeddings = self.model.image_encoder(preprocessed_crop_tensors_for_encoder)

        # 3. Iterate through encoded crops, predict masks
        per_image_results = [ [[] for _ in range(4)] for _ in range(num_original_images) ]

        for task_idx, task_info in enumerate(all_crop_tasks):
            current_crop_embedding = all_crop_embeddings[task_idx].unsqueeze(0) 
            
            crop_data_all = MaskData()
            crop_data_s, crop_data_m, crop_data_l = MaskData(), MaskData(), MaskData()

            for (points_batch_np,) in batch_iterator(self.points_per_batch, task_info["points_for_crop_np"]):
                batch_masks_all, batch_masks_s, batch_masks_m, batch_masks_l = \
                    self._predict_masks_for_crop_embedding(
                        image_embedding=current_crop_embedding,
                        points_np=points_batch_np, 
                        crop_original_size_hw=task_info["cropped_im_original_size_hw"], 
                        crop_input_to_encoder_size_hw=task_info["input_size_transformed_hw"], 
                        crop_box_xyxy=task_info["crop_box_xyxy"], 
                        original_image_size_hw=task_info["original_image_size_hw"]
                    )
                crop_data_all.cat(batch_masks_all)
                crop_data_s.cat(batch_masks_s)
                crop_data_m.cat(batch_masks_m)
                crop_data_l.cat(batch_masks_l)
            
            crop_data_all = self._filter_crop_data_internal_nms(crop_data_all, task_info["crop_box_xyxy"])
            crop_data_s = self._filter_crop_data_internal_nms(crop_data_s, task_info["crop_box_xyxy"])
            crop_data_m = self._filter_crop_data_internal_nms(crop_data_m, task_info["crop_box_xyxy"])
            crop_data_l = self._filter_crop_data_internal_nms(crop_data_l, task_info["crop_box_xyxy"])

            img_idx = task_info["original_image_idx"]
            per_image_results[img_idx][0].append(crop_data_all)
            per_image_results[img_idx][1].append(crop_data_s)
            per_image_results[img_idx][2].append(crop_data_m)
            per_image_results[img_idx][3].append(crop_data_l)
        
        # 4. Aggregate results for each original image
        final_anns_all, final_anns_s, final_anns_m, final_anns_l = [], [], [], []

        for img_idx in range(num_original_images):
            current_image_crop_boxes = [
                task["crop_box_xyxy"] for task in all_crop_tasks if task["original_image_idx"] == img_idx
            ]

            # Helper function to process each mask type
            def process_merged_data(merged_data_list, image_crops):
                merged_data = MaskData()
                for crop_data_item in merged_data_list:
                    # Ensure crop_data_item is not empty before cat, or that cat handles it
                    if crop_data_item._stats: # Check if _stats dict is not empty
                        merged_data.cat(crop_data_item)
                
                # Check if merged_data contains actual data before proceeding
                # "boxes" is a good key to check as it's fundamental for NMS
                if merged_data._stats and "boxes" in merged_data._stats and len(merged_data["boxes"]) > 0:
                    merged_data = self._filter_data_across_crops(merged_data, image_crops)
                return self.generate_curr_anns(merged_data)

            final_anns_all.append(process_merged_data(per_image_results[img_idx][0], current_image_crop_boxes))
            final_anns_s.append(process_merged_data(per_image_results[img_idx][1], current_image_crop_boxes))
            final_anns_m.append(process_merged_data(per_image_results[img_idx][2], current_image_crop_boxes))
            final_anns_l.append(process_merged_data(per_image_results[img_idx][3], current_image_crop_boxes))
            
        return final_anns_all, final_anns_s, final_anns_m, final_anns_l

    def _filter_crop_data_internal_nms(self, data: MaskData, crop_box_xyxy: List[int]) -> MaskData:
        """
        Applies NMS to masks generated from a single crop.
        Sets the 'crop_boxes' field in MaskData.
        This replaces part of the original _postprocess_crop_data.
        Assumes 'boxes' in data are already in original image coordinates.
        """
        if len(data["boxes"]) == 0:
            data["crop_boxes"] = torch.empty((0,4), dtype=torch.long, device=self.model.device if len(data.data)==0 else data["boxes"].device) # Match type
            return data

        keep_by_nms = batched_nms(
            data["boxes"].float(),
            data["iou_preds"], # Stability scores could also be used here
            torch.zeros_like(data["boxes"][:, 0]),  # categories
            iou_threshold=self.box_nms_thresh, # NMS within a crop
        )
        data.filter(keep_by_nms)
        
        # All these masks came from this crop_box_xyxy
        data["crop_boxes"] = torch.tensor(
            [crop_box_xyxy for _ in range(len(data["rles"]))], 
            device=data["boxes"].device
        )
        return data

    def _predict_masks_for_crop_embedding(
        self,
        image_embedding: torch.Tensor,
        points_np: np.ndarray,
        crop_original_size_hw: Tuple[int, int],      # H,W of the crop before any resizing
        crop_input_to_encoder_size_hw: Tuple[int, int], # H,W of the crop after resizing (for postprocess_masks input_size)
        crop_box_xyxy: List[int],
        original_image_size_hw: Tuple[int, int]
    ) -> Tuple[MaskData, MaskData, MaskData, MaskData]:

        # 1. Transform point coordinates from crop_original_size_hw frame to crop_input_to_encoder_size_hw frame
        #    (which is what the prompt encoder expects after image transform)
        transformed_points_np = self.predictor.transform.apply_coords(points_np, crop_original_size_hw)
        in_points_torch = torch.as_tensor(transformed_points_np, device=self.model.device)
        in_labels_torch = torch.ones(in_points_torch.shape[0], dtype=torch.int, device=self.model.device)

        # 2. Encode prompts
        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
            points=(in_points_torch.unsqueeze(1), in_labels_torch.unsqueeze(1)),
            boxes=None, masks=None,
        )

        # 3. Decode masks
        low_res_logits, iou_predictions, mask_embeddings_out = self.model.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True, return_mask_embeddings=True,
        )

        # 4. Post-process masks (upscale to crop_original_size_hw)
        # `input_size` for postprocess_masks is the size after ResizeLongestSide but before padding
        # `original_size` for postprocess_masks is the target size, which is the crop's original H,W
        masks_logits = self.model.postprocess_masks(
            low_res_logits,
            input_size=crop_input_to_encoder_size_hw,
            original_size=crop_original_size_hw, 
        )

        # 5. Create MaskData objects and apply filters
        points_for_maskdata = torch.as_tensor(points_np, device=self.model.device) # these are the original points for the crop

        common_args_filter = (crop_box_xyxy, original_image_size_hw[1], original_image_size_hw[0], crop_original_size_hw)

        data_all = MaskData(
            masks=masks_logits.flatten(0, 1), iou_preds=iou_predictions.flatten(0, 1),
            points=points_for_maskdata.repeat_interleave(masks_logits.shape[1], dim=0),
            embeddings=mask_embeddings_out.reshape(-1, mask_embeddings_out.shape[-1])
        )
        data_all = self._filter_and_process_batch_data(data_all, *common_args_filter)
        # ... (s, m, l masks similarly)
        data_s = MaskData(
            masks=masks_logits[:, 0, :, :], iou_preds=iou_predictions[:, 0],
            points=points_for_maskdata, embeddings=mask_embeddings_out[:, 0, :]
        )
        data_s = self._filter_and_process_batch_data(data_s, *common_args_filter)

        data_m = MaskData(
            masks=masks_logits[:, 1, :, :], iou_preds=iou_predictions[:, 1],
            points=points_for_maskdata, embeddings=mask_embeddings_out[:, 1, :]
        )
        data_m = self._filter_and_process_batch_data(data_m, *common_args_filter)

        data_l = MaskData(
            masks=masks_logits[:, 2, :, :], iou_preds=iou_predictions[:, 2],
            points=points_for_maskdata, embeddings=mask_embeddings_out[:, 2, :]
        )
        data_l = self._filter_and_process_batch_data(data_l, *common_args_filter)

        return data_all, data_s, data_m, data_l

    # _filter_and_process_batch_data: This method from your original code should largely work.
    # It takes MaskData (where 'masks' are logits), filters, binarizes, gets boxes,
    # filters edge boxes, uncrops masks to original full image size, and computes RLEs.
    # Key arguments it needs:
    #   data (MaskData), crop_box_xyxy (current crop in orig img coords),
    #   orig_w, orig_h (of the full original image),
    #   im_size (H,W of the current crop - used for edge filtering relative to crop boundary)
    def _filter_and_process_batch_data(
            self, data: MaskData, crop_box_xyxy: List[int],
            orig_w: int, orig_h: int, cropped_im_size_hw: Tuple[int, int],
    ) -> MaskData:
        # Filter by predicted IoU
        if self.pred_iou_thresh > 0.0:
            keep_mask = data["iou_preds"] > self.pred_iou_thresh
            data.filter(keep_mask)
            if len(data["masks"]) == 0: return data

        # Calculate stability score (expects logits)
        data["stability_score"] = calculate_stability_score(
            data["masks"], self.model.mask_threshold, self.stability_score_offset
        )
        if self.stability_score_thresh > 0.0:
            keep_mask = data["stability_score"] >= self.stability_score_thresh
            data.filter(keep_mask)
            if len(data["masks"]) == 0: return data

        # Threshold masks to binary and calculate boxes (relative to current crop frame)
        data["masks"] = data["masks"] > self.model.mask_threshold
        data["boxes"] = batched_mask_to_box(data["masks"]) # Boxes are XYXY relative to current crop

        # Filter boxes that touch crop boundaries (of the current crop itself)
        # `is_box_near_crop_edge` needs the box_XYXY of the current crop, and the size of this crop.
        # The `crop_box` argument to `is_box_near_crop_edge` is the frame the boxes are in.
        # Here, boxes are in crop_frame, so crop_box for check is [0,0,W_crop,H_crop]
        crop_frame_for_edge_check = [0, 0, cropped_im_size_hw[1], cropped_im_size_hw[0]]
        keep_mask = ~is_box_near_crop_edge(data["boxes"], crop_frame_for_edge_check, crop_frame_for_edge_check)
        if not torch.all(keep_mask): # if any are True (near edge)
            data.filter(keep_mask)
            if len(data["masks"]) == 0: return data

        # Uncrop masks from crop frame to original image frame, then RLE
        # `uncrop_masks` takes masks relative to crop, the crop_box_xyxy, and target full image H,W
        data["masks"] = uncrop_masks(data["masks"], crop_box_xyxy, orig_h, orig_w)
        data["rles"] = mask_to_rle_pytorch(data["masks"])
        
        # Boxes also need to be uncropped to original image frame for NMS across crops later
        # The `data["boxes"]` are currently relative to the crop.
        # We need to uncrop them before `_filter_crop_data_internal_nms` or `_filter_data_across_crops`
        data["boxes"] = uncrop_boxes_xyxy(data["boxes"], crop_box_xyxy)
        
        # Points are already relative to crop. They are uncropped later by _filter_data_across_crops if needed,
        # or handled by generate_curr_anns (original `_postprocess_crop_data` uncropped points).
        # Let's ensure points are uncropped here if they are used directly later.
        # The points in MaskData are the *original prompts* relative to the crop.
        data["points"] = uncrop_points(data["points"], crop_box_xyxy)


        del data["masks"] # Free memory
        return data

    # generate_curr_anns, _filter_data_across_crops, postprocess_small_regions
    # are assumed to be the same as in your original/previous code.
    # Make sure _filter_data_across_crops gets the list of all crop_boxes for the image it's processing.
    def _filter_data_across_crops(
            self, data: MaskData, image_crop_boxes: List[List[int]], # All crop_boxes for current orig image
    ) -> MaskData:
        if len(image_crop_boxes) > 1 and len(data["boxes"]) > 0 :
            # Scores prefer masks from smaller crops. crop_boxes in MaskData should be set.
            # This assumes data["crop_boxes"] was set correctly by _filter_crop_data_internal_nms
            # and contains the crop_box_xyxy from which each mask originated.
            if not isinstance(data["crop_boxes"], torch.Tensor):
                 data["crop_boxes"] = torch.tensor(data["crop_boxes"], device=data["boxes"].device, dtype=data["boxes"].dtype)

            scores = 1.0 / box_area(data["crop_boxes"].float()) # Add .float()
            # scores = scores.to(data["boxes"].device) # Ensure device, though should be from tensor creation
            
            keep_by_nms = batched_nms(
                data["boxes"].float(), scores,
                torch.zeros_like(data["boxes"][:, 0]),
                iou_threshold=self.crop_nms_thresh,
            )
            data.filter(keep_by_nms)
        data.to_numpy()
        return data

    def generate_curr_anns(self, mask_data: MaskData) -> List[Dict[str, Any]]:
        if not mask_data._stats or "rles" not in mask_data._stats or len(mask_data["rles"]) == 0: # Early exit if no data
             return []

        if self.min_mask_region_area > 0: # No need to check len again if already done above
            mask_data = self.postprocess_small_regions(
                mask_data, self.min_mask_region_area,
                max(self.box_nms_thresh, self.crop_nms_thresh),
            )
        
        # After postprocessing, re-check if empty
        if not mask_data._stats or "rles" not in mask_data._stats or len(mask_data["rles"]) == 0:
            return []


        if self.output_mode == "coco_rle":
            mask_data["segmentations"] = [coco_encode_rle(rle) for rle in mask_data["rles"]]
        elif self.output_mode == "binary_mask":
            mask_data["segmentations"] = [rle_to_mask(rle) for rle in mask_data["rles"]]
        else:
            mask_data["segmentations"] = mask_data["rles"]

        curr_anns = []
        # Check if 'embeddings' key exists in mask_data._stats and has the correct length
        has_embeddings = ("embeddings" in mask_data._stats and 
                          len(mask_data["embeddings"]) == len(mask_data["segmentations"]))
        
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
            if has_embeddings and mask_data["embeddings"][idx] is not None:
                # Assuming embeddings are already numpy arrays by this point
                # (e.g., if MaskData.to_numpy() was called or they were stored as numpy)
                ann["embeddings"] = mask_data["embeddings"][idx] 
            curr_anns.append(ann)
        return curr_anns

    @staticmethod
    def postprocess_small_regions(
        mask_data: MaskData, min_area: int, nms_thresh: float
    ) -> MaskData:
        # Early exit if MaskData is empty or essential keys are missing
        if not mask_data._stats or "rles" not in mask_data._stats or not mask_data["rles"]:
            return mask_data
        
        # Determine original device from "boxes" if it's a tensor, otherwise default to CPU
        # This is important if to_numpy() hasn't been called yet.
        if "boxes" in mask_data._stats and isinstance(mask_data["boxes"], torch.Tensor):
            original_device = mask_data["boxes"].device
        else:
            original_device = torch.device("cpu") # Default if boxes not tensor or not present

        new_masks_tensors = []
        scores = []
        
        for rle_idx, rle in enumerate(mask_data["rles"]):
            mask = rle_to_mask(rle) 
            mask_copy = np.copy(mask) # Ensure writable
            modified_mask, changed = remove_small_regions(mask_copy, min_area, mode="holes")
            unchanged = not changed
            modified_mask, changed = remove_small_regions(modified_mask, min_area, mode="islands")
            unchanged = unchanged and not changed
            new_masks_tensors.append(torch.as_tensor(modified_mask, dtype=torch.bool).unsqueeze(0))
            scores.append(float(unchanged))

        if not new_masks_tensors: # Should not happen if initial rles check passed, but good guard
            return mask_data
        
        masks_tensor = torch.cat(new_masks_tensors, dim=0).to(original_device)
        # Recalculate boxes from the (potentially modified) masks_tensor
        recalculated_boxes = batched_mask_to_box(masks_tensor) 
        
        keep_by_nms = batched_nms(
            recalculated_boxes.float(), 
            torch.as_tensor(scores, device=original_device),
            torch.zeros_like(recalculated_boxes[:, 0]), 
            iou_threshold=nms_thresh,
        )
        
        #--- Create a new MaskData object to hold the filtered results ---
        # This is often cleaner than trying to filter multiple lists/tensors in place
        # especially when their types (list, np.array, torch.Tensor) might vary
        # or if to_numpy() might have been called.

        filtered_data_dict = {}
        
        # Filter all fields present in the original mask_data based on keep_by_nms
        # The indices in keep_by_nms refer to the original order (same as scores, masks_tensor, recalculated_boxes)
        original_indices_kept = keep_by_nms.cpu().numpy()

        for field_key, field_values in mask_data._stats.items():
            if field_key == "rles": # RLEs will be rebuilt
                continue
            if field_key == "boxes": # Boxes will be from recalculated_boxes
                continue

            if isinstance(field_values, torch.Tensor):
                filtered_data_dict[field_key] = field_values[original_indices_kept]
            elif isinstance(field_values, np.ndarray):
                filtered_data_dict[field_key] = field_values[original_indices_kept]
            elif isinstance(field_values, list):
                # Ensure the list has the same length as scores before indexing
                if len(field_values) == len(scores):
                    filtered_data_dict[field_key] = [field_values[i] for i in original_indices_kept]
                else: # If list length doesn't match, it's problematic to filter. Log or skip.
                    # print(f"Warning: List field {field_key} in postprocess_small_regions has mismatched length. Skipping filtering for this field.")
                    filtered_data_dict[field_key] = field_values # Or handle as an error
            else: # For other data types, just carry them over if they don't correspond to per-mask items
                filtered_data_dict[field_key] = field_values


        new_rles_list = []
        new_boxes_list = []

        for i, original_idx in enumerate(original_indices_kept): # i is the index in the filtered set
            if scores[original_idx] == 0.0: # Mask was changed by postprocessing and kept
                # Use the RLE from the modified mask and its recalculated box
                new_rles_list.append(mask_to_rle_pytorch(masks_tensor[original_idx].unsqueeze(0))[0])
                new_boxes_list.append(recalculated_boxes[original_idx])
            else: # Mask was unchanged and kept
                # Use the original RLE and original box (which should be at mask_data["boxes"][original_idx])
                new_rles_list.append(mask_data["rles"][original_idx])
                # Ensure original boxes are tensor for consistent stacking
                original_boxes_tensor = mask_data["boxes"]
                if isinstance(original_boxes_tensor, np.ndarray):
                    original_boxes_tensor = torch.from_numpy(original_boxes_tensor).to(original_device)
                new_boxes_list.append(original_boxes_tensor[original_idx])
        
        filtered_data_dict["rles"] = new_rles_list
        if new_boxes_list:
            filtered_data_dict["boxes"] = torch.stack(new_boxes_list)
        else: # Handle case where no masks are kept
            filtered_data_dict["boxes"] = torch.empty((0,4), dtype=torch.long, device=original_device)
            # Ensure other essential per-mask arrays are also empty if boxes is empty
            for key_to_empty in ["iou_preds", "points", "stability_score", "crop_boxes", "embeddings", "rles"]:
                 if key_to_empty in filtered_data_dict: # Check if it was even populated
                    if isinstance(filtered_data_dict[key_to_empty], torch.Tensor):
                        filtered_data_dict[key_to_empty] = torch.empty((0, *filtered_data_dict[key_to_empty].shape[1:]), dtype=filtered_data_dict[key_to_empty].dtype, device=original_device)
                    elif isinstance(filtered_data_dict[key_to_empty], np.ndarray):
                        filtered_data_dict[key_to_empty] = np.empty((0, *filtered_data_dict[key_to_empty].shape[1:]), dtype=filtered_data_dict[key_to_empty].dtype)
                    elif isinstance(filtered_data_dict[key_to_empty], list):
                        filtered_data_dict[key_to_empty] = []


        return MaskData(**filtered_data_dict)