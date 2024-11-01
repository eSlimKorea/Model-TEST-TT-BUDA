# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# YOLOX multiple images processing script

import subprocess

# Install yolox==0.3.0 without installing its dependencies
subprocess.run(["pip", "install", "yolox==0.3.0", "--no-deps"])

import os
import cv2
import numpy as np
import pybuda
import requests
import torch
from pybuda._C.backend_api import BackendDevice

torch.multiprocessing.set_sharing_strategy("file_system")
from yolox.data.data_augment import preproc as preprocess
from yolox.data.datasets import COCO_CLASSES
from yolox.exp import get_exp
from yolox.utils import demo_postprocess, multiclass_nms


def run_yolox_multiple_images(variant, image_urls, batch_size=1):

    # Set PyBuda configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.default_df_override = pybuda.DataFormat.Float16_b
    os.environ["PYBUDA_DECOMPOSE_SIGMOID"] = "1"


       # Device specific configurations
    available_devices = pybuda.detect_available_devices()
    if available_devices:
        if available_devices[0] == BackendDevice.Wormhole_B0:
            if variant not in ["yolox_nano", "yolox_s"]:
                os.environ["PYBUDA_FORK_JOIN_BUF_QUEUES"] = "1"
                os.environ["PYBUDA_FORK_JOIN_EXPAND_OUTPUT_BUFFERS"] = "1"
                os.environ["PYBUDA_FORK_JOIN_SKIP_EXPANDING_BUFFERS"] = "1"

            if variant in ["yolox_nano", "yolox_tiny"]:
                compiler_cfg.balancer_op_override(
                    "conv2d_7.dc.conv2d.1.dc.reshape.0.dc.sparse_matmul.4.lc2", "t_stream_shape", (1, 2)
                )
                compiler_cfg.balancer_op_override(
                    "conv2d_7.dc.conv2d.3.dc.reshape.0.dc.sparse_matmul.4.lc2", "t_stream_shape", (1, 2)
                )
                os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = "81920"
                if variant == "yolox_nano":
                    compiler_cfg.balancer_op_override(
                        "max_pool2d_630.dc.sparse_matmul.5.dc.sparse_matmul.1.lc2", "t_stream_shape", (1, 1)
                    )
                elif variant == "yolox_tiny":
                    compiler_cfg.balancer_op_override(
                        "max_pool2d_454.dc.sparse_matmul.5.dc.sparse_matmul.1.lc2", "t_stream_shape", (1, 1)
                    )

            elif variant == "yolox_s":
                compiler_cfg.balancer_op_override(
                    "conv2d_7.dc.conv2d.5.dc.reshape.0.dc.sparse_matmul.4.lc2", "t_stream_shape", (1, 4)
                )
                compiler_cfg.balancer_op_override(
                    "conv2d_7.dc.conv2d.1.dc.reshape.0.dc.sparse_matmul.4.lc2", "t_stream_shape", (1, 4)
                )
                compiler_cfg.balancer_op_override(
                    "conv2d_7.dc.conv2d.3.dc.reshape.0.dc.sparse_matmul.10.lc2", "t_stream_shape", (1, 4)
                )
                compiler_cfg.balancer_op_override("conv2d_33.dc.matmul.8", "t_stream_shape", (1, 1))
                os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = "4096"
                compiler_cfg.place_on_new_epoch("concatenate_1163.dc.sparse_matmul.11.lc2")
                compiler_cfg.balancer_op_override(
                    "max_pool2d_454.dc.sparse_matmul.5.dc.sparse_matmul.1.lc2", "grid_shape", (1, 2)
                )

            elif variant == "yolox_m":
                compiler_cfg.place_on_new_epoch("conv2d_811.dc.matmul.8")
                compiler_cfg.balancer_op_override(
                    "conv2d_7.dc.conv2d.1.dc.reshape.0.dc.sparse_matmul.4.lc2", "t_stream_shape", (1, 4)
                )
                compiler_cfg.balancer_op_override(
                    "conv2d_7.dc.conv2d.3.dc.reshape.0.dc.sparse_matmul.4.lc2", "t_stream_shape", (1, 6)
                )
                compiler_cfg.balancer_op_override(
                    "conv2d_7.dc.conv2d.5.dc.sparse_matmul.9.dc.sparse_matmul.1.lc2", "t_stream_shape", (5, 1)
                )
                compiler_cfg.balancer_op_override(
                    "conv2d_7.dc.conv2d.5.dc.reshape.0.dc.sparse_matmul.10.lc2", "t_stream_shape", (1, 4)
                )
                compiler_cfg.balancer_op_override(
                    "conv2d_7.dc.conv2d.3.dc.reshape.0.dc.sparse_matmul.10.lc2", "t_stream_shape", (1, 4)
                )
                compiler_cfg.place_on_new_epoch("concatenate_1530.dc.sparse_matmul.11.lc2")
                os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = "4096"
                compiler_cfg.balancer_op_override(
                    "max_pool2d_671.dc.sparse_matmul.5.dc.sparse_matmul.1.lc2", "t_stream_shape", (169, 1)
                )

            elif variant == "yolox_l":
                os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = "245760"
                compiler_cfg.place_on_new_epoch("conv2d_1644.dc.matmul.11")
                compiler_cfg.place_on_new_epoch("concatenate_1897.dc.sparse_matmul.11.lc2")

            elif variant == "yolox_darknet":
                os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = "245760"
                compiler_cfg.place_on_new_epoch("conv2d_1147.dc.matmul.11")

            elif variant == "yolox_x":
                compiler_cfg.balancer_op_override(
                    "conv2d_7.dc.conv2d.5.dc.reshape.0.dc.sparse_matmul.4.lc2", "t_stream_shape", (1, 4)
                )
                compiler_cfg.balancer_op_override(
                    "conv2d_7.dc.conv2d.1.dc.reshape.0.dc.sparse_matmul.10.lc2", "t_stream_shape", (1, 4)
                )
                compiler_cfg.balancer_op_override(
                    "conv2d_7.dc.conv2d.3.dc.sparse_matmul.9.dc.sparse_matmul.1.lc2", "t_stream_shape", (5, 1)
                )
                compiler_cfg.balancer_op_override(
                    "conv2d_7.dc.conv2d.3.dc.reshape.0.dc.sparse_matmul.10.lc2", "t_stream_shape", (1, 4)
                )
                compiler_cfg.place_on_new_epoch("concatenate_2264.dc.sparse_matmul.11.lc2")
                compiler_cfg.balancer_op_override(
                    "max_pool2d_1104.dc.sparse_matmul.5.dc.sparse_matmul.1.lc2", "t_stream_shape", (13, 1)
                )

        elif available_devices[0] == BackendDevice.Grayskull:

            if variant == "yolox_nano":
                compiler_cfg.balancer_op_override(
                    "conv2d_7.dc.conv2d.5.dc.reshape.0.dc.sparse_matmul.4.lc2", "grid_shape", (4, 1)
                )
                compiler_cfg.balancer_op_override(
                    "conv2d_7.dc.conv2d.3.dc.reshape.0.dc.sparse_matmul.4.lc2", "grid_shape", (4, 1)
                )
                compiler_cfg.balancer_op_override(
                    "conv2d_7.dc.conv2d.1.dc.reshape.0.dc.sparse_matmul.4.lc2", "grid_shape", (4, 1)
                )
                os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = "81920"
                compiler_cfg.balancer_op_override(
                    "conv2d_7.dc.conv2d.3.dc.reshape.0.dc.sparse_matmul.14.lc2", "t_stream_shape", (1, 13)
                )
                compiler_cfg.balancer_op_override(
                    "conv2d_7.dc.conv2d.1.dc.reshape.0.dc.sparse_matmul.14.lc2", "t_stream_shape", (1, 13)
                )
                compiler_cfg.balancer_op_override(
                    "conv2d_7.dc.conv2d.5.dc.reshape.0.dc.sparse_matmul.14.lc2", "t_stream_shape", (1, 13)
                )

            elif variant == "yolox_tiny":
                os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = "81920"
                compiler_cfg.balancer_op_override(
                    "max_pool2d_454.dc.sparse_matmul.5.dc.sparse_matmul.1.lc2", "t_stream_shape", (13, 1)
                )
                compiler_cfg.balancer_op_override("_fused_op_34", "t_stream_shape", (1, 1))

            elif variant == "yolox_s":
                compiler_cfg.balancer_op_override(
                    "conv2d_7.dc.conv2d.5.dc.sparse_matmul.9.dc.sparse_matmul.1.lc2", "t_stream_shape", (10, 1)
                )
                compiler_cfg.balancer_op_override(
                    "conv2d_7.dc.conv2d.5.dc.reshape.0.dc.sparse_matmul.10.lc2", "t_stream_shape", (1, 5)
                )
                compiler_cfg.balancer_op_override(
                    "conv2d_7.dc.conv2d.1.dc.reshape.0.dc.sparse_matmul.10.lc2", "t_stream_shape", (1, 5)
                )
                compiler_cfg.balancer_op_override(
                    "conv2d_7.dc.conv2d.3.dc.reshape.0.dc.sparse_matmul.10.lc2", "t_stream_shape", (1, 5)
                )
                compiler_cfg.balancer_op_override(
                    "max_pool2d_454.dc.sparse_matmul.5.dc.sparse_matmul.1.lc2", "t_stream_shape", (169, 1)
                )

            elif variant == "yolox_m":
                os.environ["PYBUDA_FORK_JOIN_BUF_QUEUES"] = "1"
                os.environ["PYBUDA_FORK_JOIN_EXPAND_OUTPUT_BUFFERS"] = "1"
                os.environ["PYBUDA_FORK_JOIN_SKIP_EXPANDING_BUFFERS"] = "1"
                compiler_cfg.balancer_op_override(
                    "concatenate_1530.dc.concatenate.7_to_concatenate_1530.dc.sparse_matmul.11.lc2_1_serialized_dram_queue.before_padded_node.nop_0",
                    "grid_shape",
                    (1, 1),
                )
                compiler_cfg.place_on_new_epoch("concatenate_1530.dc.sparse_matmul.11.lc2")
                compiler_cfg.balancer_op_override(
                    "conv2d_7.dc.conv2d.5.dc.reshape.0.dc.sparse_matmul.4.lc2", "grid_shape", (4, 2)
                )
                compiler_cfg.balancer_op_override(
                    "conv2d_7.dc.conv2d.1.dc.reshape.0.dc.sparse_matmul.4.lc2", "grid_shape", (4, 2)
                )
                compiler_cfg.balancer_op_override(
                    "conv2d_7.dc.conv2d.3.dc.reshape.0.dc.sparse_matmul.4.lc2", "grid_shape", (4, 2)
                )
                compiler_cfg.balancer_op_override(
                    "conv2d_7.dc.conv2d.5.dc.reshape.0.dc.sparse_matmul.10.lc2", "grid_shape", (1, 5)
                )
                compiler_cfg.balancer_op_override(
                    "conv2d_7.dc.conv2d.3.dc.reshape.0.dc.sparse_matmul.10.lc2", "grid_shape", (1, 5)
                )
                compiler_cfg.balancer_op_override(
                    "conv2d_7.dc.conv2d.1.dc.reshape.0.dc.sparse_matmul.10.lc2", "grid_shape", (1, 5)
                )
                compiler_cfg.place_on_new_epoch("max_pool2d_671.dc.sparse_matmul.5.dc.sparse_matmul.1.lc2")



                    # Prepare model
    weight_name = f"{variant}.pth"
    url = f"https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/{weight_name}"
    if not os.path.exists(weight_name):
        print(f"Downloading model weights from {url}")
        response = requests.get(url)
        with open(f"{weight_name}", "wb") as file:
            file.write(response.content)
        print("Model weights downloaded.")

    if variant == "yolox_darknet":
        model_name = "yolov3"
    else:
        model_name = variant.replace("_", "-")

    exp = get_exp(exp_name=model_name)
    model = exp.get_model()
    ckpt = torch.load(f"{variant}.pth", map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.eval()
    model_name = f"pt_{variant}"
    tt_model = pybuda.PyTorchModule(f"pt_{variant}", model)

    # Prepare input shape
    if variant in ["yolox_nano", "yolox_tiny"]:
        input_shape = (416, 416)
    else:
        input_shape = (640, 640)

    test_url = "http://images.cocodataset.org/val2017/000000397133.jpg"
    response = requests.get(test_url)

    with open("input.jpg", "wb") as f:
        f.write(response.content)
    img = cv2.imread("input.jpg")
    img, ratio = preprocess(img, input_shape)
    img_tensor = torch.from_numpy(img)
    img_tensor = img_tensor.unsqueeze(0)
    batch_input = torch.cat([img_tensor] * batch_size, dim=0)

    # Run inference on Tenstorrent device
    output_q = pybuda.run_inference(tt_model, inputs=[(batch_input)])
    output = output_q.get()



    # Process each image in the list
    for idx, img_url in enumerate(image_urls):
        try:
            # Download image
            response = requests.get(img_url)
            img_name = f"input_{idx}.jpg"
            with open(img_name, "wb") as f:
                f.write(response.content)
            print(f"Image {idx + 1} downloaded.")

            # Read and preprocess image
            img = cv2.imread(img_name)
            orig_img = img.copy()  # Save original image for drawing
            img, ratio = preprocess(img, input_shape)
            img_tensor = torch.from_numpy(img).unsqueeze(0)
            batch_input = torch.cat([img_tensor] * batch_size, dim=0)

          

            # Post-processing
            if isinstance(output, list):
                output = output[0]
            if isinstance(output, pybuda.Tensor):
                output_array = output.value().detach().float().numpy()
            elif isinstance(output, torch.Tensor):
                output_array = output.detach().float().numpy()
            else:
                print(f"Unexpected output type: {type(output)}")
                continue

            predictions = demo_postprocess(output_array, input_shape)[0]
            if predictions is None or predictions.shape[0] == 0:
                print(f"No objects detected in image {idx + 1}.")
                os.remove(img_name)
                continue

            boxes = predictions[:, :4]
            scores = predictions[:, 4:5] * predictions[:, 5:]
            boxes_xyxy = np.ones_like(boxes)
            boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
            boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
            boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
            boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
            boxes_xyxy /= ratio
            dets = multiclass_nms(
                boxes_xyxy, scores, nms_thr=0.45, score_thr=0.1
            )
            if dets is not None:
                final_boxes, final_scores, final_cls_inds = (
                    dets[:, :4],
                    dets[:, 4],
                    dets[:, 5],
                )
                for box, score, cls_ind in zip(
                    final_boxes, final_scores, final_cls_inds
                ):
                    class_name = COCO_CLASSES[int(cls_ind)]
                    x_min, y_min, x_max, y_max = map(int, box)
                    # Draw bounding box and label on the image
                    cv2.rectangle(
                        orig_img,
                        (x_min, y_min),
                        (x_max, y_max),
                        (0, 255, 0),
                        2,
                    )
                    cv2.putText(
                        orig_img,
                        f"{class_name} {score:.2f}",
                        (x_min, y_min - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )
                # Save the labeled image
                output_img_name = f"output_{idx + 1}.jpg"
                cv2.imwrite(output_img_name, orig_img)
                print(f"Labeled image saved as {output_img_name}.")
            else:
                print(f"No detections after NMS in image {idx + 1}.")

            # Remove downloaded input image
            os.remove(img_name)

        except Exception as e:
            print(f"Error processing image {idx + 1}: {e}")
            if os.path.exists(img_name):
                os.remove(img_name)
            continue

    # Remove downloaded model weights
    if os.path.exists(weight_name):
        os.remove(weight_name)
        print("Model weights file removed.")


if __name__ == "__main__":
    # YOLOX model variant (e.g., yolox_nano, yolox_tiny, yolox_s, yolox_m, yolox_l, yolox_x)
    variant = "yolox_nano"

    # List of image URLs to process
    image_urls = [
        "http://images.cocodataset.org/val2017/000000397133.jpg"
      
    ]

    run_yolox_multiple_images(variant=variant, image_urls=image_urls) 
