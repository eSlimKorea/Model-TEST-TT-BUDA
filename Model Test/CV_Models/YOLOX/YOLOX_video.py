import subprocess
import os
import cv2
import numpy as np
import pybuda
import requests
import torch
import logging
from queue import Queue
from pybuda._C.backend_api import BackendDevice

torch.multiprocessing.set_sharing_strategy("file_system")
from yolox.data.data_augment import preproc as preprocess
from yolox.data.datasets import COCO_CLASSES
from yolox.exp import get_exp
from yolox.utils import demo_postprocess, multiclass_nms

def run_yolox_video(variant, video_path, output_path, batch_size=1):
    """
    Process all frames of a video file using YOLOX and generate a new video file containing object detection results.
    param variant: YOLOX model variant (e.g., yolox_nano, yolox_tiny, yolox_s, yolox_m, yolox_l, yolox_x)
    param video_path: input video file path
    param output_path: output video file path

    """
    
    
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s | %(levelname)s | %(message)s')
    logging.info("Starting YOLOX video processing.")

     # Set PyBuda configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.default_df_override = pybuda.DataFormat.Float16_b
    os.environ["PYBUDA_DECOMPOSE_SIGMOID"] = "1"
    logging.info("PyBuda configuration set.")

    # Device specific configurations
    available_devices = pybuda.detect_available_devices()
    logging.info(f"Available devices: {available_devices}")
    if available_devices:
        if available_devices[0] == BackendDevice.Wormhole_B0:
            if variant not in ["yolox_nano", "yolox_s"]:
                os.environ["PYBUDA_FORK_JOIN_BUF_QUEUES"] = "1"
                os.environ["PYBUDA_FORK_JOIN_EXPAND_OUTPUT_BUFFERS"] = "1"
                os.environ["PYBUDA_FORK_JOIN_SKIP_EXPANDING_BUFFERS"] = "1"
                os.environ["TT_BACKEND_TIMEOUT"] = "7200"  

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

    # Preparing model
    weight_folder = "weight"
    weight_name = f"weight/{variant}.pth" 
    weight_path = os.path.join(weight_folder, weight_name)  


    """

    url = f"https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/{weight_name}"
    response = requests.get(url)
    with open(f"{weight_name}", "wb") as file:
        file.write(response.content)

    if variant == "yolox_darknet":
        model_name = "yolov3"
    else:
        model_name = variant.replace("_", "-")

    """

    if variant == "yolox_darknet":
        model_name = "yolov3"
    else:
        model_name = variant.replace("_", "-")

    logging.info("Preparing the model.")
    exp = get_exp(exp_name=model_name)
    model = exp.get_model()
    ckpt = torch.load(weight_name, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.eval()
    tt_model = pybuda.PyTorchModule(f"pt_{variant}", model)

    # prepare input
    if variant in ["yolox_nano", "yolox_tiny"]:
        input_shape = (416, 416)
    else:
        input_shape = (640, 640)

    # create dummy
    dummy_input = torch.randn(1, 3, input_shape[0], input_shape[1])
    try:
        dummy_output_queue = pybuda.run_inference(tt_model, inputs=[(dummy_input, )], input_count=batch_size)
        dummy_output = dummy_output_queue.get()
        logging.info("Model compiled successfully with dummy inference.")
    except Exception as e:
        logging.error(f"Model compilation failed: {e}")
        return



    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Failed to open video file {video_path}")
        return
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    logging.info(f"Video opened: width={width}, height={height}, fps={original_fps}")

    # Initialize the video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, original_fps, (width, height))
    logging.info(f"Output video writer initialized: {output_path}")

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            logging.info("End of video file reached.")
            break
        frame_count += 1
        logging.info(f"Processing frame {frame_count}")

        # Image preprocessing
        img, ratio = preprocess(frame, input_shape)
        img_tensor = torch.from_numpy(img).unsqueeze(0).float()  # (1, 3, H, W)
        batch_input = torch.cat([img_tensor] * batch_size, dim=0)  # (batch_size, 3, H, W)

        try:
            
             # Run inference on Tenstorrent device
            logging.debug(f"Before run_inference call for frame {frame_count}")
            output_queue = pybuda.run_inference(None, inputs=[(batch_input, )], input_count=batch_size)
            output = output_queue.get()
            logging.debug(f"After run_inference call for frame {frame_count}")
            logging.debug(f"Inference completed for frame {frame_count}.")
        except Exception as e:
            logging.error(f"Inference failed for frame {frame_count}: {e}")
            continue

        try:
            # Output processing
            if isinstance(output, list):
                output = output[0]
            if isinstance(output, pybuda.Tensor):
                output_array = output.value().detach().float().numpy()
            elif isinstance(output, torch.Tensor):
                output_array = output.detach().float().numpy()
            else:
                logging.error(f"Unexpected output type: {type(output)}")
                continue

            # Check output array shape
            logging.debug(f"Output array shape: {output_array.shape}")
            logging.debug(f"Output array dtype: {output_array.dtype}")
            logging.debug(f"Output array stats: min={np.min(output_array)}, max={np.max(output_array)}, mean={np.mean(output_array)}")

            # If the output array is 1-dimensional, reshape it to 2-dimensional
            if output_array.ndim == 1:
                logging.warning(f"Output array is 1-dimensional for frame {frame_count}. Reshaping to 2D.")
                output_array = output_array.reshape(1, -1)
                logging.debug(f"Reshaped output array shape: {output_array.shape}")

            # Check for NaN or Inf values in the output array
            if np.isnan(output_array).any() or np.isinf(output_array).any():
                logging.error(f"Invalid values (NaN or Inf) in output_array for frame {frame_count}")
                continue

            # Post-processing 
            predictions = demo_postprocess(output_array, input_shape)[0]
            if predictions is None or predictions.shape[0] == 0:
                logging.info(f"No objects detected in frame {frame_count}.")
                out.write(frame)  
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
            logging.debug(f"Post-processing completed for frame {frame_count}.")
        except Exception as e:
            logging.error(f"Post-processing failed for frame {frame_count}: {e}")
            continue


        #score_threshold = 0.30

        if dets is not None:
            final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
            for box, score, cls_ind in zip(final_boxes, final_scores, final_cls_inds):

                #if score < score_threshold:
                #    continue

                class_name = COCO_CLASSES[int(cls_ind)]
                x_min, y_min, x_max, y_max = map(int, box)
               
                cv2.rectangle(
                    frame,
                    (x_min, y_min),
                    (x_max, y_max),
                    (255, 0, 255),
                    1,
                )
                
                cv2.putText(
                    frame,
                    f"{class_name} {score:.2f}",
                    (x_min, y_min - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 255),
                    2,
                )
                

        
        out.write(frame)
        logging.debug(f"Frame {frame_count} written to output video.")

    
    cap.release()
    out.release()
    logging.info("Video processing completed.")


if __name__ == "__main__":

    # YOLOX Models (ex.. yolox_nano, yolox_tiny, yolox_s, yolox_m, yolox_l, yolox_x)
    variant = "yolox_m"

   
    video_path = "input_video.mp4"  
    output_path = "output.mp4"  

    run_yolox_video(variant=variant, video_path=video_path, output_path=output_path, batch_size=1)
