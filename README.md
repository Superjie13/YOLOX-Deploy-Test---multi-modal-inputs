## YOLOX Deploy Test - multi-modal inputs
### This is part of the project [StereoTracking](https://github.com/Superjie13/StereoTracking) which aims to develop a lightweight and efficient stereo vision-based airborne object tracking system.

This repo is used to test the deployment of YOLOX model on onnxruntime and opencv with multi-modal inputs (rgb, depth).
Specifically, we provide scripts to load the YOLOX model from the onnx file and run inference on images.

### Requirements:
- onnxruntime==1.8.1
- opencv==5.0.0
- pillow==10.2.0
- numpy==1.24.4

### ONNX Model Export:
If you want to export the rgb-depth YOLOX (we named it as yolox also in this repo) model to onnx, you can refer to [readme](docker/README.md) in the `docker` directory.

### Usage:
1. Download the onnx model file:
   1. [end2end.onnx](https://drive.google.com/file/d/1bckuHYDEx0CnIYENlCmaKs-vZ1x3UF98/view?usp=sharing): Exported 
      pre-trained multi-modal YOLOX model that runs end-to-end (model forward pass + post-processing + NMS). 
   2. [yolox.onnx](https://drive.google.com/file/d/1Q7cL7BbDfZ7qzfQf1FcpPGF5LO8b68lo/view?usp=drive_link): Exported
      pre-trained multi-modal YOLOX model that runs (model forward pass + post-processing) without NMS.
   3. [yolox_raw.onnx](https://drive.google.com/file/d/1gbb-x2Qq_1KbPMGDVr9gvAw-iyUf4Oso/view?usp=drive_link): Exported
      pre-trained multi-modal YOLOX model that runs only the model forward pass without post-processing and NMS.
   4. [end2end_rgb.onnx](https://drive.google.com/file/d/137hKSxrVjGmK0cj6_d9pzbjbTIApK8yV/view?usp=drive_link): Exported
      pre-trained single-modal YOLOX model that runs end-to-end (model forward pass + post-processing + NMS) on rgb images.
   5. [yolox_rgb.onnx](https://drive.google.com/file/d/1Xvf5kEvIskjSytVrAIgSu1ah-llOWImy/view?usp=drive_link): Exported
      pre-trained single-modal YOLOX model that runs (model forward pass + post-processing) without NMS on rgb images.
   6. [yolox_raw_rgb.onnx](https://drive.google.com/file/d/14yeByJtDz4XFO_px8xVFY4Tl4F0y2YIW/view?usp=drive_link): Exported
      pre-trained single-modal YOLOX model that runs only the model forward pass without post-processing and NMS on rgb images.
      
      Download the model file and place it in the root directory of this repo.
   
   __Note__: opencv currently only can load `yolox_raw.onnx` model cause the exported op set version limitation.

2. Run the test script:
   ```bash
   # run yolox_raw.onnx model on opencv
   python on_cv.py <path_to_onnx_model> data/rgb_00000.png data/disp_00000.png --show
   
   # run end2end.onnx model on onnxruntime
   python on_ort_end2end.py <path_to_onnx_model> data/rgb_00000.png data/disp_00000.png --show

   # run end2end_rgb.onnx model on onnxruntime
   python on_ort_end2end_rgb.py <path_to_onnx_model> data/rgb_00000.png --show
   
   # run yolox.onnx model on onnxruntime
   python on_ort_nms.py <path_to_onnx_model> data/rgb_00000.png data/disp_00000.png --show

   # run yolox_rgb.onnx model on onnxruntime
   python on_ort_nms_rgb.py <path_to_onnx_model> data/rgb_00000.png --show
   
   # run yolox_raw.onnx model on onnxruntime
   python on_ort_raw.py <path_to_onnx_model> data/rgb_00000.png data/disp_00000.png --show 

   # run yolox_raw_rgb.onnx model on onnxruntime
   python on_ort_raw_rgb.py <path_to_onnx_model> data/rgb_00000.png --show
   ``` 
   
### Results:
<p align="center">
    <img src="data/result.png" width="90%">
</p>

### Citation:
If you find this repo useful, please consider citing:
```
Coming soon...
```