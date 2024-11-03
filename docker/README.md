## Convert multi-modal YOLOX to ONNX using Docker
To convert multi-modal YOLOX to ONNX, we need to install the following third-party repositories in the Docker container:
- [mmyolo_rgbd](https://github.com/Superjie13/mmyolo_rgbd)
- [mmdeploy_multi_modal](https://github.com/Superjie13/mmdeploy_multi_input)

### Third-party repositories preparation
Pretrained detection model: [det_model.pth](https://drive.google.com/file/d/1v0gBbsByNTrqET3sFy525hogxsPsJYSG/view?usp=sharing).

```bash
mkdir docker/third_party && cd docker/third_party
git clone https://github.com/Superjie13/mmdeploy_multi_input.git
git clone https://github.com/Superjie13/mmyolo_rgbd.git
mkdir checkpoint 
# Download the pre-trained model to the checkpoint directory
cd ../..  # Go back to the root directory
```


### Build and run the Docker container
#### 1. Build the Docker image
```bash
docker build -f docker/Dockerfile -t deploy_mmyolo ./docker
``` 

#### 2. Run the Docker container
```bash
docker run --gpus all -v $(pwd)/docker/third_party:/root/workspace/deploy_ws -it deploy-mmyolo
```

### Inside the Docker container
#### 1. Install the conda
```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh

. ~/miniconda3/bin/activate
```

#### 2. Create a conda environment
```bash
conda create -n deploy python=3.8 -y
conda activate deploy

pip install -U openmim
pip install torch==1.13.1 torchvision==0.14.1 onnxruntime==1.8.1 onnx==1.15.0
mim install mmengine "mmcv==2.0.1" "mmdet>=3.0.0,<4.0.0"
```

#### 3. Configure mmdeploy_multi_input
```bash
cd /root/workspace/deploy_ws/mmdeploy_multi_input
export MMDEPLOY_DIR=$(pwd)
mkdir -p build && cd build
cmake -DCMAKE_CXX_COMPILER=g++-7 -DMMDEPLOY_TARGET_BACKENDS=ort -DONNXRUNTIME_DIR=${ONNXRUNTIME_DIR} ..
make -j$(nproc) && make install
cd ${MMDEPLOY_DIR}
mim install -e .
```

#### 4. Configure mmyolo_rgbd
```bash
cd /root/workspace/deploy_ws/mmyolo_rgbd
export MMYOLO_DIR=$(pwd)
mim install -v -e .
```

#### 5. Convert multi-modal YOLOX to ONNX
```bash
cd /root/workspace/deploy_ws
bash mmyolo_rgbd/deploy_example/test_yolox_s_onnxruntime_rgb_depth.sh
```
> Get more details about the configuration in `mmyolo_rgbd/configs/deploy/detection_onnxruntime_static_mm.py`dd