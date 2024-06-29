# cv_resnet_carddetection_scrfd34gkps_onnx
covert iic/cv_resnet_carddetection_scrfd34gkps to onnx


## 1. env creating
```shell
conda create -n test_gpu python=3.8 -c main -c conda-forge -y

conda activate test_gpu

conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=10.2 -c pytorch -c conda-forge -c main -y

pip install modelscope==1.15.0 openmim

mim install mmcv==1.7.1 mmcv-full==1.7.2 mmdet==2.26.0

pip install onnx==1.16.1 onnxsim==0.4.36 onnxruntime==1.18.1
```


