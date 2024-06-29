# cv_resnet_carddetection_scrfd34gkps_onnx
covert iic/cv_resnet_carddetection_scrfd34gkps to onnx

卡证检测矫正模型

## 1. env creating
```shell
conda create -n test_gpu python=3.8 -c main -c conda-forge -y
conda activate test_gpu

conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=10.2 -c pytorch -c conda-forge -c main -y
pip install modelscope==1.15.0 openmim
mim install mmcv==1.7.1 mmcv-full==1.7.2 mmdet==2.26.0
pip install onnx==1.16.1 onnxsim==0.4.36 onnxruntime==1.18.1
```

## 2. Check modelscope Damo model
```python
# check_model.py
import cv2
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

if __name__ == '__main__':
    cdc = pipeline(Tasks.card_detection, 'damo/cv_resnet_carddetection_scrfd34gkps')
    img = cv2.imread("your_image.jpeg")
    result = cdc(img)
    print(result)
```

## 3. convert to ONNX
[modelscope官方教程](https://www.modelscope.cn/docs/%E6%A8%A1%E5%9E%8B%E7%9A%84%E5%AF%BC%E5%87%BA "官网modelscope ONNX导出方法")
```python
from modelscope.models import Model
from modelscope.exporters import Exporter
model_id = 'damo/cv_resnet_carddetection_scrfd34gkps'
model = Model.from_pretrained(model_id)
output_files = Exporter.from_model(model).export_onnx(opset=13, output_dir='.',) # save to current dir
print(output_files)
```
Error should be like this:
```shell
KeyError: "The exporting of model 'scrfd' with task: 'card-detection' is not supported currently."
```
### 3.1 Change your file:
.cache/modelscope/hub/damo/cv_resnet_carddetection_scrfd34gkps/configuration.json
```json
{
    "framework": "pytorch",
    "task": "face-detection",
    "model":{
        "type":"scrfd",
        "score_thr": 0.45
    },
    "preprocessor":{
        "type":"object-detection-scrfd"
    },
    "pipeline": {
        "type": "resnet-card-detection-scrfd34gkps"
    }
}
```
### 3.2 Some bug fix:
```shell
# FileNotFoundError: [Errno 2] No such file or directory: 'data/test/images/face_detection2.jpeg'
mkdir -p data/test/images/
cp your_image.jpeg data/test/images/face_detection2.jpeg
```
### 3.3 Convert

```python
from modelscope.models import Model
from modelscope.exporters import Exporter
model_id = 'damo/cv_resnet_carddetection_scrfd34gkps'
model = Model.from_pretrained(model_id)
output_files = Exporter.from_model(model).export_onnx(opset=13, output_dir='.',) # save to current dir
print(output_files)
```

## 4. ONNX file test
