# **Model Test**
Short demos for a broad range of NLP and CV models.

## **Setup Instructions**

### Install requirements

First, create either a Python virtual environment with PyBuda installed

Installation instructions can be found at [Install TT-Buda.](https://github.com/eSlimKorea/TT-Buda-Installation).

**install the model requirements:**

```bash
pip install -r requirements.txt
```

## **Quick Start**

```bash
export PYTHONPATH=.
python3 CV_Modles/YOLOX/yolox_test.py
```
<br>

## Tested Models

| Test Model                | e75	                 |  e150	               |n150            |	n300 (dual-chip)   |  Supported Release  |
|----------------------|-----------------------|-----------------------|----------------|--------------------|---------------------|
| YOLOv5               | ⏳                   | ⏳                   | ⏳            | ❌                  | v0.19.3           |    
| YOLOv6               | ⏳                   | ⏳                   | ⏳            | ✔️                  | v0.19.3           |    
| [YOLOX](https://github.com/eSlimKorea/Model-TEST-TT-BUDA/tree/main/Model%20Test/CV_Models/YOLOX).                | ⏳                   | ⏳                   | ⏳            | ✔️                  | v0.19.3           |    
| ResNet               | ⏳                   | ⏳                   | ⏳            | ✔️                  | v0.19.3           |    
| Faster R-CNN         | ⏳                   | ⏳                   | ⏳            | ⏳                  | v0.19.3           |    
| Hand Landmark        | ⏳                   | ⏳                   | ⏳            | ⏳                  | v0.19.3           |  


> Note: for detail Support Table is [here](https://github.com/tenstorrent/tt-buda-demos/tree/main/model_demos#install-requirements).










### **Legend**
  - ✔️ : Supported on the device
  - ❌ : Not all variants supported on the device
  - ⏳ : Not Yet Started




