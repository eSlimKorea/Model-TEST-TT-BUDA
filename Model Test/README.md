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
python3 cv_models/YOLOX//yolox_test.py
```
<br>

## Tested Models

| Test Model                | e75	                 |  e150	               |n150            |	n300 (dual-chip)   |  Supported Release  |
|----------------------|-----------------------|-----------------------|----------------|--------------------|---------------------|
| YOLOv5               | ⏳                   | ⏳                   | ⏳            | ❌                  | v0.19.3           |    
| YOLOv6               | ⏳                   | ⏳                   | ⏳            | ✔️                  | v0.19.3           |    
| [YOLOX]().                | ⏳                   | ⏳                   | ⏳            | ✔️                  | v0.19.3           |    
| ResNet               | ⏳                   | ⏳                   | ⏳            | ✔️                  | v0.19.3           |    
| Faster R-CNN         | ⏳                   | ⏳                   | ⏳            | ⏳                  | v0.19.3           |    
| Hand Landmark        | ⏳                   | ⏳                   | ⏳            | ⏳                  | v0.19.3           |  


> Note: for detail Support Table is [here](https://github.com/tenstorrent/tt-buda-demos/tree/main/model_demos#install-requirements).










### **Legend**
  - ✔️ : Supported on the device
  - ❌ : Not all variants supported on the device
  - ⏳ : Not Yet Started




