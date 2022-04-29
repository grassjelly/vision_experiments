### 1. Setup Dataset
#### 1.1 Split data to val, train and test
    cd <label-studio-exported-data>
    python split_data.py

#### 1.2 Copy dataset to yolov5 folder

    cp <label-studio-exported-data> yolov5

### 2. Copy data config file to yolov5 folder

    cp astro_boy.yaml yolov5/data

### Training

    cd yolov5
    python train.py --img 640 --batch 32 --epochs 100 --data data/astro_boy.yaml --weights yolov5s.pt --workers 24 --name astro_boy_det
