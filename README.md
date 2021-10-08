## Installation

Dependencies:

    mamba env create -f environement.yml
    pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI
    pip install matplotlib

TPU Tools:

    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
    echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
    sudo apt update
    sudo apt install libedgetpu1-std

    sudo apt install edgetpu-compiler	
    sudo apt install python3-pycoral


## Resources

* https://github.com/H-arshit/UNET-On-COCO

* https://keras.io/examples/vision/oxford_pets_image_segmentation/
