# Installation


The code was tested on Ubuntu 16.04, with [Anaconda](https://www.anaconda.com/download) Python 3.6 and [PyTorch]((http://pytorch.org/)) v0.4.1. NVIDIA GPUs are needed for both training and testing.
After install Anaconda:

0. Create a new conda environment. 

    ~~~
    conda create --name CenterNet python=3.6
    ~~~
    And activate the environment.
    
    ~~~
    conda activate CenterNet
    ~~~
   
   [recommended] Or build the docker image, mount the directories with Centernet and datasetswhen running the image. GOTO #5
    ~~~
    cd $CenterNet_ROOT/docker_files
    docker build -t $docker_image_name .
    ~~~

1. Install pytorch1.2:

    ~~~
    conda install pytorch=1.2.0 torchvision -c pytorch
    ~~~
2. Install [COCOAPI](https://github.com/cocodataset/cocoapi):

    ~~~
    # COCOAPI=/path/to/clone/cocoapi
    git clone https://github.com/cocodataset/cocoapi.git $COCOAPI
    cd $COCOAPI/PythonAPI
    make
    python setup.py install --user
    ~~~

3. Clone this repo:

    ~~~
    CenterNet_ROOT=/path/to/clone/CenterNet
    git clone https://github.com/xingyizhou/CenterNet $CenterNet_ROOT
    ~~~


4. Install the requirements

    ~~~
    pip install -r requirements.txt
    ~~~

5. [Optional] Compile NMS if your want to use multi-scale testing or test ExtremeNet.

    ~~~
    cd $CenterNet_ROOT/src/lib/external
    make
    ~~~

6. Download pertained models for [detection]() or [pose estimation]() and move them to `$CenterNet_ROOT/models/`. More models can be found in [Model zoo](MODEL_ZOO.md).
