## Installation

0. Navigate into the project directory: `cd transformer-pytorch`
1. Create Python conda environment, eg. named "transformer": `conda create -n transformer python=3.10`
2. Activate ENV_NAME environment: `conda activate transformer`
3. Install the required dependencies: 
    - Run apps on CPU, run this command: `pip install -r requirements_cpu.txt`
    - Run apps on GPU:
      - Find the compatible version of Cuda with your machine in the PyTorch's website: https://pytorch.org/get-started/locally/
      - Adapt the URL to download and install Torch and Cuda at the beginning of the requirement file `requirements_gpu.txt`, e.g. https://download.pytorch.org/whl/cu118
      - Run this command: `pip install -r requirements_gpu.txt`