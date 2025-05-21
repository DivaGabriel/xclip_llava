test.ipynb是主要程式碼  
然後由於Video-LLaVA是從原本的地方clone下來的，好像不能直接push到我的repo上面  
因此還需要從原本的地方clone下來使用  

## 🛠️ Requirements and Installation
* Python >= 3.10
* Pytorch == 2.0.1
* CUDA Version >= 11.7
* Install required packages:
```bash
git clone https://github.com/DivaGabriel/xclip_llava.git
#VideoLLaVA安裝
git clone https://github.com/PKU-YuanGroup/Video-LLaVA
cd Video-LLaVA
conda create -n videollava python=3.10 -y
conda activate videollava
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
pip install decord opencv-python git+https://github.com/facebookresearch/pytorchvideo.git@28fe037d212663c6a24f373b94cc5d478c8c1a1d
#XCLIP其餘的套件安裝
cd .. 
pip install -r Requirement.txt
```