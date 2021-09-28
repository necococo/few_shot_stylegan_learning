# Few-shot style learning by stylegan2

Implementaion of [Few-shot Image Generation via Cross-domain Correspondence][1] in PyTorch.


Normally, training of gan requires about 50000-100000 images.  
Even for stylegan-ada(2020, NVIDIA), 50000/10 = 5000 images are required.  
In this paper, it was possible to do it with only 10 images.  


These are based on [stylegan2-pytorch repository][2] and [this weights][3] like written in the paper.  
I changed some lines in dataset.py, model.py and train.py.  
And I thank for the authors of the paper and stylegan2-pytorch repo.  


## Requirements
torchextractor  
With google colab, I needed the ninja-linux.  
### I have tested on:
PyTorch 1.7.1  
CUDA 11.1  

## Result
### 1000iter  
![1000iter][5]  

### 5000iter  
![5000iter][6]  

### Here are the weights I trained from 9~~10~~ goblin images.  
[weights][7] 


[1]:https://arxiv.org/pdf/2104.06820.pdf
[2]:https://github.com/rosinality/stylegan2-pytorch
[3]:https://drive.google.com/file/d/1PQutd-JboOCOZqmd95XWxWrO8gGEvRcO/
<!--[4]:./figs/e911d211.jpg-->
[5]:./figs/1000iter.png
[6]:./figs/5000iter.png
[7]:https://drive.google.com/drive/folders/1-14kuaMPomfK4kYXxo_oxtBNDHG10Bms?usp=sharing

