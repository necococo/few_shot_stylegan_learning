# Few-shot style learning by stylegan2

(My first) Implementaion of [Few-shot Image Generation via Cross-domain Correspondence][1] in PyTorch.


GAN training generally requires a large number of training images, and training on a small number of data often results in overfitting.   
But in this paper, it was possible to do it with only 10 images.  


These are based on [stylegan2-pytorch repository][2] and [this weights][3] like written in the paper.  
I changed some lines in dataset.py, model.py and train.py.  
And I thank for the authors of this paper, stylegan2-pytorch repo, and torchextractor.  


## Requirements
torchextractor  
With google colab, I needed the ninja-linux.  
### I have tested on:
PyTorch 1.7.1  
CUDA 11.1  

## Usage 
Look at the [stylegan2-pytorch repository][2]  


## Results 
### Arcimboldo:1800iter   
![1800iter][10]  
### Arcimboldo:Source images   
![Arcimboldo_concat_source][9]  

### Goblin:1000iter  
![1000iter][5]  
### Goblin:Source images  
![Goblin_concat_source][8]  


[1]:https://arxiv.org/pdf/2104.06820.pdf
[2]:https://github.com/rosinality/stylegan2-pytorch
[3]:https://drive.google.com/file/d/1PQutd-JboOCOZqmd95XWxWrO8gGEvRcO/
<!--[4]:./figs/e911d211.jpg-->
[5]:./figs/gob_1000iter.png =2560x512
[7]:https://drive.google.com/drive/folders/1-14kuaMPomfK4kYXxo_oxtBNDHG10Bms?usp=sharing
[8]:./figs/goblin_concat_img.png | height=128

[9]:./figs/arcim_concat_img.png | height=128
[10]:./figs/arcim_551800.png | height=128
[11]:./figs/arcim_5000iter.png | height=128
[12]:https://drive.google.com/drive/folders/
