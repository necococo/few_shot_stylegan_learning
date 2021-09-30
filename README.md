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
<img src="./figs/arcim_551800.png" alt="arcim_551800" width="2324" height="516">  

### Arcimboldo:Source images   
<img src="./figs/arcim_concat_img.png" alt="arcim_concat_img" width="1536"  height="256">  

### Goblin:1000iter  
<img src="./figs/gob_1000iter.png" alt="gob_1000iter" width="2560"  height="516">  

### Goblin:Source images  
!<img src="./figs/goblin_concat_img.png" alt="goblin_concat_img" width="2304"  height="256">  


[1]:https://arxiv.org/pdf/2104.06820.pdf
[2]:https://github.com/rosinality/stylegan2-pytorch
[3]:https://drive.google.com/file/d/1PQutd-JboOCOZqmd95XWxWrO8gGEvRcO/


