# investigate_stylegan
 investigate the data needed in stylegan  

ref: https://github.com/rosinality/stylegan2-pytorch     
To reproduce the results, you can refer to `samples/get_samples.py` to generate images, and then use `projector.py` to invert and restore images (to make this project self-contained, we upload the weights, you can download from the Releases). 
# Conclusions

## Prerequisite
**It is best to keep the size of the generated/restored image consistent with the size used for model training**, otherwise the performance will decrease, as shown in the figure below.   
Therefore, subsequent experiments also follow this principle  

The left side is the official weight, trained on the 1024x1024 size, reversed the aligned 512x512 image, and restored to the 256 size,  
The right side is the unofficial weight, trained on the 256x256 size, reversed the aligned 512x512 image, and restored to the 256 size. 
1024x1024->256            |  256x256->256
:-------------------------:|:-------------------------:
<img src='https://github.com/audio-visual/investigate_stylegan/blob/main/results/stylegan2-ffhq-config-f_align-000015-project_256_step500.png' width='256px'/> | <img src='https://github.com/audio-visual/investigate_stylegan/blob/main/results/550000_align-000015-project_256_step500.png' width='256px'/>

## Q1: whether human faces need to be aligned?
https://github.com/happy-jihye/FFHQ-Alignment/tree/master/FFHQ-Alignmnet  
Alignment roughly means
1) Only capture facial areas
2) Eyes on the same horizontal line  
for example:  
original images:
   ![alt original images](https://github.com/audio-visual/investigate_stylegan/blob/main/results/0tx4o3yXM64_0.png?raw=true)
aligned images:
  ![alt original images](https://github.com/audio-visual/investigate_stylegan/blob/main/results/0tx4o3yXM64_0_align.png?raw=true)
original images:
   ![alt original images](https://github.com/audio-visual/investigate_stylegan/blob/main/results/0xjr1vVzFKY_0.png?raw=true)
aligned images:
  ![alt original images](https://github.com/audio-visual/investigate_stylegan/blob/main/results/0xjr1vVzFKY_0_align.png?raw=true)
**Conclusion:**
Whether the image is aligned has a **significant impact** on the results

The images in the first row are the original images, the images in the second row are the results of inversion using official weights (trained on 1024x1024), and the images in the third row are the results of inversion using unofficial weights (trained on 256x256)

not aligned             |  aligned
:-------------------------:|:-------------------------:
<img src='https://github.com/audio-visual/investigate_stylegan/blob/main/samples/not_align/000015.png' width='50%'/> | <img src='https://github.com/audio-visual/investigate_stylegan/blob/main/samples/not_align/align-000015.png' width='50%'/>
<img src='https://github.com/audio-visual/investigate_stylegan/blob/main/results/stylegan2-ffhq-config-f_000015-project_1024_step500.png' width='50%'/> | <img src='https://github.com/audio-visual/investigate_stylegan/blob/main/results/stylegan2-ffhq-config-f_align-000015-project_1024_step500.png' width='50%'/>
<img src='https://github.com/audio-visual/investigate_stylegan/blob/main/results/550000_000015-project_256_step500.png' width='50%'/> | <img src='https://github.com/audio-visual/investigate_stylegan/blob/main/results/550000_align-000015-project_256_step500.png' width='50%'/> 

## Q2: whether human faces need to be frontalized?
**Conclusion:**
Whether the image is frontalize has a slight impact on the results  
Expressions, facial features, hair, and other **details cannot be restored**, but are generally **within an acceptable range**   

Notice that, if we want to explore the impact of head angle on the results, we should exclude other factors (such as the need for images to be of high quality and aligned) 
original             |  aligned  | official inverted | unofficial inverted
:-------------------------:|:-------------------------: | :-------------------------: | :-------------------------:
<img src='https://github.com/audio-visual/investigate_stylegan/blob/main/samples/0xjr1vVzFKY_0/000035.png' width='256px'/> | <img src='https://github.com/audio-visual/investigate_stylegan/blob/main/samples/0xjr1vVzFKY_0_align/align-000035.png' width='256px'/> | <img src='https://github.com/audio-visual/investigate_stylegan/blob/main/results/stylegan2-ffhq-config-f_align-000035-project_1024_step500.png' width='256px'/> | <img src='https://github.com/audio-visual/investigate_stylegan/blob/main/results/550000_align-000035-project_256_step500.png' width='256px'/>

## Q3: whether imaged need to be high-quality? 
**Conclusion:**
Whether the image is of high quality has a **huge impact** on the results. It is speculated that this may be due to the weak generalization of lpips.

original             |  official inverted  |  unofficial inverted
:-------------------------:|:-------------------------: | :-------------------------: 
<img src='https://github.com/audio-visual/investigate_stylegan/blob/main/samples/low_quality/0001200.jpg' width='256px'/> | <img src='https://github.com/audio-visual/investigate_stylegan/blob/main/results/stylegan2-ffhq-config-f_0001200-project_1024_step500.png' width='256px'/> | <img src='https://github.com/audio-visual/investigate_stylegan/blob/main/results/550000_0001200-project_256_step500.png' width='256px'/> 


## Others: add id_loss can improve results
You can uncomment the `id_loss` related codes to reproduce this result

lpips loss             |  id loss  |  lpips+id loss
:-------------------------:|:-------------------------: | :-------------------------: 
<img src='https://github.com/audio-visual/investigate_stylegan/blob/main/results/550000_align-000015-project_256_step500_perceptual_loss.png' width='256px'/> | <img src='https://github.com/audio-visual/investigate_stylegan/blob/main/results/550000_align-000015-project_256_step500_id_loss.png' width='256px'/> | <img src='https://github.com/audio-visual/investigate_stylegan/blob/main/results/550000_align-000015-project_256_step500_id_perceptual_loss.png' width='256px'/> 