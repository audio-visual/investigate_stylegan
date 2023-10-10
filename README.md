# investigate_stylegan
 investigate the data needed in stylegan

https://github.com/rosinality/stylegan2-pytorch

# whether human faces need to be frontalized?

## Q1: whether human faces need to be aligned?
https://github.com/happy-jihye/FFHQ-Alignment/tree/master/FFHQ-Alignmnet


# whether imaged need to be high-quality? 
结论：
1、生成图像的大小最好与模型训练用的保持一致，不然如下图所示，左侧是官方权重，在1024*1024大小上训练，逆向化对齐后的512像素图像，并还原至256大小
/home/cwy/文档/GitHub/investigate_stylegan/stylegan2-ffhq-config-f_align-000015-project_256_step500.png
右侧是非官方权重，在256*256大小上训练，逆向化对齐后的512像素图像，并还原至256大小
/home/cwy/文档/GitHub/investigate_stylegan/550000_align-000015-project_256_step500.png

2、图像是否align对结果影响巨大，无论是否为官方权重
第一行为原图，非对齐，对齐
/home/cwy/文档/GitHub/investigate_stylegan/samples/not_align/000015.png
/home/cwy/文档/GitHub/investigate_stylegan/samples/not_align/align-000015.png
第二行官方权重，非对齐，对齐，1024
/home/cwy/文档/GitHub/investigate_stylegan/stylegan2-ffhq-config-f_000015-project_1024_step500.png
/home/cwy/文档/GitHub/investigate_stylegan/stylegan2-ffhq-config-f_align-000015-project_1024_step500.png
第三行非官方权重，非对齐，对齐，256
/home/cwy/文档/GitHub/investigate_stylegan/550000_000015-project_256_step500.png
/home/cwy/文档/GitHub/investigate_stylegan/550000_align-000015-project_256_step500.png


3、图像是否frontalize对结果有影响
表情、五官、头发等细节无法还原，但总体在可接受范围内
第一张为原图，第二张为对齐后的，第三张为官方权重(1024)，第四张为非官方权重(256)
/home/cwy/文档/GitHub/investigate_stylegan/samples/0xjr1vVzFKY_0/000035.png

/home/cwy/文档/GitHub/investigate_stylegan/samples/0xjr1vVzFKY_0_align/align-000035.png

/home/cwy/文档/GitHub/investigate_stylegan/stylegan2-ffhq-config-f_align-000035-project_1024_step500.png


/home/cwy/文档/GitHub/investigate_stylegan/550000_align-000035-project_256_step500.png


4、图像是否高质量对结果影响巨大，猜测这可能是由于lpips的泛化性不强导致的
第一张为原图，第二张为官方权重(1024), 第三张为非官方权重(256)

/home/cwy/文档/GitHub/investigate_stylegan/samples/low_quality/0001200.jpg
/home/cwy/文档/GitHub/investigate_stylegan/stylegan2-ffhq-config-f_0001200-project_1024_step500.png
/home/cwy/文档/GitHub/investigate_stylegan/550000_0001200-project_256_step500.png