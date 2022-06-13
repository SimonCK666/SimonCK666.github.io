20 July, 2020

# VCIP 2020 Grand Challenge on NIR Image Colorization

## Challenge Motivation
NIR (Near-Infrared) imaging provides a unique vision with respect to illumination and object material properties which are quite different from those in visible wavelength bands. The high sensitivity of NIR sensors and the fact that it is invisible to human vision make it an indispensable input for applications such as low-light imaging, night vision surveillance and road navigation, etc. However, the monochromatic NIR images lacks color discrimination and severely differ from the well-known RGB spectrum, which makes them unnatural and unfamiliar for both human perception and CV algorithms.

The correlation between the NIR and RGB domains are more ambiguous and complex which makes such task more challenging than gray scale image colorization. In recent years we see numerous learning-based models that explore different network architectures, supervision modes, and training strategies to tackle this challenge, however the limitations are still obvious in terms of color realism and texture fidelity. Semantic correctness and instance-level color consistency are difﬁcult to be preserved. More importantly, the demand for strictly registered NIR-RGB image pairs also restricts efﬁcient development of NIR colorization models. 

This challenge call for development of more efficient algorithms to transform NIR images to RGB images. We will provide both paired NIR-RGB image pairs and unpaired RGB images from different scene categories to facilitate explorations of both pixel-level and higher-level features for accurate semantic mapping and vivid color variation.

<div style="text-align: center"><img src="/projects/NIR2RGB_VCIP_Challenge/web/thumbnail_dataset.jpg" width="880" /></div>

## The Challenge Dataset

Challenge dataset [download](https://drive.google.com/file/d/1-P6ZrqwRc1ZxnN3nI3wZjSmFjoVEaNt9/view?usp=sharing).

The thumbnails of the challenge dataset are shown in Fig. 1., which provideds provides both pixel-aglined NIR-RGB image pairs, and un-paired RGB images with similar scene categories. The purpose of providing both paired and un-paired RGB-NIR images is to facilitate the development of learning models that explore both pixel-level and higher-level features for efficient colorization with accurate semantic mapping and vivid color variation. Participants are encouraged to take advantage of both groups of data. While it is up to your choices not to use a certain data group provided, the use of image data from outside the dataset is not allowed for this challenge. 

The dataset folder structure is shown bellow, the detail for each dataset folder will be explained in detail.

```bash
├── NIR_VCIP_Challenge_dataset
│   ├── NIR
│   │   ├── country_0000_nir.png
│   │   ├── country_0001_nir.png
│   │   ├── ...
│   │   ├── field_0000_nir.png
│   │   ├── ...             
│   ├── RGB-Registered
│   │   ├── country_0000_rgb_reg.png
│   │   ├── country_0001_rgb_reg.png
│   │   ├── ...
│   │   ├── field_0000_rgb_reg.png
│   │   ├── ...
│   ├── RGB-Online
│   │   ├── country_0000_rgb_onl.png
│   │   ├── country_0001_rgb_onl.png
│   │   ├── ...
│   │   ├── field_0000_rgb_onl.png
│   │   ├── ...
│   ├── Validation
│   │   ├── validation_0000_nir.png
│   │   ├── validation_0000_rgb.png
│   │   ├── ...
│   │   ├── validation_0009_nir.png
│   │   ├── validation_0009_rgb.png
```

### 1. Pixel-aligned NIR RGB Image Pairs
The subfolders **NIR** and **RGB-Registered** contain pixel-aligned (registered) NIR-RGB image pairs. The source of the images is [1]. Following its categorization, the challenge dataset contains NIR and RGB images under the scene categories of _countryside_, _forest_, _field_ and _mountain_. We have manually excluded ones that contain over-exposed and temporally inconsistent contents. Some histogram equalization and tone mapping operations are also implemented on some image pairs. The images are cropped and resized into patches of 256\*256 pixels. There are **372** image pairs in these two subfolders.

Images from the **NIR** folder are named as "_scene-category_nnnn_nir.png_"; images from the **RGB-Registered** folder are named as "_scene-category_nnnn_rgb_reg.png_". Here "_scene-category_" indicates the scene category and "_nnnn_" indicates image serial number. 
Identical leading names "_scene-category_nnnn_" indicate these images are pixel-aligned NIR-RGB image pairs.

Please note that although the scene category information is embedded in the image file names, **testing of algorithms will not provide any scene category information**. 

### 2. Unpaired RGB Images
The subfolder **RGB-Online** contains pixel-unaligned (unregistered) RGB images downloaded from online sources corresponding to each scene category. There are **1020** images in this subfolder.

### 3. Validation Dataset
The subfolder **Validation** contains pixel-aligned (registered) NIR-RGB image pairs for the purpose of model validation. The NIR-RGB image pairs are named as "_validation_nnnn_nir.png_" and "_validation_nnnn_rgb_reg.png_" respectively. Identical "_nnnn_" values indicate paired NIR-RGB relationships. There are **18** NIR-RGB image pairs in this subfolder.

### 4. Testing Dataset
The testing dataset will not be publically available until after the final evaluations of the challenge. The testing images will be NIR images of resolution 256\*256. 

**Terms of Use** all data provided by the VCIP challenge are freely available to the participants. The data are available only for open research and educational purposes, whithin the scope of the challenge. The conference organizing comittee makes no warranties regarding the dataset, including but not limited to warranties of non-infringement or fitness for a particular purpose. The copyright of the images remains property of their respective owners. By downloading and making use of the data, you accept full responsibility for using the data.

## Evaluation Metrics

Competing algorithms will be ranked both quantitatively and qualitatively based to the following metrics, with each metric's weight specified in percentile.

### 1. Objective Evaluation: 50%

We are going to use Peak Signal-to-Noise Ratio (**_PSNR_：15%**), Structural Similiarty (**_SSIM_：15%**) [2] and Angular Error (**_AE_：20%**) as quantitative metrics to evaluate and rank the colorization results. The PSNR value is calculated according to:

<div style="text-align: center"><img src="/projects/NIR2RGB_VCIP_Challenge/web/PSNR.jpg" width="240" /></div>

Here <img src="/projects/NIR2RGB_VCIP_Challenge/web/Iout.jpg" height="22"/> and <img src="/projects/NIR2RGB_VCIP_Challenge/web/Igt.jpg" height="22"/> represent the RGB colorization result and the RGB ground truth, respectively. To provide a color similarity measure closer to human color perception, we also use the Angular Error (**_AE_**) defined as:

<div style="text-align: center"><img src="/projects/NIR2RGB_VCIP_Challenge/web/AE.jpg" width="350" /></div>

### 2. Subjective Evaluation: 50%

To more comprehensively reflect the visual quality of the colorization performance, we will also rank the visual quality for competing algorithms. Subjective comparison will be conducted in a nonreference manner, a group of judges (10+ individuals) who do not know the ground truth RGB images are to evaluate the colorization results, and rank the algorithms based on the visual quality based on the following standards:
- **_Semantic correctness and color realism_: 10%** (judges will evaluate on whether the colors are conforming to the real world scenrios);
- **_Boundary precision_: 10%** (whether the object boudnaries are cleanly separated without "bleeding" effects);
- **_Texture preservation_: 10%** (how well the textures are preserved after colorization):;
- **_Instance consistency_: 10%** (whether the same object instance are colored consistently without wrong variations);
- **_Color vividness_: 10%**.


## Submission Requirements
The submission deadline for the challenge is 11:59pm (HK time, GMT+8) 20th September, 2020. All submissions should be sent to chenjie@comp.hkbu.edu.hk, with:
1. A brief introduction of the algorithm details in formats of _docx_ or _pdf_.
2. Statistics (**_PSNR_**, **_SSIM_** and **_AE_**) on the **Validation dataset** for each data and the total average;
3. Program executable(s) and running instruction for algorithm testing.

More submission requirements may be updated later, registered participants will be notified via email.

### Reference
[1] M. Brown and S. Susstrunk, “Multi-spectral SIFT for scene category recognition,” in _IEEE Conference on Computer Vision and Pattern Recognition_, 2011, pp. 177–184.

[2] Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, “Image quality assessment: From error visibility to structural similarity,” _IEEE Transactions on Image Processing_, vol. 13, no. 4, pp. 600–612, 2004.
