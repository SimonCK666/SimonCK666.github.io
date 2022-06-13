---
layout: post
title: Conjectures over Wide Angle Super-resolution based on Narrow Angle References
publish: 
show-avatar: false
image: /projects/LFAngularSP/LFS_teaser.png
paperlink: 
codelink:
resourcelink:
---

The challenge of super-resolving a low-resolution wide-angle image (WAL) based on another high-resolution  arrow angle reference (NAH) lies in the following aspects:
- The dense correspondence betweent he WAL and the NAH. This part is essential for transfer of high resolution contents from NAH to WAL. We have some previous work on dense correspondence matching, which is highly relevent to this sub-problem.
> Jie Chen, Junhui Hou, Yun Ni, and Lap-Pui. Chau, "Accurate Light Field Depth Estimation with Superpixel Regularization over Partially Occluded Regions,'' _IEEE Transactions on Image Processing_ **(IEEE TIP)**, vol. 27, no. 10, pp. 4889-4900, 2018. [[pdf](https://arxiv.org/abs/1708.01964)] [[code](https://github.com/hotndy/LFDepth_POBR)]
  
- With a camera pose estimation that linkes the positions of the WAL and the NAH, we can use the grid postions as guide for super-resolution operations. This is highly similar to light field spatial super-resolution on which topic we have some prior research.
> Henry W. F. Yeung, Junhui Hou, Xiaoming Chen, Jie Chen, Zhibo Chen, and Yuk Ying Chung, "Light Field Spatial Super-Resolution Using Deep Spatial-Angular Interleaved CNN,'' _IEEE Transactions on Image Processing_ **(IEEE TIP)**.  

- The major challenge we face for this problem however, is the extrapolation of high frequency conetents from liminted angle of view, and the propagation of it to wider angles. We have some on-going research, which is currently under review process at a CV publication venue. The solution we proposed was called a **Stratified Labelling for Surface Consistent Parallax Correction and Occlusion Prediction**, which uses a Generative Adversarial Network (GAN) to extend the camera baseline up to 10 times wider, based on the given narrow baseline references as input. We show some preliminary results here. In these results, we used a narrow baselined 3 captures to render the scene's appearance that is from a wider baseline angle. (Currently, this problem set up is a little different from WAL super-resolution based on NAH, however the solution can be largely similar.)

<p align='center'><figure class="image">
<img src="https://hotndy.github.io/projects/Extrapolation/workshop-1.gif" width="460px">
<img src="https://hotndy.github.io/projects/Extrapolation/workshop-2.5x.gif" width="460px">
<figcaption> Input references "Workshop" (3 views)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
Camera baseline extended 2.5x by our PCOC model
</figcaption>
<img src="https://hotndy.github.io/projects/Extrapolation/workshop-5x.gif" width="460px">
<img src="https://hotndy.github.io/projects/Extrapolation/workshop-7.5x.gif" width="460px">
<figcaption> 
Camera baseline extended 5x by our PCOC model
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
extended 7.5x by our PCOC model </figcaption></figure></p>

<p align='center'><figure class="image">
<img src="https://hotndy.github.io/projects/Extrapolation/art-gallery-1.gif" width="300px">
<img src="https://hotndy.github.io/projects/Extrapolation/art-gallery-2.5x.gif" width="300px">
  <img src="https://hotndy.github.io/projects/Extrapolation/art-gallery-5x.gif" width="300px">
<figcaption> Input references "Workshop" (3 views)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
Baseline extended 2.5x
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
Baseline extended 5x
</figcaption>
</figure></p>
  
<p align='center'><figure class="image">
<img src="https://hotndy.github.io/projects/Extrapolation/bikes-1.gif" width="300px">
<img src="https://hotndy.github.io/projects/Extrapolation/bikes-2.5x.gif" width="300px">
  <img src="https://hotndy.github.io/projects/Extrapolation/bikes-5x.gif" width="300px">
<figcaption> Input references "Workshop" (3 views)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
Baseline extended 2.5x
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
Baseline extended 5x
</figcaption>
</figure></p>
