---
layout: post
title: Light Field Compression Topics
publish: IEEE TIP 2017 & IEEE TCSVT 2018
show-avatar: false
image: /projects/LFCompression/LFDictionary.jpg
paperlink: https://ieeexplore.ieee.org/document/8030107/
codelink: 
resourcelink: https://github.com/hotndy/SC-SKV
---

We propose a LF codec which is based on disparity guide Sparse Coding over a learned perspective-shifted LF dictionary based on selected Structural Key Views. Up to 80 percent bit rate reduction is achieved compared with HEVC.  
  
\[[TIP 2017 paper](https://ieeexplore.ieee.org/document/8030107/)\]
\[[TCSVT 2018 paper](https://ieeexplore.ieee.org/document/8283506/)\]  
  
<p align="center">
<img src="https://hotndy.github.io/projects/LFCompression/LFDictionary.jpg" width="500px"/>
</p>  
  
### Abstract
> Recent imaging technologies are rapidly evolving for sampling richer and more immersive representations of the 3D world. And one of the emerging technologies are the light field (LF) cameras based on micro-lens arrays. To record the directional information of the light rays, a much larger storage space and transmission bandwidth are required as compared to a conventional 2D image of similar spatial dimension, therefore the compression of LF data becomes a vital part of its application. In this paper, we propose a LF codec which is based on disparity guide Sparse Coding over a learned perspective-shifted LF dictionary based on selected Structural Key Views (SC-SKV). The SKVs are chosen by the codec such that most LF spatial information could be preserved, and these key views are used to reconstruct the entire LF. To achieve optimum dictionary efficiency, the LF is divided into several Coding Regions (CR), and the reconstruction works on each CR separately. Experiments and comparisons have been carried out which show that the proposed SC-SKV codec produces state-of-the-art compression performances in terms of PSNR, and visual quality compared with High Efficiency Video Coding (HEVC), with up to 80 percent bit rate reduction achieved.
