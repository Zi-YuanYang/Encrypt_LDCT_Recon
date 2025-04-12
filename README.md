# Encrypt_LDCT_Recon

This repository is the implementation of the encrypted LDCT reconstruction method (accepted by IEEE Transactions on Artificial Intelligence). This paper can be downloaded at [this link](https://arxiv.org/abs/2310.09101).

#### Abstract
Deep learning (DL) has made significant advancements in tomographic imaging, particularly in low-dose computed tomography (LDCT) denoising. A recent trend involves servers training powerful models with enormous self-collected data and providing application programming interfaces (APIs) for users, such as Chat-GPT. To avoid model leakage, users are required to upload their data to the server. This approach is particularly advantageous for devices with limited computational capabilities, as it offloads computation to the server, easing the workload on the devices themselves. However, this way raises public concerns about the privacy disclosure risk. Hence, to alleviate related concerns, we propose to directly denoise LDCT in the encrypted domain to achieve privacy-preserving cloud services without exposing private data to the server. Concretely, we employ homomorphic encryption to encrypt private LDCT, which is then transferred to the server model trained with plaintext LDCT for further denoising. Since fundamental DL operations, such as convolution and linear transformation, cannot be directly used in the encrypted domain, we transform the fundamental mathematic operations in the plaintext domain into the operations in the encrypted domain. Moreover, we present two interactive frameworks for linear and nonlinear models, both of which can achieve lossless operating. In this way, the proposed methods can achieve two merits, the data privacy is well protected and the server model is free from the risk of model leakage. Moreover, we provide theoretical proof to validate the lossless property of our framework. Finally, experiments were conducted to demonstrate that the transferred contents are well protected and cannot be reconstructed.

#### Citation
If our work is valuable to you, please cite our work:
```
@ARTICLE{yang2023ccnet,
  author={Yang, Ziyuan and Huangfu, Huijie and Ran, Maosong and Wang, Zhiwen and Yu, Hui and Sun, Mengyu and Zhang, Yi},
  journal={IEEE Transactions on Artificial Intelligence}, 
  title={Novel Privacy-Enhancing Framework for Low-Dose CT Denoising}, 
  year={2025}}
```
