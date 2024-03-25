# 3MOS-Multi-sources, Multi-resolutions, and Multi-scenes dataset for Optical-SAR image matching
***
## Dataset Introduction
Optical-SAR image matching is a fundamental task for image fusion and visual navigation.  However, all large-scale open SAR dataset for methods development are collected from single platform, resulting in limited satellite types and spatial resolutions. Since images captured by different sensors vary significantly in both geometric and radiometric appearance, existing methods may fail to match corresponding regions containing the same content. Besides, most of existing datasets have not been categorized based on the characteristics of different scenes. To encourage the design of more general multi-modal image matching methods, we introduce a large-scale Multi-sources, Multi-resolutions, and Multi-scenes dataset for Optical-SAR image matching(3MOS). It consists of 155K optical-SAR image pairs, including SAR data from six commercial satellites, with resolutions ranging from 1.25m to 12.5m. The data has been classified into eight scenes including urban, rural, plains, hills, mountains, water, desert, and frozen earth. Extensively experiments show that none of state-of-the-art  methods achieve consistently superior performance across different sources, resolutions and scenes. In addition, the distribution of data has a substantial impact on the matching capability of deep learning models, this proposes the domain adaptation challenge in optical-SAR image matching.

![The overview of the 3MOS dataset]https://github.com/3M-OS/3MOS/blob/main/overview.png
