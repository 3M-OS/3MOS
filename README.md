# 3MOS：Multi-sources, Multi-resolutions, and Multi-scenes dataset for Optical-SAR image matching
***
## Dataset Introduction
Optical-SAR image matching is a fundamental task for image fusion and visual navigation.  However, all large-scale open SAR dataset for methods development are collected from single platform, resulting in **limited** satellite types and spatial resolutions. Since images captured by different sensors vary significantly in both geometric and radiometric appearance, existing methods may fail to match corresponding regions containing the same content. Besides, most of existing datasets have not been categorized based on the characteristics of different scenes. To encourage the design of more general multi-modal image matching methods, we introduce a large-scale **M**ulti-sources, **M**ulti-resolutions, and **M**ulti-scenes dataset for **O**ptical-**S**AR image matching(**3MOS**). It consists of 155K optical-SAR image pairs, including SAR data from **six** commercial satellites, with resolutions ranging from **1.25m to 12.5m**. The data has been classified into **eight scenes** including urban, rural, plains, hills, mountains, water, desert, and frozen earth. Extensively experiments show that none of state-of-the-art  methods achieve consistently superior performance across different sources, resolutions and scenes. In addition, the distribution of data has a substantial impact on the matching capability of deep learning models, this proposes the domain adaptation challenge in optical-SAR image matching.
<center>
    <div style="text-align:center;">
        <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/3M-OS/3MOS/blob/main/overview.png">
    </div>
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">The overview of the 3MOS dataset</div>
</center>

*The complete image data of the dataset and the testing code will be made public after the publication of the article.*
