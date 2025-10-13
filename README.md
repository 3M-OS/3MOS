# 3MOS Dataset: the First Multi-Source, Multi-Resolution, Multi-Scene Optical-SAR Dataset

 [**Dataset Link (Baidu NetDisk)**](https://pan.baidu.com/s/1QvwUu4CECaaTVe8KlkdhZg?pwd=wr7y)           [**Paper Link**](https://link.springer.com/article/10.1007/s44267-025-00091-0)

## Overview
The **3MOS dataset** is the first comprehensive multi-source, multi-resolution, and multi-scene optical-SAR image matching dataset, designed to address the generalization challenges in multimodal image matching. It consists of **113,000 optical-SAR image pairs** collected from five SAR satellites with resolutions ranging from **3.5 m to 12.5 m**, categorized into **eight distinct scenes** (urban, rural, plains, hills, mountains, water, desert, and frozen earth).

<img src="assets\overview_01.png" alt="overview_01" style="zoom:80%;" />

## Key Features
- **Multi-Source SAR Data**: Includes SAR imagery from five satellites: GF-3, ALOS, Sentinel-1, Radarsat, and RCM（RADARSAT

  constellation mission）.

- **Multi-Resolution**: Spatial resolutions span 3.5 m (GF-3) to 12.5 m (ALOS/RCM).

- **Multi-Scene Categorization**: Images are classified into eight scenes using a practical strategy combining NASADEM elevation data and Sentinel-2 land cover data.

- **High-Quality Ground Truth**: All pairs are manually registered by experts with sub-pixel accuracy.

  <img src="assets\illustrate5_01.png" alt="illustrate5_01" style="zoom: 25%;" />

## Dataset Construction
### 1. Data Preprocessing
- Terrain correction applied to SAR data using SRTM DEM.
- Optical images georeferenced and aligned with SAR data.
- Grayscale stretching and resampling to uniform resolution (UTM coordinate system).

### 2. Manual Registration
- Expert-selected control points ensured high registration accuracy.

- Affine transformation with RANSAC minimized outliers.

- Average registration error: **<1 pixel** for most data.

 <img src="assets\new_register_01.png" alt="register" style="zoom: 25%;" />

### 3. Image Cropping
- Original images cropped into 256×256 patches with 50% overlap.
- Invalid samples (e.g., high cloud cover, featureless regions) removed via manual inspection.

### 4. Scene Categorization
Scenes were defined using:
- **Elevation differences** (NASADEM): Mountains (>150 m), Hills (50–150 m).

- **Land cover** (Sentinel-2): Urban (high built-up density), Plains (vegetation-dominated), etc.

<img src="assets\category_new_01.png" alt="category" style="zoom: 25%;" />

### 5. Data Splitting
- Randomized split into training/validation/testing sets (6:2:2).
- Balanced distribution across sources and scenes.

## Experimental Insights
### Generalization Challenges
- No single method consistently outperforms others across all sources, resolutions, or scenes.
- Deep learning models  show promise but require domain adaptation.

### Cross-Source and Cross-Scene Training
- Training with multi-source data improves generalization.

- Models trained on diverse scenes perform robustly across different scenes.

  ### 👁‍🗨👁‍🗨👁‍🗨**For comprehensive experimental results, detailed methodology, and complete analysis, please refer to the full research [paper](https://link.springer.com/article/10.1007/s44267-025-00091-0).**

## Downstream Applications
🛰**Multimodal Change Detection**: High-precision optical-SAR image matching enables reliable multimodal change detection. 3MOS supports training of satellite-specific registration networks, enhancing robustness for disaster response applications including earthquake damage assessment and flood monitoring.

✈**Visual Navigation**: Under GNSS-denied operational scenarios, optical-SAR image matching provides a viable solution for UAV visual positioning. However, existing datasets typically overlook the significant domain gaps arising from diverse SAR data sources and scene variations. 3MOS establishes a comprehensive benchmark that systematically evaluates the robustness of matching methods across diverse sources and scenes.

🌈**Other Applications related to Optical-SAR Image Matching:**  Different downstream tasks prefer different SAR data sources and resolutions. For instance, SEN1 (C-band, 5 - 20m resolution) is frequently employed for landcover classification attributed to its free-access and global coverage capabilities, and TerraSAR-X (X-band, 1m high resolution) is widely applied in precise urban change detection. In such cases, optical-SAR matching methods should robustly align data from different data sources with varying spatial resolutions. Moreover, some downstream applications require scene-specific considerations. For example, glacier remote sensing requires optical-SAR image matching in frozen earth areas, whereas building type recognition focuses on matching image pairs in urban regions. 

✨**Unlocking New Possibilities**:  Multimodal Remote Sensing Image Fusion; Multimodal Remote Sensing Image Captioning;Cross-Modal Semantic Segmentation;Multimodal Pre-training Foundation Models;Cross-Modal Image Style Transfer......

## Multimodal Image Matching Evaluation

### Feature Matching Demo (Using [MINIMA](https://github.com/LSXI7/MINIMA?tab=readme-ov-file) Framework)

This demonstration provides a complete workflow for evaluating feature matching methods on the 3MOS dataset using the MINIMA framework. In the experimental setup, random rotations (from −5$^\circ$ to 5$^\circ$) and translations (from −30 to 30 pixels) were applied to SAR images to assess method robustness. The corner error between images warped with the estimated rigid transformation matrix $\hat{R}$ and the ground truth $R$ was computed as a correctness measure. The area under the cumulative curve (AUC@T) was used as the evaluation metric, with threshold values of $T={3,5,10}$ pixels.

**1. Repository Setup & Environment Configuration & Weights Download**  

Follow the official installation instructions from  [MINIMA](https://github.com/LSXI7/MINIMA?tab=readme-ov-file)

```
git clone https://github.com/LSXI7/MINIMA.git
cd MINIMA
conda env create -f environment.yaml
conda activate minima
```

**2. Configuration Setup**

Modify lines 189-190 in `.\MIR_master\MINIMA-main\src\config\default.py`:

```
_CN.TEST.IMG0_RESIZE = 256  # Changed from 640 to match test data
_CN.TEST.IMG1_RESIZE = 256  # Changed from 640 to match test data
```

**Notes:** The default configuration of the MINIMA framework specifies 640px resolution processing, whereas our experimental setup utilizes 256px test imagery. This resolution mismatch has been empirically observed to cause performance degradation in MINIMA-LoFTR's matching accuracy. This setting, however, has little impact on MINIMA-RoMa, as the input network dimensions are uniformly adjusted to 560px.

**3. Data Preparation**

- Download and extract `3MOS.rar`to an accessible directory (e.g., `/home/user/3mos_data`)
- Place the test index file `test_feature_matching.txt`in the project root
- Ensure `demo_feature_matching.py`and `utils.py`are located in `./MINIMA-main/`

**4. Execution Command**

```
cd MINIMA-main
python demo_feature_matching.py \
  --method loftr \
  --satellite_list GF3,ALOS \
  --data_base_path /home/user/3mos_data \
  --test_data_file test_feature_matching.txt
```

**Key Parameters:**

- `--method`: Matching method (sp_lg: SuperPoint+LightGlue)
- `--satellite_list`: Satellite data sources to evaluate
- `--data_base_path`: Path to extracted 3MOS dataset
- `--test_data_file`: Index file containing test pairs

**Notes:** Results may differ from paper stochastic nature of RANSAC process

### Template Matching Demo

This is the MATLAB code to evaluate the template matching method on the 3MOS dataset, using the [HOPC](https://github.com/yeyuanxin110/HOPC) method as an example.  In the experimental setup, sub-images were randomly cropped from SAR images to serve as templates. Meanwhile, the corresponding optical images were used as reference images. Additionally, the size of the template images was set to half the size of the reference images to evaluate the matching approaches in more challenging cases, bringing about a large search space for matching algorithms.

**1. Environment Setup**

Ensure the MATLAB working directory contains the following files:

-  Demo_template_matching.m
-  HOPC_FFT toolbox

**2. Modify the parameters before running the script**

| Parameter         | Description                  | Example                                 |
| ----------------- | ---------------------------- | --------------------------------------- |
| `base_path`       | Root directory of image data | `'.\3MOS'`                              |
| `txt_file_path`   | Test data index file path    | `'test_template_matching.txt'`          |
| `satellite_types` | Satellite types to test      | `{'ALOS','GF3','SEN','RCM','Radarsat'}` |

**3. Run the script with MATLAB ** 

```
Demo_template_matching
```

**Notes:** The results may differ from those reported in the paper, as we converted the original PNG format data to JPG during data release. This conversion reduces the image file size but also slightly degrades the image signal-to-noise ratio. 

## License
The 3MOS Dataset is released under the **Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License** ([CC BY-NC-ND 4.0]([Deed - Attribution-NonCommercial-NoDerivatives 4.0 International - Creative Commons](https://creativecommons.org/licenses/by-nc-nd/4.0/deed.en))).

## Citation
If you use the 3MOS dataset in your work, please cite:

```
@article{ye20253mos,
  title={3MOS: a multi-source, multi-resolution, and multi-scene optical-SAR dataset with insights for multi-modal image matching},
  author={Ye, Yibin and Teng, Xichao and Yang, Hongrui and Chen, Shuo and Sun, Yuli and Bian, Yijie and Tan, Tao and Li, Zhang and Yu, Qifeng},
  journal={Visual Intelligence},
  volume={3},
  number={1},
  pages={1--27},
  year={2025},
  publisher={Springer}
}
```

## Acknowledgements

We sincerely thank European Space Agency (ESA), Canadian Space Agency, Japan Aerospace Exploration Agency (JAXA), and China Center for Resources Satellite Data and Application for their SAR imagery, and Google Earth for optical imagery. The Sentinel-1 data, sourced from ESA, is utilized under the [CC BY 4.0 license]( https://creativecommons.org/licenses/by/4.0/). The RCM data are provided by the RADARSAT Constellation Mission (RCM) © Government of Canada (2023). RADARSAT is an official mark of the Canadian Space Agency. Data were accessed under the Public User License Agreement (CSA-RC-AGR-0005 Rev 1.0). The Radarsat-2 data  were obtained through the Canadian program for Science and Operational Applications Research for Radarsat-2 (SOAR). The original ALOS-2 data, provided by JAXA, were used iunder the End User License Agreement (EULA). For Google Earth data, we use them in line with the principles of fair use and adhere to the [guidelines](https://www.google.com/permissions/geoguidelines/) on the official website.

Additionally, we appreciate [MINIMA](https://github.com/LSXI7/MINIMA?tab=readme-ov-file) and [HOPC](https://github.com/yeyuanxin110/HOPC) for their contribution of methodological development. We also acknowledge  [FED-HOPC](https://github.com/yyb1234-56/FED-HOPC) for its implementation of FFT-accelerated NCC (Normalized Cross-Correlation).

## Help

If you have any questions, please contact us at zhangli_nudt@163.com
