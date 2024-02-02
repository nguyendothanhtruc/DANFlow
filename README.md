# DANFlow: Depth-wise separable convolution and Attention-based Normalizing Flow

## Key features

- A pipeline for localizing abnormal regions in input images:
    - Feature extraction stage: Extract features from input image using CNN or ViT.
    - 2D Normalizing Flows stage: Estimate the probability density from extracted features.
        - Extended Inverted Residual Bottleneck (EIRB) Module for solving the instability problem as well as enhance the
          power of flows for better anomaly localization.
        - Attention module for solving noise problem in unrelated areas and intensify concentration on specific
          abnormal areas.
    - Localization stage: Output binary localization map from the heatmap created by the 2D Normalizing Flows
      stage.

#### DANFlow

<img src="images/DANFlow.png">

#### Extended Inverted Residual Bottleneck (EIRB) Module

<img src="images/EIRB.png">

## Datasets:

- **MVTecAD**:
    - Website to the original dataset: https://www.mvtec.com/company/research/datasets/mvtec-ad
    - For more information about the train, and test samples, please visit <a href="datasets/MVTecAD"><strong>
      datasets/MVTecAD</strong></a> folder.
- **BTAD**:
    - Website to the original dataset: https://avires.dimi.uniud.it/papers/btad/btad.zip (Source: BeanTech srl)
    - For more information about the train, and test samples, please visit <a href="datasets/BTAD"><strong>
      datasets/BTAD</strong></a> folder.
- **KolektorSDD2**:
    - Website to the original dataset: https://www.vicos.si/resources/kolektorsdd2
    - For more information about the train, and test samples, please visit <a href="datasets/KolektorSDD2"><strong>
      datasets/KolektorSDD2</strong></a> folder.
- **VisA**:
    - Website to the original dataset: https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar
    - For more information about the train, and test samples, please visit <a href="datasets/VisA"><strong>
      datasets/VisA</strong></a> folder.

## Benchmark

### Quantitative comparison

The below tables compares pixel-wise anomaly localization performance on the several industrial benchmark. Top
performers are marked in **bold**.

#### Experiments on MVTecAD dataset (AUROC/AUPRO)

_Second-best is displayed in <ins>underlined</ins>. In cases where PRO scores are missing, a "-" will be displayed_

| <font color="00A36C">**MVTecAD**</font> | **PatchCore**               | **PaDiM** | **CFlow**                         | **MSFlow**                      | **MemSeg**           | **FastFlow**        | <font color="ff7518">**Ours**</font> |
|-----------------------------------------|-----------------------------|-----------|-----------------------------------|---------------------------------|----------------------|---------------------|--------------------------------------|
| carpet                                  | 99.0/96.6                   | 99.1/96.2 | <ins>99.25</ins>/<ins>97.70</ins> | <b>99.4/99.6</b>                | 99.2/\_              | <b>99.4/\_</b>      | <b>99.4/99.6</b>                     |
| grid                                    | 98.7/95.1                   | 97.3/94.6 | 98.99/96.08                       | <b>99.4</b>/<ins>99.1</ins>     | <ins>99.3 </ins>/\_  | 98.3/\_             | <b>99.41/99.52</b>                   |
| leather                                 | 99.3/98.9                   | 99.2/97.8 | <ins>99.66</ins>/<ins>99.35</ins> | <b>99.7/99.9</b>                | <b>99.7/\_</b>       | 99.5/\_             | <b>99.72</b>/<ins>99.58</ins>        |
| tile                                    | 95.6/87.4                   | 94.1/86.0 | 98.01/94.34                       | <ins>98.2</ins>/<ins>95.3</ins> | 98.0/\_              | 96.3/\_             | <b>99.20/98.45</b>                   |
| wood                                    | 95.0/89.6                   | 94.9/91.1 | 96.65/95.79                       | 97.1/<ins>96.6</ins>            | <b>99.1/\_</b>       | 97.0/\_             | <ins>98.62</ins>  / <b>98.06</b>     |
| bottle                                  | 98.6/96.1                   | 98.3/94.8 | 98.98/96.80                       | <ins>99.0</ins>/<ins>98.5</ins> | <b>99.3/\_</b>       | 97.7/\_             | <b>99.27/99.1</b>                    |
| cable                                   | <ins>98.4</ins>/92.6        | 96.7/88.8 | 97.64/93.53                       | <b>98.5</b>/<ins>93.7</ins>     | 97.4/\_              | <ins>98.4 </ins>/\_ | <b>98.5/98.42</b>                    |
| capsule                                 | 98.8/95.5                   | 98.5/93.5 | 98.98/93.40                       | <ins>99.1</ins>/<ins>98.4</ins> | <b>99.3/\_</b>       | <ins>99.1 </ins>/\_ | <ins>99.07</ins>/<ins>99.47</ins>    |
| hazelnut                                | 98.7/93.9                   | 98.2/92.6 | 98.89/96.68                       | 98.7/<ins>96.6</ins>            | 98.8/\_              | <ins>99.1 </ins>/\_ | <b>99.30/98.18</b>                   |
| metal nut                               | 98.4/91.3                   | 97.2/85.6 | 98.56/91.65                       | <b>99.3/97.6</b>                | <b>99.3/\_</b>       | 98.5/\_             | <ins>99.18</ins>/<ins>96.5</ins>     |
| pill                                    | 97.4/94.1                   | 95.7/92.7 | 98.95/95.39                       | 98.8/<ins>96.0</ins>            | <b>99.5/\_</b>       | <ins>99.2 </ins>/\_ | <ins>99.15</ins>  / <b>98.66</b>     |
| screw                                   | <b>99.4</b>/<ins>97.9</ins> | 98.5/94.4 | 98.86/95.30                       | <ins>99.1</ins>/94.2            | 98.0/\_              | <b>99.4/\_</b>      | <b>99.38/99.14</b>                   |
| toothbrush                              | 98.7/91.4                   | 98.8/93.1 | 98.93/<ins>95.06</ins>            | 98.5/91.6                       | <b>99.4/\_</b>       | 98.9/\_             | <ins>99.09</ins>/<b>99.77</b>        |
| transistor                              | 96.3/83.5                   | 97.5/84.5 | 97.99/81.40                       | <ins>98.3</ins>/<b>99.8</b>     | <b>98.8/\_</b>       | 97.3/\_             | 97.80/<ins>94.13</ins>               |
| zipper                                  | 98.8/97.1                   | 98.5/95.9 | 99.08/96.60                       | <ins>99.2</ins>/<b>99.4</b>     | 98.8/\_              | 98.7/\_             | <b>99.42</b>/<ins>99.2</ins>         |
| **Average**                             |                             |           |                                   |                                 |                      |                     |                                      |
| **Texture**                             | 97.5/93.7                   | 96.9/93.1 | 98.5/96.7                         | 98.8/<ins>98.1</ins>            | <ins>99.1 </ins>/\_  | 98.1/\_             | <b>99.27/99.04</b>                   |
| **Object**                              | 98.4/93.3                   | 97.8/91.6 | 98.7/93.6                         | <ins>98.8</ins>/<ins>96.6</ins> | 98.7/\_              | 98.6/\_             | <b>99.00/98.24</b>                   |
| **Mean**                                | 98.1/93.5                   | 97.5/92.1 | 98.6/94.6                         | <ins>98.8</ins>/<ins>97.1</ins> | <ins>98.84 </ins>/\_ | 98.5/\_             | <b>99.09/98.51</b>                   | 

#### Experiments on BTAD dataset (AUROC)

| <font color="00A36C">**BTAD**</font> | **PatchCore** | **PaDim** | **MemSeg** | **FastFlow** | <font color="ff7518">**Ours**</font> |
|--------------------------------------|---------------|-----------|------------|--------------|--------------------------------------|
| 0                                    | 95.5          | 97.0      | 98.9       | **97.71**    | 97.19                                |
| 1                                    | 94.7          | 96.0      | 96.2       | 95.94        | **96.86**                            |
| 2                                    | 99.3          | 98.8      | 96.3       | 99.47        | **99.64**                            |
| **Average**                          | 96.5          | 97.3      | 97.1       | 97.71        | **97.90**                            |

#### Experiments on KolektorSDD2 dataset (AUROC)

| <font color="00A36C">**KolektorSDD2**</font> | **CFLOW** | **FastFlow** | <font color="ff7518">**Ours**</font> |
|----------------------------------------------|-----------|--------------|--------------------------------------|
| **Average**                                  | 97.4      | 98.56        | **99.22**                            |

#### Experiments on VisA dataset (AUROC)

| <font color="00A36C">**VisA**</font> | **FastFlow** | <font color="ff7518">**Ours**</font> |
|--------------------------------------|--------------|--------------------------------------|
| candle                               | 98.84        | **99.23**                            |
| capsules                             | **99.46**    | 99.40                                |
| cashew                               | 99.60        | **99.68**                            |
| chewinggum                           | 99.54        | **99.56**                            |
| fryum                                | 96.53        | **97.43**                            |
| macaroni1                            | **99.80**    | 99.77                                |
| macaroni2                            | 98.08        | **99.18**                            |
| pcb1                                 | 99.77        | **99.81**                            |
| pcb2                                 | 98.88        | **98.92**                            |
| pcb3                                 | **99.13**    | **99.13**                            |
| pcb4                                 | 98.70        | **99.13**                            |
| pipe fryum                           | **99.58**    | 99.53                                |
| **Average**                          | 98.99        | **99.23**                            |

### Complexity evaluations

|              Back bone               |  FPS  | A.d Params (M) | AUROC |
|:------------------------------------:|:-----:|:--------------:|:-----:|
|       **DeiT-base-distilled**        |       |                |       |
|              Patch Core              | 13.81 |       0        | 97.9  |
|                CFlow                 | 15.66 |      10.5      | 97.9  |
|               FastFlow               | 26.10 |      14.8      | 98.1  |
| <font color="ff7518">**Ours**</font> | 97.82 |      10.7      | 99.09 |
|             **ResNet18**             |       |                |       |
|                SPADE                 | 4.52  |       0        |   _   |
|                CFlow                 | 19.72 |      5.5       | 98.1  |
|               FastFlow               | 26.67 |      4.9       | 97.2  |
| <font color="ff7518">**Ours**</font> | 46.1  |      3.5       | 98.22 |
|            **Eva Large**             |       |                |       |
| <font color="ff7518">**Ours**</font> | 79.3  |      19.0      | 98.87 |



