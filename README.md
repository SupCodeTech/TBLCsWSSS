`Note: As the paper is still in the REVIEW stage, the project code has not yet been fully uploaded. This project is still under construction` 

The code is based on [MONAI (AI Toolkit for Healthcare Imaging)](https://github.com/Project-MONAI/MONAI/tree/dev).

# Env configuration
```bash
pip install monai
conda create -n unest python=3.7
conda activate unest
pip install -r requirements.txt
```
# Datasets

The dataset is available in the [U.S. National Institute of Allergy & Infectious Diseases (NIH) TB portals dataset](https://tbportals.niaid.nih.gov/) and DEEPPulmtb [dataset](https://drive.google.com/drive/folders/1QPinffQ59BufdGapQyLykfaqOiDhtrnX?usp=sharing).

Please look at the Data Description below for detailed information about the dataset.

After decompressing the dataset, you can get the following directory:

```none
├── Train_data
│ ├── Training_Dataset
│ │ ├── TRN_00.nii.gz
│ │ ├── TRN_000.nii.gz
│ │ ├──  …
```

## Training Data (ImageCLEF training data) Preparation

<img width="3344" height="1512" alt="Figure1" src="https://github.com/user-attachments/assets/f09192f7-7efc-4fc3-bbf5-3f8e3915a6d0" />

<img width="2456" height="1111" alt="Figure2" src="https://github.com/user-attachments/assets/be9a6ec5-f269-4e09-b441-d117c11f4fdf" />


 We need to download the following datasets:

ImageCLEF 2022 Tuberculosis - [Caverns Report](https://www.aicrowd.com/challenges/imageclef-2022-tuberculosis-caverns-report) and [Caverns Detection](https://www.aicrowd.com/challenges/imageclef-2022-tuberculosis-caverns-detection)

and unzip the Cavern Detection Train CT files (1 to 7) and the Cavern Report Train CT files (1 to 2):

```none
-------------------------------------------------------------------------------------
|                     Zip files                                    Unzip files      |
-------------------------------------------------------------------------------------
| de0e8772-594d-41ce-9e85-578c4b59e9f3_detection_train_CT_1    (TRN_000 - TRN_099)  |
| 29cc320d-1c9d-4c7c-8d12-a006499c2f2f_detection_train_CT_2    (TRN_100 - TRN_179)  |
| 3b9da027-6015-4806-bbe9-04efb760ee53_detection_train_CT_3    (TRN_180 - TRN_269)  |
| c45b6a20-e3dc-49b5-bd93-1a86dc10721c_detection_train_CT_4    (TRN_270 - TRN_339)  |
| 447557ed-45dd-4468-8381-dbf0642b4312_detection_train_CT_5    (TRN_340 - TRN_419)  |
| f9da57c9-dbb5-4a3d-bebb-0618a8aef99f_detection_train_CT_6    (TRN_420 - TRN_499)  |
| eac9b86a-7673-4b24-924a-29529de6f130_detection_train_CT_7    (TRN_500 - TRN_558)  |
------------------------------------------------------------------------------------
| 45037ba5-e1e7-4011-98c6-35ed190e204a_cavern_report_train_CT_1   (TRN_00 - TRN_29) |
| e222bfea-28db-4d3a-b7cf-b68a7e18992b_cavern_report_train_CT_2   (TRN_30 - TRN_59) |
-------------------------------------------------------------------------------------
```
they will be placed in the following directory:

```none
├── Train_data
│ ├── Original_Image_Dataset
│ │ ├── TRN_00.nii.gz
│ │ ├── TRN_000.nii.gz
│ │ ├──  …

```
# SwinUNetR
## For supervised paradigm training
```bash
python main.py
```
## For weakly supervised paradigm training

Run the example script:  
```sh
sh scripts/train_r50_SwinUNeLCsXt.sh
```
## For Symmetrized Graph Convolutional Semantic Affinity
```bash
python train/SGCSA_Module.py
```
## For Class-driven Affinity Pseudo Label Generation
```bash
python train/CLS_CAM.py
```

## Affinity Pixel-Level Pseudo Refinement Adjustment
```bash
python train/APLPRA.py
```
## Basic Supervised Semantic Segmentation
```bash
coming soon
```

## Contact
If you have any questions, please feel free to contact me via tan.joey@pelajar.upm.edu.my

## LICENSE

This repo is under the Apache-2.0 license. For commercial use, please contact the authors.
