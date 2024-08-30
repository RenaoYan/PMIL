# PMIL
Official code for "Shapley Values-enabled Progressive Pseudo Bag Augmentation for Whole-Slide Image Classification", IEEE Transaction on Medical Imaging, 2024.

[[ArXiv]](https://arxiv.org/abs/2312.05490) | [[Preprocessing]](https://github.com/RenaoYan/PMIL?tab=readme-ov-file#Preprocessing) | [[Workflow]](https://github.com/RenaoYan/PMIL?tab=readme-ov-file#Workflow) | [[Citation]](https://github.com/RenaoYan/PMIL?tab=readme-ov-file#Citation)

**Abstract**: In computational pathology, whole-slide image (WSI) classification presents a formidable challenge due to its gigapixel resolution and limited fine-grained annotations. Multiple-instance learning (MIL) offers a weakly supervised solution, yet refining instance-level information from bag-level labels remains challenging. While most of the conventional MIL methods use attention scores to estimate instance importance scores (IIS) which contribute to the prediction of the slide labels, these often lead to skewed attention distributions and inaccuracies in identifying crucial instances. To address these issues, we propose a new approach inspired by cooperative game theory: employing Shapley values to assess each instance's contribution, thereby improving IIS estimation. The computation of the Shapley value is then accelerated using attention, meanwhile retaining the enhanced instance identification and prioritization. We further introduce a framework for the progressive assignment of pseudo bags based on estimated IIS, encouraging more balanced attention distributions in MIL models. Our extensive experiments on CAMELYON-16, BRACS, TCGA-LUNG, and TCGA-BRCA datasets show our method's superiority over existing state-of-the-art approaches, offering enhanced interpretability and class-wise insights.

## Preprocessing
### Dataset Spliting
Adopt a k-fold cross-validation protocol to split the dataset. Then obtain a DATA_SPLIT.csv at the specified DATA_SPLIT_DIRECTORY.

### Patch Tiling
Build patches for each whole slide images at a certain resolution, refer to [Build-Patch-for-Sdpc](https://github.com/RenaoYan/Build-Patch-for-Sdpc) or [CLAM](https://github.com/mahmoodlab/CLAM).

### Feature Extraction
Extract features for each whole slide images using a proper feature encoder, such as ResNet pretrained from ImageNet, [UNI](https://github.com/mahmoodlab/UNI) and etc. Then obtain the following folder structure at the specified FEATURE_DIRECTORY:
```bash
FEATURE_DIRECTORY/
	├── slide_001.pt
	├── slide_002.pt
	├── slide_003.pt
	└── ```
```

## Workflow
Train MIL models with Progressive Pseudo Bag Augmentation.

Apply **Shapley values** to estimate instance importance scores for PMIL.
```bash
python main.py --csv_dir DATA_SPLIT_DIRECTORY --feat_dir FEATURE_DIRECTORY --ckpt_dir CKPT_DIRECTORY --logger_dir LOGGER_DIRECTORY --metrics shap
```
Apply **attention scores** to estimate instance importance scores for PMIL.
```bash
python main.py --csv_dir DATA_SPLIT_DIRECTORY --feat_dir FEATURE_DIRECTORY --ckpt_dir CKPT_DIRECTORY --logger_dir LOGGER_DIRECTORY --metrics attn
```

Key configurations are as follows:
- General
  - `MIL_model`: network backbone.
  - `metric2save`: metrics to save best model.
  - `num_classes`: classification number. 
- Progressive pseudo bag augmentation
  - `search_rate`: refer $\mu$ in this paper.
  - `sample_rate`: refer $\tau$ in this paper.
  - `max_pseudo_num`: refer $M_{max}$ in this paper.
  - `pseudo_step`: refer $\Delta M$ in this paper.
  - `metrics`: IIS estimation metrics to sort.
- MIL training
  - `rounds`: rounds to train. 
  - `epochs`: MIL epochs to train in each round.
- Directory
  - `csv_dir`: csv dir to split data. 
  - `feat_dir`: train/val/test dir for features.
  - `ckpt_dir`: dir to save models. 
  - `logger_dir`: tensorboard dir.

## Citation
If you find our work useful in your research or if you use parts of this code please consider citing our paper:
```txt
@article{yan2023shapley,
  title={Shapley values-enabled progressive pseudo bag augmentation for whole slide image classification},
  author={Yan, Renao and Sun, Qiehe and Jin, Cheng and Liu, Yiqing and He, Yonghong and Guan, Tian and Chen, Hao},
  journal={arXiv preprint arXiv:2312.05490},
  year={2023}
}
```
