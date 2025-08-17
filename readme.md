This is the official repositiry for [FLIQA-AD: a Fusion Model with Large Language Model for Better Diagnose and MMSE Prediction of Alzheimer’s Disease](https://aclanthology.org/2025.naacl-short.49/) (Chen et al., NAACL 2025)

```
abstract = "Tracking a patient's cognitive status early in the onset of the disease provides an opportunity to diagnose and intervene in Alzheimer's disease (AD). However, relying solely on magnetic resonance imaging (MRI) images with traditional classification and regression models may not fully extract finer-grained information. This study proposes a multi-task Fusion Language Image Question Answering model (FLIQA-AD) to perform AD identification and Mini Mental State Examination (MMSE) prediction. Specifically, a 3D Adapter is introduced in Vision Transformer (ViT) model for image feature extraction. The patient electronic health records (EHR) information and questions related to the disease work as text prompts to be encoded. Then, an ADFormer model, which combines self-attention and cross-attention mechanisms, is used to capture the correlation between EHR information and structure features. After that, the extracted brain structural information and textual content are combined as input sequences for the large language model (LLM) to identify AD and predict the corresponding MMSE score. Experimental results demonstrate the strong discrimination and MMSE prediction performance of the model, as well as question-answer capabilities."
```
+ Installation :
You can refer to the environment download method of the 
[lavis library](https://github.com/salesforce/LAVIS) and install the missing packages

+ Dataset :
You need to download it from the [adni official](https://adni.loni.usc.edu/ )  website and do the basic preprocessing as described in the paper. You need to make a csv file and the corresponding image path.


If you're using FLIQA-AD in your research or applications, please cite it using this BibTeX:
```
@inproceedings{chen-etal-2025-fliqa,
    title = "{FLIQA}-{AD}: a Fusion Model with Large Language Model for Better Diagnose and {MMSE} Prediction of {A}lzheimer{'}s Disease",
    author = "Chen, Junhao  and Ding, Zhiyuan  et al.
    booktitle = "Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 2: Short Papers)",
    month = apr,
    year = "2025",
    address = "Albuquerque, New Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.naacl-short.49/",
    doi = "10.18653/v1/2025.naacl-short.49",
    pages = "587--594",
    ISBN = "979-8-89176-190-2",
}
```

## Acknowledgement
[lavis](https://github.com/salesforce/LAVIS) \
[BLIP-2](https://github.com/salesforce/LAVIS/tree/main/projects/blip2) \
[MedBLIP](https://github.com/Qybc/MedBLIP)

