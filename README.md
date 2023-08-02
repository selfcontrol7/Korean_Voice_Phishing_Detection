[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/selfcontrol7/Korean_Voice_Phishing_Detection/HEAD)

# AI-based_Korean_Voice_Phishing_Detection
## Detection of Korean Voice Phishing

This repository hosts our code related to the research on Korean voice phishing detection using various approaches, including Machine Learning (ML), Deep Learning (DL), Language Models (LM), Hybrid Models, and Federated Learning (FL).

The structure of the repository is as follows:
- **ML_DL_models [ML/DL_Models]**: This folder contains the implementation code for machine learning-based and deep learning-based detection models. [Link to the paper 1](https://www.dbpia.co.kr/Journal/articleDetail?nodeId=NODE10583070), [Link to the paper 2](https://doi.org/10.3745/PKIPS.y2021m11a.297)
- **KoBERT [Language_Models]**: This folder comprises the implementation code for language model-based detection models. [Link to the paper](https://doi.org/10.3745/KTSDE.2022.11.10.437)
- **Attention [Hybrid_Models]**: This folder includes the implementation code for hybrid detection models. [Link to the paper 1](https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE11113590), [Link to the paper](https://www.mdpi.com/2227-7390/11/14/3217)
- **[Federated_Learning] (To be uploaded soon)**: This folder contains the code related to the Federated Learning approach for detection. [Link to the paper](https://)
- **Data_Collection_Preprocessing**: This folder contains code for raw data preprocessing and dataset creation.

## Datasets

In our research, we primarily created and employed the Korean Call Content Vishing (KorCCVi) dataset, a comprehensive collection of transcriptions of voice phishing attempts in Korean. This dataset was curated to address the increasing incidence of voice phishing activities in Korea and provide a robust foundation for training and testing diverse voice phishing detection models.

The KorCCVi dataset features a wide variety of scenarios, mimicking the diverse strategies employed by voice phishers. It includes both successful and failed phishing attempts, making it an invaluable resource for studying the linguistic and paralinguistic characteristics inherent in voice phishing activities. This breadth of data helps in creating models that are capable of recognizing and mitigating a range of voice phishing tactics.

Details of the collection, preprocessing, and structure of the KorCCVi dataset can be found in the 'Data_Collection_Preprocessing' folder. Moreover, the exact implementation of the models on the KorCCVi dataset can be found within each respective approach folder (ML_Models, DL_Models, Language_Models, Hybrid_Models, Federated_Learning).

## Related Work

For additional experiments and more detailed discussions about the approaches used in this project, please refer to the work done by another user who forked this project: [https://github.com/kimdesok/Text-classification-of-voice-phishing-transcipts](https://github.com/kimdesok/Text-classification-of-voice-phishing-transcipts)

## Citations

Should you wish to cite our papers, you may use the following:

Attention Paper:
```bibtex
M. K. Moussavou Boussougou and D.-J. Park, “Attention-Based 1D CNN-BiLSTM Hybrid Model Enhanced with FastText Word Embedding for Korean Voice Phishing Detection,” Mathematics, vol. 11, no. 14, p. 3217, Jul. 2023, doi: 10.3390/math11143217.
M. K. Moussavou Boussougou, M.-G. Park, and D.-J. Park, “An Attention-Based CNN-BiLSTM Model for Korean Voice Phishing Detection,” Proceedings of the Korean Institute of Information Scientists and Engineers Korea Computer Congress; Korean Institute of Information Scientists: Jeju, Republic of Korea, pp. 1139–1141, June. 2022.
```

KoBERT Paper:
```bibtex
M. K. M. Boussougou and D.-J. Park, “Exploiting Korean Language Model to Improve Korean Voice Phishing Detection,” KIPS Transactions on Software and Data Engineering, vol. 11, no. 10, pp. 437–446, Oct. 2022.
```

ML/DL Paper:
```bibtex
M. K. M. Boussougou, S. Jin, D. Chang, and D.-J. Park, “Korean Voice Phishing Text Classification Performance Analysis Using Machine Learning Techniques,” Proceedings of the Korea Information Processing Society Conference, pp. 297–299, Nov. 2021.
M. K. M. Boussougou and D.-J. Park, “A Real-time Efficient Detection Technique of Voice Phishing with AI,” Proceedings of the Korean Institute of Information Scientists and Engineers Korea Computer Congress; Korean Institute of Information Scientists: Jeju, Republic of Korea, vol. 11, no. 10, pp. 768–770, June. 2021.
```
