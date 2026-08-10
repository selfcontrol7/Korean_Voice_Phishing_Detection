[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/selfcontrol7/Korean_Voice_Phishing_Detection/HEAD)

# AI-based_Korean_Voice_Phishing_Detection
## Detection of Korean Voice Phishing

This repository hosts our code related to the research on Korean voice phishing detection using various approaches, including Machine Learning (ML), Deep Learning (DL), Language Models (LM), Hybrid Models, Federated Learning (FL), and Multimodal (audio + text) learning.

> **License and data terms:** the source code is MIT-licensed (see [LICENSE](LICENSE)). The speech data is **not ours to license** — please read [DATA.md](DATA.md) for provenance, terms, and known limitations of the KorCCVi corpus before using any data in this repository.

The structure of the repository is as follows:
- **Multimodal**: the most recent and actively developed work — lightweight acoustic detection (MFCC, eGeMAPS, Wav2Vec2.0) and multimodal audio–text fusion on the KorCCVi v2 corpus, including preprocessing, feature extraction, model training/evaluation, call-level aggregation, channel-robustness analysis, and on-device benchmarking. Also contains the published dataset manifests and the full speaker-diarized transcripts (`Multimodal/data/`). [Link to the KCC 2025 paper](https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE12318548)
- **ML_DL_models [ML/DL_Models]**: This folder contains the implementation code for machine learning-based and deep learning-based detection models. [Link to the paper 1](https://www.dbpia.co.kr/Journal/articleDetail?nodeId=NODE10583070), [Link to the paper 2](https://doi.org/10.3745/PKIPS.y2021m11a.297)
- **KoBERT [Language_Models]**: This folder comprises the implementation code for language model-based detection models. [Link to the paper](https://doi.org/10.3745/KTSDE.2022.11.10.437)
- **Attention [Hybrid_Models]**: This folder includes the implementation code for hybrid detection models. [Link to the paper 1](https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE11113590), [Link to the paper](https://www.mdpi.com/2227-7390/11/14/3217)
- **FL model with KoBERTA [Federated_Learning / FL]**: These folders contain the code related to the Federated Learning approach for detection. [Link to the paper](https://www.dbpia.co.kr/Journal/articleDetail?nodeId=NODE11488126)
- **Multilingual_BT_approach**: back-translation data augmentation experiments.
- **Data_Collection_Preprocessing**: This folder contains code for raw data preprocessing and dataset creation.

## Datasets

In our research, we primarily created and employed the Korean Call Content Vishing (KorCCVi) dataset. It began as a collection of transcriptions of voice phishing attempts in Korean; the current version (**KorCCVi v2**) is a multimodal corpus of **1,417 calls (706 vishing / 711 non-vishing), 39,429 utterance-level segments, and roughly 67 hours of speech**, combining real voice-phishing recordings published by the Financial Supervisory Service with legitimate telephone conversations from AI Hub.

This repository publishes the parts of the corpus that we are able to share: the segment manifests (with labels, transcript text, and speaker-diarization fields) and the full ASR transcript JSON files for all 1,417 calls, under `Multimodal/data/`. The audio itself and the precomputed feature arrays are **not** distributed here — see [DATA.md](DATA.md) for the terms of the two data providers and how to obtain the source material.

> **Important limitation:** in KorCCVi v2, the data source and the class label are perfectly correlated (all vishing calls come from FSS, all non-vishing calls from AI Hub). Very high scores on this corpus can partly reflect recording-channel differences rather than vishing content. Please read [DATA.md](DATA.md) §4 before reporting results on this data.

Details of the collection and preprocessing can be found in [DATA.md](DATA.md) and in the `Multimodal/preprocessing/` folder; the earlier text-only pipeline is in 'Data_Collection_Preprocessing'. The exact implementation of the models can be found within each respective approach folder.

## Related Work

For additional experiments and more detailed discussions about the approaches used in this project, please refer to the work done by another user who forked this project: [https://github.com/kimdesok/Text-classification-of-voice-phishing-transcipts](https://github.com/kimdesok/Text-classification-of-voice-phishing-transcipts)

## Citations

Should you wish to cite our papers, you may use the following:

Multimodal Paper:
```bibtex
M. K. Moussavou Boussougou, M. Song, B. Jeong, Y. Hwang, and D.-J. Park, “Multimodal Detection of Korean Voice Phishing Using Audio and Text Fusion,” Proceedings of the Korea Computer Congress (KCC), Korean Institute of Information Scientists and Engineers, pp. 1467–1469, 2025.
```

Attention Papers:
```bibtex
M. K. Moussavou Boussougou and D.-J. Park, “Attention-Based 1D CNN-BiLSTM Hybrid Model Enhanced with FastText Word Embedding for Korean Voice Phishing Detection,” Mathematics, vol. 11, no. 14, p. 3217, Jul. 2023, doi: 10.3390/math11143217.
```
```bibtex
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
