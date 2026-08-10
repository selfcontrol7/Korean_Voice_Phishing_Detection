# Data Statement for Korean Call Content Vishing (KorCCVi)

This document explains where the data used in this repository comes from, what is and is not distributed here, and how to obtain the underlying material yourself. It exists because the code in this repository is openly licensed (see [LICENSE](LICENSE)) but **the data is not ours to license.**

**Short version:** the code is MIT-licensed and free to use. The speech data is not redistributable by us; you must obtain it from the two original providers.

---

## 1. What this repository does and does not contain

**Contained:**


| Item                                                                                                                                                                          | Location                                                                                                    |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| Source code (preprocessing, feature extraction, models, training, evaluation)                                                                                                 | `Multimodal/`, `Attention/`, `KoBERT/`, `ML_DL_models/`, `Federated_Learning/`, `Multilingual_BT_approach/` |
| Dataset manifests: segment boundaries, labels, feature paths, **transcript text**, and **speaker-diarization labels** (`speaker`, `speaker_name`)                             | `Multimodal/data/*.jsonl`                                                                                   |
| Full ASR transcript JSON files (produced with Naver CLOVA) for all 1,417 calls: segment timestamps, text, word-level timings, per-segment confidence, and speaker diarization | `Multimodal/data/transcripts/{vishing,non_vishing}/`                                                        |


Speaker labels are **call-local**: speaker `"1"` in one call has no relation to speaker `"1"` in another call. No global speaker identity exists in the corpus.

**Not contained:**

- audio recordings (no `.wav`, `.mp3`, `.mp4` files). We may be able to provide the FSS audio files upon request;
- precomputed feature arrays (`.npy`: MFCC, eGeMAPS, Wav2Vec2.0, KoBERT, KR-SBERT). We may be able to provide them as well upon request.

> **Please read this carefully if you plan to reuse the manifests or transcripts.** The transcript text and its derivatives are provided to make the experimental protocol reproducible and inspectable. They are **derived from third-party recordings and are not relicensed by us.** The terms in §2 apply to them exactly as they apply to the audio. See also the privacy notice in §5.

---



## 2. Provenance and terms

KorCCVi is not a single dataset that we own. It is a **compilation derived from two independent third-party sources**, one per class. We cannot grant, license, or sublicense rights to either. You must obtain each yourself.

### 2.1 Vishing class (label = 1): Financial Supervisory Service (FSS)

Real voice-phishing call recordings published by the Korean Financial Supervisory Service (FSS) on its public "보이스피싱 지킴이" (Voice Phishing Guard) service, made available by FSS from recordings shared by victims.

- Source: [https://www.fss.or.kr/fss/main/sub1voice.do?menuNo=200012](https://www.fss.or.kr/fss/main/sub1voice.do?menuNo=200012)
- 654 recordings were collected; recordings containing multiple calls were split, yielding **706 vishing calls**.
- **Terms:** governed by FSS. Confirm permitted use with FSS directly for your intended purpose. We make no representation on your behalf.



### 2.2 Non-vishing class (label = 0): AI Hub

Legitimate telephone conversations from the AI Hub dataset「저품질 전화망 음성인식 데이터」 (*Low-quality telephone network speech recognition data*), operated by the National Information Society Agency (NIA).

- Source: [https://aihub.or.kr](https://aihub.or.kr)
- Approximately 6,500 hours were reviewed across education, public-service and e-commerce domains; samples were selected from categories deliberately distinct from financial transactions, then undersampled to **711 non-vishing calls** to approximately balance the vishing class.
- **Terms:** AI Hub distributes this dataset under its own agreement, which requires **per-user application and approval**. Redistribution is not permitted, so we cannot provide it. Apply through the AI Hub portal.



### 2.3 Transcription: Naver CLOVA Speech

All transcripts were produced with **Naver Cloud Platform CLOVA Speech** (`domain: general`, `lang: ko`, diarization enabled, `boostings: []`), applied identically to both classes. Transcripts are therefore **derived works** of both the source recordings and a commercial ASR service, and inherit the constraints above in addition to CLOVA's own terms of service.

---



## 3. Corpus composition (KorCCVi v2)


| Level    | Total    | Non-vishing (0) | Vishing (1) |
| -------- | -------- | --------------- | ----------- |
| Calls    | 1,417    | 711             | 706         |
| Segments | 39,429   | 12,255          | 27,174      |
| Duration | ≈ 66.9 h | ≈ 21 h 46 m     | ≈ 45 h 07 m |


Splits are **grouped at call level** (`random_state=42`), so no call contributes segments to more than one split:


| Split      | Calls | Segments | Non-vishing | Vishing |
| ---------- | ----- | -------- | ----------- | ------- |
| Train      | 1,134 | 31,443   | 9,813       | 21,630  |
| Validation | 141   | 4,061    | 1,296       | 2,765   |
| Test       | 142   | 3,925    | 1,146       | 2,779   |


---



## 4. Known limitation: source and label are perfectly correlated

Anyone building on this corpus should be aware of the following, and we would rather state it prominently than have it discovered downstream.

**Every vishing call comes from FSS and every non-vishing call comes from AI Hub. There are no FSS non-vishing calls and no AI Hub vishing calls.** The recording channel is therefore perfectly correlated with the class label, and a classifier can achieve very high scores by recognising the *source* rather than the *fraud*.

Measured consequences (reported in our acoustic paper, §2.1 of the citations below):

- A probe trained on **eleven channel-only acoustic statistics** (no linguistic content whatsoever) separates the two sources at test F1 ≈ 0.98.
- Under a **channel-matched evaluation** (GSM codec round-trip applied to the non-vishing audio to close the channel gap), frozen handcrafted-feature classifiers lose ≈ 30 F1 points and their false-positive rate rises from ≈ 1 % to 92–100 %.
- A frozen **Wav2Vec2.0** representation is essentially unaffected under the same test, and is the channel-robust configuration we recommend.

**Practical implication:** high accuracy on KorCCVi alone should not be read as evidence of real-world vishing detection performance. Evaluations that include a channel-matched or independently-collected condition are substantially more credible. No public Korean telephone-channel *non-vishing* corpus currently exists; building one remains an open and valuable contribution.

---



## 5. Privacy, ethics, and takedown

The vishing recordings are real calls, published by the Financial Supervisory Service as public awareness and education material. **Identifying information was removed by FSS before publication**, and we have not observed victim personal data in the transcripts.

For clarity, since automated scans of the transcript text do surface name-like and numeric strings:

- **Personal names appearing in the vishing transcripts may be fictitious identities used *within the scam scripts***: fraudsters impersonating prosecutors, investigators or bank staff, and naming invented suspects. They are part of the fraud content that FSS publishes deliberately, not victim identities.
- **Numeric strings are monetary amounts, years, and case references** (6–8 digits). No bank account numbers, resident registration numbers, or telephone numbers were found.

Even so, this remains sensitive material describing real criminal incidents, so please:

- do not use it to identify, contact, or profile any individual;
- do not redistribute it, in original or derived form (see §2. the terms are the providers', not ours);
- follow any human-subjects policy applying at your institution; and
- handle it in line with PIPA (개인정보 보호법).

**Takedown / correction:** if you are an affected individual, or a rights holder at FSS, AI Hub, or NIA, and you believe material in this repository should be removed or amended, please contact us at the address in §7. We will act promptly.

---



## 6. How to cite

If you use this code or build on this work, please cite the relevant paper(s):

```bibtex
@article{boussougou2023attention,
  title   = {Attention-Based 1D CNN-BiLSTM Hybrid Model Enhanced with FastText
             Word Embedding for Korean Voice Phishing Detection},
  author  = {Moussavou Boussougou, Milandu Keith and Park, Dong-Joo},
  journal = {Mathematics},
  volume  = {11},
  number  = {14},
  pages   = {3217},
  year    = {2023},
  doi     = {10.3390/math11143217}
}

@inproceedings{boussougou2025multimodal,
  title     = {Multimodal Detection of Korean Voice Phishing Using Audio and
               Text Fusion},
  author    = {Moussavou Boussougou, Milandu Keith and Song, M. and Jeong, B.
               and Hwang, Y. and Park, Dong-Joo},
  booktitle = {Proceedings of the Korea Computer Congress (KCC)},
  pages     = {1467--1469},
  year      = {2025}
}
```

Please also cite the two data providers, the Financial Supervisory Service (FSS) and AI Hub (NIA), as the sources of the underlying recordings.

*(An acoustic-feature paper covering the lightweight on-device pipeline and the source-confounding analysis summarised in §4 is currently under review; this section will be updated when it appears.)*

---



## 7. Contact

- **Milandu Keith Moussavou Boussougou**: Department of Computer Science and Engineering, Soongsil University, Seoul, Republic of Korea. E-mail: [mbmk92@soongsil.ac.kr, mbmk92@gmail.com](mailto:mbmk92@soongsil.ac.kr)
- **Prof. Dong-Joo Park:** School of Computer Science and Engineering, Soongsil University, Seoul, Republic of Korea. E-mail: [djpark@ssu.ac.kr](mailto:djpark@ssu.ac.kr)

For data-access questions, please note that we can help with **methodology, code, and the transcription/segmentation pipeline**. We cannot supply the AI Hub source data (apply through the AI Hub portal, §2.2), but we may be able to provide the FSS source data (audio files) upon request (§2.1).

---

*Last updated: 2026-08-06*