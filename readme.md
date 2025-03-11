# Fooling the Forgers: A Multi-Stage Framework for Audio Deepfake Detection

## 📜 Overview
Audio deepfakes pose a significant threat to digital trust and security. In this paper, we introduce a **multi-stage framework** for detecting audio deepfakes using **Generative Adversarial Networks (GANs) and contrastive learning**. Our approach enhances detection performance by:

- Extracting phonetic, speaker identity, and prosodic features using **Pre-trained Models (PTMs)**.
- Augmenting training data with **HiFi-GAN** to increase robustness against manipulations.
- Applying **contrastive learning** to improve discriminability between real and synthetic speech.

This framework outperforms existing methodologies in both accuracy and robustness.

## 📌 Key Contributions
✅ Use of **contrastive learning** to refine embeddings and improve detection.
✅ Integration of **GAN-based augmentation** to enhance training diversity.
✅ Evaluation across multiple datasets to ensure **cross-lingual generalization**.
✅ Performance comparison against **state-of-the-art (SOTA) models**.

---

## 🏆 Results
### 🔹 Baseline Model Performance
| PTM  | Dataset | Accuracy (%) | F1-Score (%) | AUC (%) | EER (%) |
|------|---------|--------------|--------------|---------|---------|
| XLS-R | ASV | 89.0 | 89.0 | 89.5 | 0.38 |
| XLS-R | ITW | 86.0 | 86.0 | 86.5 | 0.06 |
| XLS-R | D-E | 80.0 | 80.0 | 80.5 | 0.03 |
| Whisper | ASV | 87.0 | 87.0 | 87.5 | 1.00 |

### 🔹 GAN-Augmented Model Performance
| PTM  | Dataset | Accuracy (%) | F1-Score (%) | AUC (%) | EER (%) |
|------|---------|--------------|--------------|---------|---------|
| XLS-R | ASV | 92.0 | 92.2 | 93.5 | 0.37 |
| XLS-R | ITW | 89.5 | 89.5 | 90.3 | 0.05 |
| Whisper | ASV | 91.2 | 90.9 | 92.5 | 0.98 |

### 🔹 Contrastive Learning Model Performance
| PTM  | Dataset | Accuracy (%) | F1-Score (%) | AUC (%) | EER (%) |
|------|---------|--------------|--------------|---------|---------|
| XLS-R | ASV | 92.5 | 91.8 | 93.0 | 0.35 |
| XLS-R | ITW | 90.0 | 89.3 | 90.5 | 0.04 |

### 🔹 **Final Proposed Model (GAN + Contrastive Learning) Performance**
| PTM  | Dataset | Accuracy (%) | F1-Score (%) | AUC (%) | EER (%) |
|------|---------|--------------|--------------|---------|---------|
| XLS-R | ASV | **93.0** | **92.3** | **94.0** | **0.33** |
| XLS-R | ITW | **91.0** | **90.3** | **91.5** | **0.02** |
| Whisper | ASV | **91.5** | **90.8** | **92.5** | **0.85** |

### 📊 **Comparison with SOTA**
The proposed method **achieves lower Equal Error Rate (EER)** and outperforms prior deepfake detection models, demonstrating superior robustness across datasets.

---

## 📂 Datasets Used
- **ASVSpoof 2019 (LA)** - Logical Access dataset for deepfake detection.
- **In-the-Wild (ITW)** - Real-world deepfake samples from public figures.
- **DECRO** - Cross-lingual dataset for evaluating generalization.

---

## 🛠 Methodology
1. **Feature Extraction:** Pre-trained models extract speaker and phonetic features.
2. **GAN Augmentation:** HiFi-GAN generates synthetic training samples.
3. **Contrastive Learning:** Enhances feature separation for improved classification.
4. **Multi-Stage Classification:** Combines multiple approaches to refine accuracy.

---

## 🚀 Future Work
🔹 Exploring additional **Pre-Trained Models** for better feature extraction.
🔹 Enhancing **GAN-based augmentation** strategies for greater diversity.
🔹 Real-time **deepfake detection deployment** for security applications.

---

## 📜 Citation
If you find this work useful, please cite:
```bibtex
@inproceedings{ali2025fooling,
  title={Fooling the Forgers: A Multi-Stage Framework for Audio Deepfake Detection},
  author={Rafiq Ali, Gautam Siddharth Kashyap, Zohaib Hasan Siddiqui, Mohammad Anas Azeez, Shantanu Kumar, Navin Kamuni, Jiechao Gao},
  booktitle={ICASSP},
  year={2025}
}
```

---

## 🤝 Collaborators
- **Rafiq Ali** (DSEU, India)
- **Gautam Siddharth Kashyap** (Jamia Hamdard, India)
- **Zohaib Hasan Siddiqui** (Jamia Hamdard, India)
- **Mohammad Anas Azeez** (Jamia Hamdard, India)
- **Shantanu Kumar** (Amazon, USA)
- **Navin Kamuni** (Tech Mahindra, USA)
- **Jiechao Gao** (University of Virginia, USA)

---

## ⭐ Acknowledgments
Special thanks to the **ICASSP 2025 committee** for selecting this work for presentation.

📌 *For updates, follow this repository! 🚀*
