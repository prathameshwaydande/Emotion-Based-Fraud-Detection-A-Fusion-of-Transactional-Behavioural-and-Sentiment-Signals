#

  Emotion-Based Fraud Detection: Fusion of Transactional, Behavioural & Sentiment Signals

MSc Financial Technology (FinTech), University of Essex — 2025
Author: Prathamesh Yeshudas Waydande

---

## 📖 Project Overview
Financial fraud is one of the most persistent challenges in the digital economy.
Traditional fraud detection methods mainly rely on **transactional anomalies** and often generate excessive false positives.

This project proposes a **fusion-based fraud detection framework** that integrates three complementary dimensions:
- **Transactional data** (Kaggle Credit Card Fraud dataset — *not redistributed here*)
- **Behavioural biometrics** (synthetic keystroke dynamics, e.g. typing speed, dwell time, error rates)
- **Sentiment/emotional signals** (synthetic polarity and subjectivity features inspired by deception psychology)

The central thesis: **fraud is not only a numerical anomaly but also a behavioural and emotional deviation.**

---

## 🗂  Repository Structure
fraud-fusion-detection/
│──    fraud_fusion_starter.py # Core feature generation & model pipeline
│──    runner.py # Orchestration script (training)
│──    predict.py # Inference script (predictions)
│──    requirements.txt # Python dependencies
│──    README.md # Project documentation
│──    data/
└──    project/ # Generated outputs (transactions.csv, keystrokes.csv, text.csv, model, meta.json)


---

## ⚙️️ ️ Setup Instructions

1. Clone this repository:
   ```bash
   git clone git@github.com:prathameshwaydande/fraud-fusion-detection.git
   cd fraud-fusion-detection
                               
pip install -r requirements.txt

python runner.py

## Notes
- Kaggle dataset (`creditcard.csv`) is NOT included due to license.
- Synthetic keystroke & sentiment features are generated automatically.
- Outputs (model + metadata) are saved under `data/project/`.

## 📊 Dataset Information  

- **Kaggle Credit Card Fraud Detection dataset** is used for transactional data.  
- ⚠️ The dataset is **not uploaded to this repository** because Kaggle’s license does not allow redistribution.  

👉 You can download it directly from Kaggle here:  
🔗 [Credit Card Fraud Detection Dataset on Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud)  

After downloading:
1. Unzip if necessary and locate the file `creditcard.csv`.  
2. Place it in the project root folder (same level as `runner.py`).  
3. The pipeline will automatically detect and use it during training.  



