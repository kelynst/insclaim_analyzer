# 🏥  insclaim_analyzer  
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python tool for **exploring and analyzing health insurance claims data**.  
It uses **Pandas** and **Matplotlib/Seaborn** to provide insights into claim amounts, claim status, patient demographics, and provider performance.  

This project uses an open dataset from [Kaggle](https://www.kaggle.com/) (Health Insurance Claim Dataset).  
⚠️ *Data is synthetic/public — no real patient information is included.*  

---

## 📌  Project Overview  
- **Goal** → Build an analyzer to quickly summarize and visualize insurance claim data.  
- **Approach** → Load claims from Kaggle dataset (`insurance_claims.csv`), clean fields, run summaries, and generate charts.  
- **Status** → Portfolio project, extendable for fraud detection, cost analysis, or RCM automation.  

---

## 📂  Repo Structure  
insclaim_analyzer/  
│── analyze_claims.py       # Main analysis script  
│── requirements.txt        # Dependencies  
│── insurance_claims.csv    # Example Kaggle dataset (public/synthetic)  
│── README.md               # Documentation  
│── .gitignore              # Ignore .venv, caches, large files  
│── outputs/              # Example charts/visualizations

---

## ✅  Features  
- Load Kaggle insurance claim dataset (CSV).  
- Clean missing/duplicate claim records.  
- Summarize claim amounts (mean, median, min, max).  
- Group by **claim status**, **provider**, or **patient demographics**.  
- Visualize results with bar charts, histograms, and boxplots.  

---

## 📦  Requirements  
- Python 3.10+  
- `pip` (Python package manager)  
- Dependencies listed in `requirements.txt`  

To install requirements manually:  
```bash
pip install -r requirements.txt
```
## 1. 🚀  Installation
```bash
git clone https://github.com/kelynst/insclaim_analyzer.git
cd insclaim_analyzer
```

## 2. 🌱  Create a virtual environment
```bash
python3 -m venv .venv
```

## 3. ⚡  Activate the virtual environment
-macOS/Linux
```bash
source .venv/bin/activate
 ```

-Windows (PowerShell)
```bash
.venv\Scripts\Activate
```

## 4. 📦  Install dependencies
``` bash
pip install -r requirements.txt
```

## ▶️  Usage 
-Run analysis on the Kaggle dataset
```bash 
python analyze_claims.py insurance_claims.csv
```

 **Terminal Output (example):**
 ```
✅ Claims Analysis Complete
• Total Claims: 10,000
• Unique Patients: 8,200
• Average Billed Amount: $3,245.19
• Claim Status Breakdown:
	•	Paid: 72%
	•	Denied: 18%
	•	Pending: 10%
```


📊 Example chart output saved: `outputs/claim_status_distribution.png`

---

## 🔮 Future Improvements  
- Add fraud detection with ML models.  
- Compare **Billed vs Paid amounts** by provider.  
- Support multiple Kaggle claim datasets.  
- Export results to Excel or database.  

---

##  🤝  Contributing  
Fork the repo and submit pull requests with improvements (new visualizations, cleaning functions, ML add-ons).  

---

##  ⚠️  Notes  
- Dataset comes from **Kaggle** and is synthetic/public.  
- Always use HIPAA-compliant handling for real insurance claim data.  
- Large files are excluded in `.gitignore`.  

---

##  📜  License  
MIT License — see LICENSE.  