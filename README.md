TravelSIM Advisor
An ML-powered mobile operator recommendation tool for tourists

Developed for learning by Balaji Gopinath

🌍 Overview

TravelSIM Advisor is a lightweight Machine Learning–powered web app that helps travelers choose the best mobile network operator for their trip abroad.

Tourists often buy SIM cards blindly at airports, not realizing that operators vary dramatically in:

Coverage

4G/5G network maturity

Spectrum/band support

Regional availability

This project solves that problem using a combination of telecom domain logic + ML clustering + itinerary awareness.

🎯 Purpose

This project was built as part of my journey to:

Strengthen my AI/ML fundamentals as a beginner

Apply ML to real-world business problems

Combine my experience in Program & Project Management and Process Excellence with practical AI

Build an end-to-end solution: data → ML → API → UI → deployment

✈️ What the app does

Users can:

Select their destination country

Enter their itinerary cities

Choose their priority:

Best Overall

Coverage

4G Experience

5G Readiness

Receive ranked operator recommendations powered by ML.

🧠 How the ML works

A KMeans clustering model groups operators based on telecom maturity using engineered features:

2G / 3G / 4G / 5G support

Spectrum band richness

Founded year (proxy for maturity/scale)

Network technology indicators

Each cluster is assigned a tech-maturity score, which is combined with:

Coverage score (based on spectrum/bands)

Itinerary score (city-region matching)

to compute the final recommendation ranking.

🔧 Tech Stack
ML & Data

Python

Pandas

scikit-learn (KMeans + StandardScaler Pipeline)

Joblib (model serialization)

Backend

Flask (API + routing)

Frontend

HTML5, CSS3 (Glassmorphism, gradient UI)

Responsive layout

Deployment

Render (Free tier)

📁 Project Structure
.
├── app.py
├── operator_cluster_model.pkl
├── 2025_06_16_MobileOperators.csv
├── requirements.txt
├── templates/
│   └── index.html
└── static/
    └── style.css

🚀 Running Locally
1. Clone the repo
git clone <your-repo-url>
cd <your-folder>

2. Install dependencies
pip install -r requirements.txt

3. Run the Flask server
python app.py

4. Open in browser
http://127.0.0.1:5000

🧪 ML Model Training (optional)

If you want to retrain the model:

Open the provided Google Colab notebook (or your own).

Upload 2025_06_16_MobileOperators.csv

Run the feature engineering + KMeans pipeline

Save the output as operator_cluster_model.pkl

Replace the file in the project root

💡 Why this project matters

This project demonstrates the combination of:

Telecom understanding

Machine Learning basics

Product thinking

Workflow automation mindset

UI/UX attention to detail

Full lifecycle execution (PM → Dev → Deploy)

A simple but meaningful example of how AI/ML can support real business decisions.

🌐 Live Demo

Render Deploy Link:
https://telecom-operator-ml.onrender.com/

📫 Contact / Connect

If you’d like to collaborate or discuss AI, telecom, PM, or automation — feel free to reach out.
