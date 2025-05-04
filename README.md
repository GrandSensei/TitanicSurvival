## 🚢 Titanic Survival Prediction Web App
This is a simple python project to help me get a hang of the python libraries used for Data Analysis.
It is a simple web app to predict if a passenger survived the Titanic disaster. 
Funny thing in the csv file only two people are missing from embarking. From deduction you can assume that they were the original people boarding before Jack and his friend won the tickets in poker. 
But that should still mean that these two people should have been labelled embarked under the previous owner''s name.
Hence, we can conclude that Jack and the friend were most likely hallucinations of Rose...

The model is trained on the famous Kaggle Titanic dataset.

# 📌 Features

* Clean Streamlit web interface
* Logistic Regression model for prediction
* Real-time input via sidebar controls
* Categorical encoding handled for Sex, Embarked, and Title
* Displays survival prediction instantly

# 🧠 Machine Learning Model

Model: Logistic Regression, Linear Regression
Training Data: Cleaned version of the Titanic dataset from Kaggle
Features used:
* Pclass
* Sex (encoded)
* Age
* SibSp
* Parch
* Fare
* Embarked (one-hot encoded)
* Title (one-hot encoded)
* FamilySize

# 🛠 How to Run Locally

1. Clone the repository
```
git clone https://github.com/yourusername/titanic-survival-app.git
cd titanic-survival-app
```
2. Set up virtual environment (recommended)
```
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

4. Train the model (if not already done)

```
python main.py
```

5. Run the Streamlit app
```
streamlit run app.py
```
