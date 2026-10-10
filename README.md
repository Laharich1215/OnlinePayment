# Online Payment Fraud Detection

This code implements Random Forest, Logistic Regression, and a hybrid stacking model for classifying payment transactions as legitimate (0) or fraudulent (1).

## Dataset expected
The script expects `onlinefraud.csv` with a target column named `isFraud`. The sample rows in the report show PaySim-style fields such as `step`, `type`, `amount`, balance columns, `isFlaggedFraud`, and `isFraud`.

**Dataset note:** The report's abstract mentions the Kaggle Credit Card Fraud Detection dataset, but the implementation/sample output uses `onlinefraud.csv` and `isFraud`, which is a different schema from the common credit-card dataset whose target is `Class`. If your CSV uses `Class`, change `target = "isFraud"` in the script to `target = "Class"`.

## Windows / VS Code setup
Put `onlinefraud.csv` beside `train_fraud_detection.py`, then run:

```powershell
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python train_fraud_detection.py
```

By default, the script uses a stratified sample of up to 500,000 rows for a more practical local run. To use all rows (may need substantial RAM and time):

```powershell
python train_fraud_detection.py --max-rows 0
```

To use a CSV at another location:

```powershell
python train_fraud_detection.py --data "C:\path\to\onlinefraud.csv"
```

## Outputs
The script creates an `outputs` directory containing model metrics, an accuracy comparison chart, and confusion-matrix images.

Accuracy from the report is not guaranteed to reproduce; assess fraud-class precision, recall, and F1-score as well.
