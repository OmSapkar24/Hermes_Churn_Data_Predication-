# Hermes Churn Prediction

## Project Overview
This project aims to predict customer churn for the Hermes brand using a synthetic dataset. Customer churn refers to the likelihood of customers leaving a service, and predicting churn can help businesses take proactive measures to retain them.

---

## Dataset Information
The dataset includes 1,000 records with the following features:
- **Gender**: Male or Female.
- **Age**: Customer's age in years.
- **TenureMonths**: Number of months the customer has been with the service.
- **MonthlyCharges**: Monthly fee paid by the customer.
- **TotalCharges**: Total amount paid by the customer.
- **NumSupportCalls**: Number of support calls made by the customer.
- **IsSeniorCitizen**: Whether the customer is a senior citizen (0: No, 1: Yes).
- **Partner**: Whether the customer has a partner.
- **Dependents**: Whether the customer has dependents.
- **InternetService**: Type of internet service used.
- **Contract**: Contract type (e.g., Month-to-month, One year, Two year).
- **PaymentMethod**: How the customer pays for the service.
- **Churn**: Target variable (0: Not churned, 1: Churned).

---

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/your-username/Hermes-Churn-Prediction.git
cd Hermes-Churn-Prediction
pip install -r requirements.txt
```

---

## Usage

1. Open the Jupyter Notebook:

```bash
jupyter notebook Hermes_Churn_Data.ipynb
```

2. Run the notebook step-by-step to:
   - Load and preprocess the dataset.
   - Train a Decision Tree and XGBoost model.
   - Evaluate model performance.

---

## Results
- **Decision Tree**: Accuracy: 65%, ROC-AUC: 0.70
- **XGBoost**: Accuracy: 70%, ROC-AUC: 0.75

### Feature Importance (XGBoost)
![Feature Importance](results/feature_importance.png)

---

## Files
- `Hermes_Churn_Data.ipynb`: Jupyter Notebook with the full implementation.
- `hermes_churn_data_complex.csv`: Synthetic dataset used for the project.
- `requirements.txt`: Python libraries required to run the project.
- `results/`: Folder containing visualizations and evaluation metrics.

---

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments
- Synthetic dataset generated for demonstration purposes.
- Libraries used: pandas, scikit-learn, XGBoost, matplotlib.

---

## Contributing
Feel free to fork this repository, create a feature branch, and submit a pull request for improvements or additional features.
