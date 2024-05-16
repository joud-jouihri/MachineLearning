from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load the pre-trained model and preprocessor
model = joblib.load('best_model.pkl')
preprocessor = joblib.load('preprocessor.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    print("Received form data:", request.form)
    try:
        # Get data from the HTML form and preprocess it
        month = request.form['month']
        age = float(request.form['age'])
        occupation = request.form['occupation']
        annual_income = float(request.form['annualIncome'])
        monthly_inhand_salary = float(request.form['monthlyInhandSalary'])
        num_bank_accounts = float(request.form['numBankAccounts'])
        num_credit_card = float(request.form['numCreditCard'])
        interest_rate = float(request.form['interestRate'])
        delay_from_due_date = float(request.form['delayFromDueDate'])
        num_of_delayed_payment = float(request.form['numOfDelayedPayment'])
        changed_credit_limit = float(request.form['changedCreditLimit'])
        num_credit_inquiries = float(request.form['numCreditInquiries'])
        credit_mix = request.form['creditMix']
        outstanding_debt = float(request.form['outstandingDebt'])
        credit_utilization_ratio = float(request.form['creditUtilizationRatio'])
        credit_history_age = float(request.form['creditHistoryAge'])
        payment_of_min_amount = request.form['paymentOfMinAmount']
        total_emi_per_month = float(request.form['totalEMIPerMonth'])
        amount_invested_monthly = float(request.form['amountInvestedMonthly'])
        payment_behaviour = request.form['paymentBehaviour']

        df = pd.DataFrame({
            'Month': [month],
            'Age': [age],
            'Occupation': [occupation],
            'Annual_Income': [annual_income],
            'Monthly_Inhand_Salary': [monthly_inhand_salary],
            'Num_Bank_Accounts': [num_bank_accounts],
            'Num_Credit_Card': [num_credit_card],
            'Interest_Rate': [interest_rate],
            'Delay_from_due_date': [delay_from_due_date],
            'Num_of_Delayed_Payment': [num_of_delayed_payment],
            'Changed_Credit_Limit': [changed_credit_limit],
            'Num_Credit_Inquiries': [num_credit_inquiries],
            'Credit_Mix': [credit_mix],
            'Outstanding_Debt': [outstanding_debt],
            'Credit_Utilization_Ratio': [credit_utilization_ratio],
            'Credit_History_Age': [credit_history_age],
            'Payment_of_Min_Amount': [payment_of_min_amount],
            'Total_EMI_per_month': [total_emi_per_month],
            'Amount_invested_monthly': [amount_invested_monthly],
            'Payment_Behaviour': [payment_behaviour]
        })

        processed_data = preprocessor.transform(df)

        # Make predictions
        prediction = model.predict(processed_data)

        # Extract the predicted label from the numpy array
        predicted_label = prediction[0]

        return render_template('result.html', prediction=predicted_label)

    except Exception as e:
        return render_template('result.html', prediction='Error: ' + str(e))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
