import replicate
import streamlit as st
import numpy as np
import pickle
from utils import icon
from streamlit_image_select import image_select

# Load models
with open('inflation_precit.pkl', 'rb') as model_file_inflation:
    model = pickle.load(model_file_inflation)

with open('sallary.pkl', 'rb') as model_file_salary:
    model_sal = pickle.load(model_file_salary)

with open('gold.pkl', 'rb') as model_file_gold:
    model_gold = pickle.load(model_file_gold)

with open('bse.pkl', 'rb') as model_file_bse:
    model_bse = pickle.load(model_file_bse)

# Function to calculate debt repayment plan
def debt_repayment_plan(current_debt, years, interest_rate):
    annual_payment = current_debt * (interest_rate / (1 - (1 + interest_rate) ** -years))
    remaining_debt = current_debt
    repayment_plan = []
    for year in range(1, years + 1):
        interest_payment = remaining_debt * interest_rate
        principal_payment = annual_payment - interest_payment
        remaining_debt -= principal_payment
        repayment_plan.append((year, principal_payment, interest_payment, remaining_debt))
    return repayment_plan

# Function to calculate expected monthly income next year
def expected_monthly_income_nextyear(current_income, income_increase_percentage):
    nextyear_income = current_income + (current_income * income_increase_percentage)
    return nextyear_income

# Function to calculate expected monthly expenses next year
def expected_monthly_expenses_nextyear(current_monthly_expense, inflation_percentage):
    nextyear_monthly_expense = current_monthly_expense + (current_monthly_expense * inflation_percentage)
    return nextyear_monthly_expense

# Streamlit app
st.title('Model Prediction')

# Sidebar for input
st.sidebar.title('Input Features')

# Input feature
with st.sidebar.form("my_form"):
    st.info("Yo! Start here ↓", icon="👋🏾")
    with st.expander(":rainbow[Financial Information]"):
        # Advanced Settings (for the curious minds!)
        year = st.number_input("Current Year", value=2022, step=1, format="%d")
        user_salary = st.number_input("Monthly income", value=0, step=10000, format="%d")
        user_expense = st.number_input("Monthly expense", value=0, step=10000, format="%d")
        user_debt = st.number_input("Loan", value=0, step=10000, format="%d")
        user_years = st.number_input("Tenure of Liabilities (in years)", value=0, step=1, format="%d")
        user_interest_rate = st.slider("Interest Rate (in %)", min_value=0.0, max_value=100.0, step=0.01, value=0.0)
        submitted = st.form_submit_button("Submit")

    # Button to make predictions and calculations
    if submitted:
        # Make prediction
        user_data = np.array([[year]])  # Adjust input based on your model's requirements
        predic_inflation = model.predict(user_data)
        predic_salary = model_sal.predict(user_data)
        predic_gold = model_gold.predict(user_data)
        predic_bse = model_bse.predict(user_data)

        # Display predictions
        st.write('Inflation Prediction of Entered year:', predic_inflation[0])
        st.write('Salary Prediction:', predic_salary[0])
        st.write('Gold Prediction:', predic_gold[0])
        st.write('BSE Prediction:', predic_bse[0])

        # Display debt repayment plan
        debt_plan = debt_repayment_plan(user_debt, user_years, user_interest_rate)
        st.write("Debt Repayment Plan")
        st.write("| Year | Principal Payment | Interest Payment | Remaining Debt |")
        st.write("|------|------------------|------------------|----------------|")
        for year, principal_payment, interest_payment, remaining_debt in debt_plan:
            st.write(f"| {year} | {principal_payment} | {interest_payment} | {remaining_debt} |")

        # Calculate expected income and expenses
        expected_income = expected_monthly_income_nextyear(user_salary, predic_salary[0])
        expected_expenses = expected_monthly_expenses_nextyear(user_expense, predic_inflation[0])
        st.write('Expected Monthly Income Next Year:', expected_income)
        st.write('Expected Monthly Expenses Next Year:', expected_expenses)
        
        # Calculate savings
        savings = (user_salary * 12) - ((user_expense * 12) + (user_debt / user_years))
        st.write(f"Savings: ₹{savings:.2f}")
        
        # Make prediction for the year
        predic_gold = model_gold.predict(np.array([[year]]))
        predic_bse = model_bse.predict(np.array([[year]]))
        
        # Compare gold and BSE predictions and print recommendation
        if predic_gold[0] > predic_bse[0]:
            st.write("Recommendation: It's recommended to invest more in gold, since gold rates are higher in this financial year.")
        else:
            st.write("Recommendation: It's recommended to invest more in BSE, since returns in BSE are higher in this financial year.")