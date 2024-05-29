import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost.sklearn import XGBClassifier
import pandas as pd
import pickle
import time

historical_columns = ['step', 'type', 'amount', 'oldBalanceOrig', 'newBalanceOrig',
           'oldBalanceDest', 'newBalanceDest', 'errorBalanceOrig',
           'errorBalanceDest', 'isFraud']

# Create an empty DataFrame with the specified columns
historical_df = pd.DataFrame(columns=historical_columns)

with open("model.pkl","rb") as file:
    model = pickle.load(file)


st.title("Financial Fraud Detection")

nav = st.sidebar.radio("Navigation",["Home","Single Record Prediction","Multi Record Prediction","History","Insights"])
if nav == "Home":
    st.image("./images.jpg",width=550)
    st.write("Welcome to our financial fraud prediction web app. Input your data for personalized predictions. Explore historical analytics for deeper insights. Stay ahead of potential threats with our tailored guidance. Secure your financial future confidently using our platform.")
if nav == "Single Record Prediction":
    st.header("Single Record Prediction")
    st.text("Enter transaction record")

    col1,col2 = st.columns(2)
    with col1:
        transaction_type = col1.selectbox("Type",['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT', 'CASH_IN'])
    with col2:
        transaction_amount = col2.number_input("Amount")
        
    col3,col4,col5 = st.columns(3)
    with col3:
        col3.text_input("OriginID")
        pass
    with col4:
        origin_old = col4.number_input("OriginOldBalance")
    with col5:
        origin_new = col5.number_input("OriginNewBalance")

    col6,col7,col8 = st.columns(3)
    with col6:
        col6.text_input("DestinationID")
    with col7:
        dest_old = col7.number_input("DestinationOldBalance")
    with col8:
        dest_new = col8.number_input("DestinationNewBalance")

    type_dict = {
        'PAYMENT':0,
        'TRANSFER':1,
        'CASH_OUT':2,
        'DEBIT':3,
        'CASH_IN':4
    }


    
    
    if st.button("Predict Result"):
        error_orig = origin_new + transaction_amount - origin_old
        error_dest = dest_new + transaction_amount - dest_old
        # Create a dictionary containing the transaction record
        transaction_record = {
            'step': [1],  # Replace with the appropriate value
            'type': [type_dict[transaction_type]],  # Assuming 'type_dict' maps transaction types to integers
            'amount': [transaction_amount],
            'oldBalanceOrig': [origin_old],
            'newBalanceOrig': [origin_new],
            'oldBalanceDest': [dest_old],
            'newBalanceDest': [dest_new],
            'errorBalanceOrig': [origin_new + transaction_amount - origin_old],
            'errorBalanceDest': [dest_new + transaction_amount - dest_old]
        }

        # Create a DataFrame from the transaction record
        input_df = pd.DataFrame(transaction_record)

        # Make predictions using the model
        prediction = model.predict(input_df)
        input_df['isFraud'] = prediction
        # historical_df = historical_df.append(input_df,ignore_index=True)

        # Output the prediction result
        if prediction[0] == 0:
            st.write("Not-Fraud")
        elif prediction[0] == 1:
            st.write("Fraud")
        else:
            st.write("Unknown")


if nav == "Multi Record Prediction":
    st.header("Multi Record Prediction")
    csvfile = st.file_uploader('Upload csv file containing transaction data',type=['csv'])
    try :
        multi_input_df = pd.read_csv(csvfile)
    except Exception as e:
        print("Exception occurred",e)
    

    if st.button('Predict Results'):
        multi_prediction = model.predict(multi_input_df)
        result_df = multi_input_df
        result_df['isFraud'] = multi_prediction
        st.write("Prediction Results Generated")
        st.write(result_df)
        # historical_df.concat(multi_input_df,ignore_index=True)


if nav == "Insights":
    st.write("insights")