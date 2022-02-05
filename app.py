import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor
import pickle
import numpy as np
from sklearn.metrics import accuracy_score

with open('intreglab.pkl','rb') as f:
    regressor = pickle.load(f)
    

def main():
    from PIL import Image
    image = Image.open('tymlo.png')
    st.image(image,use_column_width=False)
    st.write("""
    # Interest rate prediction App
    # **This webapp predicts the peer 2 peer interest rate with ML explaination**!
    # """)
    st.write('---')
    st.write('<style>div.Widget.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
    funded_amnt = st.sidebar.number_input('Funded Amount(1000 to 35000)')
    annual_inc = st.sidebar.number_input('Annual Income (7500 to 1000000)')
    installment = st.sidebar.number_input('Installment (30.44 to 1388.44)')
    dti = st.sidebar.number_input('Debt to Income (0 to 34)')
    open_acc = st.sidebar.number_input('Number of Open Account individual have  (1 to 38)')
    total_acc = st.sidebar.number_input('Total Number of Accounts (3 to 68)')
    optionterminmonth= ['Thirty_Six','Sixty']
    TermInMonths_sb = st.sidebar.selectbox("Term in Months", optionterminmonth, key="radiob1")
    optionemplength = ['one','two','three','four','five','six','seven','eight','nine','ten']
    emp_length_sb = st.sidebar.selectbox("Emplyment length in year", optionemplength, key="selectb1")
    home_ownership_sb = st.sidebar.selectbox('Home Ownership', ['mortgage','rent','own','none'])
    loan_status_sb = st.sidebar.selectbox('loan status', ['current','fully_paid','chargedoff','latethirtyonetoonetwenty','ingraceperiod','latesixteentothirty','Default'])
    purpose_sb = st.sidebar.selectbox('Purpose', ['debt_consolidation','credit_card','home_improvement','other','major_purchase','small_business','car','medical','wedding','house','vacation','renewable_energy'])
    

    result = ""
    if st.button('Predict'):
        if TermInMonths_sb == "Thirty_Six":
            TermInMonths=1
        elif TermInMonths_sb == "Sixty":
            TermInMonths=0

        if emp_length_sb == "one":
            emp_length=4
        elif emp_length_sb == "two":
            emp_length=9
        elif emp_length_sb == "three":
            emp_length=8
        elif emp_length_sb == "four":
            emp_length=2
        elif emp_length_sb == "five":
            emp_length=1
        elif emp_length_sb == "six":
            emp_length=6
        elif emp_length_sb == "seven":
            emp_length=5
        elif emp_length_sb == "eight":
            emp_length=0
        elif emp_length_sb == "nine":
            emp_length=3
        elif emp_length_sb == "ten":
            emp_length=7

        if home_ownership_sb == "mortgage":
            home_ownership=0
        elif home_ownership_sb == "none":
            home_ownership=1
        elif home_ownership_sb == "own":
            home_ownership=2
        elif home_ownership_sb == "rent":
            home_ownership=3

        if loan_status_sb == "chargedoff":
            loan_status=0
        elif loan_status_sb == "current":
            loan_status=1
        elif loan_status_sb == "Default":
            loan_status=2
        elif loan_status_sb == "fully_paid":
            loan_status=3
        elif loan_status_sb == "ingraceperiod":
            loan_status=4
        elif loan_status_sb == "latesixteentothirty":
            loan_status=5
        elif loan_status_sb == "latethirtyonetoonetwenty":
            loan_status=6
        

        if purpose_sb == "car":
            purpose=0
        elif purpose_sb == "credit_card":
            purpose=1
        elif purpose_sb == "debt_consolidation":
            purpose=2
        elif purpose_sb == "home_improvement":
            purpose=3
        elif purpose_sb == "house":
            purpose=4
        elif purpose_sb == "major_purchase":
            purpose=5
        elif purpose_sb == "medical":
            purpose=6
        elif purpose_sb == "moving":
            purpose=7
        elif purpose_sb == "other":
            purpose=8
        elif purpose_sb == "renewable_energy":
            purpose=9
        elif purpose_sb == "small_business":
            purpose=10
        elif purpose_sb == "vacation":
            purpose=11
        elif purpose_sb == "wedding":
            purpose=12
        


        X = [[ funded_amnt,annual_inc,installment, dti,open_acc, total_acc,TermInMonths,emp_length, home_ownership,loan_status,purpose]]
        shap.initjs()
        st.set_option('deprecation.showPyplotGlobalUse', False)
       
        result = regressor.predict(X)
        st.write(result)
        data_for_prediction = pd.DataFrame(X, columns= ['funded_amnt','annual_inc','installment','dti','open_acc','total_acc','TermInMonths','emp_length', 'home_ownership','loan_status','purpose'])
        #data_for_prediction = pd.DataFrame({'funded_amnt':[funded_amnt],'annual_inc':[annual_inc],'installment':[installment],'dti':[dti],'open_acc':[open_acc],'total_acc':[total_acc],'TermInMonths':[TermInMonths],'emp_length':[emp_length],'home_ownership':[home_ownership],'loan_status':[loan_status],purpose':[loan_status,purpose],'total_acc':[k]})
        explainer = shap.TreeExplainer(regressor)
        shap_values = explainer.shap_values(data_for_prediction)
        st.header('Feature Importance')
        plt.title('Feature importance based on SHAP values')
        shap.summary_plot(shap_values, data_for_prediction)
        st.pyplot(bbox_inches='tight')
        st.write('---')
        plt.title('Feature importance based on SHAP values (Bar)')
        shap.summary_plot(shap_values,data_for_prediction, plot_type="bar")
        st.pyplot(bbox_inches='tight')
        
        
if __name__== '__main__' :
    main()



    
   