import streamlit as st
import pandas as pd
import joblib

@st.cache_resource
def load_model():
    return joblib.load('churn_hgb_model.pkl')

model = load_model()

st.title("🚀 تطبيق التوقع الذكي لخدمات العملاء")
st.write("أدخل بيانات العميل في النموذج أدناه لمعرفة النتيجة بدقة.")

with st.form("prediction_form"):
    st.subheader("بيانات العميل")
    
    monthly_charge = st.number_input("Monthly Charge ($)", min_value=0.0, value=50.0)
    internet_type = st.selectbox("Internet Type", ["DSL", "Fiber Optic", "Cable", "None"])
    contract = st.selectbox("Contract Type", ["Month-to-Month", "One Year", "Two Year"])
    
   
    submit_button = st.form_submit_button(label="احصل على التوقع (Predict)")


if submit_button:
    # جمع البيانات في DataFrame بنفس ترتيب وأسماء أعمدة التدريب
    input_data = pd.DataFrame({
        'Monthly Charge': [monthly_charge],
        'Internet Type': [internet_type],
        'Contract': [contract]
    })
    
    try:
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data) if hasattr(model, "predict_proba") else None
        
        st.success("تم التوقع بنجاح! 🎉")
        st.write(f"**نتيجة التوقع:** {prediction[0]}")
        
        if prediction_proba is not None:
            st.write(f"**نسبة الثقة (Probability):** {prediction_proba[0].max() * 100:.2f}%")
            
    except Exception as e:
        st.error(f"حدث خطأ أثناء التوقع: {e}")