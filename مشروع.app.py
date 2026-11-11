import streamlit as st
import pandas as pd
import joblib

# 1. تحميل الموديل (وملف الـ Preprocessing لو حافظه لوحده، أو البايبلين مع بعض)
model = joblib.load('model.pkl')  # حط اسم ملف الموديل بتاعك هنا

# لو عندك preprocessor أو imputer حافظه لوحده:
# preprocessor = joblib.load('preprocessor.pkl')

st.title("🚀 تطبيق التوقع الذكي (Machine Learning App)")
st.write("أدخل البيانات المطلوبة في الأسفل لكي يقوم الموديل بالتوقع بدقة.")

# 2. إنشاء نموذج لإدخال البيانات من المستخدم (واجهة المستخدم)
# أمثلة لأنواع المدخلات (عدّلها حسب الأعمدة بتاعتك):
with st.form("prediction_form"):
    st.subheader("بيانات المدخلات")
    
    # مدخل رقمي (مثلاً Monthly Charge اللي كنت بتكلمه عنه)
    monthly_charge = st.number_input("Monthly Charge ($)", min_value=0.0, value=50.0)
    
    # مدخل نصي أو اختيار من متعدد (Categorical)
    internet_type = st.selectbox("Internet Type", ["DSL", "Fiber Optic", "Cable", "None"])
    
    contract = st.selectbox("Contract Type", ["Month-to-Month", "One Year", "Two Year"])
    
    # زرار التوقع
    submit_button = st.form_submit_button(label="احصل على التوقع (Predict)")

# 3. عند الضغط على زر التوقع
if submit_button:
    # جمع البيانات في DataFrame بنفس ترتيب الأعمدة اللي الموديل اتدرب عليها
    input_data = pd.DataFrame({
        'Monthly Charge': [monthly_charge],
        'Internet Type': [internet_type],
        'Contract': [contract]
        # ضيف باقي الأعمدة هنا بنفس الأسماء
    })
    
    try:
        # لو مدمج الـ Imputer والـ Scaling جوه Pipeline واحد، ابعت البيانات للموديل مباشرة:
        # (البايبلين هيتولى معالجة أي قيم نال أو تحويلات تلقائياً)
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data) if hasattr(model, "predict_proba") else None
        
        st.success("تم التوقع بنجاح! 🎉")
        st.write(**نتيجة التوقع:** {prediction[0]})
        
        if prediction_proba is not None:
            st.write(f"نسبة الثقة (Probability): {prediction_proba[0].max() * 100:.2f}%")
            
    except Exception as e:
        st.error(f"حدث خطأ أثناء التوقع: {e}")