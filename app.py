# ============================================================
# Alexandria Weather Forecast
# AI-powered next-day temperature prediction
# ============================================================
import os
import pickle
import requests
import numpy as np
import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras import Sequential
from tensorflow.keras.layers import GRU, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
# ============================================================
# 1. Project Paths
# ============================================================
# تحديد مسار فولدر المشروع تلقائيًا
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(
    PROJECT_DIR,
    "alexandria_temperature_model.keras"
)
FEATURE_SCALER_FILE = os.path.join(
    PROJECT_DIR,
    "alexandria_feature_scaler.pkl"
)
TARGET_SCALER_FILE = os.path.join(
    PROJECT_DIR,
    "alexandria_target_scaler.pkl"
)
DATA_FILE = os.path.join(
    PROJECT_DIR,
    "alexandria_weather_data.csv"
)
# ============================================================
# 2. Project Settings
# ============================================================
LOOKBACK = 7
# إحداثيات الإسكندرية
LATITUDE = 31.2001
LONGITUDE = 29.9187
TIMEZONE = "Africa/Cairo"
# ============================================================
# 3. Page Settings
# ============================================================
st.set_page_config(
    page_title="Alexandria Weather Forecast",
    page_icon="☀️",
    layout="wide"
)
# ============================================================
# 4. Website Background
# ============================================================
# خلفية سماء زرقاء مع شمس باستخدام CSS
st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(
                circle at 85% 15%,
                #fffbd0 0%,
                #fff08a 4%,
                #ffd83d 8%,
                rgba(255, 216, 61, 0.45) 12%,
                transparent 19%
            ),
            linear-gradient(
                180deg,
                #55bce9 0%,
                #8ed5f2 42%,
                #dff5fc 100%
            );
        background-attachment: fixed;
    }
    /* مساحة الموقع */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    /* ألوان النص */
    h1, h2, h3, p, label {
        color: #082b45 !important;
    }
    /* شكل الأزرار */
    .stButton > button {
        background-color: rgba(255, 255, 255, 0.92);
        color: #082b45;
        border: 2px solid rgba(255, 255, 255, 0.95);
        border-radius: 14px;
        font-size: 18px;
        font-weight: 700;
        padding: 12px;
        transition: 0.2s;
    }
    /* Hover */
    .stButton > button:hover {
        background-color: #fff3a6;
        color: #082b45;
        border-color: #ffffff;
    }
    </style>
    """,
    unsafe_allow_html=True
)
# ============================================================
# 5. Load Weather Data
# ============================================================
@st.cache_data
def load_weather_data():
    data = pd.read_csv(DATA_FILE)
    # تنظيف أسماء الأعمدة
    data.columns = (
        data.columns
        .astype(str)
        .str.strip()
    )
    # أسماء الأعمدة المحتملة للتاريخ
    possible_date_columns = [
        "date",
        "Date",
        "DATE",
        "time",
        "Time",
        "datetime",
        "Datetime"
    ]
    date_column = None
    for column in possible_date_columns:
        if column in data.columns:
            date_column = column
            break
    # لو مفيش تاريخ
    if date_column is None:
        st.error(
            "لم يتم العثور على عمود التاريخ."
        )
        st.write(
            "الأعمدة الموجودة:"
        )
        st.write(
            list(data.columns)
        )
        st.stop()
    # توحيد اسم التاريخ
    if date_column != "date":
        data = data.rename(
            columns={
                date_column: "date"
            }
        )
    # تحويل التاريخ
    data["date"] = pd.to_datetime(
        data["date"],
        errors="coerce"
    )
    # حذف التواريخ غير الصحيحة
    data = data.dropna(
        subset=["date"]
    )
    # ترتيب البيانات
    data = data.sort_values(
        "date"
    ).reset_index(drop=True)
    return data
# ============================================================
# 6. Load Prediction Model
# ============================================================
@st.cache_resource
def load_prediction_model():
    return load_model(
        MODEL_FILE
    )
# ============================================================
# 7. Load Feature Scaler
# ============================================================
@st.cache_resource
def load_feature_scaler():
    with open(
        FEATURE_SCALER_FILE,
        "rb"
    ) as file:
        return pickle.load(file)
# ============================================================
# 8. Load Target Scaler
# ============================================================
@st.cache_resource
def load_target_scaler():
    with open(
        TARGET_SCALER_FILE,
        "rb"
    ) as file:
        return pickle.load(file)
# ============================================================
# 9. Load Everything
# ============================================================
df = load_weather_data()
model = load_prediction_model()
feature_scaler = load_feature_scaler()
target_scaler = load_target_scaler()
# ============================================================
# 10. Required Columns
# ============================================================
required_columns = [
    "Temperature_Mean",
    "Temperature_Max",
    "Temperature_Min"
]
# التأكد من وجود الأعمدة
missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]
if missing_columns:
    st.error(
        f"Missing columns: {missing_columns}"
    )
    st.write(
        list(df.columns)
    )
    st.stop()
# ============================================================
# 11. Website Header
# ============================================================
st.title(
    "☀️ Alexandria Weather Forecast"
)
st.write(
    "AI-powered next-day temperature prediction"
)
st.divider()
# ============================================================
# 12. Main Buttons
# ============================================================
col1, col2 = st.columns(
    [1, 1]
)
# ============================================================
# 13. Predict Tomorrow Button
# ============================================================
with col1:
    st.subheader(
        "🌡️ Temperature Prediction"
    )
    predict_button = st.button(
        "☀️ توقع حرارة بكرة",
        use_container_width=True
    )
# ============================================================
# 14. Retrain Button
# ============================================================
with col2:
    st.subheader(
        "🔄 تجديد البيانات"
    )
    st.caption(
        "بيجدد البيانات بتاعة الحرارة اليومية "
        "ثم يعيد تدريب النموذج."
    )
    retrain_button = st.button(
        "🔄 Retrain Model",
        use_container_width=True
    )
# ============================================================
# 15. Predict Tomorrow Function
# ============================================================
def predict_next_day():
    # آخر 7 أيام
    recent_data = df[
        required_columns
    ].tail(LOOKBACK)
    # التأكد من وجود بيانات كافية
    if len(recent_data) < LOOKBACK:
        raise ValueError(
            "لا توجد بيانات كافية للتنبؤ."
        )
    # تحويل البيانات إلى أرقام
    values = recent_data.values.astype(float)
    # Scaling
    scaled_values = (
        feature_scaler.transform(values)
    )
    # تجهيز شكل البيانات للموديل
    X_input = np.expand_dims(
        scaled_values,
        axis=0
    )
    # Prediction
    prediction_scaled = model.predict(
        X_input,
        verbose=0
    )
    # إرجاع درجة الحرارة للوحدة الأصلية
    prediction = (
        target_scaler.inverse_transform(
            prediction_scaled
        )
    )
    return float(
        prediction[0][0]
    )
# ============================================================
# 16. Prediction Button Action
# ============================================================
if predict_button:
    try:
        prediction = predict_next_day()
        latest_date = df.iloc[-1]["date"]
        tomorrow = (
            latest_date
            + pd.Timedelta(days=1)
        )
        st.success(
            "تم التنبؤ بنجاح!"
        )
        st.metric(
            label=(
                "🌡️ درجة الحرارة المتوقعة يوم "
                f"{tomorrow.strftime('%d/%m/%Y')}"
            ),
            value=f"{prediction:.1f} °C"
        )
    except Exception as error:
        st.error(
            "حدث خطأ أثناء التنبؤ."
        )
        st.exception(error)
# ============================================================
# 17. Download Recent Weather
# ============================================================
def download_recent_weather():
    url = (
        "https://archive-api.open-meteo.com/v1/archive"
    )
    # آخر 30 يوم
    start_date = (
        pd.Timestamp.today()
        - pd.Timedelta(days=30)
    ).strftime("%Y-%m-%d")
    end_date = (
        pd.Timestamp.today()
    ).strftime("%Y-%m-%d")
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": start_date,
        "end_date": end_date,
        "daily":
            "temperature_2m_mean,"
            "temperature_2m_max,"
            "temperature_2m_min",
        "timezone": TIMEZONE
    }
    response = requests.get(
        url,
        params=params,
        timeout=30
    )
    response.raise_for_status()
    result = response.json()
    if "daily" not in result:
        raise ValueError(
            "لم يتم استرجاع بيانات الطقس."
        )
    daily = result["daily"]
    new_data = pd.DataFrame(
        {
            "date": daily["time"],
            "Temperature_Mean":
                daily["temperature_2m_mean"],
            "Temperature_Max":
                daily["temperature_2m_max"],
            "Temperature_Min":
                daily["temperature_2m_min"]
        }
    )
    new_data["date"] = pd.to_datetime(
        new_data["date"]
    )
    return new_data
# ============================================================
# 18. Build GRU Model
# ============================================================
def build_gru_model():
    new_model = Sequential(
        [
            GRU(
                64,
                input_shape=(
                    LOOKBACK,
                    3
                )
            ),
            Dropout(0.2),
            Dense(
                32,
                activation="relu"
            ),
            Dense(1)
        ]
    )
    new_model.compile(
        optimizer="adam",
        loss="mse"
    )
    return new_model
# ============================================================
# 19. Create Sequences
# ============================================================
def create_sequences(
    feature_values,
    target_values
):
    X = []
    y = []
    for i in range(
        LOOKBACK,
        len(feature_values)
    ):
        X.append(
            feature_values[
                i - LOOKBACK:i
            ]
        )
        y.append(
            target_values[i]
        )
    return (
        np.array(X),
        np.array(y)
    )
# ============================================================
# 20. Retrain Button Action
# ============================================================
if retrain_button:
    try:
        # ----------------------------------------------------
        # Download latest weather
        # ----------------------------------------------------
        with st.spinner(
            "🔄 بيتم تحديث بيانات الإسكندرية..."
        ):
            new_weather = (
                download_recent_weather()
            )
        # ----------------------------------------------------
        # Combine old + new data
        # ----------------------------------------------------
        combined = pd.concat(
            [
                df[
                    [
                        "date",
                        "Temperature_Mean",
                        "Temperature_Max",
                        "Temperature_Min"
                    ]
                ],
                new_weather
            ],
            ignore_index=True
        )
        # ----------------------------------------------------
        # Remove duplicate dates
        # ----------------------------------------------------
        combined = (
            combined
            .drop_duplicates(
                subset=["date"],
                keep="last"
            )
            .sort_values("date")
            .reset_index(drop=True)
        )
        # ----------------------------------------------------
        # Remove missing values
        # ----------------------------------------------------
        combined = combined.dropna(
            subset=required_columns
        )
        # ----------------------------------------------------
        # Check data size
        # ----------------------------------------------------
        if len(combined) <= LOOKBACK:
            st.error(
                "عدد البيانات غير كافٍ لإعادة التدريب."
            )
            st.stop()
        # ----------------------------------------------------
        # Features
        # ----------------------------------------------------
        features = combined[
            required_columns
        ].values.astype(float)
        # ----------------------------------------------------
        # Target
        # ----------------------------------------------------
        target = combined[
            "Temperature_Mean"
        ].values.reshape(-1, 1)
        # ----------------------------------------------------
        # Create new scalers
        # ----------------------------------------------------
        new_feature_scaler = (
            MinMaxScaler()
        )
        new_target_scaler = (
            MinMaxScaler()
        )
        # Scaling
        scaled_features = (
            new_feature_scaler
            .fit_transform(features)
        )
        scaled_target = (
            new_target_scaler
            .fit_transform(target)
        )
        # ----------------------------------------------------
        # Create sequences
        # ----------------------------------------------------
        X, y = create_sequences(
            scaled_features,
            scaled_target
        )
        # ----------------------------------------------------
        # Retrain model
        # ----------------------------------------------------
        with st.spinner(
            "🧠 بيتم إعادة تدريب النموذج..."
        ):
            new_model = build_gru_model()
            early_stopping = EarlyStopping(
                monitor="val_loss",
                patience=10,
                restore_best_weights=True
            )
            new_model.fit(
                X,
                y,
                epochs=100,
                batch_size=32,
                validation_split=0.2,
                callbacks=[
                    early_stopping
                ],
                verbose=0
            )
        # ----------------------------------------------------
        # Save model
        # ----------------------------------------------------
        new_model.save(
            MODEL_FILE
        )
        # ----------------------------------------------------
        # Save feature scaler
        # ----------------------------------------------------
        with open(
            FEATURE_SCALER_FILE,
            "wb"
        ) as file:
            pickle.dump(
                new_feature_scaler,
                file
            )
        # ----------------------------------------------------
        # Save target scaler
        # ----------------------------------------------------
        with open(
            TARGET_SCALER_FILE,
            "wb"
        ) as file:
            pickle.dump(
                new_target_scaler,
                file
            )
        # ----------------------------------------------------
        # Save updated data
        # ----------------------------------------------------
        combined.to_csv(
            DATA_FILE,
            index=False
        )
        # ----------------------------------------------------
        # Clear Streamlit cache
        # ----------------------------------------------------
        st.cache_data.clear()
        st.cache_resource.clear()
        # ----------------------------------------------------
        # Success
        # ----------------------------------------------------
        st.success(
            "✅ تم تجديد البيانات وإعادة تدريب النموذج بنجاح!"
        )
        st.rerun()
    except Exception as error:
        st.error(
            "حدث خطأ أثناء تحديث البيانات أو إعادة التدريب."
        )
        st.exception(error)
# ============================================================
# 21. Temperature Trend
# ============================================================
st.divider()
st.subheader(
    "📈 Temperature Trend"
)
chart_data = df.tail(30).copy()
chart_data = chart_data.set_index(
    "date"
)
st.line_chart(
    chart_data[
        [
            "Temperature_Mean",
            "Temperature_Max",
            "Temperature_Min"
        ]
    ]
)
# ============================================================
# 22. Last 7 Days
# ============================================================
st.subheader(
    "📅 Last 7 Days"
)
last_7 = df.tail(7).copy()
display_data = last_7[
    [
        "date",
        "Temperature_Mean",
        "Temperature_Max",
        "Temperature_Min"
    ]
].copy()
display_data["date"] = (
    display_data["date"]
    .dt.strftime("%Y-%m-%d")
)
display_data = display_data.rename(
    columns={
        "date": "Date",
        "Temperature_Mean":
            "Mean °C",
        "Temperature_Max":
            "Max °C",
        "Temperature_Min":
            "Min °C"
    }
)
st.dataframe(
    display_data,
    use_container_width=True,
    hide_index=True
)
# ============================================================
# 23. Footer
# ============================================================

st.divider()

st.markdown(
    "### ☀️ Powered by **Hady Mohamed Kamal** 👨‍💻"
)

st.caption(
    "🌊 Alexandria Weather Forecast • AI & Machine Learning"
)

