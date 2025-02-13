import streamlit as st
import pickle
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# Carica il modello, il MinMaxScaler e i LabelEncoders
with open("flight_price_predictor.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler_and_encoders.pkl", "rb") as f:
    scaler, label_encoders = pickle.load(f)

# Carica l'ordine delle feature
with open("feature_order.pkl", "rb") as f:
    feature_order = pickle.load(f)

# Interfaccia utente
st.title("Flight Price Predictor")

# Input utente
airline = st.selectbox("Compagnia Aerea", label_encoders["airline"].classes_)
source_city = st.selectbox("Città di Partenza", label_encoders["source_city"].classes_)
destination_city = st.selectbox("Città di Destinazione", label_encoders["destination_city"].classes_)
departure_time = st.selectbox("Orario di Partenza", label_encoders["departure_time"].classes_)
arrival_time = st.selectbox("Orario di Arrivo", label_encoders["arrival_time"].classes_)
stops = st.selectbox("Numero di Scali", label_encoders["stops"].classes_)
travel_class = st.selectbox("Classe", label_encoders["class"].classes_)
duration = st.number_input("Durata del Volo (in ore)", min_value=0.0, max_value=20.0, step=0.1)
days_left = st.number_input("Giorni Rimasti alla Partenza", min_value=0, max_value=365, step=1)

# Converti input in formato numerico
input_data = {
    "airline": airline,
    "source_city": source_city,
    "destination_city": destination_city,
    "departure_time": departure_time,
    "arrival_time": arrival_time,
    "stops": stops,
    "class": travel_class,
    "duration": duration,
    "days_left": days_left,
}

# Applicare il Label Encoding alle variabili categoriche
encoded_input = []
for feature, value in input_data.items():
    if feature in label_encoders:
        le = label_encoders[feature]
        encoded_input.append(le.transform([value])[0])
    else:
        encoded_input.append(value)

# Separiamo correttamente le variabili numeriche (durata e giorni rimasti) dalle categoriali
numerical_features = np.array([encoded_input[-2], encoded_input[-1]]).reshape(1, -1)  # Le ultime due sono numeriche
categorical_features = np.array(encoded_input[:-2]).reshape(1, -1)  # Le prime 6 sono categoriali

# Normalizzazione solo sulle caratteristiche numeriche
numerical_features_scaled = scaler.transform(numerical_features)

# Uniamo le variabili numeriche scalate con quelle categoriali
input_features = np.hstack([categorical_features, numerical_features_scaled])

# Assicurati che le feature siano nell'ordine corretto
# Prima otteniamo la lista degli indici corrispondenti ai nomi delle feature
feature_order_list = list(input_data.keys())

# Controlla che tutti gli elementi di feature_order siano presenti in feature_order_list
missing_features = [f for f in feature_order if f not in feature_order_list]
if missing_features:
    st.error(f"Le seguenti feature sono mancanti: {', '.join(missing_features)}")

# Poi riorganizziamo le feature secondo l'ordine richiesto
feature_indices = [feature_order_list.index(f) for f in feature_order if f in feature_order_list]

# Riorganizziamo le feature
input_features_ordered = input_features[:, feature_indices]

# Predizione
if st.button("Prevedi Prezzo"):
    predicted_price = model.predict(input_features_ordered)[0]
    st.write(f"Prezzo Stimato: {predicted_price:.2f} EUR")





    
