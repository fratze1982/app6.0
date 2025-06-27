import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.inspection import PartialDependenceDisplay
import matplotlib.pyplot as plt

st.set_page_config(page_title="KI-Vorhersage für Lackrezepturen", layout="wide")
st.title("\U0001F3A8 KI-Vorhersage für Lackrezepturen")

# --- Datei-Upload ---
uploaded_file = st.file_uploader("\U0001F4C1 CSV-Datei hochladen (mit ; getrennt)", type=["csv"])
if uploaded_file is None:
    st.warning("Bitte lade eine CSV-Datei hoch.")
    st.stop()

# --- CSV einlesen ---
try:
    df = pd.read_csv(uploaded_file, sep=";", decimal=",")
    st.success("✅ Datei erfolgreich geladen.")
except Exception as e:
    st.error(f"❌ Fehler beim Einlesen der Datei: {e}")
    st.stop()

st.write("\U0001F9FE Gefundene Spalten:", df.columns.tolist())

# --- Zielgrößen aus numerischen Spalten dynamisch auswählen ---
numerische_spalten = df.select_dtypes(include=[np.number]).columns.tolist()

if not numerische_spalten:
    st.error("❌ Keine numerischen Spalten im Datensatz gefunden.")
    st.stop()

zielspalten = st.multiselect(
    "\U0001F3AF Zielgrößen auswählen (numerische Spalten)",
    options=numerische_spalten,
    default=[numerische_spalten[0]]
)

if not zielspalten:
    st.warning("Bitte mindestens eine Zielgröße auswählen.")
    st.stop()

# --- Eingabe- und Zielvariablen trennen ---
X = df.drop(columns=zielspalten, errors="ignore")
y = df[zielspalten].copy()

# Spaltentypen bestimmen
kategorisch = X.select_dtypes(include="object").columns.tolist()
numerisch = X.select_dtypes(exclude="object").columns.tolist()

# One-Hot-Encoding
X_encoded = pd.get_dummies(X)

# Fehlende Werte bereinigen
df_encoded = X_encoded.copy()
df_encoded[y.columns] = y
df_encoded = df_encoded.dropna()

X_clean = df_encoded[X_encoded.columns]
y_clean = df_encoded[y.columns]

if X_clean.empty or y_clean.empty:
    st.error("❌ Keine gültigen Daten zum Trainieren.")
    st.stop()

# --- Modelltraining ---
modell = MultiOutputRegressor(RandomForestRegressor(n_estimators=150, random_state=42))
modell.fit(X_clean, y_clean)

# --- Benutzer-Eingabeformular ---
st.sidebar.header("\U0001F527 Parameter anpassen")
user_input = {}

for col in numerisch:
    try:
        min_val = float(df[col].min())
        max_val = float(df[col].max())
        mean_val = float(df[col].mean())
        user_input[col] = st.sidebar.slider(col, min_val, max_val, mean_val)
    except:
        continue

for col in kategorisch:
    options = sorted(df[col].dropna().unique())
    user_input[col] = st.sidebar.selectbox(col, options)

input_df = pd.DataFrame([user_input])
input_encoded = pd.get_dummies(input_df)

# Fehlende Spalten auffüllen
for col in X_clean.columns:
    if col not in input_encoded.columns:
        input_encoded[col] = 0
input_encoded = input_encoded[X_clean.columns]

# --- Vorhersage ---
prediction = modell.predict(input_encoded)[0]

st.subheader("\U0001F52E Vorhergesagte Zielgrößen")
for i, ziel in enumerate(zielspalten):
    st.metric(label=ziel, value=round(prediction[i], 2))

# --- Zieloptimierung per Zufallssuche ---
st.subheader("\U0001F3AF Zieloptimierung: Welche Formulierung erfüllt deine Vorgaben?")

if set(["KostenGesamtkg", "Glanz20", "Glanz60", "Glanz85", "Kratzschutz"]).intersection(set(zielspalten)):

    with st.expander("⚙️ Zielwerte definieren"):
        if "Glanz20" in zielspalten:
            ziel_glanz20 = st.number_input("\U0001F506 Glanz20 (±2)", value=30.0)
        if "Glanz60" in zielspalten:
            ziel_glanz60 = st.number_input("\U0001F31F Glanz60 (±2)", value=60.0)
        if "Glanz85" in zielspalten:
            ziel_glanz85 = st.number_input("\U0001F4A1 Glanz85 (±2)", value=80.0)
        if "Kratzschutz" in zielspalten:
            ziel_kratzschutz = st.number_input("\U0001F6E1️ Kratzschutz (±1)", value=3.0)
        if "KostenGesamtkg" in zielspalten:
            max_kosten = st.number_input("\U0001F4B0 Maximale Kosten €/kg", value=2.0)

    steuerbare_rohstoffe = [
        "Lackslurry", "Wasser", "Byk1770", "Albawhite70", "Omycarb2JSV",
        "Sylysia256", "AcrysolRM2020E", "MowilithLDM7416",
        "AlbedingkAC2003", "Byk1785", "AcrysolRM8WE"
    ]

    anzahl_varianten = 1000
    simulierte_formulierungen = []

    if st.button("\U0001F680 Starte Zielsuche"):
        for _ in range(anzahl_varianten):
            zufall = {
                roh: np.random.uniform(df[roh].min(), df[roh].max())
                for roh in steuerbare_rohstoffe if roh in df.columns
            }
            simulierte_formulierungen.append(zufall)

        sim_df = pd.DataFrame(simulierte_formulierungen)

        sim_encoded = pd.get_dummies(sim_df)
        for col in X_clean.columns:
            if col not in sim_encoded.columns:
                sim_encoded[col] = 0
        sim_encoded = sim_encoded[X_clean.columns]

        y_pred = modell.predict(sim_encoded)

        treffer_idx = []
        for i, y in enumerate(y_pred):
            passt = True
            if "KostenGesamtkg" in zielspalten and y[zielspalten.index("KostenGesamtkg")] > max_kosten:
                passt = False
            if "Glanz20" in zielspalten and abs(y[zielspalten.index("Glanz20")] - ziel_glanz20) > 2:
                passt = False
            if "Glanz60" in zielspalten and abs(y[zielspalten.index("Glanz60")] - ziel_glanz60) > 2:
                passt = False
            if "Glanz85" in zielspalten and abs(y[zielspalten.index("Glanz85")] - ziel_glanz85) > 2:
                passt = False
            if "Kratzschutz" in zielspalten and abs(y[zielspalten.index("Kratzschutz")] - ziel_kratzschutz) > 1:
                passt = False
            if passt:
                treffer_idx.append(i)

        if treffer_idx:
            treffer_df = sim_df.iloc[treffer_idx].copy()
            vorhersagen_df = pd.DataFrame(
                [y_pred[i] for i in treffer_idx],
                columns=zielspalten
            )
            ergebnis_df = pd.concat(
                [treffer_df.reset_index(drop=True), vorhersagen_df.reset_index(drop=True)],
                axis=1
            )
            st.success(f"✅ {len(ergebnis_df)} passende Formulierungen gefunden!")
            st.dataframe(ergebnis_df)

            csv = ergebnis_df.to_csv(index=False).encode("utf-8")
            st.download_button("\U0001F4C5 Ergebnisse als CSV herunterladen", data=csv, file_name="passende_formulierungen.csv")
        else:
            st.error("❌ Keine passenden Formulierungen gefunden. Probiere mehr Toleranz oder andere Zielwerte.")
else:
    st.info("ℹ️ Um die Zielsuche zu aktivieren, wähle mindestens eine dieser Zielgrößen aus: Glanz20, Glanz60, Glanz85, Kratzschutz oder KostenGesamtkg.")
