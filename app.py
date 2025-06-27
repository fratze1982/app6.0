import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.inspection import PartialDependenceDisplay
import matplotlib.pyplot as plt
import seaborn as sns

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

    with st.expander("⚙️ Zielwerte, Toleranzen & Gewichtung definieren"):
        zielwerte = {}
        toleranzen = {}
        gewichtung = {}

        for ziel in zielspalten:
            zielwerte[ziel] = st.number_input(f"Zielwert für {ziel}", value=float(df[ziel].mean()))
            toleranzen[ziel] = st.number_input(f"Toleranz für {ziel} (±)", value=2.0 if "Glanz" in ziel else 1.0)
            gewichtung[ziel] = st.slider(f"Gewichtung für {ziel}", min_value=0.0, max_value=5.0, value=1.0, step=0.1)

    steuerbare_rohstoffe = [
        "Lackslurry", "Wasser", "Byk1770", "Albawhite70", "Omycarb2JSV",
        "Sylysia256", "AcrysolRM2020E", "MowilithLDM7416",
        "AlbedingkAC2003", "Byk1785", "AcrysolRM8WE"
    ]

    st.sidebar.header("\U0001F6E0️ Rohstoffe für Zielsuche fixieren und begrenzen")
    fixierte_werte = {}
    rohstoffgrenzen = {}

    for roh in steuerbare_rohstoffe:
        if roh in df.columns:
            fixieren = st.sidebar.checkbox(f"{roh} fixieren?")
            if fixieren:
                fix_wert = st.sidebar.number_input(f"{roh} Wert", value=float(df[roh].mean()))
                fixierte_werte[roh] = fix_wert
            else:
                min_val = float(df[roh].min())
                max_val = float(df[roh].max())
                if min_val == max_val:
                    min_val -= 0.01
                    max_val += 0.01
                rohstoffgrenzen[roh] = st.sidebar.slider(f"Grenzen für {roh}", min_val, max_val, (min_val, max_val))

    anzahl_varianten = 1000
    simulierte_formulierungen = []
    score_liste = []

    if st.button("\U0001F680 Starte Zielsuche"):
        for _ in range(anzahl_varianten):
            zufall = {}
            for roh in steuerbare_rohstoffe:
                if roh in df.columns:
                    if roh in fixierte_werte:
                        zufall[roh] = fixierte_werte[roh]
                    else:
                        min_bereich, max_bereich = rohstoffgrenzen[roh]
                        zufall[roh] = np.random.uniform(min_bereich, max_bereich)
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
            score = 0
            passt = True
            for ziel in zielspalten:
                delta = abs(y[zielspalten.index(ziel)] - zielwerte[ziel])
                score += delta * gewichtung[ziel]
                if delta > toleranzen[ziel]:
                    passt = False
            if passt:
                score_liste.append((i, score))

        if score_liste:
            score_liste.sort(key=lambda x: x[1])
            treffer_idx = [i for i, s in score_liste]

            treffer_df = sim_df.iloc[treffer_idx].copy()
            vorhersagen_df = pd.DataFrame(
                [y_pred[i] for i in treffer_idx],
                columns=zielspalten
            )

            ergebnis_df = pd.concat(
                [treffer_df.reset_index(drop=True), vorhersagen_df.reset_index(drop=True)],
                axis=1
            )
            ergebnis_df.insert(0, "Score", [round(s, 2) for _, s in score_liste])

            st.success(f"✅ {len(ergebnis_df)} passende Formulierungen gefunden!")
            st.dataframe(ergebnis_df)

            # --- Download ---
            csv = ergebnis_df.to_csv(index=False).encode("utf-8")
            st.download_button("\U0001F4C5 Ergebnisse als CSV herunterladen", data=csv, file_name="passende_formulierungen.csv")

            # --- Diagramm: Zielwerte ---
            st.subheader("\U0001F4CA Diagramm: Zielgrößen der Top 10 Formulierungen")
            if len(ergebnis_df) > 0:
                top10 = ergebnis_df.head(10).copy()
                fig, ax = plt.subplots(figsize=(10, 5))
                for ziel in zielspalten:
                    ax.plot(top10["Score"], top10[ziel], label=ziel, marker="o")
                ax.set_xlabel("Score (niedriger = besser)")
                ax.set_ylabel("Zielwert")
                ax.set_title("Zielgrößen im Vergleich (Top 10)")
                ax.legend()
                st.pyplot(fig)

            # --- Radar-Diagramm ---
            st.subheader("\U0001F52C Radar-Diagramm der Top 3")
            if len(ergebnis_df) >= 3:
                radar_data = ergebnis_df.head(3)[zielspalten].copy()
                labels = list(zielspalten)
                num_vars = len(labels)
                angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
                labels += labels[:1]

                fig, ax = plt.subplots(subplot_kw=dict(polar=True))
                for idx, row in radar_data.iterrows():
                    values = row.tolist()
                    values += values[:1]
                    ax.plot(angles, values, label=f"Formulierung {idx+1}")
                    ax.fill(angles, values, alpha=0.1)
                ax.set_thetagrids(np.degrees(angles), labels)
                ax.set_title("Radarvergleich Zielgrößen")
                ax.legend(loc="upper right")
                st.pyplot(fig)

else:
    st.info("ℹ️ Um die Zielsuche zu aktivieren, wähle mindestens eine dieser Zielgrößen aus: Glanz20, Glanz60, Glanz85, Kratzschutz oder KostenGesamtkg.")
