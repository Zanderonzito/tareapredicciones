import requests
import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_KEY")
TEAM_ID = int(os.getenv("TEAM_ID", "0"))

CSV_PATH = "datos_catolica.csv"

# Modelos para predecir Resultado
LR_MODEL = "modelo_lineal.pkl"       
REGRESION_MODEL = "modelo_multiple.pkl"
RF_MODEL = "modelo_rf.pkl"           

# Modelos para predecir Goles (UC vs Rival)
REGRESION_GOLES_MODEL = "modelo_goles_regresion.pkl"
RF_GOLES_MODEL = "modelo_goles_rf.pkl"

headers = {
    "x-apisports-key":  API_KEY,
    "x-apisports-host": "v3.football.api-sports.io"
}

FEATURES = ["n_partido", "es_local"]

# -------------------------------------------------------
# CARGA DE DATOS (API o CSV cache)
# -------------------------------------------------------
def cargar_datos():
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        if all(c in df.columns for c in ["resultado", "goles_totales", "goles_uc", "goles_rival"]):
            print(f"cargando desde csv... {len(df)} partidos encontrados")
            return df
        os.remove(CSV_PATH)
        print("borre el csv viejo, le faltaban columnas")

    lista = []
    print("conectando a la api...")

    for year in [2022, 2023, 2024]:
        try:
            resp = requests.get(
                f"https://v3.football.api-sports.io/fixtures?team={TEAM_ID}&season={year}",
                headers=headers
            ).json()

            for p in resp.get("response", []):
                if p["fixture"]["status"]["short"] != "FT":
                    continue

                gh = p["goals"]["home"]
                ga = p["goals"]["away"]
                es_local = p["teams"]["home"]["id"] == TEAM_ID

                goles_uc    = gh if es_local else ga
                goles_rival = ga if es_local else gh

                if goles_uc > goles_rival:    res = 1.0
                elif goles_uc == goles_rival: res = 0.5
                else:                         res = 0.0

                lista.append({
                    "resultado":     res,
                    "goles_totales": gh + ga,
                    "goles_uc":      goles_uc,
                    "goles_rival":   goles_rival,
                    "condicion":     "Local" if es_local else "Visita"
                })

        except Exception as e:
            print(f"fallo temporada {year}: {e}")

    df = pd.DataFrame(lista)
    df.insert(0, "n_partido", range(1, len(df) + 1))
    df.to_csv(CSV_PATH, index=False)
    print(f"descarga lista, {len(df)} partidos en total")
    return df


# -------------------------------------------------------
# FUNCION AUXILIAR — prepara X e y y hace el split
# la uso en ambos modelos para que el split sea identico
# -------------------------------------------------------
def preparar_split_resultado(df):
    df_m = df.copy()
    df_m["es_local"] = (df_m["condicion"] == "Local").astype(int)

    X = df_m[FEATURES]
    y = df_m["resultado"]

    return train_test_split(X, y, test_size=0.2, random_state=42)

def preparar_split_goles(df):
    df_m = df.copy()
    df_m["es_local"] = (df_m["condicion"] == "Local").astype(int)

    X = df_m[FEATURES]
    # Queremos predecir ambas cosas a la vez
    y = df_m[["goles_uc", "goles_rival"]]

    return train_test_split(X, y, test_size=0.2, random_state=42)


# -------------------------------------------------------
# GRAFICOS
# -------------------------------------------------------
def hacer_graficos(df):
    plt.style.use("ggplot")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Analisis UC 2022-2024", fontsize=14, fontweight="bold")

    sns.heatmap(
        df[["resultado", "goles_totales", "goles_uc", "goles_rival"]].corr(),
        annot=True, cmap="Blues", ax=axes[0, 0]
    )
    axes[0, 0].set_title("mapa de calor")

    sns.boxplot(x="condicion", y="goles_totales", data=df,
                hue="condicion", palette="Set1", legend=False, ax=axes[0, 1])
    axes[0, 1].set_title("goles por condicion")

    df["resultado"].map({1.0: "Ganado", 0.5: "Empate", 0.0: "Perdido"}) \
                   .value_counts() \
                   .plot(kind="pie", autopct="%1.1f%%", ax=axes[1, 0])
    axes[1, 0].set_title("historico de resultados")
    axes[1, 0].set_ylabel("")

    sns.regplot(x="n_partido", y="goles_totales", data=df,
                ax=axes[1, 1], line_kws={"color": "red"})
    axes[1, 1].set_title("tendencia de goles")

    plt.tight_layout()
    plt.show()


# =======================================================
# RESULTADOS - (ENTRENAMIENTO MODELO , PREDICCIÓN , COMPARACIÓN , RANDOM FOREST) //AYUDADO POR CLAUDE
#entrena solo con X_train, evalua en X_test
#guarda el modelo en LR_MODEL para usarlo en prediccion
#con pickle, para no re-entrenar el modelo
# random forest con mismo split que la regresion lineal (random_state=42)
# =======================================================

def entrenar_lineal(df):
    print("\n--- ENTRENAMIENTO REGRESION LINEAL SIMPLE ---\n")
    X_train, X_test, y_train, y_test = preparar_split_resultado(df)
    
    modelo = LinearRegression()
    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)
    r2   = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"  R²   : {r2:.4f}")
    print(f"  RMSE : {rmse:.4f}")

    with open(LR_MODEL, "wb") as f:
        pickle.dump(modelo, f)
    print(f"\nmodelo guardado en '{LR_MODEL}'")

def predecir_lineal(df):
    print("\n--- PREDICCION REGRESION LINEAL ---\n")
    if not os.path.exists(LR_MODEL):
        print(f"no se encontro '{LR_MODEL}', entrena primero con la opcion 3")
        return

    with open(LR_MODEL, "rb") as f:
        modelo = pickle.load(f)

    sig_partido = len(df) + 1
    cond_str = input("condicion (Local/Visita) [Enter=Local]  : ").strip()
    condicion = cond_str.capitalize() if cond_str else "Local"
    es_local = 1 if condicion == "Local" else 0

    X_nuevo = pd.DataFrame([{"n_partido": sig_partido, "es_local": es_local}])

    valor = float(np.clip(modelo.predict(X_nuevo)[0], 0.0, 1.0))
    etiqueta = "GANA UC" if valor >= 0.75 else ("EMPATE" if valor >= 0.25 else "PIERDE UC")

    print(f"\n  partido #{sig_partido} | {condicion}")
    print(f"  score predicho : {valor:.4f}")
    print(f"  resultado      : {etiqueta}")


def entrenar_multiple(df):
    print("\n--- ENTRENAMIENTO REGRESION LINEAL MULTIPLE ---\n")
    X_train, X_test, y_train, y_test = preparar_split_resultado(df)
    
    modelo = LinearRegression()
    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)
    print(f"  R²   : {r2_score(y_test, y_pred):.4f}")
    print(f"  RMSE : {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")

    with open(REGRESION_MODEL, "wb") as f:
        pickle.dump(modelo, f)
    print(f"\nmodelo guardado en '{REGRESION_MODEL}'")

def predecir_multiple(df):
    print("\n--- PREDICCION REGRESION LINEAL MULTIPLE ---\n")
    if not os.path.exists(REGRESION_MODEL):
        print(f"no se encontro '{REGRESION_MODEL}', entrena primero con la opcion 5")
        return

    with open(REGRESION_MODEL, "rb") as f:
        modelo = pickle.load(f)

    sig_partido = len(df) + 1
    cond_str = input("condicion (Local/Visita) [Enter=Local]  : ").strip()
    condicion = cond_str.capitalize() if cond_str else "Local"
    es_local = 1 if condicion == "Local" else 0

    X_nuevo = pd.DataFrame([{"n_partido": sig_partido, "es_local": es_local}])

    valor = float(np.clip(modelo.predict(X_nuevo)[0], 0.0, 1.0))
    etiqueta = "GANA UC" if valor >= 0.75 else ("EMPATE" if valor >= 0.25 else "PIERDE UC")

    print(f"\n  partido #{sig_partido} | {condicion}")
    print(f"  score predicho : {valor:.4f}")
    print(f"  resultado      : {etiqueta}")


def entrenar_rf(df):
    print("\n--- ENTRENAMIENTO RANDOM FOREST ---\n")
    X_train, X_test, y_train, y_test = preparar_split_resultado(df)
    
    modelo = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)
    print(f"  R²   : {r2_score(y_test, y_pred):.4f}")
    print(f"  RMSE : {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")

    with open(RF_MODEL, "wb") as f:
        pickle.dump(modelo, f)
    print(f"\nmodelo guardado en '{RF_MODEL}'")


def predecir_rf(df):
    print("\n--- PREDICCION RANDOM FOREST ---\n")
    if not os.path.exists(RF_MODEL):
        print(f"no se encontro '{RF_MODEL}', entrena primero con la opcion 7")
        return

    with open(RF_MODEL, "rb") as f:
        modelo = pickle.load(f)

    sig_partido = len(df) + 1
    cond_str = input("condicion (Local/Visita) [Enter=Local]  : ").strip()
    condicion = cond_str.capitalize() if cond_str else "Local"
    es_local = 1 if condicion == "Local" else 0

    X_nuevo = pd.DataFrame([{"n_partido": sig_partido, "es_local": es_local}])

    valor = float(np.clip(modelo.predict(X_nuevo)[0], 0.0, 1.0))
    etiqueta = "GANA UC" if valor >= 0.75 else ("EMPATE" if valor >= 0.25 else "PIERDE UC")

    print(f"\n  partido #{sig_partido} | {condicion}")
    print(f"  score predicho : {valor:.4f}")
    print(f"  resultado      : {etiqueta}")


def comparar_modelos(df):
    print("\n--- COMPARACION DE MODELOS (RESULTADO) ---\n")
    if not all(os.path.exists(m) for m in [LR_MODEL, REGRESION_MODEL, RF_MODEL]):
        print("Faltan modelos. Por favor entrena los 3 modelos primero (opciones 3, 5 y 7).")
        return

    with open(LR_MODEL, "rb") as f: lr = pickle.load(f)
    with open(REGRESION_MODEL, "rb") as f: mlr = pickle.load(f)
    with open(RF_MODEL, "rb") as f: rf = pickle.load(f)

    _, X_test, _, y_test = preparar_split_resultado(df)

    for nombre, modelo in [("Reg Lineal", lr), ("Reg Multiple", mlr), ("Random Forest", rf)]:
        y_pred = modelo.predict(X_test)
        print(f"  {nombre:<15} -> R²: {r2_score(y_test, y_pred):.4f} | RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")


# =======================================================
# GOLES // AYUDADO POR GEMINI
# BASADOS EN LA MISMA LOGICA DE RESULTADOS, PERO AHORA PREDECIMOS LOS GOLES DE LA UC Y SU RIVAL(CALERA,PERO NO TENEMOS COMO IDENTIFICAR AL RIVAL)
# =======================================================

def entrenar_goles_regresion(df):
    print("\n--- ENTRENANDO GOLES: REGRESION MULTIPLE ---\n")
    X_train, X_test, y_train, y_test = preparar_split_goles(df)

    modelo_goles_regresion = LinearRegression()
    modelo_goles_regresion.fit(X_train, y_train)

    y_pred = modelo_goles_regresion.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"  Margen de error (MAE) : +/- {mae:.2f} goles")
    
    with open(REGRESION_GOLES_MODEL, "wb") as f:
        pickle.dump(modelo_goles_regresion, f)
    print(f"modelo guardado en '{REGRESION_GOLES_MODEL}'")


def predecir_goles_regresion(df):
    print("\n--- PREDICIENDO GOLES: REGRESION MULTIPLE ---\n")
    if not os.path.exists(REGRESION_GOLES_MODEL):
        print(f"No se encontro '{REGRESION_GOLES_MODEL}'. Entrena la opción 10 primero.")
        return

    with open(REGRESION_GOLES_MODEL, "rb") as f:
        modelo = pickle.load(f)

    sig_partido = len(df) + 1
    cond_str = input("condicion (Local/Visita) [Enter=Local]  : ").strip()
    condicion = cond_str.capitalize() if cond_str else "Local"
    es_local = 1 if condicion == "Local" else 0

    X_nuevo = pd.DataFrame([{"n_partido": sig_partido, "es_local": es_local}])
    
    # Clip para evitar predicciones matemáticas de goles negativos (-0.5)
    prediccion = np.clip(modelo.predict(X_nuevo)[0], 0, None)
    
    goles_uc_pred = int(round(prediccion[0]))
    goles_rival_pred = int(round(prediccion[1]))

    print(f"\n  Partido #{sig_partido} | {condicion}")
    print(f"  Marcador (Reg. Múltiple): UC {goles_uc_pred} - {goles_rival_pred} Rival")


def entrenar_goles_rf(df):
    print("\n--- ENTRENANDO GOLES: RANDOM FOREST ---\n")
    X_train, X_test, y_train, y_test = preparar_split_goles(df)

    modelo_goles_rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    modelo_goles_rf.fit(X_train, y_train)

    y_pred = modelo_goles_rf.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"  Margen de error (MAE) : +/- {mae:.2f} goles")
    
    with open(RF_GOLES_MODEL, "wb") as f:
        pickle.dump(modelo_goles_rf, f)
    print(f"modelo guardado en '{RF_GOLES_MODEL}'")


def predecir_goles_rf(df):
    print("\n--- PREDICIENDO GOLES: RANDOM FOREST ---\n")
    if not os.path.exists(RF_GOLES_MODEL):
        print(f"No se encontro '{RF_GOLES_MODEL}'. Entrena la opción 12 primero.")
        return

    with open(RF_GOLES_MODEL, "rb") as f:
        modelo = pickle.load(f)

    sig_partido = len(df) + 1
    cond_str = input("condicion (Local/Visita) [Enter=Local]  : ").strip()
    condicion = cond_str.capitalize() if cond_str else "Local"
    es_local = 1 if condicion == "Local" else 0

    X_nuevo = pd.DataFrame([{"n_partido": sig_partido, "es_local": es_local}])
    
    prediccion = np.clip(modelo.predict(X_nuevo)[0], 0, None)
    
    goles_uc_pred = int(round(prediccion[0]))
    goles_rival_pred = int(round(prediccion[1]))

    print(f"\n  Partido #{sig_partido} | {condicion}")
    print(f"  Marcador (Random Forest): UC {goles_uc_pred} - {goles_rival_pred} Rival")


# -------------------------------------------------------
# MENU PRINCIPAL
# -------------------------------------------------------
def main():
    df = cargar_datos()

    opciones = {str(i) for i in range(1, 16)}
    while True:
        print("\n" + "="*35)
        print("         MENÚ PRINCIPAL")
        print("="*35)
        print("\n  [ Exploración ]")
        print("  1. Ver datos históricos")
        print("  2. Gráficos y Tendencias")
        
        print("\n  [ PREDICCIÓN DE RESULTADO (Ganar/Empate/Perder) ]")
        print("  3. Entrenar Regresión Lineal")
        print("  4. Predecir con Regresión Lineal")
        print("  5. Entrenar Regresión Múltiple")
        print("  6. Predecir con Regresión Múltiple")
        print("  7. Entrenar Random Forest")
        print("  8. Predecir con Random Forest")
        print("  9. Comparar modelos de resultados")

        print("\n  [ PREDICCIÓN DE GOLES EXACTOS ]")
        print("  10. Entrenar predicción de goles (Regresión Múltiple)")
        print("  11. Predecir marcador (Regresión Múltiple)")
        print("  12. Entrenar predicción de goles (Random Forest)")
        print("  13. Predecir marcador(Random Forest)")
        
        print("\n  14. Salir")

        op = input("\n>> ").strip()
        if op not in opciones:
            print("Eso no es una opción válida")
            continue

        if   op == "1": print(df.tail(20).to_string())
        elif op == "2": hacer_graficos(df)
        elif op == "3": entrenar_lineal(df)
        elif op == "4": predecir_lineal(df)
        elif op == "5": entrenar_multiple(df)
        elif op == "6": predecir_multiple(df)
        elif op == "7": entrenar_rf(df)
        elif op == "8": predecir_rf(df)
        elif op == "9": comparar_modelos(df)
        elif op == "10": entrenar_goles_regresion(df)
        elif op == "11": predecir_goles_regresion(df)
        elif op == "12": entrenar_goles_rf(df)
        elif op == "13": predecir_goles_rf(df)
        elif op == "14": break


if __name__ == "__main__":
    main()
    main()
