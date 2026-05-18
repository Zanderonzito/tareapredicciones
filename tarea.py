import requests #si da error cambiar interprete a uno que tenga requests instalado (ej: python 3.11).
import pandas as pd
import numpy as np
import os
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
# entrena solo con X_train, evalua en X_test
# SIN PICKLE, los modelos viven en la memoria RAM
# random forest con mismo split que la regresion lineal (random_state=42)
# =======================================================

def entrenar_resultado(df, modelo, nombre_modelo):
    print(f"\n--- ENTRENAMIENTO: {nombre_modelo} (RESULTADOS) ---\n")
    X_train, X_test, y_train, y_test = preparar_split_resultado(df)
    
    modelo.fit(X_train, y_train)

    # Predicciones para ambos conjuntos
    y_train_pred = modelo.predict(X_train)
    y_test_pred = modelo.predict(X_test)

    # Métricas en conjunto de entrenamiento
    r2_train = r2_score(y_train, y_train_pred)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
    mae_train = mean_absolute_error(y_train, y_train_pred)

    # Métricas en conjunto de prueba
    r2_test = r2_score(y_test, y_test_pred)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
    mae_test = mean_absolute_error(y_test, y_test_pred)

    print("  [ Métricas Entrenamiento ]")
    print(f"    R²   : {r2_train:.4f}")
    print(f"    RMSE : {rmse_train:.4f}")
    print(f"    MAE  : {mae_train:.4f}")
    
    print("\n  [ Métricas Prueba ]")
    print(f"    R²   : {r2_test:.4f}")
    print(f"    RMSE : {rmse_test:.4f}")
    print(f"    MAE  : {mae_test:.4f}")
    
    print(f"\nModelo de {nombre_modelo} guardado exitosamente.")
    return modelo

def predecir_resultado(df, modelo, nombre_modelo):
    print(f"\n--- PREDICCION: {nombre_modelo} (RESULTADOS) ---\n")
    if modelo is None:
        print(f"No se encontro el modelo de {nombre_modelo} en memoria. Entrena primero en el menú.")
        return

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

def comparar_modelos(df, mod_mlr, mod_rf):
    print("\n--- COMPARACION DE MODELOS (RESULTADO) ---\n")
    if mod_mlr is None or mod_rf is None:
        print("Faltan modelos. Por favor entrena los 2 modelos primero (opciones 3 y 5).")
        return

    _, X_test, _, y_test = preparar_split_resultado(df)

    for nombre, modelo in [("Reg Multiple", mod_mlr), ("Random Forest", mod_rf)]:
        y_pred = modelo.predict(X_test)
        print(f"  {nombre:<15} -> R²: {r2_score(y_test, y_pred):.4f} | RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")


# =======================================================
# GOLES // AYUDADO POR GEMINI
# BASADOS EN LA MISMA LOGICA DE RESULTADOS, PERO AHORA PREDECIMOS LOS GOLES DE LA UC Y SU RIVAL
# SIN PICKLE, guardamos en memoria RAM
# =======================================================

def entrenar_goles(df, modelo, nombre_modelo):
    print(f"\n--- ENTRENANDO GOLES: {nombre_modelo} ---\n")
    X_train, X_test, y_train, y_test = preparar_split_goles(df)

    modelo.fit(X_train, y_train)

    # Predicciones para ambos conjuntos
    y_train_pred = modelo.predict(X_train)
    y_test_pred = modelo.predict(X_test)

    # Métricas en conjunto de entrenamiento
    r2_train = r2_score(y_train, y_train_pred)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
    mae_train = mean_absolute_error(y_train, y_train_pred)

    # Métricas en conjunto de prueba
    r2_test = r2_score(y_test, y_test_pred)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
    mae_test = mean_absolute_error(y_test, y_test_pred)

    print("  [ Métricas Entrenamiento ]")
    print(f"    R²   : {r2_train:.4f}")
    print(f"    RMSE : {rmse_train:.4f}")
    print(f"    MAE  : {mae_train:.4f}")
    
    print("\n  [ Métricas Prueba ]")
    print(f"    R²   : {r2_test:.4f}")
    print(f"    RMSE : {rmse_test:.4f}")
    print(f"    MAE  : {mae_test:.4f}")
    
    print(f"\nModelo de {nombre_modelo} guardado exitosamente en RAM.")
    return modelo

def predecir_goles(df, modelo, nombre_modelo):
    print(f"\n--- PREDICIENDO GOLES: {nombre_modelo} ---\n")
    if modelo is None:
        print(f"No se encontro el modelo de {nombre_modelo} en memoria. Entrena primero en el menú.")
        return

    sig_partido = len(df) + 1
    cond_str = input("condicion (Local/Visita) [Enter=Local]  : ").strip()
    condicion = cond_str.capitalize() if cond_str else "Local"
    es_local = 1 if condicion == "Local" else 0

    X_nuevo = pd.DataFrame([{"n_partido": sig_partido, "es_local": es_local}])
    
    # Clip para evitar predicciones matemáticas de goles negativos
    prediccion = np.clip(modelo.predict(X_nuevo)[0], 0, None)
    
    goles_uc_pred = int(round(prediccion[0]))
    goles_rival_pred = int(round(prediccion[1]))

    print(f"\n  Partido #{sig_partido} | {condicion}")
    print(f"  Marcador ({nombre_modelo}): UC {goles_uc_pred} - {goles_rival_pred} Rival")

def comparar_modelos_goles(df, mod_mlr, mod_rf):
    print("\n--- COMPARACION DE MODELOS (GOLES) ---\n")
    if mod_mlr is None or mod_rf is None:
        print("Faltan modelos. Por favor entrena los 2 modelos primero (opciones 8 y 10).")
        return

    _, X_test, _, y_test = preparar_split_goles(df)

    for nombre, modelo in [("Reg Multiple", mod_mlr), ("Random Forest", mod_rf)]:
        y_pred = modelo.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        print(f"  {nombre:<15} -> R²: {r2:.4f} | RMSE: {rmse:.4f} | MAE: {mae:.4f}")


# -------------------------------------------------------
# MENU PRINCIPAL
# -------------------------------------------------------
def main():
    df = cargar_datos()

    modelos = {
        "mlr_res": None,
        "rf_res": None,
        "mlr_goles": None,
        "rf_goles": None
    }

    opciones = {str(i) for i in range(1, 14)}
    while True:
        print("\n" + "="*35)
        print("         MENÚ PRINCIPAL")
        print("="*35)
        print("\n  [ Exploración ]")
        print("  1. Ver datos históricos")
        print("  2. Gráficos y Tendencias")
        
        print("\n  [ PREDICCIÓN DE RESULTADO (Ganar/Empate/Perder) ]")
        print("  3. Entrenar Regresión Múltiple")
        print("  4. Predecir con Regresión Múltiple")
        print("  5. Entrenar Random Forest")
        print("  6. Predecir con Random Forest")
        print("  7. Comparar modelos de resultados")

        print("\n  [ PREDICCIÓN DE GOLES EXACTOS ]")
        print("  8. Entrenar predicción de goles (Regresión Múltiple)")
        print("  9. Predecir marcador (Regresión Múltiple)")
        print("  10. Entrenar predicción de goles (Random Forest)")
        print("  11. Predecir marcador (Random Forest)")
        print("  12. Comparar modelos de goles")
        
        print("\n  13. Salir")

        op = input("\n>> ").strip()
        if op not in opciones:
            print("Eso no es una opción válida")
            continue

        if   op == "1": print(df.tail(20).to_string())
        elif op == "2": hacer_graficos(df)
        elif op == "3": modelos["mlr_res"] = entrenar_resultado(df, LinearRegression(), "Regresión Múltiple")
        elif op == "4": predecir_resultado(df, modelos["mlr_res"], "Regresión Múltiple")
        elif op == "5": modelos["rf_res"]  = entrenar_resultado(df, RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42), "Random Forest")
        elif op == "6": predecir_resultado(df, modelos["rf_res"], "Random Forest")
        elif op == "7": comparar_modelos(df, modelos["mlr_res"], modelos["rf_res"])
        elif op == "8":  modelos["mlr_goles"] = entrenar_goles(df, LinearRegression(), "Regresión Múltiple")
        elif op == "9":  predecir_goles(df, modelos["mlr_goles"], "Regresión Múltiple")
        elif op == "10": modelos["rf_goles"]  = entrenar_goles(df, RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42), "Random Forest")
        elif op == "11": predecir_goles(df, modelos["rf_goles"], "Random Forest")
        elif op == "12": comparar_modelos_goles(df, modelos["mlr_goles"], modelos["rf_goles"])
        elif op == "13": break
if __name__ == "__main__": main()
