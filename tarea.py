import requests #SI APARECE ERROR : CAMBIAR INTERPRETE DE PYTHON 3.11 (GLOBAL)
import pandas as pd
import numpy as np
import os
import pickle #GUARDAR MODELO ENTRENADO
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
LR_MODEL = "modelo_lineal.pkl"   # pkl de regresion lineal
RF_MODEL = "modelo_rf.pkl"       # pkl de random forest

headers = {
    "x-apisports-key":  API_KEY,
    "x-apisports-host": "v3.football.api-sports.io"
}

# columnas que usan ambos modelos
FEATURES = ["goles_uc", "goles_rival", "goles_totales", "n_partido", "es_local"]


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
def preparar_split(df):
    df_m = df.copy()
    df_m["es_local"] = (df_m["condicion"] == "Local").astype(int)

    X = df_m[FEATURES]
    y = df_m["resultado"]

    # 80% entrenamiento, 20% prueba — random_state fijo para reproducibilidad
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test


# -------------------------------------------------------
# GRAFICOS EXPLORATORIOS
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


# -------------------------------------------------------
# ENTRENAMIENTO — REGRESION LINEAL
# entrena solo con X_train, evalua en X_test (datos que no vio)
# guarda el modelo en LR_MODEL para usarlo en prediccion
# -------------------------------------------------------
def entrenar_lineal(df):
    print("\n--- ENTRENAMIENTO REGRESION LINEAL MULTIPLE---\n")

    X_train, X_test, y_train, y_test = preparar_split(df)
    print(f"entrenamiento: {len(X_train)} partidos | prueba: {len(X_test)} partidos")

    # entrenar solo con X_train
    modelo = LinearRegression()
    modelo.fit(X_train, y_train)

    # evaluar con X_test
    y_pred = modelo.predict(X_test)

    r2   = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)

    print(f"\n  R²   : {r2:.4f}")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  MAE  : {mae:.4f}")

    # coeficientes del modelo
    print("\n  coeficientes:")
    for feat, coef in zip(FEATURES, modelo.coef_):
        print(f"    {feat:>15} : {coef:+.4f}")
    print(f"    {'intercepto':>15} : {modelo.intercept_:+.4f}")

    # grafico real vs predicho sobre el conjunto de prueba
    plt.figure(figsize=(6, 5))
    plt.scatter(y_test, y_pred, alpha=0.6, color="darkorange")
    plt.plot([0, 1], [0, 1], "r--", label="prediccion perfecta")
    plt.xlabel("resultado real (test)")
    plt.ylabel("resultado predicho")
    plt.title("Real vs Predicho — Regresion Lineal")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # guardar solo el modelo entrenado con X_train
    with open(LR_MODEL, "wb") as f:
        pickle.dump(modelo, f)
    print(f"\nmodelo guardado en '{LR_MODEL}'")

    return modelo


# ------------------------------------------------------
# PREDICCION — REGRESION LINEAL
# con pickle, para no re entrenar el modelo
# ------------------------------------------------------
def predecir_lineal(df):
    print("\n--- PREDICCION REGRESION LINEAL MULTIPLE---\n")

    if not os.path.exists(LR_MODEL):
        print(f"no se encontro '{LR_MODEL}', entrena primero con la opcion 3")
        return

    with open(LR_MODEL, "rb") as f:
        modelo = pickle.load(f)
    print(f"modelo cargado desde '{LR_MODEL}'")

    media_uc    = round(df["goles_uc"].mean(), 1)
    media_rival = round(df["goles_rival"].mean(), 1)
    sig_partido = len(df) + 1

    print(f"\npromedios historicos: UC={media_uc} | rival={media_rival}")
    print("(presiona Enter para usar el promedio)\n")

    try:
        guc_str  = input(f"goles esperados UC    [Enter={media_uc}]: ").strip()
        grv_str  = input(f"goles esperados rival [Enter={media_rival}]: ").strip()
        cond_str = input("condicion (Local/Visita) [Enter=Local]  : ").strip()

        goles_uc    = float(guc_str)  if guc_str  else media_uc
        goles_rival = float(grv_str)  if grv_str  else media_rival
        condicion   = cond_str.capitalize() if cond_str else "Local"

        if condicion not in ("Local", "Visita"):
            print("condicion no reconocida, se usa Local")
            condicion = "Local"

    except ValueError:
        print("entrada invalida, se usan valores por defecto")
        goles_uc, goles_rival, condicion = media_uc, media_rival, "Local"

    es_local      = 1 if condicion == "Local" else 0
    goles_totales = int(goles_uc) + int(goles_rival)

    X_nuevo = pd.DataFrame([{
        "goles_uc":      goles_uc,
        "goles_rival":   goles_rival,
        "goles_totales": goles_totales,
        "n_partido":     sig_partido,
        "es_local":      es_local
    }])[FEATURES]

    # clip para mantener el resultado en el rango valido [0, 1]
    valor = float(np.clip(modelo.predict(X_nuevo)[0], 0.0, 1.0))

    if valor >= 0.75:
        etiqueta = "GANA UC"
    elif valor >= 0.25:
        etiqueta = "EMPATE"
    else:
        etiqueta = "PIERDE UC"

    print(f"\n  partido #{sig_partido} | {condicion} | {int(goles_uc)}-{int(goles_rival)}")
    print(f"  score predicho : {valor:.4f}")
    print(f"  resultado      : {etiqueta}")


# -------------------------------------------------------
# ENTRENAMIENTO — RANDOM FOREST
# mismo split que la regresion lineal (random_state=42)
# el modelo solo ve X_train durante el fit
# -------------------------------------------------------
def entrenar_rf(df):
    print("\n--- ENTRENAMIENTO RANDOM FOREST ---\n")

    X_train, X_test, y_train, y_test = preparar_split(df)
    print(f"entrenamiento: {len(X_train)} partidos | prueba: {len(X_test)} partidos")

    modelo = RandomForestRegressor(
        n_estimators=100,  # cantidad de arboles
        max_depth=5,       # profundidad maxima para evitar sobreajuste
        random_state=42
    )
    # fit solo sobre el conjunto de entrenamiento
    modelo.fit(X_train, y_train)

    # metricas sobre el conjunto de prueba (datos no vistos)
    y_pred = modelo.predict(X_test)

    r2   = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)

    print(f"\n  R²   : {r2:.4f}")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  MAE  : {mae:.4f}")

    # importancia de variables segun el bosque
    importancias = pd.Series(modelo.feature_importances_, index=FEATURES).sort_values(ascending=False)
    print("\n  importancia de variables:")
    for feat, val in importancias.items():
        barra = "#" * int(val * 30)
        print(f"    {feat:>15}: {val:.4f}  {barra}")

    # grafico real vs predicho
    plt.figure(figsize=(6, 5))
    plt.scatter(y_test, y_pred, alpha=0.6, color="steelblue")
    plt.plot([0, 1], [0, 1], "r--", label="prediccion perfecta")
    plt.xlabel("resultado real (test)")
    plt.ylabel("resultado predicho")
    plt.title("Real vs Predicho — Random Forest")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # guardar el modelo entrenado solo con X_train
    with open(RF_MODEL, "wb") as f:
        pickle.dump(modelo, f)
    print(f"\nmodelo guardado en '{RF_MODEL}'")

    return modelo


# -------------------------------------------------------
# PREDICCION — RANDOM FOREST
# carga el pkl, nunca re-entrena con los datos actuales
# -------------------------------------------------------
def predecir_rf(df):
    print("\n--- PREDICCION RANDOM FOREST ---\n")

    if not os.path.exists(RF_MODEL):
        print(f"no se encontro '{RF_MODEL}', entrena primero con la opcion 6")
        return

    with open(RF_MODEL, "rb") as f:
        modelo = pickle.load(f)
    print(f"modelo cargado desde '{RF_MODEL}'")

    media_uc    = round(df["goles_uc"].mean(), 1)
    media_rival = round(df["goles_rival"].mean(), 1)
    sig_partido = len(df) + 1

    print(f"\npromedios historicos: UC={media_uc} | rival={media_rival}")
    print("(presiona Enter para usar el promedio)\n")

    try:
        guc_str  = input(f"goles esperados UC    [Enter={media_uc}]: ").strip()
        grv_str  = input(f"goles esperados rival [Enter={media_rival}]: ").strip()
        cond_str = input("condicion (Local/Visita) [Enter=Local]  : ").strip()

        goles_uc    = float(guc_str)  if guc_str  else media_uc
        goles_rival = float(grv_str)  if grv_str  else media_rival
        condicion   = cond_str.capitalize() if cond_str else "Local"

        if condicion not in ("Local", "Visita"):
            print("condicion no reconocida, se usa Local")
            condicion = "Local"

    except ValueError:
        print("entrada invalida, se usan valores por defecto")
        goles_uc, goles_rival, condicion = media_uc, media_rival, "Local"

    es_local      = 1 if condicion == "Local" else 0
    goles_totales = int(goles_uc) + int(goles_rival)

    X_nuevo = pd.DataFrame([{
        "goles_uc":      goles_uc,
        "goles_rival":   goles_rival,
        "goles_totales": goles_totales,
        "n_partido":     sig_partido,
        "es_local":      es_local
    }])[FEATURES]

    valor = float(np.clip(modelo.predict(X_nuevo)[0], 0.0, 1.0))

    if valor >= 0.75:
        etiqueta = "GANA UC"
    elif valor >= 0.25:
        etiqueta = "EMPATE"
    else:
        etiqueta = "PIERDE UC"

    print(f"\n  partido #{sig_partido} | {condicion} | {int(goles_uc)}-{int(goles_rival)}")
    print(f"  score predicho : {valor:.4f}")
    print(f"  resultado      : {etiqueta}")


# -------------------------------------------------------
# MENU PRINCIPAL
# -------------------------------------------------------
def main():
    df = cargar_datos()

    opciones = {"1", "2", "3", "4", "5", "6", "7", "8", "9"}
    while True:
        print("\n--- menu ---")
        print("")
        print("  [ datos ]")
        print("  1. ver datos")
        print("  2. graficos")
        print("")
        print("  [ regresion lineal multiple ]")
        print("  3. entrenar modelo lineal multiple")
        print("  4. predecir con modelo lineal")
        print("")
        print("  [ random forest ]")
        print("  5. entrenar random forest")
        print("  6. predecir con random forest")
        print("")
        print("  [ comparar modelos ]")
        print("  7. ver metricas de ambos modelos")
        print("")
        print("  8. salir")

        op = input("\n>> ").strip()
        if op not in opciones:
            print("eso no es una opcion valida")
            continue

        if   op == "1": print(df.head(20).to_string())
        elif op == "2": hacer_graficos(df)
        elif op == "3": entrenar_lineal(df)
        elif op == "4": predecir_lineal(df)
        elif op == "5": entrenar_rf(df)
        elif op == "6": predecir_rf(df)
        elif op == "7": comparar_modelos(df)
        elif op == "8": break


# -------------------------------------------------------
# COMPARACION — muestra metricas de ambos modelos juntas
# usa el mismo split para que la comparacion sea justa
# -------------------------------------------------------
def comparar_modelos(df):
    print("\n--- COMPARACION DE MODELOS ---\n")

    if not os.path.exists(LR_MODEL) or not os.path.exists(RF_MODEL):
        print("faltan modelos entrenados. entrena primero ambos (opciones 3 y 5)")
        return

    # cargamos los modelos guardados — no re-entrenamos
    with open(LR_MODEL, "rb") as f:
        lr = pickle.load(f)
    with open(RF_MODEL, "rb") as f:
        rf = pickle.load(f)

    # mismo split que en entrenamiento para comparar sobre los mismos datos de prueba
    _, X_test, _, y_test = preparar_split(df)

    resultados = {}
    for nombre, modelo in [("Regresion Lineal", lr), ("Random Forest", rf)]:
        y_pred = modelo.predict(X_test)
        resultados[nombre] = {
            "R²":   round(r2_score(y_test, y_pred), 4),
            "RMSE": round(np.sqrt(mean_squared_error(y_test, y_pred)), 4),
            "MAE":  round(mean_absolute_error(y_test, y_pred), 4),
        }

    print(f"  {'modelo':<20} {'R²':>8} {'RMSE':>8} {'MAE':>8}")
    print(f"  {'-'*46}")
    for nombre, m in resultados.items():
        print(f"  {nombre:<20} {m['R²']:>8} {m['RMSE']:>8} {m['MAE']:>8}")

    mejor = max(resultados, key=lambda k: resultados[k]["R²"])
    print(f"\n  mejor R² en test: {mejor}")


if __name__ == "__main__":
    main()
