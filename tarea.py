import requests
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

# features para predecir RESULTADO — incluye goles porque ya ocurrieron
# y se usan para entrenar/evaluar el modelo
FEAT_RESULTADO = ["goles_uc", "goles_rival", "goles_totales", "n_partido", "es_local"]

# features para predecir GOLES — no puede incluir goles porque son el target
FEAT_GOLES = ["n_partido", "es_local"]


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
# SPLITS
# mismo random_state en ambos modelos para comparacion justa
# -------------------------------------------------------
def preparar_split_resultado(df):
    df_m = df.copy()
    df_m["es_local"] = (df_m["condicion"] == "Local").astype(int)
    X = df_m[FEAT_RESULTADO]
    y = df_m["resultado"]
    return train_test_split(X, y, test_size=0.2, random_state=42)

def preparar_split_goles(df):
    df_m = df.copy()
    df_m["es_local"] = (df_m["condicion"] == "Local").astype(int)
    X = df_m[FEAT_GOLES]
    y = df_m[["goles_uc", "goles_rival"]]
    return train_test_split(X, y, test_size=0.2, random_state=42)


# -------------------------------------------------------
# GRAFICOS DE DIAGNOSTICO AL ENTRENAR
# -------------------------------------------------------
def graficar_entrenamiento_resultado(y_test, y_pred, nombre_modelo):
    residuos = y_test.values - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Diagnostico — {nombre_modelo} (Resultado)", fontsize=13, fontweight="bold")

    # real vs predicho
    axes[0].scatter(y_test, y_pred, alpha=0.6, color="steelblue", edgecolors="white", linewidths=0.4)
    axes[0].plot([0, 1], [0, 1], "r--", linewidth=1.5, label="prediccion perfecta")
    axes[0].set_xlabel("resultado real")
    axes[0].set_ylabel("resultado predicho")
    axes[0].set_title("Real vs Predicho (test)")
    axes[0].legend()

    # histograma de residuos
    # centrado en 0 = modelo sin sesgo
    axes[1].hist(residuos, bins=15, color="steelblue", edgecolor="white", alpha=0.85)
    axes[1].axvline(0, color="red", linestyle="--", linewidth=1.5, label="residuo = 0")
    axes[1].axvline(residuos.mean(), color="orange", linestyle="-", linewidth=1.5,
                    label=f"media = {residuos.mean():.3f}")
    axes[1].set_xlabel("residuo (real - predicho)")
    axes[1].set_ylabel("frecuencia")
    axes[1].set_title("Histograma de Residuos")
    axes[1].legend()

    plt.tight_layout()
    plt.show()


def graficar_entrenamiento_goles(y_test, y_pred, nombre_modelo):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(f"Diagnostico — {nombre_modelo} (Goles)", fontsize=13, fontweight="bold")

    etiquetas = ["goles_uc", "goles_rival"]
    colores   = ["steelblue", "darkorange"]

    for i, (etiq, color) in enumerate(zip(etiquetas, colores)):
        real     = y_test.values[:, i]
        predicho = y_pred[:, i]
        residuos = real - predicho

        mn, mx = min(real.min(), predicho.min()), max(real.max(), predicho.max())
        axes[i, 0].scatter(real, predicho, alpha=0.6, color=color, edgecolors="white", linewidths=0.4)
        axes[i, 0].plot([mn, mx], [mn, mx], "r--", linewidth=1.5, label="prediccion perfecta")
        axes[i, 0].set_xlabel(f"{etiq} real")
        axes[i, 0].set_ylabel(f"{etiq} predicho")
        axes[i, 0].set_title(f"Real vs Predicho — {etiq}")
        axes[i, 0].legend()

        axes[i, 1].hist(residuos, bins=12, color=color, edgecolor="white", alpha=0.85)
        axes[i, 1].axvline(0, color="red", linestyle="--", linewidth=1.5, label="residuo = 0")
        axes[i, 1].axvline(residuos.mean(), color="black", linestyle="-", linewidth=1.5,
                            label=f"media = {residuos.mean():.3f}")
        axes[i, 1].set_xlabel("residuo (real - predicho)")
        axes[i, 1].set_ylabel("frecuencia")
        axes[i, 1].set_title(f"Histograma de Residuos — {etiq}")
        axes[i, 1].legend()

    plt.tight_layout()
    plt.show()


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


# =======================================================
# PREDICCION DE RESULTADO
# usa FEAT_RESULTADO = goles_uc, goles_rival, goles_totales, n_partido, es_local
# en prediccion se le piden los goles al usuario
# =======================================================

def entrenar_resultado(df, modelo, nombre_modelo):
    print(f"\n--- ENTRENAMIENTO: {nombre_modelo} (RESULTADO) ---\n")

    X_train, X_test, y_train, y_test = preparar_split_resultado(df)
    print(f"entrenamiento: {len(X_train)} partidos | prueba: {len(X_test)} partidos")
    print(f"features usadas: {FEAT_RESULTADO}\n")

    modelo.fit(X_train, y_train)

    y_train_pred = modelo.predict(X_train)
    y_test_pred  = modelo.predict(X_test)

    print("  [ Metricas Entrenamiento ]")
    print(f"    R²   : {r2_score(y_train, y_train_pred):.4f}")
    print(f"    RMSE : {np.sqrt(mean_squared_error(y_train, y_train_pred)):.4f}")
    print(f"    MAE  : {mean_absolute_error(y_train, y_train_pred):.4f}")

    print("\n  [ Metricas Prueba ]")
    r2_test = r2_score(y_test, y_test_pred)
    print(f"    R²   : {r2_test:.4f}")
    print(f"    RMSE : {np.sqrt(mean_squared_error(y_test, y_test_pred)):.4f}")
    print(f"    MAE  : {mean_absolute_error(y_test, y_test_pred):.4f}")

    gap = r2_score(y_train, y_train_pred) - r2_test
    if abs(gap) > 0.15:
        print(f"\n  aviso: gap R² train-test = {gap:+.4f} (posible sobreajuste)")
    else:
        print(f"\n  gap R² train-test = {gap:+.4f} (ok)")

    graficar_entrenamiento_resultado(y_test, y_test_pred, nombre_modelo)

    print(f"\nModelo '{nombre_modelo}' listo en memoria.")
    return modelo


def predecir_resultado(df, modelo, nombre_modelo):
    print(f"\n--- PREDICCION: {nombre_modelo} (RESULTADO) ---\n")

    if modelo is None:
        print(f"No hay modelo de '{nombre_modelo}' en memoria. Entrena primero.")
        return

    sig_partido = len(df) + 1
    media_uc    = round(df["goles_uc"].mean(), 1)
    media_rival = round(df["goles_rival"].mean(), 1)

    print(f"promedios historicos: UC={media_uc} | rival={media_rival}")
    print("(presiona Enter para usar el promedio)\n")

    try:
        guc_str  = input(f"goles UC esperados    [Enter={media_uc}]  : ").strip()
        grv_str  = input(f"goles rival esperados [Enter={media_rival}] : ").strip()
        cond_str = input("condicion (Local/Visita) [Enter=Local]    : ").strip()

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
    goles_totales = goles_uc + goles_rival

    X_nuevo = pd.DataFrame([{
        "goles_uc":      goles_uc,
        "goles_rival":   goles_rival,
        "goles_totales": goles_totales,
        "n_partido":     sig_partido,
        "es_local":      es_local
    }])[FEAT_RESULTADO]

    valor    = float(np.clip(modelo.predict(X_nuevo)[0], 0.0, 1.0))
    etiqueta = "GANA UC" if valor >= 0.75 else ("EMPATE" if valor >= 0.25 else "PIERDE UC")

    print(f"\n  partido #{sig_partido} | {condicion} | {int(goles_uc)}-{int(goles_rival)}")
    print(f"  score predicho : {valor:.4f}")
    print(f"  resultado      : {etiqueta}")


def comparar_modelos(df, mod_mlr, mod_rf):
    print("\n--- COMPARACION DE MODELOS (RESULTADO) ---\n")

    if mod_mlr is None or mod_rf is None:
        print("Faltan modelos. Entrena los 2 primero (opciones 3 y 5).")
        return

    _, X_test, _, y_test = preparar_split_resultado(df)

    print(f"  {'modelo':<18} {'R²':>8} {'RMSE':>8} {'MAE':>8}")
    print(f"  {'-'*44}")
    r2_vals = []
    for nombre, modelo in [("Reg Multiple", mod_mlr), ("Random Forest", mod_rf)]:
        y_pred = modelo.predict(X_test)
        r2     = r2_score(y_test, y_pred)
        rmse   = np.sqrt(mean_squared_error(y_test, y_pred))
        mae    = mean_absolute_error(y_test, y_pred)
        r2_vals.append(r2)
        print(f"  {nombre:<18} {r2:>8.4f} {rmse:>8.4f} {mae:>8.4f}")

    plt.figure(figsize=(6, 4))
    bars = plt.bar(["Reg Multiple", "Random Forest"], r2_vals,
                   color=["darkorange", "steelblue"], edgecolor="white", width=0.5)
    for bar, val in zip(bars, r2_vals):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{val:.4f}", ha="center", va="bottom", fontsize=11)
    plt.ylim(0, max(r2_vals) * 1.2)
    plt.ylabel("R² en test")
    plt.title("Comparacion R² — Resultado")
    plt.tight_layout()
    plt.show()


# =======================================================
# PREDICCION DE GOLES
# usa FEAT_GOLES = n_partido, es_local
# no puede usar goles como features porque son el target
# =======================================================

def entrenar_goles(df, modelo, nombre_modelo):
    print(f"\n--- ENTRENANDO GOLES: {nombre_modelo} ---\n")

    X_train, X_test, y_train, y_test = preparar_split_goles(df)
    print(f"entrenamiento: {len(X_train)} partidos | prueba: {len(X_test)} partidos")
    print(f"features usadas: {FEAT_GOLES}")
    print("(nota: no se pueden usar goles como features porque son lo que se quiere predecir)\n")

    modelo.fit(X_train, y_train)

    y_train_pred = modelo.predict(X_train)
    y_test_pred  = modelo.predict(X_test)

    print("  [ Metricas Entrenamiento ]")
    print(f"    R²   : {r2_score(y_train, y_train_pred):.4f}")
    print(f"    RMSE : {np.sqrt(mean_squared_error(y_train, y_train_pred)):.4f}")
    print(f"    MAE  : {mean_absolute_error(y_train, y_train_pred):.4f}")

    print("\n  [ Metricas Prueba ]")
    r2_test = r2_score(y_test, y_test_pred)
    print(f"    R²   : {r2_test:.4f}")
    print(f"    RMSE : {np.sqrt(mean_squared_error(y_test, y_test_pred)):.4f}")
    print(f"    MAE  : {mean_absolute_error(y_test, y_test_pred):.4f}")

    gap = r2_score(y_train, y_train_pred) - r2_test
    if abs(gap) > 0.15:
        print(f"\n  aviso: gap R² train-test = {gap:+.4f} (posible sobreajuste)")
    else:
        print(f"\n  gap R² train-test = {gap:+.4f} (ok)")

    graficar_entrenamiento_goles(y_test, y_test_pred, nombre_modelo)

    print(f"\nModelo '{nombre_modelo}' listo en memoria.")
    return modelo


def predecir_goles(df, modelo, nombre_modelo):
    print(f"\n--- PREDICIENDO GOLES: {nombre_modelo} ---\n")

    if modelo is None:
        print(f"No hay modelo de '{nombre_modelo}' en memoria. Entrena primero.")
        return

    sig_partido = len(df) + 1
    cond_str    = input("condicion (Local/Visita) [Enter=Local]: ").strip()
    condicion   = cond_str.capitalize() if cond_str else "Local"

    if condicion not in ("Local", "Visita"):
        print("condicion no reconocida, se usa Local")
        condicion = "Local"

    es_local = 1 if condicion == "Local" else 0
    X_nuevo  = pd.DataFrame([{"n_partido": sig_partido, "es_local": es_local}])

    prediccion       = np.clip(modelo.predict(X_nuevo)[0], 0, None)
    goles_uc_pred    = int(round(prediccion[0]))
    goles_rival_pred = int(round(prediccion[1]))

    print(f"\n  Partido #{sig_partido} | {condicion}")
    print(f"  Marcador estimado: UC {goles_uc_pred} - {goles_rival_pred} Rival")


def comparar_modelos_goles(df, mod_mlr, mod_rf):
    print("\n--- COMPARACION DE MODELOS (GOLES) ---\n")

    if mod_mlr is None or mod_rf is None:
        print("Faltan modelos. Entrena los 2 primero (opciones 8 y 10).")
        return

    _, X_test, _, y_test = preparar_split_goles(df)

    print(f"  {'modelo':<18} {'R²':>8} {'RMSE':>8} {'MAE':>8}")
    print(f"  {'-'*44}")
    r2_vals = []
    for nombre, modelo in [("Reg Multiple", mod_mlr), ("Random Forest", mod_rf)]:
        y_pred = modelo.predict(X_test)
        r2     = r2_score(y_test, y_pred)
        rmse   = np.sqrt(mean_squared_error(y_test, y_pred))
        mae    = mean_absolute_error(y_test, y_pred)
        r2_vals.append(r2)
        print(f"  {nombre:<18} {r2:>8.4f} {rmse:>8.4f} {mae:>8.4f}")

    plt.figure(figsize=(6, 4))
    bars = plt.bar(["Reg Multiple", "Random Forest"], r2_vals,
                   color=["darkorange", "steelblue"], edgecolor="white", width=0.5)
    for bar, val in zip(bars, r2_vals):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f"{val:.4f}", ha="center", va="bottom", fontsize=11)
    lim = max(abs(v) for v in r2_vals)
    plt.ylim(min(r2_vals) - lim * 0.3, max(r2_vals) + lim * 0.3)
    plt.ylabel("R² en test")
    plt.title("Comparacion R² — Goles")
    plt.tight_layout()
    plt.show()


# -------------------------------------------------------
# MENU PRINCIPAL
# -------------------------------------------------------
def main():
    df = cargar_datos()

    modelos = {
        "mlr_res":   None,
        "rf_res":    None,
        "mlr_goles": None,
        "rf_goles":  None
    }

    opciones = {str(i) for i in range(1, 14)}
    while True:
        print("\n" + "="*38)
        print("           MENU PRINCIPAL")
        print("="*38)

        print("\n  [ Gráficos ]")
        print("  1. Ver datos historicos")
        print("  2. Graficos y tendencias")

        print("\n  [ Prediccion de Resultado (G/E/P) ]")
        print("  3. Entrenar Regresion Multiple")
        print("  4. Predecir con Regresion Multiple")
        print("  5. Entrenar Random Forest")
        print("  6. Predecir con Random Forest")
        print("  7. Comparar modelos de resultado")

        print("\n  [ Prediccion de Goles Exactos ]")
        print("  8. Entrenar goles (Regresion Multiple)")
        print("  9. Predecir marcador (Regresion Multiple)")
        print("  10. Entrenar goles (Random Forest)")
        print("  11. Predecir goles (Random Forest)")
        print("  12. Comparar modelos de goles")

        print("\n  13. Salir")

        op = input("\n>> ").strip()
        if op not in opciones:
            print("Eso no es una opcion valida")
            continue

        if   op == "1":  print(df.tail(20).to_string())
        elif op == "2":  hacer_graficos(df)
        elif op == "3":  modelos["mlr_res"]   = entrenar_resultado(df, LinearRegression(), "Regresion Multiple")
        elif op == "4":  predecir_resultado(df, modelos["mlr_res"],   "Regresion Multiple")
        elif op == "5":  modelos["rf_res"]    = entrenar_resultado(df, RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42), "Random Forest")
        elif op == "6":  predecir_resultado(df, modelos["rf_res"],    "Random Forest")
        elif op == "7":  comparar_modelos(df, modelos["mlr_res"], modelos["rf_res"])
        elif op == "8":  modelos["mlr_goles"] = entrenar_goles(df, LinearRegression(), "Regresion Multiple")
        elif op == "9":  predecir_goles(df, modelos["mlr_goles"],     "Regresion Multiple")
        elif op == "10": modelos["rf_goles"]  = entrenar_goles(df, RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42), "Random Forest")
        elif op == "11": predecir_goles(df, modelos["rf_goles"],      "Random Forest")
        elif op == "12": comparar_modelos_goles(df, modelos["mlr_goles"], modelos["rf_goles"])
        elif op == "13": break


if __name__ == "__main__":
    main()
