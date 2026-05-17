import requests
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
from dotenv import load_dotenv

load_dotenv()  # lee el .env automaticamente

API_KEY = os.getenv("API_KEY")
TEAM_ID = int(os.getenv("TEAM_ID"))
CSV_PATH = 'datos_catolica.csv'


headers = {
    'x-apisports-key': API_KEY,
    'x-apisports-host': 'v3.football.api-sports.io'
}


def cargar_datos():
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        # verifico que tenga todas las columnas que necesito
        if all(c in df.columns for c in ['resultado','goles_totales','goles_uc','goles_rival']):
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

            for p in resp.get('response', []):
                # me salto los que no terminaron
                if p['fixture']['status']['short'] != 'FT':
                    continue

                gh = p['goals']['home']
                ga = p['goals']['away']
                es_local = p['teams']['home']['id'] == TEAM_ID

                goles_uc    = gh if es_local else ga
                goles_rival = ga if es_local else gh

                # saco el resultado como numero pa poder hacer regresion despues
                if goles_uc > goles_rival:   res = 1
                elif goles_uc == goles_rival: res = 0.5
                else:                         res = 0

                # nota: corners y amarillas son aproximados, la api gratis no los da directo
                lista.append({
                    'resultado':     res,
                    'goles_totales': gh + ga,
                    'goles_uc':    goles_uc,
                    'goles_rival': goles_rival,
                    'condicion':     'Local' if es_local else 'Visita'
                })

        except Exception as e:
            print(f"fallo temporada {year}: {e}")

    df = pd.DataFrame(lista)
    df.insert(0, 'n_partido', range(1, len(df)+1))
    df.to_csv(CSV_PATH, index=False)
    print(f"descarga lista, {len(df)} partidos en total")
    return df


def hacer_graficos(df):
    plt.style.use('ggplot')
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Analisis UC 2022-2024', fontsize=14, fontweight='bold')

    # correlacion entre todas las variables numericas
    sns.heatmap(
        df[['resultado','goles_totales','goles_uc','goles_rival']].corr(),
        annot=True, cmap='Blues', ax=axes[0,0]
    )
    axes[0,0].set_title('mapa de calor')

    # goles segun si juegan de local o visita
    sns.boxplot(x='condicion', y='goles_totales', data=df,
                hue='condicion', palette='Set1', legend=False, ax=axes[0,1])
    axes[0,1].set_title('goles por condicion')

    df['resultado'].replace({1:'Ganado', 0.5:'Empate', 0:'Perdido'}) \
                   .value_counts() \
                   .plot(kind='pie', autopct='%1.1f%%', ax=axes[1,0])
    axes[1,0].set_title('historico de resultados')
    axes[1,0].set_ylabel('')

    # linea de tendencia pa ver si meten mas o menos goles con el tiempo
    sns.regplot(x='n_partido', y='goles_totales', data=df,
                ax=axes[1,1], line_kws={'color':'red'})
    axes[1,1].set_title('tendencia de goles')

    plt.tight_layout()
    plt.show()

def calcular_r2(df):
    df_modelo = df.copy()
    df_modelo['es_local'] = (df_modelo['condicion'] == 'Local').astype(int)

    features = ['goles_uc', 'goles_rival', 'goles_totales', 'n_partido', 'es_local']
    X = df_modelo[features]
    y = df_modelo['resultado']

    modelo = LinearRegression()
    modelo.fit(X, y)

    y_pred = modelo.predict(X)
    r2 = r2_score(y, y_pred)

    print("        R² DEL MODELO      ")
    print(f"  R²           : {r2:.4f}")
    print(f"  Explicado    : {r2*100:.2f}%")

    return r2

def predecir_partido(df):
    X = df[['n_partido']]

    # un modelo por variable, todos con regresion lineal simple
    m_resultado  = LinearRegression().fit(X, df['resultado'])
    m_goles      = LinearRegression().fit(X, df['goles_totales'])

    sig = pd.DataFrame({'n_partido': [len(df)+1]})

    r = m_resultado.predict(sig)[0]
    g = m_goles.predict(sig)[0]

    print("  prediccion para UC vs La Calera\n")
    print(f"  resultado esperado : {'GANA UC' if r > 0.5 else 'NO gana UC'}")
    print(f"  goles totales      : {g:.2f}  ")

def regresion_multiple(df):
    df_modelo = df.copy()
    df_modelo['es_local'] = (df_modelo['condicion'] == 'Local').astype(int)

    features = ['goles_uc', 'goles_rival', 'goles_totales', 'n_partido', 'es_local']
    X = df_modelo[features]
    y = df_modelo['resultado']

    modelo = LinearRegression()
    modelo.fit(X, y)

    print("     REGRESIÓN MÚLTIPLE - UC 2022-2024   ")
    print("\n  Coeficientes del modelo:")
    for feat, coef in zip(features, modelo.coef_):
        print(f"    {feat:>15} : {coef:+.4f}")
    print(f"    {'intercepto':>15} : {modelo.intercept_:+.4f}")


    return modelo

def main():
    df = cargar_datos()

    opciones = {'1','2','3','4','5','6'}
    while True:
        print("\n--- menu ---")
        print("1. ver datos")
        print("2. graficos")
        print("3. prediccion vs calera")
        print("4. ver R² del modelo")
        print("5. regresion multiple")
        print("6. salir")

        op = input(">> ").strip()
        if op not in opciones:
            print("eso no es una opcion valida")
            continue

        if   op == '1': print(df.head(20).to_string())
        elif op == '2': hacer_graficos(df)
        elif op == '3': predecir_partido(df)
        elif op == '4': calcular_r2(df)
        elif op == '5': regresion_multiple(df)
        elif op == '6': break


if __name__ == "__main__":
    main()
