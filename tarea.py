import urllib.request
import json
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

temporadas = ["2020", "2021", "2022", "2023", "2024", "2025", "2026"]

todos_partidos = []

for año in temporadas:

    url = f"https://www.thesportsdb.com/api/v1/json/3/eventsseason.php?id=4627&s={año}"

    respuesta = urllib.request.urlopen(url)
    data = json.loads(respuesta.read())

    if data["events"] is not None:
        for partido in data["events"]:
            todos_partidos.append(partido)

lista = []

# tomamos los resultados de los partidos de catolica de visita y local y les asignamos un valor segun el resultado
for partido in todos_partidos:

    local = partido["strHomeTeam"].lower()
    visita = partido["strAwayTeam"].lower()

    if partido["intHomeScore"] is not None and partido["intAwayScore"] is not None:

        goles_local = int(partido["intHomeScore"])
        goles_visita = int(partido["intAwayScore"])

        if "católica" in local or "catolica" in local:

            if goles_local > goles_visita:
                resultado = 1
            elif goles_local == goles_visita:
                resultado = 0.5
            else:
                resultado = 0

            lista.append(resultado)

        elif "católica" in visita or "catolica" in visita:

            if goles_visita > goles_local:
                resultado = 1
            elif goles_visita == goles_local:
                resultado = 0.5
            else:
                resultado = 0

            lista.append(resultado)


df = pd.DataFrame({"resultado": lista})

x = np.array(range(1, len(df) + 1)).reshape(-1, 1)
y = np.array(df["resultado"])

modelo = LinearRegression()
modelo.fit(x, y)

# prediccion
proximo = len(df) + 1

prediccion = modelo.predict([[proximo]])[0]

if prediccion < 0:
    prediccion = 0

if prediccion > 1:
    prediccion = 1

y_pred = modelo.predict(x)

r2 = r2_score(y, y_pred)
mae = mean_absolute_error(y, y_pred)

# resultados

print()
print("Probabilidad estimada de que Universidad Católica gane su proximo partido:",
      round(prediccion * 100, 2), "%")

