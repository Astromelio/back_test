import pandas as pd
import numpy as np
from statistics import mean
from statistics import median

info = pd.read_json('ws.json')
devs = []
camb = []
info = info[info.astype(str)['prices'] != '[]'].reset_index(drop=True)
for x in range(0, len(info.index)):
    price = np.array(info.prices[x])
    cambio = np.diff(price)
    camb.append(cambio)

info["cambio"] = camb
camb_total = []
camb_final = []
cambio_real = []
for x in range(0, len(info.prices)):
    camb_tot = [((n - info.prices[x][0]) / (info.prices[x][0])) * 100 for n in info.prices[x]]
    camb_real = [((info.prices[x][n] - info.prices[x][n - 15]) / (info.prices[x][n - 15])) * 100 for n in
                 range(15, len(info.prices[x]) - 1)]
    camb_total.append(camb_tot)
    camb_final.append(camb_tot[-1])
    cambio_real.append(camb_real)
info['resta'] = camb_total
info['camb_final'] = camb_final
info['cambio_real'] = cambio_real

df = pd.read_csv('result.csv')
df['time_entry'] = pd.to_datetime(df['time_entry'], unit='ms').round('min')
info['time'] = pd.to_datetime(info['time']).round('min')
complete = pd.merge(info, df, left_on='time', right_on='time_entry')
importante = complete[
    ["symbol_x", "time_x", "prices", "cambio", "resta", "camb_final", "realizedPnl", "side", "cambio_real"]].copy()
importante['side2'] = 1
importante.loc[importante['side'] == "BUY", 'side2'] = -1
importante['resta'].to_numpy()
importante['zeros'] = 'no cero'
zeros = []
zeros2 = []
derivadas = []
derivadas2 = []
derivadas_real = []
zeros_real = []
cambiost = []
cambiost_real = []
for x in range(0, len(importante.resta)):
    zero = np.where(np.logical_and(np.array(importante.resta[x]) >= -0.002, np.array(importante.resta[x]) <= 0.002))[0]
    derivada = np.array(importante.resta[x])[zero]
    asign2 = np.sign(importante.cambio_real[x])
    signchange2 = ((np.roll(asign2, 1) - asign2) != 0).astype(int)
    zero_real = np.where(signchange2 == 1)[0]
    derivada_real = np.array(importante.cambio_real[x])[zero_real]
    asign = np.sign(importante.resta[x])
    signchange = ((np.roll(asign, 1) - asign) != 0).astype(int)
    zero2 = np.where(signchange == 1)[0]
    derivada2 = np.array(importante.resta[x])[zero2]
    cambiot = np.diff(zero2)
    cambiot_real = np.diff(zero_real)
    cambiost.append(cambiot)
    cambiost_real.append(cambiot_real)
    zeros2.append(zero2)
    derivadas2.append(derivada2)
    zeros.append(zero)
    derivadas.append(derivada)
    derivadas_real.append(derivada_real)
    zeros_real.append(zero_real)

importante['zeros'] = zeros
importante['derivadas'] = derivadas
importante['zeros2'] = zeros2
importante['derivadas2'] = derivadas2
importante['cambiot'] = cambiost
df_cero = importante[['zeros2', 'derivadas2', 'side2', 'realizedPnl']].copy()
df_cero['zero_real'] = zeros_real
df_cero['derivada_real'] = derivadas_real
df_cero['cambiot_real'] = cambiost_real
lismax = []
proms = []
medianas = []
for i in df_cero.cambiot_real:
    maximo = max(i)
    prom = mean(i)
    mediana = median(i)
    lismax.append(maximo)
    medianas.append(mediana)
    proms.append(prom)
df_cero['max_camb'] = lismax
df_cero['prom'] = proms
df_cero['mediana'] = medianas
signos_derivada = []
segundis = []
algos = []
precios = []
transacciones = []
lados = []
pnls = []
camb2 = []
camb4 = []
n = 0
for pos in range(0, len(df_cero.cambiot_real)):
    for i in range(0, len(df_cero.cambiot_real[pos])):

        if (df_cero.cambiot_real[pos][i] > 35):
            posicion = df_cero.zero_real[pos][i + 1]
            pre = importante.resta[pos][posicion + 15]
            segs = df_cero.cambiot_real[pos][i]
            cambi = df_cero.derivada_real[pos][i]
            try:
                cambi2 = df_cero.derivada_real[pos][i + 5]
            except:
                cambi2 = 0
            try:
                cambi4 = df_cero.derivada_real[pos][i + 10]
            except:
                cambi4 = 0

            algo = df_cero.zero_real[pos][i + 1] + 15
            lado = importante.side2[pos]
            transacion = n
            pnl = importante.camb_final[pos]
            signos_derivada.append(cambi)
            segundis.append(segs)
            algos.append(algo)
            precios.append(pre)
            transacciones.append(transacion)
            lados.append(lado)
            pnls.append(pnl)
            camb2.append(cambi2)
            camb4.append(cambi4)

    n += 1

df_transacciones = pd.DataFrame()
df_transacciones['transacion'] = transacciones
df_transacciones['sig'] = signos_derivada
df_transacciones['segs'] = segundis
df_transacciones['seg_real'] = algos
df_transacciones['dev'] = precios
df_transacciones['side'] = lados
df_transacciones['Pnl'] = pnls
df_transacciones['camb2'] = camb2
df_transacciones['camb4'] = camb4
