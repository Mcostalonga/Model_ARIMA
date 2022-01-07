# Importando bibliotecas

import pandas as pd
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.arima_model import ARIMA
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker

# Usando a função read_csv do pandas para realizar a leitura do arquivo com os dados de temperatura
# link para os dados: https://machinelearningmastery.com/autoregression-models-time-series-forecasting-python/
# header=0 indica que a primeira linha do arquivo é o cabeçalho das informações
# index_col=0 indica que a primeira coluna será considerada como o index.
series = pd.read_csv('daily-min-temperatures.csv', header=0, index_col=0, parse_dates=True)

# Imprime os 5 primeiros valores dos dados lidos e guardados na variável "series"
print(series.head())

# Imprime os últimos 5 valores dos dados.
print(series.tail())

# Pode-se imprimir os cinco primeiros e últimos valores, por padrão, se a quantidade de dados for muito grande
# se não for especificado nada.
print(series)

# Plotando os dados.
# figsize é usado para alterar a dimensão do gráfico no sentido (horizontal, vertical)
fig = series.plot(figsize=(15, 6), color=(0.2, 0.2, 0.7))

# Adicionando legenda aos eixos x e y
fig.set_ylabel('Temperatura mínima [°C]', font='Times New Roman', fontsize=16, fontweight='bold')
fig.set_xlabel('Data', font='Times New Roman', fontsize=16, fontweight='bold')

# Adicionando título ao gráfico
fig.set_title('Temperatura mínima ao longo de 10 anos',
              font='Times New Roman', fontsize=18, fontweight='bold')

# Alterando a fonte dos valores dos eixos
plt.setp(fig.get_xticklabels(), font='Times New Roman', fontsize=16)
plt.setp(fig.get_yticklabels(), font='Times New Roman', fontsize=16)
plt.show(block=False)
plt.pause(2)
plt.close("all")

### Percebe-se que há um componente de sazonalidade forte, o qual pode ser neutralizado para transformar os dados ###
### estacionários, realizando a diferença sazonal. Isso é feito realizando a subtração do valor observado em um dia ###
### com o valor observado naquele mesmo dia em um ano anterior, por exemplo. ###


### Os coeficientes de autocorrelação ajudam na determinação dos parâmetros para o modelo arima, como o lag ###
### (defasagem) para se obter valores com significância estatística. ###

# Plotando os coeficientes de autocorrelação
fig1, ax1 = plt.subplots(figsize=(15, 6))
plot_acf(series, lags=200, title='', ax=ax1)

# Adicionando legenda aos eixos x e y
ax1.set_ylabel('Coeficiente de autocorrelação', font='Times New Roman', fontsize=16, fontweight='bold')
ax1.set_xlabel('Lags', font='Times New Roman', fontsize=16, fontweight='bold')

# Alterando a fonte dos valores dos eixos
plt.setp(ax1.get_xticklabels(), font='Times New Roman', fontsize=16)
plt.setp(ax1.get_yticklabels(), font='Times New Roman', fontsize=16)
plt.show(block=False)
plt.pause(2)
plt.close("all")

# Dividindo os dados.
# Nesse caso específico, usaremos as informações até o dia 21 de dezembro de 1990 para dados de treino
# e o dados restantes serão usados para testar o modelo.

# Determinando o ponto de separação dos dados. Usaremos os 10 dias restantes para testar o modelo.
separacao = len(series) - 10

# Dados para o treino do modelo
dados_de_treino = series[:separacao]

# Dados para o teste do modelo
dados_de_teste = series[separacao:]


# Para mais informações sobre a diferenciação, acesso o link a seguir.
# https://machinelearningmastery.com/make-sample-forecasts-arima-python/
# Como comentado acima, os dados apresentados apresentam uma sazonalidade e, por isso,
# podemos transformar a série temporal em uma série estacionária.

# Função para a diferenciação dos dados
def diferenca(dados, intervalo=1):
    dif = list()
    for i in range(intervalo, len(dados)):
        valor = dados[i] - dados[i - intervalo]
        dif.append(valor)
    return np.array(dif)


# Função para inverter o valor que foi diferenciado
def inverte_diferenca(valor_original, yhat, intervalo=1):
    return yhat + valor_original[-intervalo]


# Diferença sazonal
X = dados_de_treino.values
dias_por_ano = 365
diferenciacao = diferenca(X, dias_por_ano)

# Modelo ARIMA(p,d,q)
# p: número de lags (ou defasagem), d: número de diferenciações entre as observações, q: ordem da média móvel.
modelo = ARIMA(diferenciacao, order=(9, 1, 3))
ajuste_modelo = modelo.fit()

# Imprimindo o resumo do modelo
print(ajuste_modelo.summary())

inicio = len(diferenciacao)
fim = inicio + 9
previsao = ajuste_modelo.predict(start=inicio, end=fim)

# Inverte a diferenciação de previsão para algo usável
historico = [x for x in X]
dia = 1

prev_lista = []
dados_teste = []
dias = range(1, 11)

for yhat in previsao:
    inversao = inverte_diferenca(historico, yhat, dias_por_ano)
    prev_lista.append(inversao)
    dados_teste.append(dados_de_teste.values[dia - 1])
    historico.append(inversao)
    dia += 1

fig2, ax2 = plt.subplots(figsize=(15, 6))
plt.plot(dias, prev_lista, color=(1, 0, 0), label='Previsão', linestyle='--')
plt.plot(dias, dados_teste, color=(0.2, 0.2, 0.7), label='Valor de teste')

# Adicionando legenda aos eixos x e y
ax2.set_ylabel('Temperatura [°C]', font='Times New Roman', fontsize=16, fontweight='bold')
ax2.set_xlabel('Dias', font='Times New Roman', fontsize=16, fontweight='bold')

# Alterando a fonte dos valores dos eixos
plt.setp(ax2.get_xticklabels(), font='Times New Roman', fontsize=16)
plt.setp(ax2.get_yticklabels(), font='Times New Roman', fontsize=16)

# Limites do eixo x
ax2.set_xlim([0, 10])

# Alterando espaçamento dos ticks do eixo x
espacamento = 1
ax2.xaxis.set_major_locator(ticker.MultipleLocator(espacamento))

# Mostrando legenda
plt.legend()


# Mostrando o gráfico de previsão
plt.show(block=False)
plt.pause(2)
plt.close("all")

