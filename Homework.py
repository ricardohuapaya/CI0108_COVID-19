#!/usr/bin/env python
# coding: utf-8

# # Modelos SARS COV-2

# ### Librerias

# In[227]:


import pandas as pd 
import matplotlib.pyplot as plt 
import math as m
import numpy as np
from scipy.integrate import odeint


# ### Datos que se usan 

# In[228]:


covid_confirmed = pd.read_excel("time_series_covid19_confirmed_global.xlsx")
covid_recovered = pd.read_excel("time_series_covid19_recovered_global.xlsx")
covid_death = pd.read_excel("time_series_covid19_deaths_global.xlsx")


# ### Dataframe

# In[229]:


CR = covid_confirmed['Costa Rica'][covid_confirmed['Costa Rica']>0]
CR2 = covid_recovered['Costa Rica'][44:]
CR3 = covid_death['Costa Rica'][44:]
df = pd.DataFrame({'Confirmados': CR, 'Recuperados': CR2, 'Fallecidos': CR3})
df = df.reset_index()
df['Nuevos'] = df.Confirmados.diff()
df['Nuevos_rec'] = df.Recuperados.diff()
df=df.fillna(0)
df["Cambio"] = df["Nuevos"] /df["Nuevos"].shift()
df= df.replace([np.inf, -np.inf], np.nan)
df=df.fillna(0)


# ## Modelo Gompertz

# In[230]:


x= range(len(CR))
x2= range(len(CR)+10)
bb= df.Nuevos.mean()


# In[231]:


a, b, c=10000, bb, bb*10**-3
y2 = [a*m.exp(-b * m.exp(-c*i)) for i in x2]
y3 = [m.exp(-c*i)*a*(-b)*(-c)*m.exp(-b * m.exp(-c*i)) for i in x2]


# In[232]:


fig = plt.figure(facecolor='w')
ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)
ax.plot(x, CR, 'b', label='Curva de Casos')
ax.plot(x2, y2, 'r', label='Gompertz')
ax.set_xlabel('Tiempo en días')
ax.set_ylabel('Contagiados ', rotation=0)
ax.grid(b=True, which='major', c='w', lw=2, ls='-')
legend = ax.legend()
legend.get_frame().set_alpha(0.5)
for spine in ('top', 'right', 'bottom', 'left'):
    ax.spines[spine].set_visible(False)
plt.show()


# ## Modelo SIR

# ### Datos

# In[233]:


N = 1 #población
I0, R0 = 0.01, 0 #datos iniciales
S0 = N - I0 - R0 #identidad
beta , gamma = 0.5, 1/5
# tiempo
t = np.linspace(0,  (50), (50))


# In[234]:


# The SIR model differential equations.
def deriv(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

# Initial conditions vector
y0 = S0, I0, R0
# Integrate the SIR equations over the time grid, t.
ret = odeint(deriv, y0, t, args=(N, beta, gamma))
S, I, R = ret.T


# In[235]:


fig = plt.figure(facecolor='w')
ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)
ax.plot(t, S, 'b', label='Suceptibles')
ax.plot(t, I, 'r', label='Infectados')
ax.plot(t, R, 'g',  label='Resueltos')
ax.set_xlabel('Tiempo en Semanas')
ax.set_ylabel('Proporcion I S R', rotation=0,labelpad=50)
ax.set_ylim(0,1)
ax.set_title('Modelo SIR General')
ax.yaxis.set_tick_params(length=0)
ax.xaxis.set_tick_params(length=0)
ax.grid(b=True, which='major', c='w', lw=2, ls='-')
legend = ax.legend()
legend.get_frame().set_alpha(0.5)
for spine in ('top', 'right', 'bottom', 'left'):
    ax.spines[spine].set_visible(False)
plt.tight_layout()


# Datos Adaptados a CR

# In[236]:


promedio= df.Cambio.mean()+0.3
gamma = 1/15
beta = promedio * gamma


# In[237]:


N = 1 #población
I0, R0 = 0.01, 0 #datos iniciales
S0 = N - I0 - R0 #identidad
#beta , gamma = 0.08, 1/20
# tiempo
t = np.linspace(0,  (200), (200))


# In[238]:


# The SIR model differential equations.
def deriv(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

# Initial conditions vector
y0 = S0, I0, R0
# Integrate the SIR equations over the time grid, t.
ret = odeint(deriv, y0, t, args=(N, beta, gamma))
S, I, R = ret.T


# In[239]:


fig = plt.figure(facecolor='w')
ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)
ax.plot(t, S, 'b', label='Suceptibles')
ax.plot(t, I, 'r', label='Infectados')
ax.plot(t, R, 'g',  label='Resueltos')
ax.set_xlabel('Tiempo en Semanas')
ax.set_ylabel('Proporcion I S R', rotation=0,labelpad=50)
ax.set_title('Modelo SIR Costa Rica 1/7/2020')
ax.set_ylim(0,1)
ax.yaxis.set_tick_params(length=0)
ax.xaxis.set_tick_params(length=0)
ax.grid(b=True, which='major', c='w', lw=2, ls='-')
legend = ax.legend()
legend.get_frame().set_alpha(0.5)
for spine in ('top', 'right', 'bottom', 'left'):
    ax.spines[spine].set_visible(False)
plt.tight_layout()


# ## Modelo SIRM 

# In[240]:


N = 1 #población
I0, R0, M0 = 0.01, 0, 0#datos iniciales
S0 = N - I0 - R0 #identidad
beta , gamma = 0.5, 1/5
pi= 0.01
k= 0.05
pit = pi -(k * I)
# tiempo
t = np.linspace(0,  (50), (50))


# In[241]:


# The SIRM model differential equations.
def deriv(y, t, N, beta, gamma, pi, k):
    S, I, R, M= y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = (1-(pi-(k*I))) * gamma * I
    dMdt = (pi-(k*I)) * gamma * I
    return dSdt, dIdt, dRdt, dMdt

# Initial conditions vector
y0 = S0, I0, R0, M0
# Integrate the SIR equations over the time grid, t.
ret = odeint(deriv, y0, t, args=(N, beta, gamma, pi, k))
S, I, R, M = ret.T


# In[242]:


fig = plt.figure(facecolor='w')
ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)
ax.plot(t, S, 'b', label='Suceptibles')
ax.plot(t, I, 'r', label='Infectados')
ax.plot(t, R, 'g',  label='Resueltos')
ax.plot(t, M,'m',  label='Muertes')
ax.set_xlabel('Tiempo en Semanas')
ax.set_ylabel('Proporcion I S R M', rotation=0,labelpad=50)
ax.set_ylim(0,1)
ax.yaxis.set_tick_params(length=0)
ax.xaxis.set_tick_params(length=0)
ax.grid(b=True, which='major', c='w', lw=2, ls='-')
ax.set_title('Modelo SIRM')

legend = ax.legend()
legend.get_frame().set_alpha(0.5)
for spine in ('top', 'right', 'bottom', 'left'):
    ax.spines[spine].set_visible(False)
plt.tight_layout()


# Datos corregidos a CR

# In[243]:


promedio= df.Cambio.mean()+0.3
gamma = 1/15
beta = promedio * gamma


# In[253]:


N = 1 #población
I0, R0, M0 = 0.01, 0, 0#datos iniciales
S0 = N - I0 - R0 #identidad
pi= df.loc[df.index[-1], "Fallecidos"]/df.loc[df.index[-1], "Confirmados"]
k= 0.05
pit = pi -(k * I)
# tiempo
t = np.linspace(0,  (300), (300))
# The SIRM model differential equations.
def deriv(y, t, N, beta, gamma, pi, k):
    S, I, R, M= y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = (1-(pi-(k*I))) * gamma * I
    dMdt = (pi-(k*I)) * gamma * I
    return dSdt, dIdt, dRdt, dMdt

# Initial conditions vector
y0 = S0, I0, R0, M0
# Integrate the SIR equations over the time grid, t.
ret = odeint(deriv, y0, t, args=(N, beta, gamma, pi, k))
S, I, R, M = ret.T
fig = plt.figure(facecolor='w')
ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)
ax.plot(t, S, 'b', label='Suceptibles')
ax.plot(t, I, 'r', label='Infectados')
ax.plot(t, R, 'g',  label='Resueltos')
ax.plot(t, M,'m',  label='Muertes')
ax.set_xlabel('Tiempo en Semanas')
ax.set_ylabel('Proporcion I S R M', rotation=0,labelpad=50)
ax.set_title('Modelo SIRM Costa Rica 1/7/2020')
ax.set_ylim(0,1)
ax.yaxis.set_tick_params(length=0)
ax.xaxis.set_tick_params(length=0)
ax.grid(b=True, which='major', c='w', lw=2, ls='-')
legend = ax.legend()
legend.get_frame().set_alpha(0.5)
for spine in ('top', 'right', 'bottom', 'left'):
    ax.spines[spine].set_visible(False)
plt.tight_layout()


# Muertes sin confinamiento
# 

# In[254]:


fig = plt.figure(facecolor='w')
ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)
ax.plot(t, M,'m',  label='Muertes')
ax.set_xlabel('Tiempo en Semanas')
ax.set_ylabel('Proporcion Muertes', rotation=0,labelpad=50)
ax.set_title('Muertes 1/7/2020')
ax.set_ylim(0,0.005)
ax.yaxis.set_tick_params(length=0)
ax.xaxis.set_tick_params(length=0)
ax.grid(b=True, which='major', c='w', lw=2, ls='-')
legend = ax.legend()
legend.get_frame().set_alpha(0.5)
for spine in ('top', 'right', 'bottom', 'left'):
    ax.spines[spine].set_visible(False)
plt.tight_layout()


# ### Modelo SIRM con tetta confinamiento 

# In[247]:


promedio= df.Cambio.mean()+0.3
gamma = 1/15
beta = promedio * gamma
pi= df.loc[df.index[-1], "Fallecidos"]/df.loc[df.index[-1], "Confirmados"]
print(pi)


# In[248]:


N = 1 #población
tetta = 0.07 #arbitrario pero en teoría con 0.08 se contiene la pandemia
I0, R0, M0 = 0.01, 0, 0 #datos iniciales
S0 = N - I0 - R0 #identidad,¿
k= 0.05
tetta = 0.07
# tiempo
t = np.linspace(0,  (300), (300))
# The SIRM model differential equations.
def deriv(y, t, N, beta, gamma, pi, k, tetta):
    S, I, R, M, = y
    dSdt = -beta * S*(1-tetta) * (I*(1-tetta))/ N
    dIdt = beta * S*(1-tetta) * (I*(1-tetta))/ N - gamma * I
    dRdt = (1-(pi-(k*I))) * gamma * I
    dMdt = (pi-(k*I2)) * gamma * I
    return dSdt, dIdt, dRdt, dMdt
# Initial conditions vector
y0 = S0, I0, R0, M0
# Integrate the SIR equations over the time grid, t.
ret = odeint(deriv, y0, t, args=(N, beta, gamma, pi, k, tetta))
S, I, R, M  = ret.T
fig = plt.figure(facecolor='w')
ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)
ax.plot(t, S, 'b', label='Suceptibles')
ax.plot(t, I2, 'r', label='Infectados')
ax.plot(t, R, 'g',  label='Resueltos')
ax.plot(t, M,'m',  label='Muertes')
ax.set_xlabel('Tiempo en Semanas')
ax.set_ylabel('Proporcion I S R M', rotation=0,labelpad=50)
ax.set_title('Modelo SIRM Costa Rica con Confinamiento 1/7/2020')
ax.set_ylim(0,1)
ax.yaxis.set_tick_params(length=0)
ax.xaxis.set_tick_params(length=0)
ax.grid(b=True, which='major', c='w', lw=2, ls='-')
legend = ax.legend()
legend.get_frame().set_alpha(0.5)
for spine in ('top', 'right', 'bottom', 'left'):
    ax.spines[spine].set_visible(False)
plt.tight_layout()
costo = (tetta*(S[-1]+I[-1]) + (M[-1]))*100
print('Costo de la Pandemia %.2f ' %(costo), '%')


# Muertes con Confinamiento 

# In[249]:


fig = plt.figure(facecolor='w')
ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)
ax.plot(t, M,'m',  label='Muertes')
ax.set_xlabel('Tiempo en Semanas')
ax.set_ylabel('Proporcion Muertes', rotation=0,labelpad=50)
ax.set_title('Muertes 1/7/2020')
ax.set_ylim(0,0.005)
ax.yaxis.set_tick_params(length=0)
ax.xaxis.set_tick_params(length=0)
ax.grid(b=True, which='major', c='w', lw=2, ls='-')
legend = ax.legend()
legend.get_frame().set_alpha(0.5)
for spine in ('top', 'right', 'bottom', 'left'):
    ax.spines[spine].set_visible(False)
plt.tight_layout()


# #### Confinamiento con thetta = 0.08 que soluciona la pandemia

# In[258]:


N = 1 #población
tetta = 0.07 #arbitrario pero en teoría con 0.08 se contiene la pandemia
I0, R0, M0 = 0.01, 0, 0 #datos iniciales
S0 = N - I0 - R0 #identidad,¿
k= 0.05
pit = pi -(k * I)
tetta = 0.08
# tiempo
t = np.linspace(0,  (300), (300))
# The SIRM model differential equations.
def deriv(y, t, N, beta, gamma, pi, k, tetta):
    S, I, R, M, = y
    dSdt = -beta * S*(1-tetta) * (I*(1-tetta))/ N
    dIdt = beta * S*(1-tetta) * (I*(1-tetta))/ N - gamma * I
    dRdt = (1-(pi-(k*I))) * gamma * I
    dMdt = (pi-(k*I)) * gamma * I
    return dSdt, dIdt, dRdt, dMdt
# Initial conditions vector
y0 = S0, I0, R0, M0
# Integrate the SIR equations over the time grid, t.
ret = odeint(deriv, y0, t, args=(N, beta, gamma, pi, k, tetta))
S, I, R, M  = ret.T
fig = plt.figure(facecolor='w')
ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)
ax.plot(t, S, 'b', label='Suceptibles')
ax.plot(t, I3, 'r', label='Infectados')
ax.plot(t, R, 'g',  label='Resueltos')
ax.plot(t, M,'m',  label='Muertes')
ax.set_xlabel('Tiempo en Semanas')
ax.set_ylabel('Proporcion I S R M', rotation=0,labelpad=50)
ax.set_title('Modelo SIRM Costa Rica 8% de la Población Confinada')
ax.set_ylim(0,1)
ax.yaxis.set_tick_params(length=0)
ax.xaxis.set_tick_params(length=0)
ax.grid(b=True, which='major', c='w', lw=2, ls='-')
legend = ax.legend()
legend.get_frame().set_alpha(0.5)
for spine in ('top', 'right', 'bottom', 'left'):
    ax.spines[spine].set_visible(False)
plt.tight_layout()
costo = (tetta*(S[-1]+I[-1]) + (M[-1]))*100
print('Costo de la Pandemia %.2f ' %(costo), '%')
