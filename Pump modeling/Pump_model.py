# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 13:35:51 2020

@author: rashedi
"""
import numpy as np
import math
import pandas as pd
from sklearn.linear_model import LinearRegression

data=pd.read_csv('Data_from_xylect.csv')
flow60=data['Flow 60'].dropna()[:,np.newaxis]/1000
flow55=data['Flow 55'].dropna()[:,np.newaxis]/1000
flow50=data['Flow 50'].dropna()[:,np.newaxis]/1000
flow45=data['Flow 45'].dropna()[:,np.newaxis]/1000
flow40=data['Flow 40'].dropna()[:,np.newaxis]/1000
head60=data['Head 60'].dropna()[:,np.newaxis]
head55=data['Head 55'].dropna()[:,np.newaxis]
head50=data['Head 50'].dropna()[:,np.newaxis]
head45=data['Head 45'].dropna()[:,np.newaxis]
head40=data['Head 40'].dropna()[:,np.newaxis]
efficiency60=data['Overall Eff 60'].dropna()[:,np.newaxis]
efficiency55=data['Overall Eff 55'].dropna()[:,np.newaxis]
efficiency50=data['Overall Eff 50'].dropna()[:,np.newaxis]
efficiency45=data['Overall Eff 45'].dropna()[:,np.newaxis]
efficiency40=data['Overall Eff 40'].dropna()[:,np.newaxis]
input_power60=data['Power input 60'].dropna()[:,np.newaxis]*1000
input_power55=data['Power input 55'].dropna()[:,np.newaxis]*1000
input_power50=data['Power input 50'].dropna()[:,np.newaxis]*1000
input_power45=data['Power input 45'].dropna()[:,np.newaxis]*1000
input_power40=data['Power input 40'].dropna()[:,np.newaxis]*1000
#### Constants ####
d_50=0.425e-3
S_kt=0.8
g=9.81
h_init=2
Q_0=0.013
D=0.08
w_s=0
rho_s=2650
rho_w=1000
rho_m=100/(w_s/rho_s+(100-w_s)/rho_w)
C_v=(rho_m-rho_w)/(rho_s-rho_w)
L_p=30
mu_w=8.9e-4
mu_m=mu_w*(1+2.5*C_v+10.05*C_v**2+0.00273*np.exp(16.6*C_v))
eps=0.0001
V=230
I_max=9.6
current=3
I_min=(40/60)**3*I_max
if (current<I_min) | (current>I_max):
    raise Exception('Value of current is out of range!')
I=current
#### Extracting the right speed info ####
freq_max=60
freq=(I/I_max)**(1/3)*freq_max
#### Predicting True Pressure ####
HEAD60=LinearRegression().fit(flow60,head60)
HEAD55=LinearRegression().fit(flow55,head55)
HEAD50=LinearRegression().fit(flow50,head50)
HEAD45=LinearRegression().fit(flow45,head45)
HEAD40=LinearRegression().fit(flow40,head40)
#### Predicting Total Efficiency and Input Power ####
def Parameters(Q,eff,power):    
    X1=np.ones((len(Q),1))
    X2=Q
    X3=np.multiply(Q,Q)
    X4=np.multiply(X3,Q)
    X_eff=np.column_stack((X1,X2,X3))    
    X_power=np.column_stack((X1,X2,X3,X4))
    params_1=np.matmul(np.linalg.inv(np.matmul(X_eff.T,X_eff)),np.matmul(X_eff.T,eff))
    params_2=np.matmul(np.linalg.inv(np.matmul(X_power.T,X_power)),np.matmul(X_power.T,power))
    return params_1,params_2
param_eff_60,param_pow_60=Parameters(flow60,efficiency60,input_power60)
param_eff_55,param_pow_55=Parameters(flow55,efficiency55,input_power55)
param_eff_50,param_pow_50=Parameters(flow50,efficiency50,input_power50)
param_eff_45,param_pow_45=Parameters(flow45,efficiency45,input_power45)
param_eff_40,param_pow_40=Parameters(flow40,efficiency40,input_power40)
def tot_eff(Q,param_eff):
    etta=param_eff[0]+param_eff[1]*Q+param_eff[2]*Q**2
    return etta/100
def input_power(Q,param_pow):
    inp_power=param_pow[0]+param_pow[1]*Q+param_pow[2]*Q**2+param_pow[3]*Q**3
    return inp_power

Q=Q_0
dt=0.01
error=10
#for i in range(n):
while error>1e-10:
    if (freq<=60) & (freq>55):
        Power=(input_power(Q,param_pow_60)-input_power(Q,param_pow_55))/5*(freq-55)+input_power(Q,param_pow_55)
        Tot_eff=(tot_eff(Q,param_eff_60)-tot_eff(Q,param_eff_55))/5*(freq-55)+tot_eff(Q,param_eff_55)
    elif (freq<=55) & (freq>50):
        Power=(input_power(Q,param_pow_55)-input_power(Q,param_pow_50))/5*(freq-50)+input_power(Q,param_pow_50)
        Tot_eff=(tot_eff(Q,param_eff_55)-tot_eff(Q,param_eff_50))/5*(freq-50)+tot_eff(Q,param_eff_50)
    elif (freq<=50) & (freq>45):
        Power=(input_power(Q,param_pow_50)-input_power(Q,param_pow_45))/5*(freq-45)+input_power(Q,param_pow_45)
        Tot_eff=(tot_eff(Q,param_eff_50)-tot_eff(Q,param_eff_45))/5*(freq-45)+tot_eff(Q,param_eff_45)
    elif (freq<=45) & (freq>=40):
        Power=(input_power(Q,param_pow_45)-input_power(Q,param_pow_40))/5*(freq-40)+input_power(Q,param_pow_40)
        Tot_eff=(tot_eff(Q,param_eff_45)-tot_eff(Q,param_eff_40))/5*(freq-40)+tot_eff(Q,param_eff_40)
    Net_Power=Power*Tot_eff
    Re=rho_m*4*Q/math.pi/D**2*D/mu_m
    f=0.25*(math.log10(eps/3.7/D+5.74/Re**0.9))**(-2)
    a_p=8*f*rho_w*L_p/math.pi**2/D**5
    b_p=S_kt*math.pi*D**2*L_p/4
    P_loss=a_p*Q**2+b_p*C_v/Q+(rho_m-rho_w)*g*h_init
    Aux=Q
    Q=Q+dt*(math.pi*D**2/4/rho_m/L_p*(Net_Power/Q-P_loss))
    error=np.abs(Aux-Q)
print('The predicted flowrate is: {:.3f} l/s'.format(Q[-1]*1000))
h_predict=Net_Power/Q[-1]/rho_m/g
print('The predicted head at the slurry pump discharge is: {:.2f} m'.format(h_predict))
if (freq<=60) & (freq>55):
    HEAD=(HEAD60.predict([Q])-HEAD55.predict([Q]))/5*(freq-55)+HEAD55.predict([Q])
elif (freq<=55) & (freq>50):
    HEAD=(HEAD55.predict([Q])-HEAD50.predict([Q]))/5*(freq-50)+HEAD50.predict([Q])
elif (freq<=50) & (freq>45):
    HEAD=(HEAD50.predict([Q])-HEAD45.predict([Q]))/5*(freq-45)+HEAD45.predict([Q])
elif (freq<=45) & (freq>=40):
    HEAD=(HEAD45.predict([Q])-HEAD40.predict([Q]))/5*(freq-40)+HEAD40.predict([Q])
print('The discharge head based on pump curve is: {} m'.format(HEAD))