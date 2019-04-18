# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 11:58:13 2017

@author: rashedi
"""
import numpy as np
import math
import matplotlib.pyplot as plt

Length=5000;
dz=10;
W=[5000,10000,15000];
# W=24000;
quality=[];
for i in range(len(W)):
    w=W[i];
    t=100;
    g=32.2*3600**2; # ft/hr^2
    d_ti=2.441/12; 
    r_ti=d_ti/2; 
    A_ti=math.pi*r_ti**2;
    d_to=2.875/12; 
    r_to=d_to/2; 
    d_ins=2.95/12; 
    r_ins=d_ins/2;
    d_ci=6.276/12; 
    r_ci=d_ci/2;
    d_co=7/12; 
    r_co=d_co/2;
    d_h=9.625/12; 
    r_h=d_h/2;
    G=w/A_ti;
    eps_to=0.9;
    eps_ci=0.4;
    k_ins=0.04;
    k_cem=0.55;
    k_e=1.4;
    alpha=0.96;
    T_e=80;
    gamma=0.02;
    J=788;
    P_0=500;
    X_0=1;
    v_0=0.005;
    P=np.zeros(int(Length/dz))
    P[0]=P_0
    X=np.zeros(int(Length/dz))
    X[0]=X_0
    V=np.zeros(int(Length/dz))
    V[0]=v_0
    for z in range(int(Length/dz)-1):
        Z=(z+1)*dz;
        T_e=80+Z*gamma;
        T=115.1*P[z]**0.225;
        v_s=363.9*P[z]**-0.9588;
        v_w=0.01587+8.6e-5*P[z]**0.225+2e-4*P[z]**0.45;
        v=X[z]*v_s+(1-X[z])*v_w;
        V[z+1]=v;
        dvdX=v_s-v_w;
        dens=1/v;
        dens_s=1/v_s;
        q=w/dens;
        q_s=q*X[z]*dens/dens_s;
        gamma_a=w*q_s/(144*A_ti**2*g*P[z]);
        mu_s=1e-4*(82.2516+0.17815*T+6.59e-5*T**2)*2.4191;
        mu_w=2.185/(0.04012*T+5.1547e-6*T**2-1)*2.4191;
        mu=X[z]*mu_s+(1-X[z])*mu_w;
        H_w=91*P[z]**0.2574;
        L=1318*P[z]**(-0.08774);
        H_s=1119*P[z]**0.01267;
        H=H_w+X[z]*L;
        dHdX=H_s-H_w;
        Re=G*2*r_ti/mu;
        rough=0.005;
        epsd=rough/d_ti;
        lambd=epsd**1.1098/2.8257+(7.149/Re)**0.8981;
        f=(4*np.log(epsd/3.7065-5.0452/Re*np.log(lambd)))**(-2);
        vel=q/A_ti;
        vel_s=dens*v_s*X[z]*vel;
        T_s=(T-32)/1.8+273;
        B_s=235.8e-3;
        b_s=-0.625;
        mu_s=1.256;
        Tc_s=647.15;
        sigma=B_s*(1-T_s/Tc_s)**mu_s*(1+b_s*(1-T_s/Tc_s))*0.0685;
        N_w=dens*vel**2*d_ti/sigma;
        if N_w<=0.005:
            epsd=34*sigma*v_s/(vel_s**2*d_ti);
        else:
            epsd=174.8*sigma*N_w**0.302*v_s/(vel_s**2*d_ti);
        #f=(2*log10(0.27*epsd))^(-2)+0.268*epsd^1.73;
        tau_f=f*dens*vel**2/2/g/d_ti;
        dPdz=(dens-tau_f)/(1-gamma_a)/144;
        U=4;
        f_t=np.log(2*np.sqrt(alpha*t)/r_h)-0.29;
        target=0;
        flag=0;
        while target==0:
            T_h=(k_e*T_e/U/r_to+f_t*T)/(k_e/U/r_to+f_t);
            T_ci=T_h+(r_to*U*np.log(r_h/r_co))/k_cem*(T-T_h);
            F_otic=1/(1/eps_to+r_to/r_ci*(1/eps_ci-1));
            h_r=1.713e-9*F_otic*((T+460)**2+(T_ci+460)**2)*(T+460+T_ci+460);
            T_an=(T_ci+T)/2;
            mu_an=0.01827*(0.555*524.07+120)/(0.555*(T_an+460)+120)*((T_an+460)/524.07)**1.5*2.4191; # The last coeff. is centipoise to lb/(ft-hr)
            P_an=1e5;
            dens_an=P_an/287.058/((T_an-32)/1.8+273)/16.01846;
            c_an=0.25;
            k_ha=0.0265;
            beta=1/(T_an+460);
            Pr=c_an*mu_an/k_ha;
            Gr=(r_ci-r_to)**3*g*dens_an**2*beta*(T-T_ci)/(mu_an**2);
            k_hc=k_ha*0.049*(Gr*Pr)**0.333*Pr**0.074;
            h_c=k_hc/(r_to*np.log(r_ci/r_to));
            if flag==1:
                target=1
            U_to=1/r_to/(1/r_ins/(h_c+h_r)+1/k_cem*np.log(r_h/r_co)+np.log(r_ins/r_to)/k_ins);
            err=np.abs((U_to-U)/U)*100;
            U=U_to;
            if err<=0.1:
                flag=1
        Q_dot=2*math.pi*r_to*U*(T-T_h);
        c1=w*(1119*P[z]**0.01267-91*P[z]**0.2574);
        c2=w*(1119*0.01267*P[z]**(-0.98733)-91*0.2574*P[z]**(-0.7426))*dPdz;
        c3=Q_dot+w*(91*0.2574*P[z]**(-0.7426)*dPdz+w**2/(g*J*dens*A_ti**2)*(V[z+1]-V[z])/dz-1/J);
        dXdz=(-c3-c2*X[z])/c1;
        P[z+1]=P[z]+dPdz*dz;
        if X[z]>1:
            X[z]=1
        elif X[z]<0:
            X[z]=0
        else:
            X[z+1]=X[z]+dXdz*dz;
#        pause
    depth=np.linspace(0,Length,len(X))
    T=115.1*np.power(P,0.225);
    if i==1:
        X1=X*100
        T1=T
    elif i==2:
        X2=X*100
        T2=T
    else:
        X3=X*100
        T3=T
        
#f, axarr=plt.subplots(1,2)
#axarr[0].plot(depth,X1,'b',depth,X2,'--r',depth,X3,'-.g')
#axarr[1].plot(depth,T1,'b',depth,T2,'--r',depth,T3,'-.g')
f1=plt.figure(1)
plt.plot(depth,X1,'b',depth,X2,'--r',depth,X3,'-.g')
plt.gca().legend(('Qs=5000','Qs=10000','Qs=15000'))
plt.xlabel('Depth')
plt.ylabel('Steam quality (%)')
f1.savefig('SQ')
f2=plt.figure(2)
plt.plot(depth,T1,'b',depth,T2,'--r',depth,T3,'-.g')
plt.gca().legend(('Qs=5000','Qs=10000','Qs=15000'))
plt.xlabel('Depth')
plt.ylabel('Temperature (F)')
f2.savefig('T')