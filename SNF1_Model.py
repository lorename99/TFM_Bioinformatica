#*******************************************
#
# SNF1_Model.py
#
# Script for obteining dynamics of proteins 
# involved in Snf1/AMPK pathway in BY4742 
# strain
#
# Trabajo Fin de Master
#
# Master en Bioinformatica
#
# Author: Lorena Martinez España
#
# Date: 6/7/2023
#
#*******************************************

import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import odeint

#ESTE ES EL CODIGO BUENOO

def Ageing (states, t):

    Cex, Cin, Elm1a, Sak1a, Tos3a, Snf1a, Mig1a, mRNA_Gal1, Glc7a  = states
    
    
    #Parameters Carbon
    Vmax_Hxk1 = 1.7
    Vmax_Hxk2 = 1.7
    Vmax_Hxk3 = 1.7
    Vmax_Hxk4 = 1.7
    Vmax_Hxk5 = 1.7
    Vmax_Hxk6 = 1.7
    Km_Hxk1 = 100
    Km_Hxk2 = 10
    Km_Hxk3 = 50
    Km_Hxk4 = 10
    Km_Hxk5 = 1.5
    Km_Hxk6 = 1.5
    Vmax_met = 1 
    Km_met = 1 

    #Parameters Snf1/AMPK pathway
    K_Sak1a = 10
    K_Sak1i = 0.01
    K_Elm1a = 10
    K_Elm1i = 0.01
    K_Tos3a = 10
    K_Tos3i = 0.01
    Kcat_Sak1 = 1
    Km_Sak1 = 10
    Kcat_Elm1 = 1
    Km_Elm1 = 10
    Kcat_Tos3 = 1
    Km_Tos3 = 10
    Kcat_Glc7 = 1
    Km_Glc7 = 1
    Kcat_Snf1i = 1
    Km_Snf1 = 1
    K_Mig1i = 1
    K_Glc7i = 0.01
    K_Glc7a = 10
    Vh_Mig1 = 1
    Kh_Mig1 = 1
    Kdeg_mRNAGal1 = 100

    #Total concentration of protein 
    Snf1_total = 0.312033017155115
    Sak1_total = 0.048091803595057
    Elm1_total = 0.036332110032318
    Tos3_total = 0.048091803595057
    Glc7_total = 1
    Mig1_total = 0.078932126716784

    #Expressions for inactive state of the proteins
    Sak1i = Sak1_total - Sak1a
    Elm1i = Elm1_total - Elm1a
    Tos3i = Tos3_total - Tos3a
    Snf1i = Snf1_total - Snf1a
    Mig1i = Mig1_total - Mig1a
    Glc7i = Glc7_total - Glc7a


    #Equations
    dCex = -Vmax_Hxk1*Cex/(Km_Hxk1+Cex) - Vmax_Hxk2*Cex/(Km_Hxk2+Cex) - Vmax_Hxk3*Cex/(Km_Hxk3+Cex) -(Vmax_Hxk4*Cex)/(Km_Hxk4+Cex) -(Vmax_Hxk5*Cex)/(Km_Hxk5+Cex) -(Vmax_Hxk6*Cex)/(Km_Hxk6+Cex)
    dCin = -dCex - Vmax_met*Cin/(Km_met + Cin)
    dSak1a = K_Sak1i*Sak1i - K_Sak1a*Cin*Sak1a
    dElm1a = K_Elm1i*Elm1i - K_Elm1a*Cin*Elm1a
    dTos3a = K_Tos3i*Tos3i - K_Tos3a*Cin*Tos3a  
    dGlc7a = K_Glc7i*Glc7i - K_Glc7a*Glc7a
    dSnf1a = Kcat_Sak1*Sak1a*Snf1i/(Km_Sak1 + Snf1i) + Kcat_Elm1*Elm1a*Snf1i/(Km_Elm1 + Snf1i) + Kcat_Tos3*Tos3a*Snf1i/(Km_Tos3 + Snf1i) - Kcat_Glc7*Glc7a*Snf1a/(Km_Glc7 + Snf1a)
    dMig1a = K_Mig1i*Mig1i - Kcat_Snf1i*Snf1a*Mig1a/(Km_Snf1 + Mig1a)
    dmRNA_Gal1 = Vh_Mig1/(Kh_Mig1**2 + Mig1a**2) - Kdeg_mRNAGal1*mRNA_Gal1
    

        
    return dCex, dCin, dElm1a, dSak1a, dTos3a, dSnf1a, dMig1a, dmRNA_Gal1, dGlc7a



def main():

    #Initial conditions
    Cex = 0.011
    Cin = 0
    Sak1a = 0.048091
    Elm1a = 0.036332
    Tos3a = 0.04809
    Snf1a = 0.3120
    Mig1a = 0
    Glc7a = 0
    mRNA_Gal1 = 0.000998
    

    IC = [Cex, Cin, Elm1a, Sak1a, Tos3a, Snf1a, Mig1a, mRNA_Gal1, Glc7a]
    t = np.linspace(0,200,15000)
    X = odeint(Ageing, IC, t)
   
    print(X)
    plt.plot(t, X[:,0], color = 'dodgerblue')
    plt.xlabel('Time (hours)')
    plt.ylabel('[External carbon] (nM)')
    plt.savefig('CarbonoExterno.png')
    plt.clf()
    
    plt.plot(t, X[:,1], color = 'dodgerblue')
    plt.xlabel('Time (hours)')
    plt.ylabel('[Internal carbon] (nM)')
    plt.savefig('CarbonoInterno.png')
    plt.clf()
    
    plt.plot(t, X[:,2], color = 'dodgerblue')
    plt.xlabel('Time (hours)')
    plt.ylabel('[Elm1 active] (nM)')
    plt.savefig('Elm1Activa.png')
    plt.clf()    

    plt.plot(t, X[:,3], color = 'dodgerblue')
    plt.xlabel('Time (hours)')
    plt.ylabel('[Sak11 active] (nM)')
    plt.savefig('Sak1Activa.png')
    plt.clf() 

    plt.plot(t, X[:,4], color = 'dodgerblue')
    plt.xlabel('Time (hours)')
    plt.ylabel('[Tos3 active] (nM)')
    plt.savefig('Tos3Activa.png')
    plt.clf() 

    plt.plot(t, X[:,5], color = 'dodgerblue')
    plt.xlabel('Time (hours)')
    plt.ylabel('[Snf1 active] (nM)')
    plt.savefig('Snf1Activa.png')
    plt.clf() 

    plt.plot(t, X[:,6], color = 'dodgerblue')
    plt.xlabel('Time (hours)')
    plt.ylabel('[Mig1 active] (nM)')
    plt.savefig('Mig1Activa.png')
    plt.clf()

    plt.plot(t, X[:,7], color = 'dodgerblue')
    plt.xlabel('Time (hours)')
    plt.ylabel('[Gal1 mRNA] (nM)')
    plt.savefig('mRNAGal1.png')
    plt.clf()

    plt.plot(t, X[:,8], color = 'dodgerblue')
    plt.xlabel('Time (hours)')
    plt.ylabel('[Glc7 active] (nM)')
    plt.savefig('Glc7.png')
    plt.clf()


if __name__ == '__main__':
    main()