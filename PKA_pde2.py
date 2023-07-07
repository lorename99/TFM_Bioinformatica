#*******************************************
#
# PKA_pde2_Model.py
#
# Script for obteining dynamics of proteins 
# involved in Ras/cAMP/PKA in BY4742 pde2 strain
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


def Ageing (states, t):

    Cex, Cin, Cdc25a, Ira1a, Ira2a, Ras1a, Ras2a, Gpr1a, Rgs2a,\
    Gpa2a, Cyr1a, cAMP, PKAa, Msn2a, Msn4a, mRNA_Ams1, mRNA_Adh1 = states
    
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


    #Parameters Ras/cAMP/PKA pathway
    ATP = 5000000
    K_Cdc25i = 1
    K_Cdc25a = 1
    K_Ira1a = 1
    K_Ira1i = 1
    K_Ira2a = 1
    K_Ira2i = 1
    Kcat_Ras1i = 1
    Kcat_Ras2i = 1
    Kcat_Ras1a = 1
    Kcat_Ras2a = 1
    Km_Ras1i = 1
    Km_Ras1a = 1
    Km_Ras2i = 1
    Km_Ras2a = 1
    K_Gpr1a = 1
    K_Gpr1i = 1
    K_Rgs2a = 1
    K_Rgs2i = 1
    Kcat_Gpa2i = 1
    Kcat_Gpa2a = 1
    Km_Gpa2i = 1
    Km_Gpa2a = 1
    Kcat_Cyr1i_Ras1a = 1
    Kcat_Cyr1i_Ras2a = 1
    Kcat_Cyr1i_Gpa2a = 1
    Kdeg_Cyr1a = 1
    Kcat_cAMP = 1
    Km_cAMP = 1
    Kcat_Pde1 = 1
    Km_cAMP_Pde1a = 100
    K_PKAa_cAMP = 1
    Pde1a = 0.113700654876763
    Kdeg_PKAa = 1
    K_Msn2i = 1
    Kcat_Msn2a_PKA = 1
    Km_Msn2_PKA = 1
    Kcat_Msn4a_PKA = 1
    Km_Msn4_PKA = 1
    K_Msn4i = 1
    Vmax_Msn2a = 1
    Kh_Msn2a = 1
    Kdeg_mRNAAms1 = 1
    Vmax_Msn4a = 1
    Kh_Msn4a = 1
    Kdeg_mRNAAdh1 = 1

    #Total concentration of protein    
    Cdc25_total = 0.119200512588227
    Ira1_total = 0.083899609968775
    Ira2_total = 0.08535250778738
    Ras1_total = 0.156305319419943
    Ras2_total = 0.828395092419381
    Gpr1_total = 0.063926481943688
    Rgs2_total = 0.078563730739029
    Gpa2_total = 0.188148828759568
    Cyr1_total = 0.17961533478082
    PKA_total = 0.141729684926687    
    Msn2_total = 0.128658027715119
    Msn4_total = 0.184674607272751

    
    #Expressions for inactive state of the proteins
    Cdc25i = Cdc25_total - Cdc25a
    Ira1i = Ira1_total - Ira1a
    Ira2i = Ira2_total - Ira2a
    Ras1i = Ras1_total - Ras1a
    Ras2i = Ras2_total - Ras2a
    Gpr1i = Gpr1_total - Gpr1a
    Rgs2i = Rgs2_total - Rgs2a
    Gpa2i = Gpa2_total - Gpa2a
    Cyr1i = Cyr1_total - Cyr1a
    PKAi = PKA_total - PKAa
    Msn2i = Msn2_total - Msn2a
    Msn4i = Msn4_total - Msn4a
    

    #Equations
    dCex = -Vmax_Hxk1*Cex/(Km_Hxk1+Cex) - Vmax_Hxk2*Cex/(Km_Hxk2+Cex) - Vmax_Hxk3*Cex/(Km_Hxk3+Cex) - Vmax_Hxk4*Cex/(Km_Hxk4+Cex) - Vmax_Hxk5*Cex/(Km_Hxk5+Cex) - Vmax_Hxk6*Cex/(Km_Hxk6+Cex)
    dCin = -dCex - Vmax_met*Cin/(Km_met+Cin)
    dCdc25a = K_Cdc25i*Cin*Cdc25i - K_Cdc25a*Cdc25a
    dIra1a = K_Ira1i*Ira1i - K_Ira1a*Cin*Ira1a
    dIra2a = K_Ira2i*Ira2i - K_Ira2a*Cin*Ira2a
    dRas1a = Kcat_Ras1i*Ras1i*Cdc25a/(Km_Ras1i+Ras1i) - Kcat_Ras1a*Ras1a*Ira1a/(Km_Ras1a+Ras1a) - Kcat_Ras1a*Ras1a*Ira2a/(Km_Ras1a+Ras1a)
    dRas2a = Kcat_Ras2i*Ras2i*Cdc25a/(Km_Ras2i+Ras2i) - Kcat_Ras2a*Ras2a*Ira1a/(Km_Ras2a+Ras2a) - Kcat_Ras2a*Ras2a*Ira2a/(Km_Ras2a+Ras2a)
    dGpr1a = K_Gpr1i*Cex*Gpr1i - K_Gpr1a*Gpr1a
    dRgs2a = K_Rgs2i*Rgs2i - K_Rgs2a*Cin*Rgs2a
    dGpa2a = Kcat_Gpa2i*Gpa2i*Gpr1a/(Km_Gpa2i+Gpa2i) - Kcat_Gpa2a*Gpa2a*Rgs2a/(Km_Gpa2a+Gpa2a)
    dCyr1a = Kcat_Cyr1i_Ras1a*Cyr1i*Ras1a + Kcat_Cyr1i_Ras2a*Cyr1i*Ras2a + Kcat_Cyr1i_Gpa2a*Cyr1i*Gpa2a - Kdeg_Cyr1a*Cyr1a
    dcAMP = Kcat_cAMP*ATP*Cyr1a/(Km_cAMP+ATP) - Kcat_Pde1*cAMP*Pde1a/(Km_cAMP_Pde1a+cAMP)  - 4*K_PKAa_cAMP*PKAi*cAMP**4
    dPKAa =  K_PKAa_cAMP*PKAi*cAMP**4 - Kdeg_PKAa*PKAa
    dMsn2a = K_Msn2i*Msn2i - Kcat_Msn2a_PKA*Msn2a*PKAa/(Km_Msn2_PKA+Msn2a)
    dMsn4a = K_Msn4i*Msn4i - Kcat_Msn4a_PKA*Msn4a*PKAa/(Km_Msn4_PKA+Msn4a)
    dmRNA_Ams1 = Vmax_Msn2a + Msn2a**2/Kh_Msn2a*Msn2a - Kdeg_mRNAAms1*mRNA_Ams1
    dmRNA_Adh1 = Vmax_Msn4a + Msn4a**2/Kh_Msn4a*Msn4a - Kdeg_mRNAAdh1*mRNA_Adh1
    
    return dCex, dCin, dCdc25a, dIra1a, dIra2a, dRas1a, dRas2a, dGpr1a, dRgs2a,\
           dGpa2a, dCyr1a, dcAMP, dPKAa, dMsn2a, dMsn4a, dmRNA_Ams1, dmRNA_Adh1
        
    
    
def main():
    
    #Initial conditions
    Cex = 0.011
    Cin = 0
    Ras1a = 0
    Ras2a = 0
    Gpa2a = 0.188148828759568
    Cyr1a = 0
    cAMP = 0
    PKAa = 0
    Msn2a =  0
    Msn4a = 0
    Cdc25a = 0
    Ira1a = 0.083899609968775
    Ira2a = 0.08535250778738
    Gpr1a = 0.063926481943688
    Rgs2a = 0
    mRNA_Ams1 = 0
    mRNA_Adh1 =  0

    IC = [Cex, Cin, Cdc25a, Ira1a, Ira2a, Ras1a, Ras2a, Gpr1a, Rgs2a,\
           Gpa2a, Cyr1a, cAMP, PKAa, Msn2a, Msn4a, mRNA_Ams1, mRNA_Adh1]
    

    t = np.linspace(0,100,150000)
    X = odeint(Ageing,IC,t)
    
    
    plt.plot(t, X[:,0], color = 'crimson')
    plt.xlabel('Time (hours)')
    plt.ylabel('[External carbon] (nM)]')
    plt.savefig('Cex.png')
    plt.clf()

    plt.plot(t, X[:,1], color = 'crimson')
    plt.xlabel('Time (hours)')
    plt.ylabel('[Internal carbon] (nM)')
    plt.savefig('InternalCarbon.png')
    plt.clf()

    plt.plot(t, X[:,2],color = 'crimson')
    plt.xlabel('Time (hours)')
    plt.ylabel('[Cdc25 active] (nM)')
    plt.savefig('Cdc25a.png')
    plt.clf()

    plt.plot(t, X[:,3], color = 'crimson')
    plt.xlabel('Time (hours)')
    plt.ylabel('[Ira1 active] (nM)')
    plt.savefig('Ira1a.png')
    plt.clf()

    plt.plot(t, X[:,4], color = 'crimson')
    plt.xlabel('Time (hours)')
    plt.ylabel('[Ira2 active] (nM)')
    plt.savefig('Ira2a.png')
    plt.clf()

    plt.plot(t, X[:,5], color = 'crimson')
    plt.xlabel('Time')
    plt.ylabel('Concentration of active Ras1')
    plt.savefig('Ras1a.png')
    plt.clf()

    plt.plot(t, X[:,6], color = 'crimson')
    plt.xlabel('Time (hours)')
    plt.ylabel('[Ras2 active] (nM)')
    plt.savefig('Ras2a.png')
    plt.clf()

    plt.plot(t, X[:,7], color = 'crimson')
    plt.xlabel('Time (hours)')
    plt.ylabel('[Gpr1 active] (nM)')
    plt.savefig('Gpr1a.png')
    plt.clf()

    plt.plot(t, X[:,8], color = 'crimson')
    plt.xlabel('Time (hours)')
    plt.ylabel('[Rgs2 active] (nM)')
    plt.savefig('Rgs2a.png')
    plt.clf()

    plt.plot(t, X[:,9], color = 'crimson')
    plt.xlabel('Time (hours)')
    plt.ylabel('[Gpa2 active] (nM)')
    plt.savefig('Gpa2a.png')
    plt.clf()

    plt.plot(t, X[:,10], color = 'crimson')
    plt.xlabel('Time (hours)')
    plt.ylabel('[Cyr1 active] (nM)')
    plt.savefig('Cyr1a.png')
    plt.clf()

    plt.plot(t, X[:,11], color = 'crimson')
    plt.xlabel('Time (hours)')
    plt.ylabel('[cAMP]')
    plt.savefig('cAMP_pde2.png')
    plt.clf()

    plt.plot(t, X[:,12], color = 'crimson')
    plt.xlabel('Time (hours)')
    plt.ylabel('[PKA active] (nM)')
    plt.savefig('PKA_pde2.png')
    plt.clf()

    plt.plot(t, X[:,13], color = 'crimson')
    plt.xlabel('Time (hours)')
    plt.ylabel('[Msn2 active] (nM)')
    plt.savefig('Msn2a_pde2.png')
    plt.clf()

    plt.plot(t, X[:,14], color = 'crimson')
    plt.xlabel('Time (hours)')
    plt.ylabel('[Msn4 active] (nM)')
    plt.savefig('Msn4a_pde2.png')
    plt.clf()

    plt.plot(t, X[:,15], color = 'crimson')
    plt.xlabel('Time (hours)')
    plt.ylabel('[Ams1 mRNA] (nM)')
    plt.savefig('mRNA_Ams1_pde2.png')
    plt.clf()

    plt.plot(t, X[:,16], color = 'crimson')
    plt.xlabel('Time (hours)')
    plt.ylabel('[Adh1 mRNA] (nM)')
    plt.savefig('mRNA_Adh1_pde2.png')
    plt.clf()
    
if __name__ == '__main__':
    main()

