#*******************************************
#
# TORC1RTG_Model.py
#
# Script for obteining dynamics of proteins 
# involved in TORC1/Sch9 and RTG pathways
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

    Nin, Vam6a, SeaCa, Gtr1a, TORC1a, Gln3a, Gat1a,\
    mRNA_Can1, mRNA_Atg8, Mks1a, Rtg1a, Rtg3a,mRNA_Inp54, mRNA_Cit2, Nex = states


    #Parameters TORC1/RTG
    Kdeg_Nin = 0.05
    K_Vam6i = 1
    K_Vam6a = 1
    K_SeaC = 0.01
    Kdeg_SeaC = 1
    Kcat_Vam6_Gtr1 = 1
    Km_Vam6_Gtr1 = 1
    Kcat_SeaC_Gtr1 = 1
    Km_SeaC_Gtr1 = 1
    K_Gtr1_TorC1 = 1
    Kdeg_TorC1 = 1
    K_Gln3a = 0.01
    Kcat_Gln3i_TorC1 = 1
    Km_Gln3_TorC1 = 1
    K_Gat1a = 0.01
    Kcat_Gat1i_TorC1 = 1
    Km_Gat1a_TorC1 = 1
    Vmax_Gat1a = 1
    Kh_Gat1a = 1
    Kdeg_mRNACan1 = 1
    Vmax_Gln3a = 1
    Kh_Gln3a = 1
    Kdeg_Atg8 = 1
    Kcat_TORC1_Mks1i = 1
    Km_TORC1_Mks1 = 1
    Kdeg_Mks1a = 1
    Kcat_Rtg2_Rtg1i = 1
    Km_Rtg2_Rtg1i = 1
    Kcat_Mks1_Rtg1 = 1
    Km_Mks1_Rtg1 = 1
    Kcat_Rtg2_Rtg3i = 1
    Km_Rtg2_Rtg3i = 1
    Kcat_Mks1_Rtg3 = 1
    Km_Mks1_Rtg3 = 1
    Vmax_Rtg1a = 1
    Kh_Rtg1a = 1
    Kdeg_mRNAInp54 = 1
    Vmax_Rtg3a = 1
    Kh_Rtg3a = 1
    Kdeg_mRNACit2 = 1
    Rtg2 = 0.269620440792012
    KNex = 1

    #Total amount of proteins
    Vam6_total = 0.074552750684979
    SeaC_total = 0.11322816284590227
    Gtr1_total = 0.145221235121776
    TORC1_total = 0.073474195254765
    Gln3_total = 0.100095853812112
    Gat1_total = 0.168038652618799
    Mks1_total = 0.10626494466496
    Rtg1_total = 0.270170163307827
    Rtg3_total = 0.128346167425747

    

    #Expressions for inactive state of the proteins    
    Vam6i = Vam6_total - Vam6a
    SeaCi = SeaC_total - SeaCa
    Gtr1i =Gtr1_total -Gtr1a
    TORC1i = TORC1_total - TORC1a
    Gln3i = Gln3_total - Gln3a
    Gat1i = Gat1_total - Gat1a
    Mks1i = Mks1_total - Mks1a
    Rtg1i = Rtg1_total - Rtg1a
    Rtg3i = Rtg3_total - Rtg3a



    #Equations
    dNex = -KNex*Nex
    dNin =  - dNex - Kdeg_Nin*Nin
    dVam6a = K_Vam6i*Nin*Vam6i - K_Vam6a*Vam6a
    dSeaCa = K_SeaC*SeaCi - Kdeg_SeaC*Nin*SeaCa
    dGtr1a = Kcat_Vam6_Gtr1*Vam6a*Gtr1i/(Km_Vam6_Gtr1 + Gtr1i) - Kcat_SeaC_Gtr1*SeaCa*Gtr1a/(Km_SeaC_Gtr1 + Gtr1a)
    dTORC1a = K_Gtr1_TorC1*Gtr1a*TORC1i - Kdeg_TorC1*TORC1a
    dGln3a = K_Gln3a*Gln3i - Kcat_Gln3i_TorC1*TORC1a*Gln3a/(Km_Gln3_TorC1+Gln3a)
    dGat1a = K_Gat1a*Gat1i - Kcat_Gat1i_TorC1*TORC1a*Gat1a/(Km_Gat1a_TorC1+Gat1a)
    dmRNA_Can1 = Vmax_Gln3a*Gln3a**2/(Kh_Gln3a**2 + Gln3a**2) - Kdeg_mRNACan1*mRNA_Can1 #Gen controlado por Rtg1
    dmRNA_Atg8 = Vmax_Gat1a*Gat1a**2/(Kh_Gat1a**2 + Gat1a**2) - Kdeg_Atg8*mRNA_Atg8
    dMks1a = Kcat_TORC1_Mks1i * TORC1a * Mks1i /(Km_TORC1_Mks1 + Mks1i) - Kdeg_Mks1a * Mks1a
    dRtg1a = Kcat_Rtg2_Rtg1i * Rtg2 * Rtg1i / (Km_Rtg2_Rtg1i + Rtg1i) - Kcat_Mks1_Rtg1 * Mks1a * Rtg1a/ (Km_Mks1_Rtg1 + Rtg1a)
    dRtg3a = Kcat_Rtg2_Rtg3i * Rtg2 * Rtg3i / (Km_Rtg2_Rtg3i + Rtg3i) - Kcat_Mks1_Rtg3 * Mks1a * Rtg3a/ (Km_Mks1_Rtg3 + Rtg3a)
    dmRNA_Inp54 = ((Vmax_Rtg1a * ((Rtg1a)**2)) / (((Kh_Rtg1a)**2) + ((Rtg1a)**2))) - Kdeg_mRNAInp54 * mRNA_Inp54 
    dmRNA_Cit2 = ((Vmax_Rtg3a * ((Rtg3a)**2)) / (((Kh_Rtg3a)**2) + ((Rtg3a)**2))) - Kdeg_mRNACit2 * mRNA_Cit2 

    return dNin, dVam6a, dSeaCa, dGtr1a, dTORC1a, dGln3a, dGat1a, dmRNA_Can1, dmRNA_Atg8, dMks1a, dRtg1a, dRtg3a, dmRNA_Inp54, dmRNA_Cit2, dNex

    
def main():

    #Initial conditions
    Nex = 0.1
    Nin = 0
    Vam6a = 0
    SeaCa = 0.11322816284590227
    Gtr1a = 0
    TORC1a = 0
    Gat1a = 0.168038652618799
    Gln3a = 0.100095853812112
    mRNA_Can1 = 0
    mRNA_Atg8 = 0
    Mks1a = 0
    Rtg1a = 0.270170163307827
    Rtg3a = 0.128346167425747
    mRNA_Inp54 = 0
    mRNA_Cit2 = 0
   
    
    IC = [Nin, Vam6a, SeaCa, Gtr1a, TORC1a, Gln3a, Gat1a,\
         mRNA_Can1, mRNA_Atg8, Mks1a, Rtg1a, Rtg3a, mRNA_Inp54, mRNA_Cit2, Nex]
    
    t = np.linspace(0,1000,1500)
    X = odeint(Ageing, IC, t)
    
    
    plt.plot(t, X[:,0], color = 'mediumvioletred')
    plt.xlabel('Time (hours)')
    plt.ylabel('[Internal nitrogen] (nM)')
    plt.savefig('100InternalNitrogen_ModelJuneSplit.png')
    plt.clf()

    plt.plot(t, X[:,1], color = 'mediumvioletred')
    plt.xlabel('Time (hours)')
    plt.ylabel('[Vam6 active] (nM)')
    plt.savefig('100Vam6_ModelJuneSplit.png')
    plt.clf()

    plt.plot(t, X[:,2], color = 'mediumvioletred')
    plt.xlabel('Time (hours)')
    plt.ylabel('[SEACIT active] (nM)')
    plt.savefig('100SEA_ModelJuneSplit.png')
    plt.clf()

    plt.plot(t, X[:,3],color = 'mediumvioletred')
    plt.xlabel('Time (hours)')
    plt.ylabel('[Gtr1 active] (nM)')
    plt.savefig('100Gtr1_ModelJuneSplit.png')
    plt.clf()

    plt.plot(t, X[:,4], color = 'mediumvioletred')
    plt.xlabel('Time (hours)')
    plt.ylabel('[Tor1 Complex active] (nM)')
    plt.savefig('100TORC1_ModelJuneSplit.png')
    plt.clf()

    plt.plot(t, X[:,5], color = 'mediumvioletred')
    plt.xlabel('Time (hours)')
    plt.ylabel('[Gln3 active] (nM)')
    plt.savefig('100gLN3_ModelJuneSplit.png')

    plt.clf()

    plt.plot(t, X[:,6], color = 'mediumvioletred')
    plt.xlabel('Time (hours)')
    plt.ylabel('[Gat1 active] (nM)')
    plt.savefig('100Gat1_ModelJuneSplit.png')

    plt.clf()

    plt.plot(t, X[:,7], color = 'mediumvioletred')
    plt.xlabel('Time (hours)')
    plt.ylabel('[Can1 mRNA] (nM)')
    plt.savefig('100cAN1_ModelJuneSplit.png')
    plt.clf()
    
    plt.plot(t, X[:,8], color = 'mediumvioletred')
    plt.xlabel('Time (hours)')
    plt.ylabel('[Atg8 mRNA] (nM)')
    plt.savefig('100Atg8_ModelJuneSplit.png')
    plt.clf()
    

    plt.plot(t, X[:,9], color = 'mediumseagreen')
    plt.xlabel('Time (hours)')
    plt.ylabel('[Mks1 active] (nM)')
    plt.savefig('100Mks1_ModelJuneSplit.png')
    plt.clf()

    plt.plot(t, X[:,10], color = 'mediumseagreen')
    plt.xlabel('Time (hours)')
    plt.ylabel('[Rtg1 active] (nM)')
    plt.savefig('100Rtg1_ModelJuneSplit.png')
    plt.clf()

    plt.plot(t, X[:,11], color = 'mediumseagreen')
    plt.xlabel('Time (hours)')
    plt.ylabel('[Rtg3 active] (nM)')
    plt.savefig('100Rtg3_ModelJuneSplit.png')
    plt.clf()

    plt.plot(t, X[:,12], color = 'mediumseagreen')
    plt.xlabel('Time (hours)')
    plt.ylabel('[Inp54 mRNA] (nM)')
    plt.savefig('100dddddInp54_ModelJuneSplit.png')
    plt.clf()

    plt.plot(t, X[:,13], color = 'mediumseagreen')
    plt.xlabel('Time (hours)')
    plt.ylabel('[Cit2 mRNA] (nM)')
    plt.savefig('100sdddCit2_ModelJuneSplit.png')
    plt.clf()
    
if __name__ == '__main__':
    main()
