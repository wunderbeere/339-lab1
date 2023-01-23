# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 14:40:00 2023

@author: Bhuvan
"""

import numpy as np
import matplotlib.pyplot as plt
import random as r
from math import log10


'Gaussian Random Generator'



Nsamp = 10

values = np.zeros(int(Nsamp))
trials = range(int(Nsamp))

meangausst = []
stdgausst = []
#Total arrays for plotting

for i in trials:
    values[i] = r.gauss(0,1) # value between 0 and 1

#Finding mean and std
meangauss = np.mean(values)
meangausst.append(meangauss)
stdgauss = np.std(values)
stdgausst.append(stdgauss)


Nsampv = 25

valuesv = np.zeros(int(Nsampv))
trialsv = range(int(Nsampv))

for i in trialsv:
    valuesv[i] = r.gauss(0,1) # value between 0 and 1

meangaussv = np.mean(valuesv)
meangausst.append(meangaussv)
stdgaussv = np.std(valuesv)
stdgausst.append(stdgaussv)


Nsampx = 75

valuesx = np.zeros(int(Nsampx))
trialsx = range(int(Nsampx))

for i in trialsx:
    valuesx[i] = r.gauss(0,1) # value between 0 and 1

meangaussx = np.mean(valuesx)
meangausst.append(meangaussx)
stdgaussx = np.std(valuesx)
stdgausst.append(stdgaussx)


Nsamp2 = 200

values2 = np.zeros(int(Nsamp2))
trials2 = range(int(Nsamp2))

for i in trials2:
    values2[i] = r.gauss(0,1) # value between 0 and 1
    
meangauss2 = np.mean(values2)
meangausst.append(meangauss2)
stdgauss2 = np.std(values2)
stdgausst.append(stdgauss2)


Nsampt = 500

valuest = np.zeros(int(Nsampt))
trialst = range(int(Nsampt))


for i in trialst:
    valuest[i] = r.gauss(0,1) # value between 0 and 1

meangaussl = np.mean(valuest)
meangausst.append(meangaussl)
stdgaussl = np.std(valuest)
stdgausst.append(stdgaussl)


Nsamp3 = 1e3

values3 = np.zeros(int(Nsamp3))
trials3 = range(int(Nsamp3))

for i in trials3:
    values3[i] = r.gauss(0,1) # value between 0 and 1
    
meangauss3 = np.mean(values3)
meangausst.append(meangauss3)
stdgauss3 = np.std(values3)
stdgausst.append(stdgauss3)


Nsamp4 = 1e4

values4 = np.zeros(int(Nsamp4))
trials4 = range(int(Nsamp4))

for i in trials4:
    values4[i] = r.gauss(0,1) # value between 0 and 1
    
meangauss4 = np.mean(values4)
meangausst.append(meangauss4)
stdgauss4 = np.std(values4)
stdgausst.append(stdgauss4)


Nsamp5 = 1e5

values5 = np.zeros(int(Nsamp5))
trials5 = range(int(Nsamp5))


for i in trials5:
    values5[i] = r.gauss(0,1) # value between 0 and 1

meangauss5 = np.mean(values5)
meangausst.append(meangauss5)
stdgauss5 = np.std(values5)
stdgausst.append(stdgauss5)


Nsampo = 5e5

valueso = np.zeros(int(Nsampo))
trialso = range(int(Nsampo))


for i in trialso:
    valueso[i] = r.gauss(0,1) # value between 0 and 1

meangausso = np.mean(valueso)
meangausst.append(meangausso)
stdgausso = np.std(valueso)
stdgausst.append(stdgausso)


Nsamp6 = 1e6

values6 = np.zeros(int(Nsamp6))
trials6 = range(int(Nsamp6))


for i in trials6:
    values6[i] = r.gauss(0,1) # value between 0 and 1

meangauss6 = np.mean(values6)
meangausst.append(meangauss6)
stdgauss6 = np.std(values6)
stdgausst.append(stdgauss6)


Nsamp0 = 5e6

values0 = np.zeros(int(Nsamp0))
trials0 = range(int(Nsamp0))


for i in trials0:
    values0[i] = r.gauss(0,1) # value between 0 and 1

meangauss0 = np.mean(values0)
meangausst.append(meangauss0)
stdgauss0 = np.std(values0)
stdgausst.append(stdgauss0)
print(meangausst)
print(stdgausst)


#Plotting mean
x = [10, 25, 75, 200, 500, 1e3, 1e4, 1e5, 5e5, 1e6, 5e6]
ln_N = [log10(i) for i in x]
plt.figure(figsize=(10, 7))
plt.title("Mean of Gaussian Distribution For Varying Sample Sizes")
plt.xlabel("$Log_{10}$ of Sample Size N")
plt.ylabel("Mean of Gaussian Number Distribution")
plt.plot(ln_N, meangausst, 'o')
plt.rcParams.update({'font.size': 16})
plt.show()

#Plotting std
plt.figure(figsize=(10, 7))
plt.title("STD of Gaussian Distribution For Varying Sample Sizes")
plt.xlabel("$Log_{10}$ of Sample Size N")
plt.ylabel("STD of Gaussian Number Distribution")
plt.plot(ln_N, stdgausst, 'o')
plt.rcParams.update({'font.size': 16})
plt.show()



'Linear Random Number Generator'


#Defining Linear CDF and its inverse
A = 2
def CDF(x):
    return 1/2 * A * x**2

def inverseCDF(r):
    return np.sqrt(2*r/A)



N = 10

val = np.zeros(int(N))
tri = range(int(N))

meanlt = []
stdlt = []
#Total arrays for plotting

for i in tri:
    ro = r.random() # value between 0 and 1
    val[i] = inverseCDF(ro) # store randomly generated x's
    
    
#Finding mean and std
meanl = np.mean(val)
meanlt.append(meanl)
stdl = np.std(val)
stdlt.append(stdl)


Na = 25

vala = np.zeros(int(Na))
tria = range(int(Na))


for i in tria:
    roa = r.random() # value between 0 and 1
    vala[i] = inverseCDF(roa) # store randomly generated x's
    
meanla = np.mean(vala)
meanlt.append(meanla)
stdla = np.std(vala)
stdlt.append(stdla)


N2 = 75

val2 = np.zeros(int(N2))
tri2 = range(int(N2))


for i in tri2:
    ro2 = r.random() # value between 0 and 1
    val2[i] = inverseCDF(ro2) # store randomly generated x's
    
meanl2 = np.mean(val2)
meanlt.append(meanl2)
stdl2 = np.std(val2)
stdlt.append(stdl2)


Nb = 200

valb = np.zeros(int(Nb))
trib = range(int(Nb))


for i in trib:
    rob = r.random() # value between 0 and 1
    valb[i] = inverseCDF(rob) # store randomly generated x's
    
meanlb = np.mean(valb)
meanlt.append(meanlb)
stdlb = np.std(valb)
stdlt.append(stdlb)


Nc = 500

valc = np.zeros(int(Nc))
tric = range(int(Nc))


for i in tric:
    roc = r.random() # value between 0 and 1
    valc[i] = inverseCDF(roc) # store randomly generated x's
    
meanlc = np.mean(valc)
meanlt.append(meanlc)
stdlc = np.std(valc)
stdlt.append(stdlc)


Nd = 1e3

vald = np.zeros(int(Nd))
trid = range(int(Nd))


for i in trid:
    rod = r.random() # value between 0 and 1
    vald[i] = inverseCDF(rod) # store randomly generated x's
    
meanld = np.mean(vald)
meanlt.append(meanld)
stdld = np.std(vald)
stdlt.append(stdld)


Ne = 1e4

vale = np.zeros(int(Ne))
trie = range(int(Ne))


for i in trie:
    roe = r.random() # value between 0 and 1
    vale[i] = inverseCDF(roe) # store randomly generated x's
    
meanle = np.mean(vale)
meanlt.append(meanle)
stdle = np.std(vale)
stdlt.append(stdle)


Nf = 1e5

valf = np.zeros(int(Nf))
trif = range(int(Nf))


for i in trif:
    rof = r.random() # value between 0 and 1
    valf[i] = inverseCDF(rof) # store randomly generated x's
    
meanlf = np.mean(valf)
meanlt.append(meanlf)
stdlf = np.std(valf)
stdlt.append(stdlf)


Ng = 5e5

valg = np.zeros(int(Ng))
trig = range(int(Ng))


for i in trig:
    rog = r.random() # value between 0 and 1
    valg[i] = inverseCDF(rog) # store randomly generated x's
    
meanlg = np.mean(valg)
meanlt.append(meanlg)
stdlg = np.std(valg)
stdlt.append(stdlg)


Nh = 1e6

valh = np.zeros(int(Nh))
trih = range(int(Nh))


for i in trih:
    roh = r.random() # value between 0 and 1
    valh[i] = inverseCDF(roh) # store randomly generated x's
    
meanlh = np.mean(valh)
meanlt.append(meanlh)
stdlh = np.std(valh)
stdlt.append(stdlh)


Ni = 5e6

vali = np.zeros(int(Ni))
trii = range(int(Ni))


for i in trii:
    roi = r.random() # value between 0 and 1
    vali[i] = inverseCDF(roi) # store randomly generated x's
    
meanli = np.mean(vali)
meanlt.append(meanli)
stdli = np.std(vali)
stdlt.append(stdli)


#Plotting mean
x2 = [10, 25, 75, 200, 500, 1e3, 1e4, 1e5, 5e5, 1e6, 5e6]
ln_N = [log10(i) for i in x]
plt.figure(figsize=(10, 7))
plt.title("Mean of Linear Distribution For Varying Sample Sizes")
plt.xlabel("$Log_{10}$ of Sample Size N")
plt.ylabel("Mean of Linear Number Distribution")
plt.plot(ln_N, meanlt, 'o')
plt.rcParams.update({'font.size': 16})
plt.show() 

#Plotting std
plt.figure(figsize=(10, 7))
plt.title("STD of Linear Distribution For Varying Sample Sizes")
plt.xlabel("$Log_{10}$ of Sample Size N")
plt.ylabel("STD of Linear Number Distribution")
plt.plot(ln_N, stdlt, 'o')
plt.rcParams.update({'font.size': 16})
plt.show() 


'Uniform Distribution'



Nu = 10

valu = np.zeros(int(Nu))
triu = range(int(Nu))

meanut = [] 
stdut = []
#Total arrays to plot

for i in triu:
    valu[i] = r.random() # value between 0 and 1
   
#Finding mean and std
meanu = np.mean(valu)
meanut.append(meanu)
stdu = np.std(valu)
stdut.append(stdu)


Nu1 = 25

valu1 = np.zeros(int(Nu1))
triu1 = range(int(Nu1))

for i in triu1:
    valu1[i] = r.random() # value between 0 and 1
   

meanu1 = np.mean(valu1)
meanut.append(meanu1)
stdu1 = np.std(valu1)
stdut.append(stdu1)


Nu2 = 75

valu2 = np.zeros(int(Nu2))
triu2 = range(int(Nu2))

for i in triu2:
    valu2[i] = r.random() # value between 0 and 1
   

meanu2 = np.mean(valu2)
meanut.append(meanu2)
stdu2 = np.std(valu2)
stdut.append(stdu2)


Nu3 = 200

valu3 = np.zeros(int(Nu3))
triu3 = range(int(Nu3))

for i in triu3:
    valu3[i] = r.random() # value between 0 and 1
   

meanu3 = np.mean(valu3)
meanut.append(meanu3)
stdu3 = np.std(valu3)
stdut.append(stdu3)


Nu4 = 500

valu4 = np.zeros(int(Nu4))
triu4 = range(int(Nu4))

for i in triu4:
    valu4[i] = r.random() # value between 0 and 1
   

meanu4 = np.mean(valu4)
meanut.append(meanu4)
stdu4 = np.std(valu4)
stdut.append(stdu4)


Nu5 = 1e3

valu5 = np.zeros(int(Nu5))
triu5 = range(int(Nu5))

for i in triu5:
    valu5[i] = r.random() # value between 0 and 1
   

meanu5 = np.mean(valu5)
meanut.append(meanu5)
stdu5 = np.std(valu5)
stdut.append(stdu5)


Nux = 1e4

valux = np.zeros(int(Nux))
triux = range(int(Nux))

for i in triux:
    valux[i] = r.random() # value between 0 and 1
   

meanux = np.mean(valux)
meanut.append(meanux)
stdux = np.std(valux)
stdut.append(stdux)


Nup = 1e5

valup = np.zeros(int(Nup))
triup = range(int(Nup))

for i in triup:
    valup[i] = r.random() # value between 0 and 1
   

meanup = np.mean(valup)
meanut.append(meanup)
stdup = np.std(valup)
stdut.append(stdup)


Nus = 5e5

valus = np.zeros(int(Nus))
trius = range(int(Nus))

for i in trius:
    valus[i] = r.random() # value between 0 and 1
   

meanus = np.mean(valus)
meanut.append(meanus)
stdus = np.std(valus)
stdut.append(stdus)


Nuq = 1e6

valuq = np.zeros(int(Nuq))
triuq = range(int(Nuq))

for i in triuq:
    valuq[i] = r.random() # value between 0 and 1
   

meanuq = np.mean(valuq)
meanut.append(meanuq)
stduq = np.std(valuq)
stdut.append(stduq)


Nuz = 5e6

valuz = np.zeros(int(Nuz))
triuz = range(int(Nuz))

for i in triuz:
    valuz[i] = r.random() # value between 0 and 1
   

meanuz = np.mean(valuz)
meanut.append(meanuz)
stduz = np.std(valuz)
stdut.append(stduz)


#Plotting mean
x2 = [10, 25, 75, 200, 500, 1e3, 1e4, 1e5, 5e5, 1e6, 5e6]
ln_N = [log10(i) for i in x]
plt.figure(figsize=(10, 7))
plt.title("Mean of Uniform Distribution For Varying Sample Sizes")
plt.xlabel("$Log_{10}$ of Sample Size N")
plt.ylabel("Mean of Uniform Number Distribution")
plt.plot(ln_N, meanut, 'o')
plt.rcParams.update({'font.size': 16})
plt.show() 

#Plotting std
plt.figure(figsize=(10, 7))
plt.title("STD of Uniform Distribution For Varying Sample Sizes")
plt.xlabel("$Log_{10}$ of Sample Size N")
plt.ylabel("STD of Uniform Number Distribution")
plt.plot(ln_N, stdut, 'o')
plt.rcParams.update({'font.size': 16})
plt.show() 




    