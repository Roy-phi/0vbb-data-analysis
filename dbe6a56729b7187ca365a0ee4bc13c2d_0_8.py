##Necessary Imports
import numpy as np
import numpy.random as nprd
import matplotlib.pyplot as plt

##Specify x and y data
x = [1,2,3,4,5]
y = [3.83,8.75,10.98,14.18,17.22]

##Define sigma and range and nunmber of trial parameters for a and b
sigma = 0.71
a = np.linspace(-1.5,4.5,500)
b = np.linspace(2.25,4.25,500)

##Calculate chi-squared for trial parameters
chisq = np.zeros((len(a),len(b)))

for i in range(0,len(a)):
    for j in range(0,len(b)):
        for k in range(0,5):
            chisq[i,j] = chisq[i,j] + ((y[k]-(a[i]+b[j]*x[k]))/sigma)**2

##Calculate delchisq
chisqmin = np.amin(chisq)
delchisq = chisq - chisqmin

##Create and plot contours
cont = [2.3, 6.7, 11.8]
plt.contour(a,b,delchisq,cont)
plt.show()