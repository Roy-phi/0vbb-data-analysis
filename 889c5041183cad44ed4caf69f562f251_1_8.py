##Necessary Imports
import numpy as np
import numpy.random as nprd
import matplotlib.pyplot as plt

##Specify x and y data
x = [1,2,3,4,5]
y = [3.83,8.75,10.98,14.18,17.22]

##Define sigma
sigma = 0.71

##Specify length of MCMC chain
nmcmc = 100000

a = np.zeros(nmcmc)
b = np.zeros(nmcmc)
chisq = np.zeros(nmcmc)

##Specify initial values, width of proposal density, and counter to compute acceptance ratio
accept = 0
a[0] = 1.33
sig_a = 0.1
b[0] = 3.22
sig_b = 0.05

##Loop over the data points for chisquared calculation
for k in range(0,len(x)):
    chisq[0] = chisq[0] + ((y[k]-(a[0]+b[0]*x[k]))/sigma)**2

##Go over points in the chain
for i in range(1,nmcmc):
    print("Point "+str(i+1)+" of "+str(nmcmc), end="\r")

    ##Determine trial values
    a_trial = a[i-1] + sig_a*nprd.randn(1)
    b_trial = b[i-1] + sig_b*nprd.randn(1)

    ##Set initial value of chisq for trial point
    chisq_trial = 0

    ##Calculate the value of chisq for trial point
    for k in range(0,len(x)):
        chisq_trial = chisq_trial + ((y[k]-(a_trial+b_trial*x[k]))/sigma)**2

    ##Calculate trial value of log ratio
    log_Lratio = 0.5*(chisq[i-1]-chisq_trial)

    ##Decide whether or not to accept trial value
    ##If trial point is "uphill" from previous point, accept it
    if log_Lratio >= 0:
        a[i] = a_trial
        b[i] = b_trial
        chisq[i] = chisq_trial
        accept = accept + 1

    ##If it is not, calculate the actual likelihood ratio, then pick a random number and if bigger accept, if smaller set the point as the previous point in the Markov Chain
    else:
        ratio = np.exp(log_Lratio)
        test_uniform = nprd.rand(1)

        if test_uniform <= ratio:
            a[i] = a_trial
            b[i] = b_trial
            chisq[i] = chisq_trial
            accept = accept + 1

        else:
            a[i] = a[i-1]
            b[i] = b[i-1]
            chisq[i] = chisq[i-1]
            
accept_ratio = accept/nmcmc

##Plotting and information
plt.scatter(a,b,0.01)
plt.show()

bar = [np.mean(a), np.mean(b)]
err = [np.std(a), np.std(b)]

cvr = np.cov(a,b)
covari = cvr[0,1]

print("")
print("From MCMC")
print("a           b")
print(bar)
print(err)
print("Covariance")
print(covari)
print("Acceptance Ratio")
print(accept_ratio)
