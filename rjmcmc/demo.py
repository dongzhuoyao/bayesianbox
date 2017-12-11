import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
from scipy.stats import norm
from scipy.special import gamma

np.random.seed(14)

def generate_data(n):
	y = []
	for w in np.random.sample(n):
		val = 2*np.random.randn()-10 if w < 0.2 else np.random.randn()
		val = 2*np.random.randn()+10 if w > 0.6 else val
		y.append(val)
	return y

#================== PARAMETERS =====================#
N = 500
R = 30 # length of range of data
# mu = N(xi, kappa-1)
xi = 0	
kappa = 1./(R**2)
# beta = f(g,h), sigma = Gamma(alpha, beta)
alpha = 2
g = 0.2
h = 10/(R**2)
# w = Dirichlet([delta] * k)
delta = 1 
kmax = 20
#==================================================#

def allocation(k, W, MU, SIGMA):
	z = np.zeros(N)
	for i in range(N):
		z[i] = np.argmax([W[j]*norm.pdf(y[i], MU[j], np.sqrt(SIGMA[j])) for j in range(k)])
	return z

#================== INITIALIZATION =====================#
y = generate_data(N)
ki = 1
wi = np.array([1.0])
mui = np.array([0.0])
sigmai = np.array([1.0])
zi = allocation(ki, wi, mui, sigmai)
betai = np.random.gamma(g+ki*alpha, 1./(h + sum([1./sigmai[j] for j in range(ki)])))
init_state = {"k": ki, "W": wi, "MU": mui, "SIGMA": sigmai, "z": zi, "beta": betai}
#========================================================#

def sweep(state):
	k = state["k"]
	n = np.histogram(state["z"], bins=range(k+1))[0]
	#Step(a)
	w = np.random.dirichlet(n + delta,1)[0]

	#Step(b)
	mu = np.zeros(k)
	for j in range(k):
		i = np.where(state["z"]==j)[0]
		sigmainv = 1./(state["SIGMA"][j])
		muj = sigmainv*sum([y[index] + xi*kappa for index in i]) / (sigmainv*n[j] + kappa)
		sigmaj =  np.sqrt(1. / (sigmainv*n[j] + kappa))
		mu[j] = sigmaj * np.random.randn() + muj

	if not all(b >= a for a, b in zip(mu, mu[1:])):
		mu = state['MU']

	sigma = np.zeros(k)
	for j in range(k):
		i = np.where(state["z"]==j)[0]
		alphaj = alpha + 0.5*n[j]
		betaj = state["beta"] + 0.5*sum([(y[index] - mu[j])**2 for index in i])
		sigma[j] = 1./np.random.gamma(alphaj,1./betaj)
	
	#Step(c)
	z = allocation(k, w, mu, sigma)

	#Step(d)
	beta = np.random.gamma(g+k*alpha, 1./(h + sum([1./sigma[j] for j in range(k)])))

	# #Step(e)
	b = [1] + [0.5]*(kmax-2) + [0]
	d = [0] + [0.5]*(kmax-2) + [1]
	choice = np.random.rand()

	if (choice < 0.5 or k == 0) and k+1 != kmax: #split
		jstar = np.random.randint(k)
		u1, u2, u3 = np.random.beta(2,2), np.random.beta(2,2), np.random.beta(1,1)
		wj1, wj2 = w[jstar]*u1, w[jstar]*(1-u1)

		if(wj1 < wj2 and not any([wj1 <= weight <= wj2 for weight in w if weight != w[jstar]])):
			muj1 = mu[jstar] - u2*np.sqrt(sigma[jstar])*np.sqrt(wj2/wj1)
			muj2 = mu[jstar] + u2*np.sqrt(sigma[jstar])*np.sqrt(wj1/wj2)
			sigmaj1 = u3*(1-u2**2)*sigma[jstar]*w[jstar]/wj1
			sigmaj2 = (1-u3)*(1-u2**2)*sigma[jstar]*w[jstar]/wj2
			
			l = np.where(z==j)[0]
			l1, l2 = 0, 0
			likelihood, Palloc = 0.0, 1.0
			for index in l:
				prob1 = norm.pdf(y[index], muj1, np.sqrt(sigmaj1))
				prob2 = norm.pdf(y[index], muj2, np.sqrt(sigmaj2))
				probj = norm.pdf(y[index], mu[jstar], np.sqrt(sigma[jstar]))
				if prob1 > prob2:
					l1 += 1
					likelihood += np.log(prob1) - np.log(probj)
					Palloc *= (wj1*prob1)/(wj1*prob1 + wj2*prob2)
				else:
					l2 += 1
					likelihood += np.log(prob2) - np.log(probj)
					Palloc *= (wj2*prob2)/(wj1*prob1 + wj2*prob2)

			A = np.exp(likelihood) * (k+1) *(np.power(wj1, l1)*np.power(wj2, l2))/(np.power(w[jstar], l1+l2))
			A *= np.sqrt(kappa/(2*np.pi)) * np.exp(-0.5*kappa*((muj1-xi)**2 + (muj2-xi)**2 + (mu[jstar]-xi)**2))
			A *= np.power(beta, alpha)/gamma(alpha) * np.power((sigmaj1*sigmaj2/sigma[jstar]), -alpha-1)*np.exp(-beta*(1.0/sigmaj1 + 1.0/sigmaj2 - 1.0/sigma[jstar]))
			A *= float(d[k+1])/(float(b[k])*Palloc) * 1.0/(scipy.stats.beta.pdf(u1, 2, 2) * scipy.stats.beta.pdf(u2,2,2) * scipy.stats.beta.pdf(u3, 1,1))
			A *= (w[jstar]*np.abs(muj1-muj2)*sigmaj1*sigmaj2)/(u2*(1-u2**2)*u3*(1-u3)*sigma[jstar])

			if(np.random.rand() < min(1, A) and np.isfinite(A)):
				index = 0
				while(muj2 > mu[index] and index < k-1):
					index += 1
				k = k+1
				w = np.array(list(w[:index]) + [wj1, wj2] + list(w[index+1:]))
				mu = np.array(list(mu[:index]) + [muj1, muj2] + list(mu[index+1:]))
				sigma = np.array(list(sigma[:index]) + [sigmaj1, sigmaj2] + list(sigma[index+1:]))
				z = allocation(k, w, mu, sigma)
				beta = np.random.gamma(g+k*alpha, 1./(h + sum([1./sigma[j] for j in range(k)])))
	# elif choice >= 0.5 or k == kmax: #combine
	# 	# j1 = np.random.randint(k-1)
	# 	# j2 = j1 + 1
	# 	# wj = w[j1] + w[j2]
	# 	# muj = (w[j1]*mu[j1] + w[j2]*mu[j2])/wj
	# 	# sigmaj = (w[j1]*(mu[j1]**2+sigma[j1]**2) + w[j2]*(mu[j2]**2+sigma[j2]**2))/wj - muj**2
	# 	pass
		

	#DONE
	return {"k": k, "W": w, "MU": mu, "SIGMA": sigma, "z": z, "beta": beta,}

state = init_state
for i in range(200):
	new_state = sweep(state)
	print new_state["k"]
	state = new_state
	if state["k"] >= 2:
		count, bins, ignored = plt.hist(y, 100, normed = True , alpha=0.75, color='cyan')
		plt.plot(bins, sum([state["W"][j]*norm.pdf(bins, state["MU"][j], np.sqrt(state["SIGMA"][j])) for j in range(state["k"])]), linewidth=3, color="magenta")
		plt.show()

# sweep(init_state)
# state = init_state
# finalw = None
# finalmu = None
# finalsigma = None
# for i in range(50):
# 	print i
# 	new_state = sweep(state)
# 	if i == 25:
# 		finalw, finalmu, finalsigma = new_state["W"], new_state["MU"], new_state["SIGMA"]
# 	if i > 25:
# 		finalw += new_state["W"]
# 		finalmu += new_state["MU"]
# 		finalsigma += new_state["SIGMA"]
# 	state = new_state
# finalw /= 25
# finalmu /= 25
# finalsigma /= 25
# print finalw, finalmu, finalsigma
# count, bins, ignored = plt.hist(y, 100, normed = True , alpha=0.75, color='cyan')
# plt.plot(bins, sum([finalw[j]*norm.pdf(bins, finalmu[j], np.sqrt(finalsigma[j])) for j in range(state["k"])]), linewidth=3, color='magenta')
# plt.show()

