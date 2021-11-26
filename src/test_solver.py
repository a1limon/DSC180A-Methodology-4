import pandas as pd
import numpy as np
import os
from mcmc_logreg import *

def main():
    accs = []
    for num_steps in range(200000, 400000, 20000):
        path = os.path.join("data", "heart.csv")
        heart_data = pd.read_csv(path, header = None, delimiter = " ").values
        X = heart_data[:,:-1]
        # X= np.hstack((np.ones((len(X),1)),X))  # add bias/offset term
        y = np.array([1 if i ==2 else 0 for i in heart_data[:,-1]]).reshape((-1,1))
        # beta mean priors
        beta_priors = np.repeat(0.0, X.shape[1]) 

        # beta standard deviation priors
        stddevs_priors = np.repeat(1, X.shape[1])
        # standard deviation of the proposal distribution
        stddevs_proposal_dist = np.repeat(0.1, X.shape[1])
        mcmc_log_mod = mcmc_log_reg()
        # create the untrimmed distribution of beta hats
        mcmc_log_mod.mh_mcmc(y, 
                                X,
                                beta_priors, 
                                stddevs_priors,
                                stddevs_proposal_dist, 
                                num_steps,
                                random_seed=42)
        mcmc_log_mod.beta_distr = mcmc_log_mod.raw_beta_distr
        mcmc_log_mod.fit('mean')
        pred = mcmc_log_mod.predict(X)
        pred = [1 if i > .5 else 0 for i in pred.flatten()]
        yflat = y.flatten()
        correct = pred == yflat
        acc = sum(correct) / len(correct)
        print(acc)
        accs.append(acc)
    return accs

if __name__ == "__main__":
    print(main())
    