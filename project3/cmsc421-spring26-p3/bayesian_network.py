""" Bayesian networks """

from probability import BayesNet, enumeration_ask, elimination_ask, rejection_sampling, likelihood_weighting, gibbs_ask
from timeit import timeit, repeat
import pickle
import numpy as np

T, F = True, False

class DataPoint:
    """
    Represents a single datapoint gathered from one lap.
    Attributes are exactly the same as described in the project spec.
    """
    def __init__(self, muchfaster, early, overtake, crash, win):
        self.muchfaster = muchfaster
        self.early = early
        self.overtake = overtake
        self.crash = crash
        self.win = win

def generate_bayesnet():
    """
    Generates a BayesNet object representing the Bayesian network in Part 2
    returns the BayesNet object
    """
    bayes_net = BayesNet()
    # load the dataset, a list of DataPoint objects
    data = pickle.load(open("data/bn_data.p","rb"))
    # BEGIN_YOUR_CODE ######################################################

    # counting frequency of points 
    total_points = len(data) 
    mf_points = 0 
    early_points = 0

    for point in data: 
        if point.muchfaster: 
            mf_points += 1 
        if point.early: 
            early_points += 1
    
    # constructing root nodes, prior events 
    bayes_net.add(("MuchFaster", "", mf_points / total_points))
    bayes_net.add(("Early", "", early_points / total_points)) 


    def cond_prob(data, conditions, condition_vals, target): 
        # frequency of parent conditions in data 
        subset = [p for p in data if all(
            getattr(p, a) == v for a, v in zip(conditions, condition_vals)
        )]
        # frequency of target in the subset 
        target_count = sum(1 for p in subset if getattr(p, target))
        return target_count / len(subset)


    # P(O=T | MuchFaster, Early) 
    overtake_cpt = {
        (T, T): cond_prob(data, ["muchfaster", "early"], [T, T], "overtake"), 
        (T, F): cond_prob(data, ["muchfaster", "early"], [T, F], "overtake"),
        (F, T): cond_prob(data, ["muchfaster", "early"], [F, T], "overtake"), 
        (F, F): cond_prob(data, ["muchfaster", "early"], [F, F], "overtake")  
    }
    bayes_net.add(("Overtake", "MuchFaster Early", overtake_cpt)) 

    # P(Crash=T | Overtake, Early) 
    crash_cpt = {
        (T, T): cond_prob(data, ["overtake", "early"], [T, T], "crash"), 
        (T, F): cond_prob(data, ["overtake", "early"], [T, F], "crash"),
        (F, T): cond_prob(data, ["overtake", "early"], [F, T], "crash"), 
        (F, F): cond_prob(data, ["overtake", "early"], [F, F], "crash")
    }
    bayes_net.add(("Crash", "Overtake Early", crash_cpt)) 

    # P(Win=T | Overtake, Crash) 
    win_cpt = {
        (T, T): cond_prob(data, ["overtake", "crash"], [T, T], "win"), 
        (T, F): cond_prob(data, ["overtake", "crash"], [T, F], "win"),
        (F, T): cond_prob(data, ["overtake", "crash"], [F, T], "win"), 
        (F, F): cond_prob(data, ["overtake", "crash"], [F, F], "win") 
    }
    bayes_net.add(("Win", "Overtake Crash", win_cpt))  
    
    # END_YOUR_CODE ########################################################
    return bayes_net

def find_best_overtake_condition(bayes_net):
    """
    Finds the optimal condition for overtaking the car, as described in Part 3
    Returns the optimal values for (MuchFaster,Early)
    """
    # BEGIN_YOUR_CODE ######################################################
    
    # P(muchfaster | win=T, crash=F)
    prob_mf = enumeration_ask('MuchFaster', dict(Win=T, Crash=F), bayes_net)
    optimal_mf_val = T if prob_mf[T] > prob_mf[F] else F 

    # P(early | win=T, crash=F) 
    prob_early = enumeration_ask('Early', dict(Win=T, Crash=F), bayes_net)
    optimal_early_val = T if prob_early[T] > prob_early[F] else F

    return (optimal_mf_val, optimal_early_val)  
    
    # END_YOUR_CODE ########################################################

def main():
    bayes_net = generate_bayesnet()
    cond = find_best_overtake_condition(bayes_net)
    print("Best overtaking condition: MuchFaster={}, Early={}".format(cond[0],cond[1]))

if __name__ == "__main__":
    main()

