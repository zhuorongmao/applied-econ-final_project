import numpy as np

import itertools
import pandas as pd

def GeneratePermutations(nOrders):

    IDs = ""

    for i in range(nOrders):

        IDs = IDs + str(i)

    res = np.array(list(itertools.permutations(IDs, nOrders - 1))).astype("int")

 

    return res


def generate_random_numbers(nOrders, prob_behavior):
    # Generate n random numbers that sum up to (1 - x)
    random_numbers = np.random.dirichlet(np.ones(nOrders-1))
    scaled_numbers = random_numbers * (1 - prob_behavior)
    scaled_numbers[-1] += prob_behavior
    return scaled_numbers

 

def RunSingleSimulation(nOrders, perm):

    assert(nOrders > 0)

    vec = np.random.rand(perm.shape[0])
    #print ('vec',vec)

    normedSamples = vec/ np.sum(vec)
    

    pMat = np.zeros([nOrders - 1, nOrders])
    #print('pMat',pMat)

    for i in range(nOrders - 1):

        for j in range(nOrders):

            pMat[i,j] = normedSamples[np.any(perm[:, :i+1]==j, axis=1)].sum()
            

    return pMat,normedSamples

def ExtendpMat(pMat, nOrders, step=0.5):
    pMatExtend=pMat
    for p in np.arange(0,1.1,step):
        scaled_numbers=generate_random_numbers(nOrders,p)
        pMatExtend=np.vstack([pMatExtend, scaled_numbers.dot(pMat)])
    return pMatExtend


# count= # of trails out of all sims trials that results in same results of all first and all both 
def RunSimulation(nOrders,sims,SmallestUtil):
    matrix=np.zeros((nOrders-1,nOrders-1))
    all_equal=0

    perm = GeneratePermutations(nOrders)
    #print (perm)
    count=0
    sim_data_maxlist = pd.DataFrame(columns=['sim'] + ['all first']+['all both']+[f'{p:.1f} certain' for p in np.arange(0, 1.1, 0.5)])
    sim_data_sum_util=pd.DataFrame(columns=['sim'] + ['all first']+['all both']+[f'{p:.1f} certain' for p in np.arange(0, 1.1, 0.5)])
    for sim in range(sims):

        pMat,normedSamples = RunSingleSimulation(nOrders, perm)
        

        #print(pMat)

        maxMat = np.argmax(pMat, axis=1)
        maxlist=maxMat.tolist()
        
        
        if all(x==maxlist[0] for x in maxlist):
            count+=1
            
        else: 
            pMatExtend= ExtendpMat(pMat,nOrders)
            #print ('pMatExtend',pMatExtend)
            maxMatExtend=np.argmax(pMatExtend, axis=1)
            maxlistExtend=maxMatExtend.tolist()
            #print('maxlistExtend',maxlistExtend)
            
            
        
            sum_util=np.zeros((len(maxlistExtend),1))

            for k in range((len(maxlistExtend))):
                #print ("k",k)

                for i in range(perm.shape[0]):
                    #print (maxlistExtend[k],perm[i,:])
                    if maxlistExtend[k]==perm [i,0]:
                        #print('it is the first in this perm')
                        sum_util[k]+=2*normedSamples[i]
                    elif maxlistExtend[k]==perm[i,1]:
                        #print('it is the 2')
                        sum_util[k]+=1*normedSamples[i]
                    else:
                        #print (' it is the 3')
                        sum_util[k]+= SmallestUtil*normedSamples[i]
            sim_data_maxlist = sim_data_maxlist.append({'sim': sim, 'all first': maxlistExtend[0], 'all both': maxlistExtend[1], 
                                            **{f'{p:.1f} certain': maxlistExtend[i] for i, p in enumerate(np.arange(0, 1.1, 0.5), start=2)}}, 
                                           ignore_index=True)


            sim_data_sum_util = sim_data_sum_util.append({'sim': sim, 'all first': sum_util[0], 'all both': sum_util[1], 
                                              **{f'{p:.1f} certain': sum_util[i] for i, p in enumerate(np.arange(0, 1.1, 0.5), start=2)}}, 
                                             ignore_index=True)
    
    return count, sim_data_maxlist, sim_data_sum_util
    
                

import matplotlib.pyplot as plt

def plot_simulation_results(sim_data_sum_util):
    # Get column names excluding 'sim'
    columns = sim_data_sum_util.columns[1:]
    
    # Calculate the number of rows and columns for subplots
    num_cols = len(columns)
    num_rows = (num_cols + 1) // 2  # Round up to the nearest integer
    
    fig, axes = plt.subplots(num_rows, 2, figsize=(12, 8), sharey=True)
    
    # Flatten the axes if necessary
    if num_cols > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
    
    # Plot each column separately
    for i, column in enumerate(columns):
        axes[i].scatter(sim_data_sum_util['sim'], sim_data_sum_util[column], label=column)
        axes[i].set_title(f'Plot of {column} for each simulation')
        axes[i].set_xlabel('Simulation Number')
        axes[i].set_ylabel(column)
        axes[i].legend()
        axes[i].grid(True)
    
    # Hide any remaining axes
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.show()



#columm name = '0.0 certain'  '0.5 certain'   '1.0 certain' 'all both'
def plot_simulation_results_diff(sim_data_sum_util,column_name,SmallestUtil):
    # Calculate the difference between 'all first' and 'all both'
    sim_data_sum_util['difference'] = sim_data_sum_util['all first'] - sim_data_sum_util[str(column_name)]
    
    # Calculate the average difference
    avg_difference = sim_data_sum_util['difference'].mean()
    
    # Plot the difference against the simulation number
    plt.figure(figsize=(8, 6))
    plt.scatter(sim_data_sum_util['sim'], sim_data_sum_util['difference'])
    
    # Plot the horizontal line representing the average difference
    plt.axhline(y=avg_difference, color='r', linestyle='--', label='Average Difference')
    
    plt.title('Difference between "all first" and '+str(column_name)+" for each simulation")
    plt.xlabel('Simulation Number')
    plt.ylabel('Difference')
    plt.legend()
    plt.grid(True)
    
    

    
