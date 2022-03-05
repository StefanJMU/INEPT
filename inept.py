import numpy as np
from numba import jit
  
@jit(nopython=True)
def build_result(split_points : np.array, n_groups : int) :
    
    res = [split_points[n_groups - 1,-1]]
    
    index = res[0] - 1
    
    for i in range(n_groups - 2, 0, -1) :
        
        res = [split_points[i,index]] + res
        index = res[0] - 1
        
    return res
       
@jit(nopython=True)
def _optimal_partitioning(data : np.array, n_groups : int) :
    
    table = np.zeros((n_groups + 1, data.shape[0]),dtype='float') 
    split_points = np.zeros_like(table,dtype='int')
        
    for i in range(table.shape[1]) :
    
        group_range = min(i + 1, n_groups - 1 if i != table.shape[1] - 1 else n_groups)
        
        table[0,i] = cost(data,0,i) 
        for j in range(1,group_range) : 
           
            costs = np.array([table[j - 1, z - 1] + cost(data,z,i) for z in range(j-1,i + 1)])
            
            z_raw = np.argmin(costs)
            table[j,i] = costs[z_raw]            
            split_points[j,i] = z_raw + (j - 1)
            
    
    return table[n_groups - 1,-1],build_result(split_points, n_groups)
    

def interval_partitioning(data : np.array, cost, n_intervals : int = 2) :
    
    """
        Optimal interval partitioning of array data according to penality function cost.
        
        parameters
        ----------
        data : np.array
            1D numpy array containing the data to be optimally split into intervals
        cost : callable
            numba conform callable
        n_intervals : int, default = 2
            number of intervals array data is to be split in. Expected to be at least 2.
    
    """
    
    if n_intervals < 1 :
        raise ValueError(f'Expected n_intervals >= 2. Got {n_intervals}')
        
    try :
    
        score,partitioning = _optimal_partitioning(data,n_intervals)
    
    except :
        
        #todo : error handling convept
        print('Something went wrong')

