import numba
import numpy as np
from numba import jit
from typing import Callable, Literal


@jit(nopython=True)
def _build_result(split_points: np.array, n_groups: int):
    
    res = [split_points[n_groups - 1, -1]]
    index = res[0] - 1
    
    for i in range(n_groups - 2, 0, -1):
        res = [split_points[i, index]] + res
        index = res[0] - 1
    return res


def _optimal_partitioning(data: np.array, n_groups: int, cost):
    
    table = np.zeros((n_groups + 1, data.shape[0]), dtype='float')
    split_points = np.zeros_like(table, dtype='int')
        
    for i in range(table.shape[1]):
        group_range = min(i + 1, n_groups - 1 if i != table.shape[1] - 1 else n_groups)

        table[0, i] = cost(data, 0, i)
        for j in range(1, group_range):
            costs = np.array([table[j - 1, z - 1] + cost(data, z, i) for z in range(j-1, i + 1)])
            z_raw = np.argmin(costs)
            table[j, i] = costs[z_raw]
            split_points[j, i] = z_raw + (j - 1)
            
    return table, split_points
    

def interval_partitioning(data: np.array,
                          cost: Callable,
                          n_intervals: int = 2,
                          mode: Literal['only_python', 'with_python', 'no_python'] = 'with_python'):
    
    """
        Optimal interval partitioning of array data according to penalty function cost.
        
        parameters
        ----------
        data : np.array
            1D numpy array containing the data to be optimally split into intervals
        cost : callable
            function or other callable object
        n_intervals : int, default = 2
            number of intervals array data is to be split in. Expected to be at least 2.
        mode : Literal['only_python'. 'with_python', 'no_python']
            Operation mode: 'only_python' deploys no Numba acceleration.
                            'with_python' allows Python with Numba acceleration
                            'no_python' optimization without usage of Python. The cost function must not use
                             any non numba compatible features
    
    """
    
    if n_intervals < 2:
        raise ValueError(f'Expected n_intervals >= 2. Got {n_intervals}')
    if n_intervals >= data.shape[0]:
        raise ValueError('Expected n_intervals to be smaller than length of the data array.'
                         f'Length of data array is {data.shape[0]}. n_intervals is {n_intervals}.')

    try:

        if mode == 'only_python':
            table, split = _optimal_partitioning(data, n_intervals, cost)
        elif mode == 'with_python':
            table, split = jit(nopython=False)(_optimal_partitioning)(data,
                                                                      n_intervals,
                                                                      jit(nopython=False)(cost))
        elif mode == 'no_python':
            table, split = jit(nopython=True)(_optimal_partitioning)(data,
                                                                     n_intervals,
                                                                     jit(nopython=True)(cost))
        else:
            raise ValueError(f'Unknown operation mode. Got {mode}')

    except numba.NumbaError as e:
        print("""The cost function provided was incompatible to Numba. Make sure, that all features deployed
              in your cost function are compatible with Numba or consider using a different deployment mode.
              Numba raised the following exception: """)
        raise e

    return [(table[i, -1], _build_result(split, i + 1)) for i in range(1, n_intervals)]

