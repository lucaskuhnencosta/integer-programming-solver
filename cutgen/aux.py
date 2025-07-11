########################### Utilities related to mercado livre problem ###########################

solution_map={
    "Inputs/instance_0001.txt":3,
    "Inputs/instance_0002.txt":1,
    "Inputs/instance_0003.txt":4,
    "Inputs/instance_0004.txt":2,
    "Inputs/instance_0005.txt":8,
    "Inputs/instance_0006.txt":1,
    "Inputs/instance_0007.txt":4,
    "Inputs/instance_0008.txt":13,
    "Inputs/instance_0009.txt":12 ,
    "Inputs/instance_0010.txt":103,
    "Inputs/instance_0011.txt":20,
    "Inputs/instance_0012.txt":4,
    "Inputs/instance_0013.txt":13,
    "Inputs/instance_0014.txt":11,
    "Inputs/instance_0015.txt":3,
    "Inputs/instance_0016.txt":2,
    "Inputs/instance_0017.txt":2,
    "Inputs/instance_0018.txt":5,
    "Inputs/instance_0019.txt":1,
    "Inputs/instance_0020.txt":2
}

########################### Utilities to manipulate constraints ###########################

def invert_negative(A,b,num_variables):
    """
    Inverts negative coefficients in A and adjusts index (x_j -> x̄_j).
    Also removes zero-coefficient variables and adjusts RHS b.

    Parameters:
    - A: list of (index, coefficient)
    - b: right-hand-side constant
    - num_variables: offset to remap x_j to x̄_j

    Returns:
    - new_A: transformed list of (index, positive coefficient)
    - new_b: adjusted RHS

    This is called in our pre-process constraint routine
    """
    new_const_index_list = []
    summation = b
    for idx, val in A:
        if val < 0:
            new_const_index_list.append((idx + num_variables, -val))
            summation = summation - val
        elif val==0:
            continue
        else:
            new_const_index_list.append((idx, val))

    return new_const_index_list, summation

########################### Utilities to convert node index to variable/plot labels ###########################

def get_node_label(index,num_orders,num_aisles):
    """This function is used to plot the graph, it is useful to give some meaning to the variables ONLY"""
    if 0<=index<num_orders:
        return f'X_{index}'
    elif num_orders <=index < num_orders+num_aisles:
        return f'Y_{index-num_orders}'
    elif num_orders+num_aisles <=index<2*num_orders + num_aisles:
        return f'X_bar_{index-(num_orders+num_aisles)}'
    elif 2*num_orders+num_aisles <=index < 2*(num_orders+ num_aisles):
        return f'Y_bar_{index-(2*num_orders+num_aisles)}'
    else:
        raise ValueError(f'Index {index} out of range')


def get_variable_expression(index,X,Y,num_orders,num_aisles):
    """This function is used inside gurobi environment to convert cliques to Gurobi variables"""
    if 0<=index<num_orders:
        return X[index]
    elif num_orders <=index < num_orders+num_aisles:
        return Y[index-num_orders]
    elif num_orders+num_aisles <=index<2*num_orders + num_aisles:
        return 1-X[index-num_orders-num_aisles]
    elif 2*num_orders+num_aisles<=index<2*(num_orders+ num_aisles):
        return 1-Y[index-2*num_orders-num_aisles]
    else:
        raise ValueError(f'Index {index} out of range')


def extract_variable_map(X,Y,num_orders,num_aisles):
    """This function is used to extract Gurobi's variables back to the pre-processor"""
    var_map={}

    for j in range(num_orders):
        var_map[j]=X[j].X
        var_map[num_orders+num_aisles+j]=1-X[j].X

    for j in range(num_aisles):
        var_map[num_orders+j]=Y[j].X
        var_map[2*num_orders+num_aisles+j]=1-Y[j].X

    return var_map

# Re-import after environment reset
def extract_ordered_fractional_map(X, Y, num_orders, num_aisles, tolerance=1e-4):
    """
    Returns a sorted list of (index, value) tuples for all fractional variables (including bar variables),
    using the internal 0-based index system.
    """
    fractional_list = []

    # X_j and X_bar_j
    for j in range(num_orders):
        val = X[j].X
        if tolerance < val < 1 - tolerance:
            fractional_list.append((j, val))
        if tolerance < 1 - val < 1 - tolerance:
            fractional_list.append((num_orders + num_aisles + j, 1 - val))

    # Y_j and Y_bar_j
    for j in range(num_aisles):
        val = Y[j].X
        if tolerance < val < 1 - tolerance:
            fractional_list.append((num_orders + j, val))
        if tolerance < 1 - val < 1 - tolerance:
            fractional_list.append((2 * num_orders + num_aisles + j, 1 - val))

    return sorted(fractional_list, key=lambda x: x[0])



########################### Data/Clique processing routines ###########################

def remove_dominated_cliques(S_extended):
    """ This routine is called in the main notebook to remove dominated cliques from the extended clicl solution"""
    cliques=[set(c) for c in S_extended]
    maximal=[]

    for i,c1 in enumerate(cliques):
        is_dominated=False
        for j,c2 in enumerate(cliques):
            if i!=j and c1<c2:
                is_dominated=True
                break
        if not is_dominated:
            maximal.append(frozenset(c1))

    unique_maximal=[list(c) for c in set(maximal)]

    return sorted(unique_maximal, key=lambda x: (-len(x), x))

