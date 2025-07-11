from aux import invert_negative

def pre_process_constraints(parsed_data):
    """This function preprocesses all constraints to make them in the form needed for cliques
    It will take all constraints from the "mercado livre" problem and transform them into Ax<b. It returns A and b
    That means that they leave this function ready to generate clicks"""
    orders = parsed_data['orders']
    aisles = parsed_data['aisles']
    num_items = parsed_data['num_items']
    num_orders = parsed_data['num_orders']
    num_aisles = parsed_data['num_aisles']
    num_variables = num_orders + num_aisles
    quantities_orders = parsed_data['soma_pedidos']
    quantities_aisles = parsed_data['soma_corredor']
    LB = parsed_data['LB']
    UB = parsed_data['UB']

    A=[]
    b=[]

    #UB and LB constraints
    indexed_quantities_orders = list(enumerate(quantities_orders))
    indexed_quantities_orders_inv = [(idx, -val) for idx, val in indexed_quantities_orders]

    #The first row refers to the UB constraint
    indexed_quantities_orders_UB, UB = invert_negative(indexed_quantities_orders,UB,num_variables)
    A.append(indexed_quantities_orders_UB)
    b.append(UB)

    #The second row refers to the LB constraint
    indexed_quantities_orders_LB,new_LB_value=invert_negative(indexed_quantities_orders_inv,LB,num_variables)
    A.append(indexed_quantities_orders_LB)
    b.append(new_LB_value)

    #Per item constraints
    for item in range(num_items):
        order_demands = [orders[o][item] for o in range(num_orders)]
        aisle_supplies = [-aisles[a][item] for a in range(num_aisles)]
        constraint = order_demands + aisle_supplies

        indexed_item_constraint = list(enumerate(constraint))

        RHS=0

        indexed_item_constraint_fixed,RHS=invert_negative(indexed_item_constraint,RHS,num_variables)
        A.append(indexed_item_constraint_fixed)
        b.append(RHS)

    return A,b
