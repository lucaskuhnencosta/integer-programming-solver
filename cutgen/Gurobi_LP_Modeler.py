import gurobipy as gp
from gurobipy import GRB
from aux import get_variable_expression


def build_lp_model(parsed_data,best_A,clique_constraints=None,lp_filename="model.lp"):
    """
    Builds and returns an LP model of the problem with optional clique constraints.

    Parameters:
    - parsed_data: dictionary with parsed problem data
    - best_A: number of aisles to select
    - clique_constraints: list of cliques, where each clique is a list of variable indices (assumed to be 'pedido_X')
    - lp_filename: file to write the LP model

    Returns:
    - model: Gurobi model
    - pedido_X: decision variables for orders
    - corredor_Y: decision variables for aisles
    """
    print("     Building LP model...")

    num_orders = parsed_data['num_orders']
    num_items = parsed_data['num_items']
    num_aisles = parsed_data['num_aisles']
    LB = parsed_data['LB']
    UB = parsed_data['UB']
    quantity_orders = parsed_data['soma_pedidos']

    model = gp.Model()

    # Decision variables
    X = model.addVars(num_orders, vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name="X")
    Y = model.addVars(num_aisles, vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name="Y")

    # Objective
    model.setObjective(gp.quicksum(quantity_orders[i] * X[i] for i in range(num_orders)), GRB.MAXIMIZE)

    # Bounds on total picked quantity
    model.addConstr(gp.quicksum(quantity_orders[i] * X[i] for i in range(num_orders)) >= LB)
    model.addConstr(gp.quicksum(quantity_orders[i] * X[i] for i in range(num_orders)) <= UB)

    # Item coverage constraints
    for item in range(num_items):
        model.addConstr(
            gp.quicksum(X[i] * parsed_data['orders'][i][item] for i in range(num_orders)) <=
            gp.quicksum(Y[j] * parsed_data['aisles'][j][item] for j in range(num_aisles))
        )

    # Limit number of aisles
    model.addConstr(gp.quicksum(Y[i] for i in range(num_aisles)) == best_A)

    # Add optional clique constraints
    if clique_constraints:
        for idx, clique in enumerate(clique_constraints):
            model.addConstr(gp.quicksum(get_variable_expression(j,X,Y,num_orders,num_aisles) for j in clique) <= 1, name=f"clique_{idx}")

    #model.write(lp_filename)
    model.setParam('OutputFlag', 0)


    print(f"     LP model successfully built with {len(clique_constraints or [])} clique constraints.")
    return model, X, Y
