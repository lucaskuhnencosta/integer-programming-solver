from cutgen.aux import extract_ordered_fractional_map
import gurobipy as gp

def clique_detection(A,b):
    """
    This function is a first clique detection algorithm for constraint already structure in the format Ax<=b
    :param A: List of constraint indexes and values
    :param b: List of results
    :return: A list S of cliques
    """
    S=[]
    for constraint, RHS in zip(A,b):
        if len(constraint)<2:
            continue
        sorted_constraint=sorted(constraint,key=lambda x:x[1])
        if sorted_constraint[-1][1]+sorted_constraint[-2][1]>RHS:
            k=len(sorted_constraint)-1 #The index of the highest coefficient
            idx=k-1
            for s in range(k-2,-1,-1):
                _,a_js=sorted_constraint[s]
                _,a_js1=sorted_constraint[s+1]
                if a_js+a_js1>RHS:
                    idx=s
                else:
                    break
            C=[sorted_constraint[i][0] for i in range(idx,len(sorted_constraint))]
            S.append(C)
            for o in range(idx-1,-1,-1):
                _,a_jo=sorted_constraint[o]
                idx_f=None
                for f in range(k,-1,-1):
                    _,a_jf=sorted_constraint[f]
                    if a_jo+a_jf>RHS:
                        idx_f=f
                    else:
                        break
                if idx_f is not None:
                    clique_candidate=[sorted_constraint[o][0]]+[sorted_constraint[i][0] for i in range(idx_f,len(sorted_constraint))]
                    if clique_candidate:
                        S.append(clique_candidate)
    return S

def clique_extension(G, S):
    """
    Implements Algorithm 2: Clique Extension from the provided conflict graph G and clique C.

    Parameters:
    - G: A NetworkX graph representing the conflict graph.
    - C: A list of node identifiers representing the current clique.

    Returns:
    - S_extended: The extended clique including additional nodes if possible.
    """
    S_extended = []
    for clique in S:
        # 1. Let d be the vertex with the smallest degree
        d = min(clique, key=lambda node: G.degree[node])

        # 2. Candidate set L: neighbors of d not in C
        #L = {k for k in G.neighbors(d) if k not in clique}
        L = sorted(
            [k for k in G.neighbors(d) if k not in clique],
            key=lambda node: G.degree[node],
            reverse=True
        )

        # 3. Initialize extended clique C'
        C_extended = list(clique)

        # 4. While there are candidates to try
        while L:
            # 5. Pick l in L with the largest degree in G
            l = max(L, key=lambda node: G.degree[node])
            # 6. Remove l from L
            L.remove(l)

            # 7. Check if l is adjacent to all nodes in the current clique
            if all(l in G.neighbors(k) for k in C_extended):
                C_extended.append(l)

        S_extended.append(C_extended)

    return S_extended

def clique_cut_separator(parsed_data,best_A,G,S_strenghtned,max_rounds=20,min_viol=1e-4,instance_name="instance_0004.txt"):
    """
    Cutting-plane loop that reuses precomputed full cliques and conflict graph
    Only checks for violated cliques in fractional subgraph G' in each iteration

    Runs the full iterative clique cut separation loop.
    Solves LP, identifies violated cliques, adds them, and rebuilds until convergence.

    Parameters:
    - parsed_data: input data structure
    - best_A: number of aisles to select
    - all_extended_cliques: full list of precomputed cliques (already extended)
    - max_rounds: max number of cutting-plane rounds
    - min_viol: violation threshold
    - max_cliques_per_round: max cliques to generate per round

    Returns:
    - final_model: Gurobi model with all added cuts
    - all_cliques: list of all clique constraints added
    - X, Y: decision variable dictionaries
    """

    num_orders = parsed_data['num_orders']
    num_aisles = parsed_data['num_aisles']
    minW = 1 + min_viol
    all_cliques_added=set()

    for round_num in range(max_rounds):
        print(f"\n🔁 ROUND {round_num + 1}")
        model,X,Y=build_lp_model(parsed_data,best_A,clique_constraints=list(all_cliques_added),lp_filename=f"{instance_name}_round_{round_num+1}.lp")
        model.optimize()

        if model.status !=gp.GRB.OPTIMAL:
            print("❌ LP not optimal. Stopping.")
            break

        fractional_vars=extract_ordered_fractional_map(X,Y,num_orders,num_aisles)
        if not fractional_vars:
            print("✅ LP solution is integral. Done.")
            break

        fractional_indices={idx for idx,_ in fractional_vars}
        weights=dict(fractional_vars)

        G_sub=G.subgraph(fractional_indices).copy()
        if G_sub.number_of_nodes()==0:
            print("⚠️ No fractional variables in conflict graph.")
            break

        new_cliques = []
        for clique in S_strenghtned:
            if all(j in fractional_indices for j in clique):
                weight_sum=sum(weights[j] for j in clique)
                if weight_sum>minW:
                    new_cliques.append(clique)
                    print(f"⚠️ Violated clique: {clique}, weight sum = {weight_sum:.4f} > {minW:.4f}")


        if not new_cliques:
            print("🛑 No new violated cliques found.")
            break

        print(f"➕ {len(new_cliques)} new clique cuts added.")
        for clique in new_cliques:
            clique_key=tuple(sorted(clique))
            all_cliques_added.add(clique_key)

    return model, all_cliques_added, X, Y