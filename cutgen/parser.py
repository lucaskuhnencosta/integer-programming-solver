def parse_input(file_path):
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    # Parse primeira linha (o, i, a)
    o, i, a = map(int, lines[0].split())
    current_line = 1

    # Matriz de pedidos (o x i)
    orders_matrix = [[0] * i for _ in range(o)]
    for order_idx in range(o):
        parts = list(map(int, lines[current_line].split()))
        k = parts[0]
        items = parts[1:]
        for idx in range(0, 2*k, 2):  # Processa pares (item, quantidade)
            item = items[idx]
            qty = items[idx + 1]
            orders_matrix[order_idx][item] = qty
        current_line += 1

    # Matriz de corredores (a x i)
    aisles_matrix = [[0] * i for _ in range(a)]
    for aisle_idx in range(a):
        parts = list(map(int, lines[current_line].split()))
        l = parts[0]
        items = parts[1:]
        for idx in range(0, 2*l, 2):  # Processa pares (item, quantidade)
            item = items[idx]
            qty = items[idx + 1]
            aisles_matrix[aisle_idx][item] = qty
        current_line += 1

    # Parse limites da wave
    LB, UB = map(int, lines[current_line].split())

    soma_pedidos = []
    soma_corredor = []
    for pedidos_iter in range(len(orders_matrix)):
        soma = sum(orders_matrix[pedidos_iter])
        soma_pedidos.append(soma)
    for corredor in aisles_matrix:
        soma_corredor.append(sum(corredor))

    return {
        'num_orders': o,
        'num_items': i,
        'num_aisles': a,
        'soma_pedidos': soma_pedidos,
        'soma_corredor': soma_corredor,
        'orders': orders_matrix,
        'aisles': aisles_matrix,
        'LB': LB,
        'UB': UB,
    }