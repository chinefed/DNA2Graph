import networkx as nx


# Mapping from numeric group identifiers to group names
GROUP_NAME_MAP = {
    0: 'BOUNDARY',
    1: 'LINEAR',
    2: 'NON_LINEAR'
}


def is_non_linear(G):
    '''
    Check if the graph is non-linear, meaning it has nodes with degree > 2
    or it has only nodes with degree = 2 (in this case it is a loop)
    '''
    has_junction = False # has node with degree > 2
    has_endpoint = False # has node with degree 1
    for node in G.nodes:
        deg = nx.degree(G, node)
        if deg == 1:
            has_endpoint = True
        if deg > 2:
            has_junction = True
            break

    if has_junction:
        # If has a node with degree > 2, it is non-linear
        non_linear = True
    elif has_endpoint:
        # If it does not have nodes with degree > 2 and 
        # has at least one node of degree 1, it is linear
        non_linear = False
    else:
        # If it has only nodes with degree 2, it is non-linear
        # (indeed it is a loop)
        non_linear = True

    return non_linear


def classify_components(G):
    '''
    Assigns each connected component of the input graph to a group:
        - Linear (group 1)
        - Non-linear (group 2)
    Each node in the component stores the group number in the 'component_group' attribute.
    '''
    G = G.copy()

    for component in nx.connected_components(G):
        G_component = G.subgraph(component)

        v = next(iter(component)) # Get an arbitrary node
        if G.nodes[v]['is_boundary']:
            component_group = 0 # Boundary component
        elif is_non_linear(G_component):
            component_group = 2 # Non-linear component
        else:
            component_group = 1 # Linear component

        # Assign the group to each node in the component
        nx.set_node_attributes(G, {v: component_group for v in component}, 'component_group')

    return G