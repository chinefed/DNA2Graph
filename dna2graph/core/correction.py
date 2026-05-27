import time
import multiprocessing as mp

import numpy as np
import networkx as nx
from scipy.spatial import cKDTree

from dna2graph.utils import UserCancelledError
from dna2graph.core.image_utils import (
    get_distance_weighted_skeleton,
    get_boundary_skeleton_pixels
)
from dna2graph.core.graph_utils import (
    skeleton_to_graph,
    linear_walk,
    get_graph_bounding_box,
    mark_boundary_components
)


MAX_ITERATIONS = 10 # Maximum number of iterations for pruning cycles and branches
PRUNE_SHORT_CYCLES_WIRING_CAP = 10 # Wiring cap used in 'prune_short_cycles'
BOUNDARY_TOL = 50  # Tolerance for detecting boundary skeleton pixels

def prune_short_cycles(G, min_cycle_length, max_iterations):
    '''
    Remove short cycles from the graph by replacing them with a centroid node.
    '''
    # We process each connected component of the graph separately
    # (different components may require a different number of iterations to be pruned).
    components = [
            G.subgraph(component).copy()
            for component in nx.connected_components(G)
        ]
    for G_component in components:

        for _ in range(max_iterations):
            # Find a cycle basis of the component and keep only the short cycles
            short_cycles = [
                set(cycle) for cycle in nx.cycle_basis(G_component)
                if len(cycle) < min_cycle_length
            ]
            # If no short cycles are found, we stop the pruning process
            # for the current component
            if len(short_cycles) == 0:
                break

            removed_nodes = set() # Tracks nodes removed in the current iteration
            for cycle in short_cycles:
                # If the cycle contains nodes that were removed in the current iteration, skip ir
                # -> we do not remove overlapping cycles in a single iteration
                if not removed_nodes.isdisjoint(cycle):
                    continue
                
                # Compute the cycle neighborhood
                neighbors = set().union(
                    *(set(G_component.neighbors(node)) for node in cycle)
                ) - cycle

                # Remove the cycle nodes from the graph
                G_component.remove_nodes_from(cycle)
                removed_nodes.update(cycle)

                if len(neighbors) == 0:
                    # Isolated ring, skip it
                    continue

                if len(neighbors) > PRUNE_SHORT_CYCLES_WIRING_CAP:
                    # Removing this cycle would introduce a node
                    # with degree > PRUNE_SHORT_CYCLES_WIRING_CAP.
                    # Exclude this component, as it is likely corrupted.
                    G_component.clear()
                    break

                # The cycle is replaced with a node that is
                # the centroid of the cycle neighborhood
                coords = np.array(list(neighbors), dtype=np.int32)
                new_node = tuple(np.mean(coords, axis=0).astype(np.int32))
                new_node_dist = np.mean(
                    [G_component.nodes[node]['dist'] for node in neighbors]
                ).astype(np.uint8)
                G_component.add_node(new_node, dist=new_node_dist)

                # Wire the new node to the cycle neighborhood
                new_edges = [(new_node, node) for node in neighbors]
                G_component.add_edges_from(new_edges)

    # After processing all components, we combine them back into a single graph
    G = nx.compose_all(components)

    return G


def prune_short_branches(G, min_branch_length, max_iterations):
    '''
    Remove short branches that arise due to noise from the graph.
    '''
    # We process each connected component of the graph separately
    # (different components may require a different number of iterations to be pruned).
    components = [
            G.subgraph(component).copy()
            for component in nx.connected_components(G)
        ]
    for G_component in components:

        for _ in range(max_iterations):
            # Get the endpoints of the current component
            endpoints = [node for node in G_component.nodes if G.degree(node) == 1]

            # Extract the branches. We define a branch as a path starting from an endpoint.
            # and going through nodes of degree 2, until we reach a node with degree not equal to 2.
            to_remove = []
            for endpoint in endpoints:
                branch, max_steps_reached = linear_walk(
                    G_component,
                    start=endpoint,
                    v=next(iter(G_component.neighbors(endpoint))),
                    max_steps=min_branch_length
                )
                if not max_steps_reached:
                    # If the branch is shorter than the minimum length, we remove it
                    to_remove.append(branch)

            # If no branches were found, we stop the pruning process
            if not to_remove:
                break

            # Remove the branches in sorted order of length, and ensure that we do not remove
            # multiple branches that are connected to the same joint in a single iteration.
            # This is done to make the pruning process less destructive. 
            blocked_joints = set()
            for branch in sorted(to_remove, key=len):
                joint = branch[-1] # The "other" terminal wrt the endpoint we walked from
                if joint not in blocked_joints:
                    G_component.remove_nodes_from(branch[:-1])
                    blocked_joints.add(joint)

    # After processing all components, we combine them back into a single graph
    G = nx.compose_all(components)

    return G


def repair_connectivity(G, min_graph_distance, max_euclidean_distance, restrict_to_endpoints=False):
    '''
    Attempts to improve the connectivity of a spatial graph by adding edges between nodes that are close
    in Euclidean space but distant in the graph topology.
    '''
    # Copy the graph to avoid modifying the original
    G = G.copy()

    # Build a KD-tree from endpoints
    endpoints = np.array([node for node in G.nodes if G.degree(node) == 1], dtype=np.int32)
    kd_tree_endpoints = cKDTree(endpoints)

    if restrict_to_endpoints:
        kd_tree_other = kd_tree_endpoints
    else:
        all_nodes = np.array(list(G.nodes), dtype=np.int32)
        kd_tree_other = cKDTree(all_nodes)

    # For each endpoint, find nearby nodes in Euclidean space
    neighbor_indices = kd_tree_endpoints.query_ball_tree(kd_tree_other, max_euclidean_distance)

    for i, indices in enumerate(neighbor_indices):
        endpoint = tuple(endpoints[i])

        # Get nodes within min_graph_distance hops
        reachable_nodes = set(
            nx.single_source_shortest_path_length(
                G,
                source=endpoint,
                cutoff=min_graph_distance
            ).keys()
        )

        # Get spatial neighbors from KD-tree
        euclidean_neighbors = {
            tuple(kd_tree_other.data[j].astype(np.int32)) for j in indices
        }

        # Identify candidates that are close in Euclidean space but distant in the graph
        candidates = list(euclidean_neighbors - reachable_nodes)

        # Add an edge to the closest valid candidate (if any)
        if candidates:
            target = None
            min_dist = float('inf')
            for candidate in candidates:
                dist = np.linalg.norm(np.array(endpoint) - np.array(candidate))
                if dist < min_dist:
                    min_dist = dist
                    target = candidate

            G.add_edge(endpoint, target)

    return G


def check_node_structurality(G, node, structurality_depth_cutoff):
    '''
    Check if a node is structural by analyzing its stems.
    A node is structural if:
    1. It belongs to a cycle; OR
    2. It has at least two stems (i.e., neighbors) that:
    - Either lead to a node more than `structurality_depth_cutoff` steps away (via DFS), OR
    - Lead to a cycle anywhere in the graph (not necessarily including the node).
    '''

    # Check if the node has enough stems
    stems = list(G.neighbors(node))
    if len(stems) < 2:
        return False
    
    # Visited labels are shared across all stems
    # as we aim to detect also cycles involving different stems.
    # Note that if a cycle involving different stems exists,
    # then the node belongs to a cycle. Furthermore,
    # if the node belongs to a cycle, than at least two stems
    # meet structurality conditions.
    visited = {node}
    n_struct_stems = 0 # Counts the number of structural stems
    for stem in stems:
        stack = [(stem, node, 1)]

        while stack:
            v, parent, depth = stack.pop()
            # When (v, parent) was added to the stack, v was not visited.
            # When it is popped, v is still not visited.
            # If the algorithm reached this point, no other DFS traversal
            # visited v. Indeed, if v was reached from parent' ≠ parent:
            # parent ≠ parent' would have belonged to Nbr(v)
            # with parent necessarily visited---when (v, parent) was added to the stack.
            # -> DFS would have stopped due to cycle detection.
            visited.add(v)

            # If the DFS traversal depth for the current stem exceeds the cutoff
            if depth > structurality_depth_cutoff:
                # The stem is structural, increase the counter
                n_struct_stems += 1
                if n_struct_stems >= 2:
                    # If the number of structural stems >= 2,
                    # the node is structural
                    return True
                # If the number of structural stems < 2,
                # move to the next stem
                break

            # For each neighbor of the current node
            for u in G.neighbors(v):
                if u not in visited:
                    # Add u to the stack
                    stack.append((u, v, depth+1))
                elif u != parent:
                    # A cycle is detected when v is connected
                    # to a visited node that is not the parent.
                    # If this is the case, the stem is structural.
                    n_struct_stems += 1
                    if n_struct_stems >= 2:
                        return True
                    break
                
    # If no structural stems were found, the node is not structural

    return False


def prune_by_thickness(G, max_thickness, structurality_depth_cutoff):
    '''
    Prune non-structural nodes with a thickness greater than max_thickness.
    High distance nodes are likely to be noise or artifacts.
    '''
    G = G.copy()
    nodes = list(G.nodes)
    for node in nodes:
        if G.nodes[node]['dist'] <= max_thickness:
            continue

        if not check_node_structurality(G, node, structurality_depth_cutoff):
            G.remove_node(node)

    return G


def prune_small_components(G, minimum_bbox_size):
    '''
    Prune the graph to remove small components whose largest
    bounding box side is smaller than the user-defined threshold.
    '''
    G = G.copy()

    components = list(nx.connected_components(G))
    for component in components:
        G_component = G.subgraph(component)
        y_min, x_min, y_max, x_max = get_graph_bounding_box(G_component)
        largest_side = max(y_max - y_min, x_max - x_min)

        if largest_side < minimum_bbox_size:
            G.remove_nodes_from(component)

    return G


def init_worker(corrector):
    global global_corrector
    global_corrector = corrector


def worker(G):
     '''
     Worker function to process a graph in parallel.
     '''
     global global_corrector

     G = global_corrector._worker(G)

     return G


class Corrector:
    '''
    A class to process a binary mask and extract a cleaned skeleton graph.
    The processing includes:
    - Skeletonization
    - Pruning short cycles
    - Repairing connectivity
    - Pruning short branches
    - Pruning non-structural nodes by thickness
    '''
    def __init__(self,
                 trained_seg,
                 batch_size,
                 min_cycle_length,
                 min_branch_length,
                 repair_1_min_graph_distance,
                 repair_1_max_euclidean_distance,
                 minimum_bbox_size,
                 max_thickness=None,
                 structurality_depth_cutoff=None,
                 repair_2_min_graph_distance=None,
                 repair_2_max_euclidean_distance=None,
                 repair_2_restrict_to_endpoints=None,
                 stop_event=None
    ):
        self.trained_seg = trained_seg
        self.batch_size = batch_size
        self.min_cycle_length = min_cycle_length
        self.min_branch_length = min_branch_length
        self.repair_1_min_graph_distance = repair_1_min_graph_distance
        self.repair_1_max_euclidean_distance = repair_1_max_euclidean_distance
        self.minimum_bbox_size = minimum_bbox_size
        self.max_thickness = max_thickness
        self.structurality_depth_cutoff = structurality_depth_cutoff
        self.repair_2_min_graph_distance = repair_2_min_graph_distance
        self.repair_2_max_euclidean_distance = repair_2_max_euclidean_distance
        self.repair_2_restrict_to_endpoints = repair_2_restrict_to_endpoints
        self.stop_event = stop_event

    def correct(self, mask, validity_mask):
        '''
        Process the input mask to extract a cleaned skeleton graph.
        '''
        # Get weighted skeleton with distance weights
        skeleton = get_distance_weighted_skeleton(mask)

        # Convert the skeleton to a graph representation
        G = skeleton_to_graph(skeleton)

        # Identify boundary skeleton pixels and store them
        # in the 'boundary_skel_pixels' graph attribute
        boundary_skel_pixels = get_boundary_skeleton_pixels(
            skeleton,
            validity_mask,
            BOUNDARY_TOL
        )
        G.graph['boundary_skel_pixels'] = boundary_skel_pixels

        # Repair the connectivity (first pass)
        G = repair_connectivity(
            G,
            min_graph_distance=self.repair_1_min_graph_distance,
            max_euclidean_distance=self.repair_1_max_euclidean_distance,
            restrict_to_endpoints=False
        )

        # Correct components in parallel
        G = self._correct_components_in_parallel(G)

        return G
    
    def finalize(self, G):
        '''
        Finalize the graph by pruning small components, enhancing connectivity,
        and marking components with endpoints that are close to the boundary of valid regions.
        '''
        # Prune small components
        G = prune_small_components(G, self.minimum_bbox_size)

        # Skip additional refining if using trained
        # segmentation pipeline
        if not self.trained_seg:
            # Repair the connectivity (second pass)
            G = repair_connectivity(
                G,
                min_graph_distance=self.repair_2_min_graph_distance,
                max_euclidean_distance=self.repair_2_max_euclidean_distance,
                restrict_to_endpoints=self.repair_2_restrict_to_endpoints
            )

            # Run a second pass of prune_short_branches
            # if repair_2_restrict_to_endpoints is True
            # (may create short branches)
            if not self.repair_2_restrict_to_endpoints:
                G = prune_short_branches(
                    G,
                    min_branch_length=self.min_branch_length,
                    max_iterations=MAX_ITERATIONS
                )

        # Mark components with endpoints that are close to the boundary of valid regions
        boundary_skel_pixels = G.graph['boundary_skel_pixels']
        G = mark_boundary_components(G, boundary_skel_pixels)

        return G
    
    def _worker(self, G):
        '''
        Worker function to process a graph in parallel.
        This function is called by each worker process.
        '''
        # Prune short cycles
        G = prune_short_cycles(
            G,
            min_cycle_length=self.min_cycle_length,
            max_iterations=MAX_ITERATIONS
        )

        # Prune short branches
        G = prune_short_branches(
            G,
            min_branch_length=self.min_branch_length,
            max_iterations=MAX_ITERATIONS
        )

        # Skip prune by thickness if using trained
        # segmentation pipeline
        if self.trained_seg:
            return G

        # Prune non-structural nodes by thickness
        G = prune_by_thickness(
            G,
            max_thickness=self.max_thickness,
            structurality_depth_cutoff=self.structurality_depth_cutoff
        )

        return G
    
    def _correct_components_in_parallel(self, G):
        '''
        Split the graph in batches and process it in parallel.
        '''
        # Split the graph into batches of connected components
        components = list(nx.connected_components(G))
        n_components = len(components)
        batches = [
            G.subgraph(set().union(*components[i:i+self.batch_size])).copy()
            for i in range(0, n_components, self.batch_size)
        ]

        try:
            pool = mp.Pool(mp.cpu_count(), initializer=init_worker, initargs=(self,))
            result = pool.map_async(worker, batches)

            while not result.ready():
                if self.stop_event and self.stop_event.is_set():
                    pool.terminate()
                    raise UserCancelledError
                time.sleep(0.2)

            try:
                processed_batches = result.get()
            except Exception as e:
                raise RuntimeError(
                    f"Worker raised an exception: {e}"
                )

        finally:
            pool.close()
            pool.join()

        # Combine the results from all workers
        G = nx.compose_all(processed_batches)

        return G