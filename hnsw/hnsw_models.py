import numpy as np

class HNSWNode:
    def __init__(self, embedding, max_level, label=None):
        self.embedding = embedding
        self.label = label
        # Initialize neighbors for all levels, even those higher than the node's assigned level
        self.neighbors = {i: [] for i in range(max_level + 1)}

    def add_neighbor(self, neighbor, level):
        self.neighbors[level].append(neighbor)


class HNSW:
    def __init__(self, max_level, max_neighbours=200, level_probability=0.5):
        self.max_level = max_level
        self.max_neighbours = max_neighbours
        self.level_probability = level_probability
        self.layers = {i: [] for i in range(max_level + 1)}
        self.entry_point = None

    # ------------------------ #
    #    Adding Node
    # ------------------------ #
    def add_node(self, embedding, label=None):
        level = self._get_random_level()
        new_node = HNSWNode(embedding, self.max_level, label=label)
        if self.entry_point is None:
            self.entry_point = new_node
        else:
            self._insert_node(new_node, level)
        for l in range(level + 1):
            self.layers[l].append(new_node)

    def _get_random_level(self):
        level = 0
        while np.random.rand() < self.level_probability and level < self.max_level:
            level += 1
        return level

    # ------------------------------------------------ #
    #    Adding Node when Entrypoint is not Null
    # ------------------------------------------------ #
    def _insert_node(self, new_node, level):
        current_node = self.entry_point
        # list: [level_new_node, level_new_node-1, ..., 0]
        for l in reversed(range(level + 1)):
            current_node = self._search_layer(new_node.embedding, current_node, l)
            self._connect_neighbors(new_node, current_node, l)

    def _search_layer(self, query, entry_point, layer):
        current_node = entry_point
        while True:
            neighbors = current_node.neighbors[layer]
            
            # Return the closest HNSWNode in that layer otherwise None
            closest = self._find_closest(query, neighbors)
            
            # if in that level there is no neighbor, then break
            # if the distance of the query to the closest neighbor is greater than the distance of the query to the current node, then break
            if closest is None or self._distance(query, closest.embedding) >= self._distance(query, current_node.embedding):
                # If True (Neighbor Not Closer) then break
                break
            current_node = closest
        return current_node

    def _connect_neighbors(self, new_node, closest_node, level):
        if new_node.label != closest_node.label:
            closest_node.add_neighbor(new_node, level)
            new_node.add_neighbor(closest_node, level)
        if len(new_node.neighbors[level]) > self.max_neighbours:
            new_node.neighbors[level] = self._prune_neighbors(new_node.neighbors[level], new_node.embedding)

    def _prune_neighbors(self, neighbors, embedding):
        distances = self._distance_matrix(embedding, [neighbor.embedding for neighbor in neighbors])
        pruned_indices = np.argsort(distances)[:self.max_neighbours]
        return [neighbors[i] for i in pruned_indices]

    def _find_closest(self, query, neighbors):
        if not neighbors:
            return None
        neighbor_embeddings = np.array([neighbor.embedding for neighbor in neighbors])
        distances = self._distance_matrix(query, neighbor_embeddings)
        closest_idx = np.argmin(distances)
        return neighbors[closest_idx]

    def _distance_matrix(self, vec, matrix):
        # Calculate Euclidean distance in a vectorized manner
        return np.linalg.norm(matrix - vec, axis=1)

    def _distance(self, vec1, vec2):
        return np.linalg.norm(vec1 - vec2)
    
    # ------------------------------------------------ #
    #    Inference
    # ------------------------------------------------ #
    def search_knn(self, query_embedding, k):
        # Start the search from the entry point at the top level
        current_node = self.entry_point
        for level in reversed(range(self.max_level + 1)):
            current_node = self._search_layer(query_embedding, current_node, level)
        
        # Now perform a search in the bottom layer to find the top K nearest neighbors
        neighbors = current_node.neighbors[0]  # Start with neighbors in the bottom level (level 0)
        candidates = [current_node] + neighbors  # Include the current node itself
        
        # Calculate distances from the query to all candidates
        distances = [self._distance(query_embedding, node.embedding) for node in candidates]
        
        # Sort candidates by distance
        sorted_indices = np.argsort(distances)
        
        # Select the top K closest nodes
        top_k_indices = sorted_indices[:k]
        
        # Return the top K closest nodes (or embeddings)
        top_k_nodes = [candidates[i] for i in top_k_indices]
        
        return top_k_nodes

    
class HNSW_V2:
    """
    V2: we use multiple entry points to search in parallel in a given layer
    """
    def __init__(self, max_level, max_neighbours=200, level_probability=0.5):
        self.max_level = max_level
        self.max_neighbours = max_neighbours
        self.level_probability = level_probability
        self.layers = {i: [] for i in range(max_level + 1)}
        self.entry_point = None

    def add_node(self, embedding, label=None):
        level = self._get_random_level()
        new_node = HNSWNode(embedding, self.max_level, label=label)
        if self.entry_point is None:
            self.entry_point = new_node  # Set entry_point only when it's None
        self._insert_node(new_node, level)
        for l in range(level + 1):
            self.layers[l].append(new_node)

    def _get_random_level(self):
        level = 0
        while np.random.rand() < self.level_probability and level < self.max_level:
            level += 1
        return level

    def _insert_node(self, new_node, level):
        current_node = self.entry_point
        for l in reversed(range(level + 1)):
            current_node = self._search_layer(new_node.embedding, current_node, l)
            self._connect_neighbors(new_node, current_node, l)

    def _search_layer(self, query, entry_point, layer):
        # Select 3 entry points with the most neighbors
        entry_points = self._select_top_entry_points(layer)
        
        # Perform searches from each entry point
        candidates = []
        for ep in entry_points:
            result = self._single_search_layer(query, ep, layer)
            if result is not None:
                candidates.append(result)

        # Check if candidates list is empty
        if not candidates:
            return entry_point  # or return None, depending on the desired behavior
        
        # Find the closest node among all candidates
        closest = min(candidates, key=lambda node: self._distance(query, node.embedding))
        return closest

    def _select_top_entry_points(self, layer):
        # Sort nodes in the layer by the number of neighbors, in descending order
        sorted_nodes = sorted(self.layers[layer], key=lambda node: len(node.neighbors[layer]), reverse=True)
        
        # Select the top 3 nodes with the most neighbors
        top_entry_points = sorted_nodes[:3]
        
        return top_entry_points

    def _single_search_layer(self, query, entry_point, layer):
        current_node = entry_point
        while True:
            neighbors = current_node.neighbors[layer]
            closest = self._find_closest(query, neighbors)
            if closest is None or self._distance(query, closest.embedding) >= self._distance(query, current_node.embedding):
                break
            current_node = closest
        return current_node

    def _connect_neighbors(self, new_node, closest_node, level):
        if new_node.label != closest_node.label:
            closest_node.add_neighbor(new_node, level)
            new_node.add_neighbor(closest_node, level)
        if len(new_node.neighbors[level]) > self.max_neighbours:
            new_node.neighbors[level] = self._prune_neighbors(new_node.neighbors[level], new_node.embedding)

    def _prune_neighbors(self, neighbors, embedding):
        distances = self._distance_matrix(embedding, [neighbor.embedding for neighbor in neighbors])
        pruned_indices = np.argsort(distances)[:self.max_neighbours]
        return [neighbors[i] for i in pruned_indices]

    def _find_closest(self, query, neighbors):
        if not neighbors:
            return None
        neighbor_embeddings = np.array([neighbor.embedding for neighbor in neighbors])
        distances = self._distance_matrix(query, neighbor_embeddings)
        closest_idx = np.argmin(distances)
        return neighbors[closest_idx]

    def _distance_matrix(self, vec, matrix):
        return np.linalg.norm(matrix - vec, axis=1)

    def _distance(self, vec1, vec2):
        return np.linalg.norm(vec1 - vec2)
    
    def search_knn(self, query_embedding, k):
        # Start the search from the top level using multiple entry points
        current_candidates = []
        
        for level in reversed(range(self.max_level + 1)):
            # Get the top 3 entry points with the most neighbors at the current level
            entry_points = self._select_top_entry_points(level)
            
            # Perform searches from each entry point and collect candidates
            level_candidates = []
            for entry_point in entry_points:
                candidate = self._single_search_layer(query_embedding, entry_point, level)
                if candidate:
                    level_candidates.append(candidate)
            
            # Merge level candidates into the current candidates
            current_candidates.extend(level_candidates)
        
        # Remove duplicates from candidates
        current_candidates = list(set(current_candidates))
        
        # Perform a final selection of the top K nearest neighbors from the bottom level
        if current_candidates:
            distances = [self._distance(query_embedding, node.embedding) for node in current_candidates]
            sorted_indices = np.argsort(distances)
            top_k_nodes = [current_candidates[i] for i in sorted_indices[:k]]
        else:
            top_k_nodes = []

        return top_k_nodes
