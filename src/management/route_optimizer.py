"""
Route Optimizer
Optimizes routes using Dijkstra, A*, and other algorithms
"""

import numpy as np
import heapq
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Node:
    """Graph node for routing."""
    id: int
    latitude: float
    longitude: float
    cost: float = float('inf')
    previous: Optional['Node'] = None


class RouteOptimizer:
    """
    Route optimization using various algorithms.
    """
    
    def __init__(self, algorithm: str = "astar"):
        """
        Initialize route optimizer.
        
        Args:
            algorithm: Algorithm to use (dijkstra, astar, bellman_ford)
        """
        self.algorithm = algorithm.lower()
        self.graph: Dict[int, List[Tuple[int, float]]] = {}  # Adjacency list
        self.nodes: Dict[int, Node] = {}
    
    def build_graph(
        self,
        nodes: List[Dict],
        edges: List[Tuple[int, int, float]]
    ):
        """
        Build graph from nodes and edges.
        
        Args:
            nodes: List of node dictionaries with id, latitude, longitude
            edges: List of (from_id, to_id, weight) tuples
        """
        # Create nodes
        for node_data in nodes:
            node = Node(
                id=node_data['id'],
                latitude=node_data['latitude'],
                longitude=node_data['longitude']
            )
            self.nodes[node.id] = node
        
        # Create adjacency list
        self.graph = {node_id: [] for node_id in self.nodes.keys()}
        for from_id, to_id, weight in edges:
            if from_id in self.graph and to_id in self.nodes:
                self.graph[from_id].append((to_id, weight))
        
        logger.info(f"Built graph with {len(self.nodes)} nodes and {sum(len(adj) for adj in self.graph.values())} edges")
    
    def _haversine_distance(
        self,
        lat1: float,
        lon1: float,
        lat2: float,
        lon2: float
    ) -> float:
        """Calculate Haversine distance in kilometers."""
        R = 6371  # Earth radius in km
        
        phi1 = np.radians(lat1)
        phi2 = np.radians(lat2)
        delta_phi = np.radians(lat2 - lat1)
        delta_lambda = np.radians(lon2 - lon1)
        
        a = (np.sin(delta_phi / 2) ** 2 +
             np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2) ** 2)
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        
        return R * c
    
    def _heuristic(self, node1: Node, node2: Node) -> float:
        """Heuristic function for A* algorithm."""
        return self._haversine_distance(
            node1.latitude, node1.longitude,
            node2.latitude, node2.longitude
        )
    
    def dijkstra(self, start_id: int, end_id: int) -> Optional[List[int]]:
        """
        Find shortest path using Dijkstra's algorithm.
        
        Args:
            start_id: Start node ID
            end_id: End node ID
        
        Returns:
            List of node IDs representing the path, or None if no path exists
        """
        if start_id not in self.nodes or end_id not in self.nodes:
            return None
        
        # Reset nodes
        for node in self.nodes.values():
            node.cost = float('inf')
            node.previous = None
        
        start_node = self.nodes[start_id]
        start_node.cost = 0
        
        # Priority queue: (cost, node_id)
        pq = [(0, start_id)]
        visited = set()
        
        while pq:
            current_cost, current_id = heapq.heappop(pq)
            
            if current_id in visited:
                continue
            
            visited.add(current_id)
            
            if current_id == end_id:
                # Reconstruct path
                path = []
                node = self.nodes[end_id]
                while node:
                    path.append(node.id)
                    node = node.previous
                return list(reversed(path))
            
            # Explore neighbors
            for neighbor_id, weight in self.graph.get(current_id, []):
                if neighbor_id in visited:
                    continue
                
                neighbor = self.nodes[neighbor_id]
                new_cost = current_cost + weight
                
                if new_cost < neighbor.cost:
                    neighbor.cost = new_cost
                    neighbor.previous = self.nodes[current_id]
                    heapq.heappush(pq, (new_cost, neighbor_id))
        
        return None  # No path found
    
    def astar(self, start_id: int, end_id: int) -> Optional[List[int]]:
        """
        Find shortest path using A* algorithm.
        
        Args:
            start_id: Start node ID
            end_id: End node ID
        
        Returns:
            List of node IDs representing the path, or None if no path exists
        """
        if start_id not in self.nodes or end_id not in self.nodes:
            return None
        
        # Reset nodes
        for node in self.nodes.values():
            node.cost = float('inf')
            node.previous = None
        
        start_node = self.nodes[start_id]
        end_node = self.nodes[end_id]
        start_node.cost = 0
        
        # Priority queue: (f_score, node_id)
        # f_score = g_score (cost) + h_score (heuristic)
        f_score = self._heuristic(start_node, end_node)
        pq = [(f_score, 0, start_id)]  # (f_score, g_score, node_id)
        visited = set()
        
        while pq:
            _, current_g, current_id = heapq.heappop(pq)
            
            if current_id in visited:
                continue
            
            visited.add(current_id)
            
            if current_id == end_id:
                # Reconstruct path
                path = []
                node = self.nodes[end_id]
                while node:
                    path.append(node.id)
                    node = node.previous
                return list(reversed(path))
            
            # Explore neighbors
            for neighbor_id, weight in self.graph.get(current_id, []):
                if neighbor_id in visited:
                    continue
                
                neighbor = self.nodes[neighbor_id]
                tentative_g = current_g + weight
                
                if tentative_g < neighbor.cost:
                    neighbor.cost = tentative_g
                    neighbor.previous = self.nodes[current_id]
                    
                    h_score = self._heuristic(neighbor, end_node)
                    f_score = tentative_g + h_score
                    
                    heapq.heappush(pq, (f_score, tentative_g, neighbor_id))
        
        return None  # No path found
    
    def find_optimal_route(
        self,
        origin: Tuple[float, float],
        destination: Tuple[float, float],
        traffic_data: Optional[Dict] = None
    ) -> Dict:
        """
        Find optimal route between two points.
        
        Args:
            origin: (latitude, longitude) of origin
            destination: (latitude, longitude) of destination
            traffic_data: Optional traffic data to consider
        
        Returns:
            Dictionary with route information
        """
        try:
            # Find nearest nodes to origin and destination
            start_id = self._find_nearest_node(origin[0], origin[1])
            end_id = self._find_nearest_node(destination[0], destination[1])
            
            if start_id is None or end_id is None:
                raise ValueError("Could not find nodes near origin or destination")
            
            # Find path using selected algorithm
            if self.algorithm == "dijkstra":
                path = self.dijkstra(start_id, end_id)
            elif self.algorithm == "astar":
                path = self.astar(start_id, end_id)
            else:
                path = self.astar(start_id, end_id)  # Default to A*
            
            if path is None:
                return {
                    'success': False,
                    'message': 'No path found'
                }
            
            # Calculate route metrics
            total_distance = 0
            route_coordinates = []
            
            for i in range(len(path) - 1):
                node1 = self.nodes[path[i]]
                node2 = self.nodes[path[i + 1]]
                distance = self._haversine_distance(
                    node1.latitude, node1.longitude,
                    node2.latitude, node2.longitude
                )
                total_distance += distance
                route_coordinates.append((node1.latitude, node1.longitude))
            
            # Add destination
            end_node = self.nodes[path[-1]]
            route_coordinates.append((end_node.latitude, end_node.longitude))
            
            result = {
                'success': True,
                'algorithm': self.algorithm,
                'path': path,
                'coordinates': route_coordinates,
                'distance_km': total_distance,
                'num_segments': len(path) - 1,
                'origin': origin,
                'destination': destination
            }
            
            logger.info(f"Found optimal route: {total_distance:.2f} km using {self.algorithm}")
            return result
            
        except Exception as e:
            logger.error(f"Error finding optimal route: {e}")
            return {
                'success': False,
                'message': str(e)
            }
    
    def _find_nearest_node(self, latitude: float, longitude: float) -> Optional[int]:
        """Find nearest node to given coordinates."""
        if not self.nodes:
            return None
        
        min_distance = float('inf')
        nearest_id = None
        
        for node_id, node in self.nodes.items():
            distance = self._haversine_distance(
                latitude, longitude,
                node.latitude, node.longitude
            )
            if distance < min_distance:
                min_distance = distance
                nearest_id = node_id
        
        return nearest_id

