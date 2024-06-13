from statistics import median
from typing import Tuple, List
from collections import namedtuple


Point = namedtuple('Point', ['x', 'y'])
"""A class representing a single point. Attributes can be accessed like a tuple via the '[]' operators or
by name, i.e. point.x == p[0].

...

Attibutes
---------

x: float
    Value of the x coordinate of the Point.
y: float
    Value of the y coordinate of the Point.
"""

class Node:
    """Class representing a single node in the PriorityTree.
    Contains a median and a point.    
    ...

    Attributes
    ----------
    
    med: float
        Value of the median contained in the node.
    point: Point
        The Point contained in the Node.
    """

    def __init__(self, point: Point, med: float) -> None:
        self.point = point
        self.med = med

    def __repr__(self):
        return f'Node{self.point, self.med}'
    

INF = float('inf')
"""Constant for infinite value."""

INF_POINT = Point(-INF, -INF)
"""A point at negative infinity in both x and y directions. Used to represent non-existant points."""

INF_NODE = Node(INF_POINT, -INF)
"""A node with a negative inifnity point and median. Used to represent non-existant nodes."""

def from_points_help(result: List[Node], points: List[Point], start_index: int = 0) -> List[Node]:
    """Helper method for recursively constructing a priority tree (list of Nodes).

    Args:
        result (List[Node]): The current tree.
        points (List[Point]): The points that need to be contained in the tree.
        start_index (int): The index of the node we are currently located at.

    Returns:
        List[Node]: The constructed tree. The first argument.
    """
    if not points:
        return result
    first_point = points.pop(0)
    node = Node(first_point, first_point.x)
    n_nodes = len(result)
    for _ in range(start_index - n_nodes + 1):
        result.append(INF_NODE)
    result[start_index] = node
    if points:
        med = median(map(lambda point: point.x, points))
        node.med = med
        points_left = list(filter(lambda point: point.x <= med, points))
        points_right = list(filter(lambda point: point.x > med, points))
        result = from_points_help(result, points_left, start_index=2 * start_index + 1)
        result = from_points_help(result, points_right, start_index=2 * start_index + 2)
    return result


def from_points(points: List[Point]) -> List[Node]:
    """Function used to construct a priority tree from a list of Point.

    Args:
        points (List[Point]): A list of points that the priority tree will contain.
    
    Returns:
        List[Node]: The constructed tree.
    """
    if not points:
        return []
    points = list(sorted(points, key=lambda point: -point.y))
    result = []
    return from_points_help(result, points, 0)

def get_node(tree: List[Node], index: int) -> Node:
    """Get the node at the specified index in the tree or the INF_NODE if the node
    is not in the tree.

    Args:
        tree (List[Node]): The priority tree.
        index (int): Index of the node in the tree.

    Returns:
        Node: The node at the specified index or INF_NODE if such a node does not exist.
    """
    if index >= len(tree):
        return INF_NODE
    return tree[index]

def query_priority_subtree(tree: List[Node], index: int, limit_value: float) -> List[Point]:
    """Helper function for querying a subtree of the priority tree.

    Args:
        tree (List[Node]): The priority tree.
        index (int): The index of the starting node.
        limit_value (float): The limit of the y coordinate.
    
    Returns:
        List[Point]: Points from the subtree that are within the limit on the y coordinate.
    """
    result = []
    node = get_node(tree, index)
    if node is not None and node is not INF_NODE and node.point.y >= limit_value: # TODO: Replace the condition
        result.append(node.point)
        result += query_priority_subtree(tree, 2*index + 1, limit_value)
        result += query_priority_subtree(tree, 2*index + 2, limit_value)
    return result

def query(tree: List[Node], interval: Tuple[float, float], limit: float) -> List[Point]:
    """Query the priority tree for the points contained in a specified interval [x1, x2]x[y, inf].
    Meaning that there is an interval on the X coordinate, but there is no upper bound on the Y coordinate. 

    Args:
        tree (List[Node]): The priority tree.
        interval (Tuple[float, float]): The interval on the X coordinate, i.e. [x1, x2]
        limit (float): The lower limit of the Y coordinate value, i.e. [limit, infinity]

    Returns:
        List[Point]: A list of points contained within the interval [interval[0], interval[1]]x[limit, infinity], i.e. [x1, x2]x[y, infinity]
    """
    if not tree:
        return []
    result = []
    index = 0    
    node = get_node(tree, index)
    while node is not None and node is not INF_NODE and \
        (interval[0] > node.med or interval[1] < node.med):
        point = node.point
        if point.x >= interval[0] and point.x <= interval[1] and point.y >= limit:
            result.append(point)
    
        if interval[0] > node.med:
            index = 2*index + 2
        else:
            index = 2*index + 1
        
        node = get_node(tree, index)
            
    
    if node is None or node is INF_NODE:
        return result
    
    point = node.point
    if point.x >= interval[0] and point.x <= interval[1] and point.y >= limit:
        result.append(point)

    
    node_index = index
    index = 2 * node_index + 1
    node = get_node(tree, index)
    while node is not None and node is not INF_NODE and node.point.y >= limit:
        point = node.point
        if point.x >= interval[0] and point.x <= interval[1] and point.y >= limit:
            result.append(point)

        if node.med >= interval[0] and node.med <= interval[1]:
            result.extend(query_priority_subtree(tree, 2 * index + 2, limit))
            index = 2 * index + 1
        else:
            index = 2 * index + 2

        node = get_node(tree, index)
    
    index = 2 * node_index + 2
    node = get_node(tree, index)
    while node is not None and node is not INF_NODE and node.point.y >= limit:
        point = node.point
        if point.x >= interval[0] and point.x <= interval[1] and point.y >= limit:
            result.append(point)

        if node.med >= interval[0] and node.med <= interval[1]:
            result.extend(query_priority_subtree(tree, 2 * index + 1, limit))
            index = 2 * index + 2
        else:
            index = 2 * index + 1
        node = get_node(tree, index)
    return result




TEST_POINTS = [Point(-5.5, -5.0),
               Point(6.5, 2.0),
               Point(-3.5, 2.5),
               Point(4.5, 1.0),
               Point(2.0, -2.0),
               Point(-1.0, 4.0),
               Point(2.5, 0.5),
               Point(-0.5, -0.5),
               Point(1.0, 1.0),
               Point(-2.0, -2.5),
               Point(3.0, 4.0),
               Point(7.0, 0.0),
               Point(-4.5, -1.0),
               Point(3.5, -1.5),
               Point(5.5, -3.0),
               Point(3.5, 2.5),
               Point(6.0, 5.0),
               Point(-3.0, 0.0)]

tree = from_points(TEST_POINTS)
result = query(tree, (-1.5, 5.0), -1.75)

expected = [Point(x=-1.0, y=4.0), Point(x=1.0, y=1.0), Point(x=-0.5, y=-0.5), Point(x=3.0, y=4.0), Point(x=3.5, y=2.5), Point(x=2.5, y=0.5), Point(x=3.5, y=-1.5), Point(x=4.5, y=1.0)]

print(result)

assert sorted(result) == sorted(expected)