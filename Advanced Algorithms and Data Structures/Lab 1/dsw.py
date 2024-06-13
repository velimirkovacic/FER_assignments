from typing import Optional

class Node:
    """
    Class representing a single node of a binary tree containing integer values.
    ...
    Attributes
    ----------
    value: int
        Value stored in the node.
    parent: Node, optional
        Parent of the current node. Can be None.
    left: Node, optional
        Left child of the current node. Can be None.
    right: Node, optional
        Right child of the current node. Can be None.
    """
    def __init__(self, value) -> None:
        self.value = value
        self.parent = self.right = self.left = None
    
    def set_left_child(self, node) -> None:
        """
        Set the the left child of self to the given node.
        Sets the node's parent to self (if it is not None).
        Args:
            node (Node, optional): the node to set as the child.
        """
        self.left = node
        if node is not None:
            node.parent = self

    def set_right_child(self, node) -> None:
        """
        Set the the right child of self to the given node.
        Sets the node's parent to self (if it is not None).
        Args:
            node (Node, optional): the node to set as the child.
        """
        self.right = node
        if node is not None:
            node.parent = self
    
    def left_rotate(self, tree) -> None:
        """
        Left rotate the tree around given node.
        Args:
            tree (BinaryTree): The tree to rotate. 
        """
        right = self.right
        if not right:
            return
        parent = self.parent
        if parent:
            if parent.left == self:
                parent.set_left_child(right)
            if parent.right == self:
                parent.set_right_child(right)
        else:
            right.parent = None
            tree.root = right
        temp = right.left
        right.set_left_child(self)
        self.set_right_child(temp)
    
    def right_rotate(self, tree) -> None:
        """
        Right rotate the tree around given node.
        Args:
            tree (BinaryTree): The tree to rotate. 
        """
        left = self.left
        if not left:
            return
        parent = self.parent
        if parent:
            if parent.left == self:
                parent.set_left_child(left)
            if parent.right == self:
                parent.set_right_child(left)
        else:
            left.parent = None
            tree.root = left
        temp = left.right
        left.set_right_child(self)
        self.set_left_child(temp)

    
    def node_count(self) -> int:
        """
        Recursively count the number of child nodes.
        Returns:
            int: The number of nodes.
        """
        return 1 + (self.right.node_count() if self.right else 0) + (self.left.node_count() if self.left else 0)

    def __repr__(self) -> str:
        return "(" + str(self.value) + ")"


class BinaryTree:
    """
    Class repreesenting a binary tree, consisting of Nodes.
    ...
    Attributes
    ----------
    root : Node, optional
        the root node of the BinaryTree of type Node (or None)
    """
    def __init__(self, root: Node) -> None:
        self.root = root

    def right_backbone(self) -> None:
        curr = self.root
        while curr is not None:
            curr_left = curr.left
            if curr_left is not None:
                curr.right_rotate(self)
                curr = curr_left
            else:
                curr = curr.right
    
    def set_root(self, node: Optional[Node]) -> None:
        """
        Set the root of the tree to the provided node and set the node's parent to None (if the node is not None).
        Args:
            node (Node, optional): The Node object to set as the root (whose parent is set to None)
        """
        self.root = node
        if self.root is not None:
            self.root.parent = None
    
    def insert(self, value: int) -> bool:
        """
        Insert the given integer value into the tree at the right position.
        Args:
            value (int): The value to insert
        Returns:
            bool: True if the element was not already in the tree (insertion was successful), otherwise False.
        """
        node = self.root
        if node is None:
            self.set_root(Node(value))
            return True
        while node is not None:
            if value < node.value:
                if node.left is None:
                    node.set_left_child(Node(value))
                    break
                else:
                    node = node.left
            elif value > node.value:
                if node.right is None:
                    node.set_right_child(Node(value))
                    break
                else:
                    node = node.right
            else:
                return False
        return True

    def node_count(self) -> int:
        """
        Count the number of nodes in the tree. Return 0 if root is None.
        Returns:
            int: Number of nodes in the tree.
        """
        return self.root.node_count() if self.root else 0

    def __repr__(self) -> str:
        """
        Get the string representation of the Node.
        Returns:
            str: A string representation which can create the Node object.
        """
        
    
        return f"Node({self.root.value})"

    def ispisStabla(self):
        self.ispisRek(self.root, 0)


    def ispisRek(self, node, razina):
        if node != None:
            self.ispisRek(node.right, razina + 1)
            print(' ' * 4 * razina + '-> ' + str(node.value))
            self.ispisRek(node.left, razina + 1)

        

from math import floor, ceil, log, pow


def DSW(tree: BinaryTree) -> None:
    """
    Balances the binary tree using right backbone

    Args:
        tree (BinaryTree): The tree o balanse.
    """
    
    
    tree.right_backbone()

    current = tree.root

    n = tree.node_count()
    h = ceil(log(n + 1, 2))
    i = 2**(h - 1) - 1

    print(i)
    to_rotate = []
    for j in range(0, n - i):
        if current == None:
            break
        
        to_rotate += [current]

        if current.right != None:
            current = current.right.right
    
    print(to_rotate)

    for node in to_rotate:
        if node.right == None:
            continue
        right = node.right
        node.left_rotate(right)
        if node == tree.root:
            tree.root = right

    while i > 1:
        i = i//2
        to_rotate = []
        current = tree.root
        for j in range(0, i):
            if current == None:
                break
            
            to_rotate += [current]

            if current.right != None:
                current = current.right.right
        for j in range(0, i):
            node = to_rotate[j]
            if node.right == None:
                continue
            right = node.right

            node.left_rotate(right)
            if node == tree.root:
                tree.root = right
    
tree = BinaryTree(Node(2))

tree.insert(4)

tree.insert(9)
tree.insert(11)
tree.insert(13)
tree.insert(19)
tree.insert(27)

tree.right_backbone()
DSW(tree)

tree.ispisStabla()
