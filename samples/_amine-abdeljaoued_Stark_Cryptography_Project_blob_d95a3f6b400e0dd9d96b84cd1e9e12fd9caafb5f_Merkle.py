#Author: amine abdeljaoued

from hashlib import sha256 


class Node:
    
    def __init__(self,left,value,right,label):
        self.left, self.right = left, right
        self.value = value
        self.label = label
            
    def __str__(self):
        return self.value

        

def is_pow2(n):
    if n==1:
        return True
    else:
        if n%2==1: return False
        return is_pow2(n/2)

def binaryTreePaths(root):
    #Source: https://stackoverflow.com/questions/41471115/find-all-root-to-leaf-paths-in-a-binary-tree-in-python
    if root is None: 
        return []
    if (root.left == None and root.right == None):
        return [root.label]
    return [root.label + '->'+ l for l in 
             binaryTreePaths(root.right) + binaryTreePaths(root.left)]
    
class MerkleTree:
    
    def __init__(self,data):
        self.root = self.build(data)
        
    def build(self,data):
        if not is_pow2(len(data)):
            while not is_pow2(len(data)):
                data.append(data[-1]) #Padding
        
        nodes = [Node(None,sha256(str(d).encode()).hexdigest(),None,str(d) ) for d in data]
        return self.build_rec(nodes)
        
    def build_rec(self,data):
        if len(data)==2:
            new_hash = sha256( (str(data[0].value) + str(data[1].value)).encode()).hexdigest()
            root = Node(data[0], new_hash ,data[1], data[0].label + data[1].label )
            return root
        else:
            new_nodes = []
            for i in range(0,len(data),2):
                new_nodes.append(Node(data[i], sha256((str(data[i].value) + str(data[i+1].value) ).encode()).hexdigest() , data[i+1], data[i].label + data[i+1].label)) 
            
            return self.build_rec(new_nodes)

    def get_paths(self):
        paths = binaryTreePaths(self.root)
        return [i.split('->') for i in paths]

            

#Test
""" data = [32,54,65,23,55,66,77,88]
tree = MerkleTree(data)
paths = tree.get_paths()

path,proof = merkle_proof(tree,paths,'54')
print("path: ",path)
print("proof: ",proof)
verify_proof(tree.root,path,proof) """


            