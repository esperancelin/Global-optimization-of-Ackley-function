# -*- coding: utf-8 -*-
"""
MCTS to handle the splitting of the sample.
@author: rongh
"""

import random
import math
from copy import deepcopy

from meta import MCTSMeta
import samples as samples
import numpy as np

class Node:
    def __init__(self, splitpara, parent):
        self.splitpara = splitpara
        self.parent = parent
        self.N = 0
        self.Q = 0
        self.children = {}

    def add_children(self, children: dict) -> None:
        for child in children:
            self.children[child.splitpara] = child

    def value(self, explore: float = MCTSMeta.EXPLORATION):
        if self.N == 0:
            return 0 
        else:
            return self.Q / self.N + explore * math.sqrt(math.log(self.parent.N) / self.N)

class MCTS:
    def __init__(self, state,original_x,orignial_fom):
        self.root_state = deepcopy(state)
        self.root = Node(None, None)
        self.run_time = 0
        self.node_count = 0
        self.num_rollouts = 0
        self.original_x=original_x
        self.original_fom=orignial_fom
        
    def search(self):
        num_rollouts = 0
        while num_rollouts<MCTSMeta.n_itr:
            node, state = self.select_node()
            outcome = self.roll_out(state)
            
            self.back_propagate(node, outcome)
            num_rollouts += 1
            
        self.num_rollouts = num_rollouts


    def select_node(self) -> tuple:
        node = self.root
        state = deepcopy(self.root_state)

        while len(node.children) != 0:  ##########if there are nodes already, random choose the maximun valued node, or if it has not been visited
            children = node.children.values()
            max_value = max(children, key=lambda n: n.value()).value()
            max_nodes = [n for n in children if n.value() == max_value]
            node = random.choice(max_nodes)
            state.split_replace(n_cluster=2,random_state=node.splitpara)

            if node.N == 0:
                return node, state

        if self.expand(node, state):#########if no children expand XX children, choose a randon node 
            node = random.choice(list(node.children.values()))
            state.split_replace(n_cluster=2,random_state=node.splitpara)

        return node, state  ##########return the selected node and state

    def expand(self, parent: Node, state) -> bool:######################expand the node XX times and return a boolean
        if not state.splitable():
            return False

        children = [Node(np.random.randint(100), parent) for i in range(MCTSMeta.n_children)]   ##################expand children 
        parent.add_children(children)
        return True  

    def roll_out(self, state):
        
        while state.splitable():
            state.split_replace(n_cluster=2,random_state=np.random.randint(100))
        est_fom=state.get_outcome()
        return est_fom
        

    def back_propagate(self, node: Node, outcome) -> None:
        reward = outcome
        while node is not None:
            node.N += 1
            node.Q += reward
            node = node.parent


    def suggest_best_move(self):
        if not self.root_state.splitable():
            return -1

        max_value = max(self.root.children.values(), key=lambda n: n.N).N
        max_nodes = [n for n in self.root.children.values() if n.N == max_value]
        best_child = random.choice(max_nodes)

        return best_child.splitpara

    def update(self, splitpara):
        if splitpara in self.root.children:
            self.root_state.split_replace(n_cluster=2,random_state=splitpara)
            self.root = self.root.children[splitpara]
            return

        self.root_state.split_replace(n_cluster=2,random_state=splitpara)
        self.root = Node(None, None)


