"""
Path.py

Desingned to manage the matcing part of the problem.
After matchings are done, will pass it to PathAlgorithms.py

Note
----
a. For single/parallel tweezer in classical, we use Linear Sum Assignments (modified Jonker-Volgenan and Hungarian).
b. Machine learning method should be good at matching problems ie. bipatie matching, graph neural networks,
    * I left some ports (PathGeneratorSemiClassical, PathGeneratorReinforcedLearning) for future developments just in case -- Chun-Wei

Contributors
------------
Chun-Wei Liu (cl3762)

"""

import numpy as np
import math
from scipy.optimize import linear_sum_assignment
from scipy.spatial import Delaunay

# Open source imports
import networkx as nx
from shapely.geometry import Point, Polygon

# Local imports
from lib.RearrangingAlgo.PathAlgoritms import *

class PathGeneratorSingle():
    def __init__(self, atom_points,
                 target_trap_points, reservior_trap_points, total_trap_points, graph, cost, parent=None):
        """
        Generate path via Linear sum assignments (LSAP)

        Problem definition:
            Pathfinding Problem

        Input:
            :param num_atoms: total atom number (int)
            :param atom_points: target trap points (np.array)
            :param target_trap_points: target trap points (np.array)
            :param reservior_trap_points: reservior trap points (np.array)
            :param total_trap_points: total trap points (np.array)
        
        Output:
            :param path_total: Non-collision path (list)
        """

        self.atom_points_list = atom_points.tolist()
        self.target_trap_points_list = target_trap_points.tolist()
        self.reservior_trap_points_list = reservior_trap_points.tolist()
        self.total_trap_points_list = total_trap_points.tolist()
        self.graph = graph
        self.distance_matrix = cost
        self.num_targets = len(self.target_trap_points_list)
        
        
    #############################
    #                           #
    #      Main Algolrithm      #
    #                           #
    #############################

    ## Stable
    def lsap_general_lattice(self):
        '''
        Main Algolrithm for general lattices
        '''
        ## Linear sum assignment, LSAP will generate atom-target matching 
        ## Transform target selection index to match total trap index
        self.target_selection, self.atom_selection, self.target_selection_index_in_total_traps = self.atom_target_matching_LSAP()

        #path_total = Path_NonCollision_GraphTheory(self.atom_points_list, self.atom_selection, self.target_selection_index_in_total_traps, self.total_trap_points_list).path()
        path_total = Path_NonCollision_GraphTheory_Update(self.atom_points_list, self.atom_selection, self.target_selection_index_in_total_traps, self.total_trap_points_list, self.graph).path()
        
        return path_total

    ## Stable
    def lsap_rectangular_lattice(self):
        '''
        Main Algolrithm for rectangular lattice
        '''
        ## Linear sum assignment, LSAP will generate atom-target matching 
        ## Transform target selection index to match total trap index
        self.target_selection, self.atom_selection, self.target_selection_index_in_total_traps = self.atom_target_matching_LSAP()

        ## LSAP is suppose to be universal
        gap_small = self.rectangular_lattice_configs()

        print("Path ordering...")
        
        path_total = Path_NonCollision_Search(gap_small, self.atom_points_list, self.atom_selection, self.target_selection_index_in_total_traps, self.total_trap_points_list).path()
    
        #path_total = Path_NonCollision_Search_update(gap_small, self.atom_points_list, self.atom_selection, self.target_selection_index_in_total_traps, self.total_trap_points_list).path()
        
        return path_total

    def one_d_lattice(self):
        self.target_selection, self.atom_selection, self.target_selection_index_in_total_traps = self.atom_target_matching_LSAP()
        path_total = Path_NonCollision_GraphTheory_Update(self.atom_points_list, self.atom_selection, self.target_selection_index_in_total_traps, self.total_trap_points_list, self.graph).path()

        return path_total

    def MAPF_tweezers(self):
        '''
        Main Algolrithm for general lattices with multiple AODs
        '''
        ## Linear sum assignment, LSAP will generate atom-target matching 
        ## Transform target selection index to match total trap index
        self.target_selection, self.atom_selection, self.target_selection_index_in_total_traps = self.atom_target_matching_LSAP()

        #path_total = Path_NonCollision_GraphTheory(self.atom_points_list, self.atom_selection, self.target_selection_index_in_total_traps, self.total_trap_points_list).path()
        path_total = Path_NonCollision_GraphTheory_MAPF(self.atom_points_list, self.atom_selection, self.target_selection_index_in_total_traps, self.total_trap_points_list, self.graph).path()
        
        return path_total

    # TODO: Need to migrate to parallel class
    # Not that stable, some knobs to fine tune
    def parallel_rectangular_lattice_flank(self):
        '''
        Main Algolrithm for general lattices, inspired by QuEra

        '''
        path_total = Path_NonCollision_Parallel_Sorting_Flank(self.atom_points_list, self.target_trap_points_list, self.reservior_trap_points_list, self.total_trap_points_list).path()

        return path_total
    #############################
    #                           #
    #           Tools           #
    #                           #
    #############################


    def atom_target_matching_LSAP(self):
        ## Atom-Target selections via linear sum assignment
        target_selection, atom_selection = linear_sum_assignment(self.distance_matrix)

        ## Target index within total trap configurations
        ## Since we now successfuly matches atoms to their corresponding target traps.
        target_selection_index_in_total_traps = []
        
        for target_selection_index in target_selection:
            target_selection_index_in_total_traps.append(self.total_trap_points_list.index(self.target_trap_points_list[target_selection_index]))

        return target_selection, atom_selection, target_selection_index_in_total_traps


    def rectangular_lattice_configs(self):
        ## The totol number of trap on one side
        num_trap_one_side = math.ceil(math.sqrt(2 * self.num_targets))

        if (num_trap_one_side - math.sqrt(self.num_targets)) % 2 != 0:
            num_trap_one_side += 1

        ## The gap here represents the small gap of each trap
        gap_small = 1 / (num_trap_one_side - 1)

        return gap_small
 
class PathGeneratorParallel():
    def __init__(self, atom_points, target_trap_points, reservior_trap_points, total_trap_points, atom_mesh, target_trap_mesh, reservior_trap_mesh, total_trap_mesh, graph_row, graph_column, row_index, column_index, parent=None):
        """
        Generate path via Parallel algorithms

        Problem definition:
            Pathfinding Problem

        Input:
            :param atom_points: target trap points (np.array)
            :param target_trap_points: target trap points (np.array)
            :param reservior_trap_points: reservior trap points (np.array)
            :param total_trap_points: total trap points (np.array)
        
        Output:
            :param path_total: Non-collision path (list)
        """
        self.atom_points_list = atom_points.tolist()
        self.target_trap_points_list = target_trap_points.tolist()
        self.reservior_trap_points_list = reservior_trap_points.tolist()
        self.total_trap_points_list = total_trap_points.tolist()
        self.atom_mesh = atom_mesh
        self.target_trap_mesh = target_trap_mesh
        self.reservior_trap_mesh = reservior_trap_mesh
        self.total_trap_mesh = total_trap_mesh
        self.graph_row = graph_row
        self.graph_column = graph_column
        self.row_index = row_index
        self.column_index = column_index
        
        
    #############################
    #                           #
    #      Main Algolrithm      #
    #                           #
    #############################

    def parallel_rectangular_lattice_compact(self):
        '''
        Main Algolrithm for general lattices, inspired by Tetris.
        '''
        path_total = Path_NonCollision_Parallel_Sorting_Compact(self.atom_points_list, self.atom_mesh, self.target_trap_mesh, self.reservior_trap_mesh, self.total_trap_mesh, self.graph_row, self.graph_column, self.row_index, self.column_index).path()

        return path_total

class PathGeneratorSemiClassical():
    def __init__(self, parent=None):
        """
        Generate path via Linear sum assignments (LSAP) and reinforced learning
        
        Problem Definition:
            Matching Problem

        Reeinforced learning:
            Use DQN to update the policy on atom - target matching policy
            based on "total sorting time".

        Classical Methods: 
            Already a pathfinding expert.

        :param : 
        """
        super().__init__(parent)

    def atom_target_matching(self):
        NotImplementedError()

    def atom_target_matching(self):
        NotImplementedError()

    def path_generating(self):
        NotImplementedError()

    def rearranging(self):
        NotImplementedError()

    def reforced_learning(self):
        NotImplementedError()

class PathGeneratorReinforcedLearning():
    def __init__(self, parent=None):
        """
        Generate path via reinforced learning

        Problem Definition:
            Pathfinding Problem
        :param : 
        """
        super().__init__(parent)

    def policy(self):
        NotImplementedError()

    