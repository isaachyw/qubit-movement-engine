"""
PathAlgorithms.py

The algorithm library that stores classical algorithms.

Contents
--------
1. Path_NonCollision_Parallel_Sorting_Compact (Fall 2022): **Class Object**
    (Stable) Break rectangular lattice into vertical and horizontal 1D lattices.

2. Path_NonCollision_GraphTheory_Update (Winter 2021): **Class Object**
    (Stable) The non-collision algorithm via Graph Theory + LSAP + reordering + split.

3. Path_NonCollision_Search (Fall 2021) : **Class Object**
    (Stable) The non-collision only for rectangular lattice.

4. Path_NonCollision_GraphTheory (Spring 2022): **Class Object**
    (Stable) The non-collision algorithm via Graph Theory + LSAP +  Exclude + reordering.

5. Path_NonCollision_Parallel_Sorting_Flank (Fall 2022): **Class Object**
    (Unstable) Break rectangular lattice into vertical and horizontal 1D lattices. 

5. Layer by layer or onion algorithm (Fall 2021):
    (In Github history) Not useful since we have LSAP and Parallel algos.

Contributors
------------
Chun-Wei Liu (cl3762)
"""
import numpy as np
import math
from scipy.spatial import Delaunay
from scipy.optimize import linear_sum_assignment


## Tool imports
import copy
import random
from itertools import permutations
import itertools

## Open sourse imports
from tqdm import tqdm
import networkx as nx

## Local lib imports
from lib.RearrangingAlgo import OutputData
from utils import check_folder

## Multi process import
from multiprocessing import Process, Manager
#import concurrent.futures as cf
#import matplotlib.pyplot as plt

import time

#####################################
#          Optimized Algo.          #
#   Use those two for experiments   #
#####################################
## Stable, and the most advansed version
## Works for all possible geometries, dimensions
## Use split to replace exclude
class Path_NonCollision_GraphTheory_Update(object):
    '''
    This is a general method that can generate non-collision path
    for all graph via graph theory.

    Attribut
    --------
    atom_points : lst
        The initial atom position.
    atom_selecttion : lst
        The atom-target matching expanded in the basis of (a1, a2, ... aj), 
        where aj is the atom index and j is target index.
    target_index_in_total : lst
        The target index expanded in the basis of overall trap data.
    graph : graph
        The graph object that contains edge and node information of 
        the overall trap.

    Note
    ----
    This is the final implementation of the "Enhanced defect free atomic array" paper, also called LSAP II.
    '''
    def __init__(self, atom_points, atom_selecttion, target_index_in_total,
                 total_trap_points, graph):
        self.atom_points = atom_points
        self.atom_selecttion = atom_selecttion
        self.total_trap_points = total_trap_points
        self.target_index_in_total = target_index_in_total
        self.graph = graph
 
    def reordering(self, path_total):
        '''
        A cleaner version of the reordering algorithm, including reordering and spliting.

        Note
        ----

        # Case 4 and 5: The surpreme case that require splits
        
            Case 4: If the begining and the final of a move is a subset of another path
            Case 5: If the target trap of current (following) move is in the following (current) move
                    at the same time. 

        # Case 1-3: Requires reordering 

        The initial trap in folling move in current move or target in current move in following move.
        
            Case 1: If the target is occupied
            Case 2: If the move encounters a filled trap
            Case 3: If the target is the subset of another move
        '''

        i = 0
        #print(f"initial path: {path_total}\n")
        solvable_counter = 0
        pbar = tqdm(total=len(path_total)-1)

        ## The main control loop
        while i < len(path_total) - 1:
        
            ## The flag that indicate all cases has passed
            ## If all passed, current move +1
            all_pass = True

            ## Inner loop for next move
            for re_idx, next_move in enumerate(path_total[i+1:]):
                
                ## Identify a non-solvable path
                if solvable_counter > len(path_total[i+1:]) + 2:
                    raise RuntimeError('Unable to obtain non-collision path via LSAP (split and reordering).')

                ## Entering main reordering algorithm
                current_move = path_total[i]

                ## Splitting cases can not resolved by reordering, thus must process pior
                ## Case 4-1, 4-2, 5
                ## Split case 1
                if (current_move[0] in next_move) and (current_move[-1] in next_move):
                    all_pass = False
                    if current_move[0] == current_move[1]: # Idle atoms
                        path_total[i] = next_move[:next_move.index(current_move[0])] + [current_move[0]]
                    else:
                        path_total[i] = next_move[:next_move.index(current_move[0])] + current_move
                    path_total[i + re_idx + 1] = next_move[next_move.index(current_move[0]):]
                    break

                ## Split case 2
                elif (next_move[0] in current_move) and (next_move[-1] in current_move):
                    all_pass = False
                    if next_move[0] == next_move[1]: # Idle atoms
                        path_total[i] = current_move[:current_move.index(next_move[0])] + [next_move[0]]
                    else:
                        path_total[i] = current_move[:current_move.index(next_move[0])] + next_move
                    path_total[i + re_idx + 1] = current_move[current_move.index(next_move[0]):]
                    break
                
                ## Split case 3
                elif (current_move[-1] in next_move) and (next_move[-1] in current_move):
                    all_pass = False
                    path_total[i + re_idx + 1] = next_move[:next_move.index(current_move[-1]) + 1]
                    path_total[i] = current_move[:current_move.index(next_move[-1]) + 1]
                    break
                
                ## Reordering cases
                ## Case 1, 2, 3
                elif (next_move[0] in current_move) or (current_move[-1] in next_move):
                    path_total.append(path_total.pop(i))
                    all_pass = False
                    solvable_counter += 1
                    break
                    
            # If the current move pass all the cases, then move on
            if all_pass:
                i += 1
                solvable_counter = 0
                pbar.update(1)

        pbar.close()
        return path_total

    
    def make_move_selection(self, graph):
        '''
        Taking the atom-target matching distance matrix and LSAP
        and making move selections that stores the path of each pairs.
        '''
        move_selection = []
        ## Generating the paths and select just one randomly
        ## The intention is to avoid the iter.product function
        for index, a_selection_index in enumerate(tqdm(self.atom_selecttion)):#enumerate(self.atom_selecttion):#
            
            ## I just choose position as basis
            start = tuple(self.atom_points[a_selection_index])
            end = tuple(self.total_trap_points[self.target_index_in_total[index]])
            
            # move_selection_each_pair = []
            
            if end == start:
                move_selection.append([list(start), list(end)])
            
            ## Since it's possible that we have multiple shortest path, we can randomly select one
            ## But dragged the algorithm
            # else:
            #     for p in nx.all_shortest_paths(graph, source=start, target=(end)):
            #         move_selection_each_pair.append([list(el) for el in p])
            #     move_selection.append(random.choice(move_selection_each_pair))

            ## For better performance, we just select the first one
            else:
                move_selection.append([list(move) for move in nx.shortest_path(graph, source=start, target=end)])

        return move_selection


    # Define path from distance metrix selection
    def path(self):
        '''
        Creating path from atom-target mapping distance matrix.
        And pass the total_path to reordering.
        '''
        ## Create the non-collision path based on atom configuration and graph
        print("Path libaray generating...")
        path_total = self.make_move_selection(self.graph)

        ## Call the reordering
        print("Path reordering...")
        path_total = self.reordering(path_total)

        return path_total

## Stable, and the most advansed version
## The parallel sorting algorithm for compact lattice
## In theory should work for all bravis lattices
class Path_NonCollision_Parallel_Sorting_Compact(object):
    """
    The tetris algorithm.
    """
    def __init__(self, atom_points_list, atom_mesh, target_trap_mesh, reservior_trap_mesh, total_trap_mesh, graph_row, graph_column, row_index, column_index):
        self.atom_points_list = atom_points_list ## The actual atom position
        self.atom_mesh = atom_mesh ## The meshed atom position
        self.target_trap_mesh = target_trap_mesh ## The meshed target trap position
        self.reservior_trap_mesh = reservior_trap_mesh ## The meshed reservior trap position
        self.total_trap_mesh = total_trap_mesh ## The meshed total trap position
        self.graph_row = graph_row ## The graph with edges only in row direction
        self.graph_column = graph_column ## The graph with edges only in column direction
        self.row_index = row_index ## A list of y direction spatial points in the mesh
        self.column_index = column_index ## A list of x direction spatial points in the mesh

        ## Atom reated
        ###
        ## Atom column in each row: dict = {row (from 0 to N): [cols]}
        self.atom_idx_row_basis = {row: np.argwhere(atom != 0).flatten().tolist() for row, atom in enumerate(self.atom_mesh)}
        ## Atom row in each column: dict = {col (from 0 to N): [row]}
        self.atom_idx_column_basis = {row: np.argwhere(atom != 0).flatten().tolist() for row, atom in enumerate(self.atom_mesh.transpose())}
        ## Atom column in each row: dict = {row (iff non_empty): [cols]}
        self.pure_atom_idx_row_basis = self.nonempty_dict(self.atom_idx_row_basis)

        ## Target related
        ###
        ## Target column in each row: dict = {row (from 0 to N): [cols]}
        self.target_idx_row_basis = {row: np.argwhere(trap != 0).flatten().tolist() for row, trap in enumerate(self.target_trap_mesh)}
        ## Target row in each column: dict = {col (from 0 to N): [row]}
        self.target_idx_column_basis = {row: np.argwhere(trap != 0).flatten().tolist() for row, trap in enumerate(self.target_trap_mesh.transpose())}
        ## Target column in each row: dict = {row (iff non_empty): [cols]}
        self.pure_target_idx_row_basis = self.nonempty_dict(self.target_idx_row_basis)
        ## Target row in each row: dict = {column (iff non_empty): [row]}
        self.pure_target_idx_column_basis = self.nonempty_dict(self.target_idx_column_basis)
        self.first_target_row = list(self.pure_target_idx_row_basis)[0]
        self.last_target_row = list(self.pure_target_idx_row_basis)[-1]

    def path(self):
        """
        The main funciton of the class. Will general non-collision path with parallism.
        Potentially better than QuEra.
        """
        path = []
        defect_col = []
        ## Horizontal Compression 
        ###
        ## Row by row, one can interpret it as a deviding atoms into blocks.
        initial_case = True
        for r, row in enumerate(self.row_index):
            row_path = []
            num_atom_row = len(self.atom_idx_row_basis[r])
            ## If no atom in that row, just pass
            if num_atom_row == 0:
                pass
            
            ## If non-empty row
            else:
                row_atom = self.atom_mesh[r, self.pure_atom_idx_row_basis[r]]

                ## Extract leftmost target column number
                if r not in list(self.pure_target_idx_row_basis):
                    target_row = self.nearst_non_empty_row(self.pure_target_idx_row_basis, r)
                else:
                    target_row = r
                target_num_row = len(self.target_idx_row_basis[target_row]) ## The total number of targets in each row
                
                ## Check initial case, will refresh the defect_column 
                if initial_case:
                    ## 1. Push atom to align with the leftmost target
                    ## 2. record defect_column
                    ## destination will record the atom position after row compression
                    if num_atom_row < target_num_row:
                        ## Accessing atom position, arrange atoms from leftmost target column
                        destination = self.total_trap_mesh[r, self.pure_target_idx_row_basis[target_row][:num_atom_row]]
                        ## The defect column numbers, the remaining unfilled target columns
                        defect_col = self.pure_target_idx_row_basis[target_row][num_atom_row:]
                        defect_col.sort()
                        initial_case = False
                    ## Atom number enough to cover all the targets in the row
                    else:
                        ## Accessing atom position, all targets are filled
                        destination = self.total_trap_mesh[r, self.target_idx_row_basis[target_row]]
                        ## There is no defects in this case
                        defect_col = []
                
                ## Not the initial case, will require special treatments
                else:
                    ## If the atom number in this row is larger than number of defects in the previous row
                    num_defect = len(defect_col)
                    if num_atom_row > num_defect:
                        ## If number of atoms in this row is no less than the target trap number in this row
                        if num_atom_row >= target_num_row:
                            ## Accessing atom position, all targets are filled 
                            destination = self.total_trap_mesh[r, self.target_idx_row_basis[target_row]]

                        ## If the number of atom in this row is less than the target trap number in this row
                        else:
                            ## Compute destination in this row, 
                            ## destination columns are (defects columns + targets avilible by the remaining number of atoms)
                            ## The next line is important, will always keep defects in the piority
                            destination_col = copy.deepcopy(defect_col)
                            i = 0
                            for t, t_col in enumerate(self.target_idx_row_basis[target_row]):
                                ## Break point if atoms are delpeted, the t that left dead is the index for the last filled target
                                if i + len(defect_col) > num_atom_row:
                                    break

                                if t_col not in defect_col:
                                    destination_col.append(t_col)
                                    i += 1

                            ## Record destination
                            destination = self.total_trap_mesh[r, [destination_col[:num_atom_row]]][0]

                            ## Update defect columns, are the empty targets columns + previous defects columns
                            ## Always keep the accending order and remove duplicated
                            unfilled_target = list(set(self.target_idx_row_basis[target_row]) - set(destination_col[:num_atom_row]))
                            defect_col = list(set(unfilled_target + defect_col))
                    ## If atom number in a row is not enough to fill all the defects
                    else:
                        ## Destinations are now the filled defect columns starting from the leftmost site.        
                        destination_col = defect_col[0:num_atom_row]
                        destination = self.total_trap_mesh[r, [destination_col]][0] # Bad indexing
                        unfilled_taget = filter(lambda x: x > destination_col[-1], self.target_idx_column_basis[target_row])
                        defect_col = list(unfilled_taget)

                ## Take the destination list and create minima cost path for this row with make_selection
                ## Reorder the path to get the actual non-collision
                cost_row = self.distance_matrix(destination, row_atom, r = 2)
                row_path = self.reordering(self.make_path(self.graph_row, row_atom, destination, cost_row))
            path += row_path

        ## Check point
        ###
        ## Check if we have enough atom in each column to assemble this column
        ## Begin by update atom after row compression
        self.update_atom(path)

        ## Then check column by column
        for i, (col, target_rows) in enumerate(self.pure_target_idx_column_basis.items()):
            if len(self.pure_atom_idx_column_basis[col]) < len(target_rows):
                # print("Atom number not enough to assemble column. Need LSAP. Abort.")
                raise RuntimeError("Atom number not enough to assemble column. Abort.")   

        ## Vertical compression
        ## Column by column, finally achieving a compressed array
        for i, (col, target_rows) in enumerate(self.pure_target_idx_column_basis.items()):
            ## Atoms are all the atoms in the corresponding columns
            column_atom = self.atom_mesh[self.pure_atom_idx_column_basis[col], col]

            ## Destinations are all the targets in corresponding columns
            destination = self.total_trap_mesh[target_rows, col]
            
            ## The usual matching and pathfinding
            cost_column = self.distance_matrix(destination, column_atom, r = 2)
            column_path = self.reordering(self.make_path(self.graph_column, column_atom, destination, cost_column))
            
            ## Accumulate the path
            path += column_path

        return path
            
    def make_path(self, graph, atom_point, destination_point, distance_matrix):
        """
        Linear sum assignment problem. Finding the minimum total cost to move atom to destination
        with the weight assigned by the distance_marix. Previous called make_selection.

        Note
        ----
        LSAP matching algorithm was previously done in higher level. Now it's the time to reconsider it.
        """
        ## LSAP
        target_selection, atom_selection = linear_sum_assignment(distance_matrix)

        ## Decide path
        move_selection = []

        ## Generating the paths and select just one randomly
        ## The intention is to avoid the iter.product function
        for index, a_selection_index in enumerate(tqdm(atom_selection)):#enumerate(self.atom_selecttion):#
            
            ## I just choose position as basis
            start = tuple(atom_point[a_selection_index])
            end = tuple(destination_point[target_selection[index]])
            
            if end == start:
                move_selection.append([list(start), list(end)])

            ## For better performance, we just select the first one
            else:
                move_selection.append([list(move) for move in nx.shortest_path(graph, source=start, target=end)])

        return move_selection

    def reordering(self, path_total):
        '''
        A cleaner version of the reordering algorithm, including reordering and spliting.

        Note
        ----

        # Case 4 and 5: The surpreme case that require splits
        
            Case 4: If the begining and the final of a move is a subset of another path
            Case 5: If the target trap of current (following) move is in the following (current) move
                    at the same time. 

        # Case 1-3: Requires reordering 

        The initial trap in folling move in current move or target in current move in following move.
        
            Case 1: If the target is occupied
            Case 2: If the move encounters a filled trap
            Case 3: If the target is the subset of another move
        '''

        i = 0
        #print(f"initial path: {path_total}\n")
        solvable_counter = 0
        pbar = tqdm(total=len(path_total)-1)

        ## The main control loop
        while i < len(path_total) - 1:
        
            ## The flag that indicate all cases has passed
            ## If all passed, current move +1
            all_pass = True

            ## Inner loop for next move
            for re_idx, next_move in enumerate(path_total[i+1:]):
                
                ## Identify a non-solvable path
                if solvable_counter > len(path_total[i+1:]) + 2:
                    raise RuntimeError('Unable to obtain non-collision path via LSAP (split and reordering).')

                ## Entering main reordering algorithm
                current_move = path_total[i]

                ## Splitting cases can not resolved by reordering, thus must process pior
                ## Case 4-1, 4-2, 5
                ## Split case 1
                if (current_move[0] in next_move) and (current_move[-1] in next_move):
                    all_pass = False
                    if current_move[0] == current_move[1]: # Idle atoms
                        path_total[i] = next_move[:next_move.index(current_move[0])] + [current_move[0]]
                    else:
                        path_total[i] = next_move[:next_move.index(current_move[0])] + current_move
                    path_total[i + re_idx + 1] = next_move[next_move.index(current_move[0]):]
                    break

                ## Split case 2
                elif (next_move[0] in current_move) and (next_move[-1] in current_move):
                    all_pass = False
                    if next_move[0] == next_move[1]: # Idle atoms
                        path_total[i] = current_move[:current_move.index(next_move[0])] + [next_move[0]]
                    else:
                        path_total[i] = current_move[:current_move.index(next_move[0])] + next_move
                    path_total[i + re_idx + 1] = current_move[current_move.index(next_move[0]):]
                    break
                
                ## Split case 3
                elif (current_move[-1] in next_move) and (next_move[-1] in current_move):
                    all_pass = False
                    path_total[i + re_idx + 1] = next_move[:next_move.index(current_move[-1]) + 1]
                    path_total[i] = current_move[:current_move.index(next_move[-1]) + 1]
                    break
                
                ## Reordering cases
                ## Case 1, 2, 3
                elif (next_move[0] in current_move) or (current_move[-1] in next_move):
                    path_total.append(path_total.pop(i))
                    all_pass = False
                    solvable_counter += 1
                    break
                    
            # If the current move pass all the cases, then move on
            if all_pass:
                i += 1
                solvable_counter = 0
                pbar.update(1)

        pbar.close()
        return path_total

    def distance_matrix(self, target_trap_points, atom_points, r = 2):
        ## Defining the distant matrix, this will assign an atom to a specific target trap.
        cost_matrix = np.zeros((target_trap_points.shape[0], atom_points.shape[0]))
        print("Generating distance matrix...")
        for t, trap in enumerate(target_trap_points):
            for a, atom in enumerate(atom_points): 
                ## Faster but questionable method
                cost_matrix[t][a] = abs(trap[0] - atom[0])**r + abs(trap[1] - atom[1])**r
        print("Done\n")

        return cost_matrix
    
    def update_atom(self, path_reservior):
        """
        Update atom location after evoluting the path.

        Parameters
        ----------
        path: list
            The path until now.
        """
        for move in path_reservior:
            self.atom_points_list[self.atom_points_list.index(move[0])] = move[-1]

        self.atom_mesh = self.trap_grid(self.atom_points_list, self.row_index, self.column_index)
        self.atom_idx_row_basis = {row: np.argwhere(atom != 0).flatten().tolist() for row, atom in enumerate(self.atom_mesh)}
        self.atom_idx_column_basis = {row: np.argwhere(atom != 0).flatten().tolist() for row, atom in enumerate(self.atom_mesh.transpose())}
        self.pure_atom_idx_row_basis = self.nonempty_dict(self.atom_idx_row_basis)
        self.pure_atom_idx_column_basis = self.nonempty_dict(self.atom_idx_column_basis)

    def trap_grid(self, data_location, row_index, column_index):
        """
        Based on the total trap location, regroup the specified trap data for better accessing data
        trough row and column indeces.

        Returns
        -------
        trap_grid: np.array
            A array with size equals to total trap size. The elements are non-zero if the trap attribute is correct.
            For example, a 1D array with 10 total traps.
            Target array: np.array([0, 0, 0, (0, 3), (0, 4), (0, 5), 0, 0, 0, 0])
            Reservior array: np.array([(0, 0), (0, 1), (0, 2), 0, 0, 0, (0, 6), (0, 7), (0, 8), (0, 9)])
        """
        ## Create trap grid
        trap_grid = np.zeros((len(row_index), len(column_index)), dtype=np.ndarray)


        for trap in data_location:
            trap_grid[row_index.index(trap[1])][column_index.index(trap[0])] = trap

        return trap_grid

    def nearst_non_empty_row(self, data_dict, current_row):
        return min(list(data_dict), key=lambda x:abs(x-current_row))

    def nonempty_dict(self, data_dict):
        non_empty_data = {}
        for i, (row, data) in enumerate(data_dict.items()):
            if len(data) != 0:
                non_empty_data[row] = data
        return non_empty_data


## Underconstruction
class Path_NonCollision_GraphTheory_MAPF(object):
    '''
    This is a general method that can generate non-collision path
    for all graph via graph theory. And a modified version to adapt Multi-agent
    Path Finding algorithms. As a initial trial, we are taking CBS algorithm to
    devide

    Attribut
    --------
    atom_points : lst
        The initial atom position.
    atom_selecttion : lst
        The atom-target matching expanded in the basis of (a1, a2, ... aj), 
        where aj is the atom index and j is target index.
    target_index_in_total : lst
        The target index expanded in the basis of overall trap data.
    graph : graph
        The graph object that contains edge and node information of 
        the overall trap.

    Note
    ----
    This is not exactly CBS algorithm, instead, it is a multi-agent adaption to the LSAPII.
    '''
    def __init__(self, atom_points, atom_selecttion, target_index_in_total,
                 total_trap_points, graph):
        self.atom_points = atom_points
        self.atom_selecttion = atom_selecttion
        self.total_trap_points = total_trap_points
        self.target_index_in_total = target_index_in_total
        self.graph = graph
    
    def make_move_selection(self, graph):
        '''
        Taking the atom-target matching distance matrix and LSAP
        and making move selections that stores the shortest path of each pairs.
        '''
        print("Path generating...")
        move_selection = []
        ## Generating the paths and select just one randomly
        ## The intention is to avoid the iter.product function
        for index, a_selection_index in enumerate(tqdm(self.atom_selecttion)):#enumerate(self.atom_selecttion):#
            
            ## I just choose position as basis
            start = tuple(self.atom_points[a_selection_index])
            end = tuple(self.total_trap_points[self.target_index_in_total[index]])
            
            # move_selection_each_pair = []
            
            if end == start:
                move_selection.append([list(start), list(end)])
            
            ## Since it's possible that we have multiple shortest path, we can randomly select one
            ## But dragged the algorithm
            # else:
            #     for p in nx.all_shortest_paths(graph, source=start, target=(end)):
            #         move_selection_each_pair.append([list(el) for el in p])
            #     move_selection.append(random.choice(move_selection_each_pair))

            ## For better performance, we just select the first one
            else:
                move_selection.append([list(move) for move in nx.shortest_path(graph, source=start, target=end)])
        print(f'Done\n')

        return move_selection

    def distribute_to_agent(self, path_set, num_agent):
        """
        Distribute the shortest path set to agents.
        
        param:
            path_set : list
                A 1D list of shortest path among atom and target sites.
            num_agent : int
                Avilible number of agents (AOD).

        Note:
            Current implementation is based on sequential chunk split. One consideration of
            this part will hope this step eventually cause less conflicts among agents.
        """
        ## The current implementation split a list into chuncks, one can access the path
        ## of each agent by path_agents[index of agent]
        chunck_size = math.ceil(len(path_set)/num_agent)
        for i in range(0, num_agent):    
            yield path_set[i*chunck_size:(i+1)*chunck_size]

    def reformat(self, agent_path):
        ''' Now we assume that the path is now non-collisional,
        ## we then no longer require the nested list structure
        ## and since there will be a lot of list conparison algo required
        ## we unzip the list and encode actions (g, d, m, w) into in to path
        ## WARNING: It is NOT TRUE that we have non-collision now, reasons are
        ## listed in the documentation of the code on github.
        #
        returns:
            path: list
                [[v, t0, 'g'], ..., [xn, tn, 'm'], [u, tm, 'd']] 
        '''

        # TODO: Encode time info in the above loop
        ## Encoding time information to each site
        # TODO: Remove idle atom
        grab_time = 2
        drop_time = 2
        move_time = 1
        path = []
        ## Initial time starts from 0
        time_total = 0
        for moves in agent_path:
            ## Skip the idel atoms
            if len(moves) == 1 and (moves[0] == moves[-1]):
                pass
            else:
                for s, site in enumerate(moves):
                    move_length = len(moves)
                    ## grab
                    if s == 0:
                        time_total += grab_time
                        path.append([site, time_total, 'g'])
                    ## drop
                    elif s == move_length - 1:
                        time_total += drop_time
                        path.append([site, time_total, 'd'])
                    else:
                        time_total += move_time
                        path.append([site, time_total, 'm'])
        return path

    def high_level_reordering(self, agent_path):
        """
        Check conflcts among agents, and if true, add wait operation by adding
        a "wait tolken" in the path of lower ranking agent(s). Be aware that the action
        "adding wait tolken" is not competible with low_level_reordering.

        Input : 
            path: list
                [agent1, agent2, ..., agentn]
        Output :
            path: list
                [agent1, agent2, ..., agentn]
        Note:
            The definition of high level conflict means different agents arrive
            at the same node at the same time. And in our current implementation
            the piority of agents is determined by their assigned index, agent with
            smaller index has higher piority.
        """
        ## Detect conflict
        ## Lower index agent have higher piority
        ## Comparing latter ones to the unpassed-highest-ranked agent n
        wait_time = 0.5
        agen_num = len(agent_path)
        for n in range(agen_num - 1):
            for main_site in agent_path[n]:
                ## Can be parallelize for each (path_total[n:])
                for m in range(n + 1, agen_num):
                    for s, site in enumerate(agent_path[m]):
                        if main_site[:2] == site[:2]:
                            agent_path[m].insert(s + 1, [site[0], site[1] + wait_time,'w'])
                            ## Add wait time to all the following moves
                            for i in range(s+2, len(agent_path[n+1])):
                                agent_path[m][i][1] += wait_time
            return agent_path

    def low_level_reordering(self, path):
        '''
        A cleaner version of the reordering algorithm, including reordering and spliting.
        And as a adapt version to the multi-agent algorithm, it is neccessary to embed time
        information into path at this step.
        
        We are able to implant time in a deeper loop, but for now, we just run this sequence in
        the very last.

        input:
            agent_path : list
                [[[u], [x1], [x2], ..., [v]], ...]
         
        Note
        ----
        Sec. A, Notes on data structure
            * Reordering best work with grouping move in path together since the basic element is move
            when reordering. 
            * But for time check, a plain 1D list works best since there is only compare action.
            
        Sec. B, Notes on reordering
        # Case 4 and 5: The surpreme case that require splits
        
            Case 4: If the begining and the final of a move is a subset of another path
            Case 5: If the target trap of current (following) move is in the following (current) move
                    at the same time. 

        # Case 1-3: Requires reordering 

        The initial trap in folling move in current move or target in current move in following move.
        
            Case 1: If the target is occupied
            Case 2: If the move encounters a filled trap
            Case 3: If the target is the subset of another move
        '''
        print(f'Reordering...')
        i = 0
        #print(f"initial path: {path_total}\n")
        solvable_counter = 0
        pbar = tqdm(total=len(path)-1)

        ## The main control loop
        while i < len(path) - 1:
        
            ## The flag that indicate all cases has passed
            ## If all passed, current move +1
            all_pass = True

            ## Inner loop for next move
            for re_idx, next_move in enumerate(path[i+1:]):
                
                ## Identify a non-solvable path
                if solvable_counter > len(path[i+1:]) + 2:
                    raise RuntimeError('Unable to obtain non-collision path via LSAP (split and reordering).')

                ## Entering main reordering algorithm
                current_move = path[i]

                ## Splitting cases can not resolved by reordering, thus must process pior
                ## Case 4-1, 4-2, 5
                ## Split case 1
                if (current_move[0] in next_move) and (current_move[-1] in next_move):
                    all_pass = False
                    if current_move[0] == current_move[1]: # Idle atoms
                        path[i] = next_move[:next_move.index(current_move[0])] + [current_move[0]]
                    else:
                        path[i] = next_move[:next_move.index(current_move[0])] + current_move
                    path[i + re_idx + 1] = next_move[next_move.index(current_move[0]):]
                    break

                ## Split case 2
                elif (next_move[0] in current_move) and (next_move[-1] in current_move):
                    all_pass = False
                    if next_move[0] == next_move[1]: # Idle atoms
                        path[i] = current_move[:current_move.index(next_move[0])] + [next_move[0]]
                    else:
                        path[i] = current_move[:current_move.index(next_move[0])] + next_move
                    path[i + re_idx + 1] = current_move[current_move.index(next_move[0]):]
                    break
                
                ## Split case 3
                elif (current_move[-1] in next_move) and (next_move[-1] in current_move):
                    all_pass = False
                    path[i + re_idx + 1] = next_move[:next_move.index(current_move[-1]) + 1]
                    path[i] = current_move[:current_move.index(next_move[-1]) + 1]
                    break
                
                ## Reordering cases
                ## Case 1, 2, 3
                elif (next_move[0] in current_move) or (current_move[-1] in next_move):
                    path.append(path.pop(i))
                    all_pass = False
                    solvable_counter += 1
                    break
                    
            # If the current move pass all the cases, then move on
            if all_pass:
                i += 1
                solvable_counter = 0
                pbar.update(1)

        pbar.close()
        print(f'Done.\n')
        return path

    # Define path from distance metrix selection
    def path(self):
        '''
        Creating path from atom-target mapping distance matrix.
        And pass the total_path to reordering.
        '''
        ## Create the non-collision path based on atom configuration and graph
        path_total = self.low_level_reordering(self.make_move_selection(self.graph))
        print(f'Reordered path: {path_total}')

        ## Path into chuncks and distribute to agents
        # TODO: migrate number of agent to ini
        num_agent = 2
        path_total = list(self.distribute_to_agent(path_total, num_agent))

        ## Embed time info.
        path_total = list(map(self.reformat, path_total))

        ## High-level conflict resolve outside agents
        print(f"Path reordering for {num_agent} agents")
        path_total = self.high_level_reordering(path_total)
        for agent, path in enumerate(path_total):
            print(f'Agent {agent}; path:{path}\n')

        return path_total

#####################################
#          Bookeeping Algo.         #
#   Use those for references        #
#####################################

## Stable, refering to version 2 in source code 
## working version for rectangular lattice
class Path_NonCollision_Search(object):
    '''
    Improving more on the readibility.
    An improvement to enhance efficiency, for rectangular lattice.

    Note
    ----
    For book keeping puposes, we leave this older version of the code here. In this iteration, we were
    using searching algorithms to excluse Case 4, the cases that requires splits. But it turns out that 
    if N > 300, it will be exteremely difficult for the algorithm to successed.
    '''
    def __init__(self, gap_small, atom_points, atom_selecttion,
                 target_index_in_total, total_trap_points):
        self.atom_points = atom_points
        self.atom_selecttion = atom_selecttion
        self.gap_small = gap_small
        self.total_trap_points = total_trap_points
        self.target_index_in_total = target_index_in_total

    def reordering(self,path_total):
        '''
        The reordering part. Will take the definite total path 
        and reordering to achieve non-colling path.
        '''
        # Need relorder total path
        move_index = 0
        #self.atom_points = [[0.0, 0.0], [0.3333333333333333, 0.6666666666666667], [0.3333333333333333, 0.0], [0.0, 1.0]]
        #path_total = [[[0.0, 0.0], [0.0, 0.3333333333333333], [0.3333333333333333, 0.3333333333333333]], [[0.0, 1.0], [0.0, 0.6666666666666667], [0.3333333333333333, 0.6666666666666667]], [[0.3333333333333333, 0.0], [0.3333333333333333, 0.3333333333333333], [0.6666666666666666, 0.3333333333333333]], [[0.3333333333333333, 0.6666666666666667], [0.6666666666666666, 0.6666666666666667]]]
        #print("atom_points:", self.atom_points)
        #print("path_total_initial:", path_total)
        # Set a psudo for reordering
        atom_points_psudo = copy.deepcopy(self.atom_points)
        path_total_initial = copy.deepcopy(path_total)
        #TODO: The path reordering should be in another class.
        #print("len(path_total)", len(path_total))
        print("Path reordering...")
        pbar = tqdm(total=len(path_total))
        while move_index < len(path_total):
            #print("move_index:", move_index)
            Move = path_total[move_index]
            caseIII_flag = False
            #print("Move index:", move_index)
            #print("Move", Move)
            # Case 3: If the trap is in the path of another move following the list
            #print("Move:", Move)
            #print("move_index:", move_index)
            if move_index + 1 < len(path_total):
                for next_Move in path_total[move_index + 1:]:
                    if caseIII_flag == True:
                        break
                    for nm in next_Move:
                        if (abs(nm[0] - Move[-1][0]) +
                                abs(nm[1] - Move[-1][1])) < 0.00000001:
                            #print("Move:", Move)
                            #print("nm:", nm)
                            #print("next_Move", next_Move)
                            #print("Case III")
                            #print("move_index:", move_index)
                            path_total.append(path_total.pop(move_index))
                            #print("path_total after Case III:", path_total)
                            caseIII_flag = True
                            break
            if caseIII_flag == True:
                continue
            # Case 1: If the target trap of the move is occupied
            if Move[-1] in atom_points_psudo:
                if len(Move) == 2 and (abs(Move[0][0] - Move[1][0]) +
                                       abs(Move[0][1] - Move[1][1]) <
                                       0.00000001):
                    #print("Case I, but ok")
                    move_index += 1
                    pbar.update(1)
                    continue
                else:
                    #print("Case I")
                    #print("move_index:", move_index)
                    #print("Move[-1]", Move[-1])
                    #print("atom_points", atom_points)
                    path_total.append(path_total.pop(move_index))
                    #print("path_total after Case I:", path_total)
                    continue
            # Case 2: If another trap along the path of the move is already filled
            caseII_flag = False
            for a in atom_points_psudo:
                if caseII_flag == True:
                    break
                for mv in Move[1:]:
                    if (abs(a[0] - mv[0]) + abs(a[1] - mv[1])) < 0.00000001:
                        #print("Case II")
                        #print("atom_points_initial:", atom_points)
                        #print("path_total_initial", path_total_initial)
                        #print("atom_points_psudo:", atom_points_psudo)
                        #print("path_total before:", path_total)
                        #print("move_index:", move_index)
                        path_total.append(path_total.pop(move_index))
                        #print("path_total after Case II:", path_total)
                        caseII_flag = True
                        break
            if caseII_flag == True:
                continue
            else:
                # Change the atom position if well sorted
                move_done = path_total[move_index]
                atom_points_psudo[atom_points_psudo.index(
                    move_done[0])] = move_done[-1]
                move_index += 1
                pbar.update(1)
        pbar.close()
        return path_total

    def make_tape_to_move(self, start, x_num, y_num, move_selection, move_tape):
        '''
        Taking the four-direction letter moving tape,
        and turn them into move_selection, a coordinate based tape.
        '''
        # This is a time comsuming computation
        element = random.choice(list(set(permutations(move_tape))))
        move = []
        move.append(start)
        for e, ele in enumerate(element):
            if ele == 'h':
                if x_num > 0:
                    move.append([move[-1][0] + self.gap_small, move[-1][1]])
                elif x_num < 0:
                    move.append([move[-1][0] - self.gap_small, move[-1][1]])
            elif ele == 'p':
                if y_num > 0:
                    move.append([move[-1][0], move[-1][1] + self.gap_small])
                elif y_num < 0:
                    move.append([move[-1][0], move[-1][1] - self.gap_small])
        move_selection.append(move) 

        return move_selection

    def make_move_tape(self, x_num, y_num, move_tape):
        '''
        Creating letter move tapes from 
        atom-target matching distance matrix.
        '''
        if x_num != 0:
            for horizontal in range(abs(x_num)):
                move_tape.append('h')
        if y_num != 0:
            for prependicular in range(abs(y_num)):
                move_tape.append('p')
        return move_tape
    
    def make_move_selection(self):
        '''
        Taking the atom-target matching distance matrix
        and making move selections that stores the path of each pairs.
        '''
        move_selection = []
        # Generating the paths and select just one randomly
        # The intention is to avoid the iter.product function
        for index, a_selection_index in enumerate(tqdm(self.atom_selecttion)):
            # I just choose position as basis
            start = self.atom_points[a_selection_index]
            end = self.total_trap_points[self.target_index_in_total[index]]
            #print("start:", start)
            #print("end:", end)
            x_num = round((end[0] - start[0]) / self.gap_small)
            y_num = round((end[1] - start[1]) / self.gap_small)
            move_tape = []
            # move_selection_each_pair = []
            if (x_num == 0) and (y_num == 0):
                move = []
                move.append(start)
                move.append(end)
                move_selection.append(move)
            # convert letters p and h into positions
            else:
                move_tape = self.make_move_tape(x_num, y_num, move_tape)
                move_selection = self.make_tape_to_move(start, x_num, y_num, move_selection, move_tape)
                  
        return move_selection

    def check_exclusion(self, move_selection):
        '''
        Check if the move selections do not contain any cases that will fail the
        reordering part.
        '''
        # The search algo. is here
        print("Check if selected the invalid path...")
        exclude = False
        for idx_pair, pair in enumerate(tqdm(move_selection)):
            for idx_pair_s, pair_s in enumerate(move_selection):
                cond_start = False
                cond_end = False
                if idx_pair_s == idx_pair:
                    continue
                else:
                    for idx_point_s, point_s in enumerate(pair_s):
                        if (abs(pair[0][0] - point_s[0]) +
                                abs(pair[0][1] - point_s[1])) < 0.00000001:
                            cond_start = True

                        if (abs(pair[-1][0] - point_s[0]) +
                                abs(pair[-1][1] -
                                    point_s[1])) < 0.00000001:
                            cond_end = True
                    if (cond_start == True) and (cond_end == True):
                        exclude = True
                        break
            if exclude == True:
                break
        return exclude

    def path(self):
        '''
        Creating path from atom-target mapping distance matrix.
        And pass the total_path to reordering.
        '''
        print("Path libaray generating...")
        pass_assigned = False
        while pass_assigned == False:
            move_selection = self.make_move_selection()

            avilible_selections = []

            # Return False if the path passes the test.
            exclude = self.check_exclusion(move_selection)
            
            if exclude == False:
                avilible_selections = move_selection

            if len(avilible_selections) != 0:
                print("Selected path passes the test.")
                pass_assigned = True
            else:
                print(
                    "Selected path failed the test, re-selecting initiate...")

        # Call the reordering
        path_total = avilible_selections

        path_total = self.reordering(path_total)
        return path_total

## Stable
## Quasi lattice/ Universal/ Best for close and compact target-reservior matching
class Path_NonCollision_GraphTheory(object):
    '''Parallel sorting for rectangular lattice. This algorithm requires a specific type of rectangular lattice configuration.

    Note
    ----
    A rough algorithm scheme that we gathered from QuEra's talk.
    # 1. Check the number of missing atoms in target trap horizontaly       
    # 2. Rearrange the reservior atoms to match the missing in target traps.
    # 3. Give total rearrange row by row 
    '''
    def __init__(self, atom_points, atom_selecttion, target_index_in_total,
                 total_trap_points, tri_total):
        self.atom_points = atom_points
        self.atom_selecttion = atom_selecttion
        self.total_trap_points = total_trap_points
        self.target_index_in_total = target_index_in_total
        self.tri_total = tri_total # array


    def find_neighbors(self, pindex, triang):
        '''
        Finding neighboring (NN) points in Delaunay triangulation
        For setting up the possible path.
        '''
        return triang.vertex_neighbor_vertices[
            1][triang.vertex_neighbor_vertices[0][pindex]:triang.
               vertex_neighbor_vertices[0][pindex + 1]]


    def add_edges_neighbrs(self, graph):
        ''' 
        Setup the graph for the entire trap collection
        By adding edges
        No diagonal edges
        '''
        #print("total trap points:", self.total_trap_points)
        for idx_initial, initial in enumerate(self.total_trap_points):
            neighbor_indices = self.find_neighbors(idx_initial, self.tri_total)
            for idx_final in neighbor_indices:
                initial = tuple(initial)
                final = tuple(self.total_trap_points[idx_final])
                
                # We don't want diagonal moves
                if (initial[0] - final[0] == 0) or (initial[1] - final[1] == 0):
                    distance = abs(initial[0] - final[0])**2 + abs(initial[1] - final[1])**2
                    #print(f'initial: {initial}')
                    #print(f'final: {final}\n')
                    graph.add_edge(initial, final, weight = distance)


    def add_edges(self, graph):
        ''' 
        Setup the graph for the entire trap collection
        By adding edges
        '''
        for idx_initial, initial in enumerate(self.total_trap_points):
            neighbor_indices = self.find_neighbors(idx_initial, self.tri_total)
            for idx_final in neighbor_indices:
                initial = tuple(initial)
                final = tuple(self.total_trap_points[idx_final])
                distance = abs(initial[0] - final[0])**2 + abs(initial[1] - final[1])**2
                graph.add_edge(initial, final, weight = distance)


    def dijsktra(self, graph, initial, end):
        '''
        The algo finding shortest path in graph theory
        shortest paths is a dict of nodes
        whose value is a tuple of (previous node, weight)
        '''
        shortest_paths = {initial: (None, 0)}
        current_node = initial
        visited = set()

        while current_node != end:
            visited.add(current_node)
            destinations = graph.edges[current_node]
            weight_to_current_node = shortest_paths[current_node][1]

            for next_node in destinations:
                weight = graph.weights[(current_node,
                                        next_node)] + weight_to_current_node
                if next_node not in shortest_paths:
                    shortest_paths[next_node] = (current_node, weight)
                else:
                    current_shortest_weight = shortest_paths[next_node][1]
                    if current_shortest_weight > weight:
                        shortest_paths[next_node] = (current_node, weight)

            next_destinations = {
                node: shortest_paths[node]
                for node in shortest_paths if node not in visited
            }
            if not next_destinations:
                return "Route Not Possible"
            # next node is the destination with the lowest weight
            current_node = min(next_destinations,
                               key=lambda k: next_destinations[k][1])

        # Work back through destinations in shortest path
        path = []
        while current_node is not None:
            path.append(current_node)
            next_node = shortest_paths[current_node][0]
            current_node = next_node
        # Reverse path
        path = path[::-1]
        return path

    # Reordering the path
    def reordering(self, path_total):
        '''
        The reordering part. Will take the definite total path 
        and reordering to achieve non-colling path.
        '''
        # Need relorder total path
        move_index = 0
        #self.atom_points = [[0.0, 0.0], [0.3333333333333333, 0.6666666666666667], [0.3333333333333333, 0.0], [0.0, 1.0]]
        #path_total = [[[0.0, 0.0], [0.0, 0.3333333333333333], [0.3333333333333333, 0.3333333333333333]], [[0.0, 1.0], [0.0, 0.6666666666666667], [0.3333333333333333, 0.6666666666666667]], [[0.3333333333333333, 0.0], [0.3333333333333333, 0.3333333333333333], [0.6666666666666666, 0.3333333333333333]], [[0.3333333333333333, 0.6666666666666667], [0.6666666666666666, 0.6666666666666667]]]
        #print("atom_points:", self.atom_points)
        #print("path_total_initial:", path_total)
        # Set a psudo for reordering
        atom_points_psudo = copy.deepcopy(self.atom_points)
        path_total_initial = copy.deepcopy(path_total)
        #TODO: The path reordering should be in another class.
        #print("len(path_total)", len(path_total))
        print("Path reordering...")
        pbar = tqdm(total=len(path_total))
        while move_index < len(path_total):
            #print("move_index:", move_index)
            Move = path_total[move_index]
            caseIII_flag = False
            #print("Move index:", move_index)
            #print("Move", Move)
            # Case 3: If the trap is in the path of another move following the list
            #print("Move:", Move)
            #print("move_index:", move_index)
            if move_index + 1 < len(path_total):
                for next_Move in path_total[move_index + 1:]:
                    if caseIII_flag == True:
                        break
                    for nm in next_Move:
                        if (abs(nm[0] - Move[-1][0]) +
                                abs(nm[1] - Move[-1][1])) < 0.00000001:
                            #print("Move:", Move)
                            #print("nm:", nm)
                            #print("next_Move", next_Move)
                            #print("Case III")
                            #print("move_index:", move_index)
                            path_total.append(path_total.pop(move_index))
                            #print("path_total after Case III:", path_total)
                            caseIII_flag = True
                            break
            if caseIII_flag == True:
                continue
            # Case 1: If the target trap of the move is occupied
            if Move[-1] in atom_points_psudo:
                if len(Move) == 2 and (abs(Move[0][0] - Move[1][0]) +
                                       abs(Move[0][1] - Move[1][1]) <
                                       0.00000001):
                    #print("Case I, but ok")
                    move_index += 1
                    pbar.update(1)
                    continue
                else:
                    #print("Case I")
                    #print("move_index:", move_index)
                    #print("Move[-1]", Move[-1])
                    #print("atom_points", atom_points)
                    path_total.append(path_total.pop(move_index))
                    #print("path_total after Case I:", path_total)
                    continue
            # Case 2: If another trap along the path of the move is already filled
            caseII_flag = False
            for a in atom_points_psudo:
                if caseII_flag == True:
                    break
                for mv in Move[1:]:
                    if (abs(a[0] - mv[0]) + abs(a[1] - mv[1])) < 0.00000001:
                        #print("Case II")
                        #print("atom_points_initial:", atom_points)
                        #print("path_total_initial", path_total_initial)
                        #print("atom_points_psudo:", atom_points_psudo)
                        #print("path_total before:", path_total)
                        #print("move_index:", move_index)
                        path_total.append(path_total.pop(move_index))
                        #print("path_total after Case II:", path_total)
                        caseII_flag = True
                        break
            if caseII_flag == True:
                continue
            else:
                # Change the atom position if well sorted
                move_done = path_total[move_index]
                atom_points_psudo[atom_points_psudo.index(list(
                    move_done[0]))] = move_done[-1]
                move_index += 1
                pbar.update(1)
        pbar.close()
        return path_total


    def check_exclusion(self, move_selection):
        '''
        Check if the move selections do not contain any cases that will fail the
        reordering part.
        '''
        # The search algo. is here
        print("Check if selected the invalid path...")
        exclude = False
        for idx_pair, pair in enumerate(tqdm(move_selection)):
            for idx_pair_s, pair_s in enumerate(move_selection):
                cond_start = False
                cond_end = False
                if idx_pair_s == idx_pair:
                    continue
                else:
                    for idx_point_s, point_s in enumerate(pair_s):
                        if (abs(pair[0][0] - point_s[0]) +
                                abs(pair[0][1] - point_s[1])) < 0.00000001:
                            cond_start = True

                        if (abs(pair[-1][0] - point_s[0]) +
                                abs(pair[-1][1] -
                                    point_s[1])) < 0.00000001:
                            cond_end = True
                    if (cond_start == True) and (cond_end == True):
                        exclude = True
                        break
            if exclude == True:
                break
        return exclude


    def make_move_selection(self, graph):
        '''
        Taking the atom-target matching distance matrix and LSAP
        and making move selections that stores the path of each pairs.
        '''
        move_selection = []
        # Generating the paths and select just one randomly
        # The intention is to avoid the iter.product function
        for index, a_selection_index in enumerate(tqdm(self.atom_selecttion)):
            # I just choose position as basis
            start = tuple(self.atom_points[a_selection_index])
            end = tuple(self.total_trap_points[self.target_index_in_total[index]])
            #print("start:", start)
            #print("end:", end)
            move_selection_each_pair = []
            if end == start:
                move_selection.append([list(start), list(end)])
            else:
                for p in nx.all_shortest_paths(graph, source=start, target=(end)):
                    move_selection_each_pair.append([list(el) for el in p])
                move_selection.append(random.choice(move_selection_each_pair))

        return move_selection


    def lattice_graph(self):
        '''
        Setup the graph for the entire trap collection
        '''
        graph = nx.Graph()

        # Setup the graph for the entire trap collection
        # By adding edges
        # self.add_edges(graph)
        
        # By adding NN edges (No diagonal path for rectangular lattice)
        self.add_edges_neighbrs(graph)
        #nx.draw(graph)
        #plt.show()
        return graph


    # Define path from distance metrix selection
    def path(self):
        '''
        Creating path from atom-target mapping distance matrix.
        And pass the total_path to reordering.
        '''
        # Defining graph
        print("Defining graph...")
        graph = self.lattice_graph()

        print("Path libaray generating...")
        pass_assigned = False
        while pass_assigned == False:

            move_selection = self.make_move_selection(graph)
            
            avilible_selections = []

            exclude = self.check_exclusion(move_selection)

            if exclude == False:
                avilible_selections = move_selection

            if len(avilible_selections) != 0:
                print("Selected path passes the test.")
                pass_assigned = True
            else:
                print(
                    "Selected path failed the test, re-selecting initiate...")

        # Call the reordering
        path_total = avilible_selections
        #print("path_total:", path_total)
        #print("atom points:", self.atom_points)
        # Quiasi Lattice need more conditions.
        path_total = self.reordering(path_total)
        return path_total

## Not Stable
## Can Work Parallel sorting inspired by QuEra, not ideal solution
class Path_NonCollision_Parallel_Sorting_Flank(object):
    '''# Parallel sorting for rectangular lattice
        #########################################################################
        # 1. Check the number of missing atoms in target trap horizontaly       #
        # 2. Rearrange the reservior atoms to match the missing in target traps.#
        # 3. Give total rearrange row by row                                    #
        #########################################################################

    '''
    def __init__(self, atom_points_list, target_trap_points_list, reservior_trap_points_list, total_trap_points_list):
        self.atom_points_list = atom_points_list
        self.target_trap_points_list = target_trap_points_list
        self.reservior_trap_points_list = reservior_trap_points_list
        self.total_trap_points_list = total_trap_points_list
        self.left_reservior, self.right_reservior = self.break_reservior()
        self.row_index, self.column_index = self.trap_indexes()
    
    ## Main function
    def path(self):
        """
        The main function for generating parallel path.
        """
        ## Overall path
        overall_path = []

        ## Create reservior graph
        reservior_graph = self.lattice_graph_reservior()

        ## Scanning through the initial configuration to get the row demand list
        row_demand = self.scan_traps()

        ## Based on the demand list, determine the donors and receivers
        donors, receivers = self.donor_reservior_match(self.target_grid, self.reservior_grid, self.atom_grid, row_demand)

        ## Baseed on the selection, get the cost matrix
        cost = self.donor_cost(donors, receivers, reservior_graph)

        ## Match each donors to a reveiver
        receivers_selection, donors_selection = linear_sum_assignment(cost)

        ## Obtain path and update atom location
        ## TODO: Tobe replace by parallel sorting
        path_reservior = self.make_move_selection(reservior_graph, donors, receivers, receivers_selection, donors_selection)
        self.update_reservior(path_reservior)
        path_reservior = self.reordering(self.add_idel_atoms(path_reservior))
        # print(f'path_reservior: {path_reservior}')

        ## Row graph for final sorting
        overall_row_graph = self.lattice_graph_pararell()

        ## Get atom and target grid data (duplicated somewhere)
        atom_grid = self.trap_grid(self.atom_points_list)
        target_grid = self.trap_grid(self.target_trap_points_list)

        ## Generate path for each row
        for r in range(len(self.row_index)):
            target = [t for t in target_grid[r] if t != 0]
            atom = [a for a in atom_grid[r] if a != 0]
            cost_row = self.row_cost(target, atom, overall_row_graph, r = 2)
            target_selection, atom_selection = linear_sum_assignment(cost_row)
            
            ## TODO: Tobe replace by parallel sorting
            path = self.make_move_selection(overall_row_graph, atom, target, target_selection, atom_selection)
            
            ## Single tweezer
            overall_path += self.reordering(path)
            
            ## For parallel tweezer sorting
            # overall_path.append(self.reordering(path))
        
        return path_reservior + overall_path
    
    #################
    # Graph related #
    #################

    def find_neighbors(self, pindex, triang):
        '''
        Finding neighboring points in Delaunay triangulation
        For setting up the possible path.

        Parameters
        ----------
        pindex: int
            The data point index.
        triang: object
            The calculated triangulation data.
        '''
        return triang.vertex_neighbor_vertices[
            1][triang.vertex_neighbor_vertices[0][pindex]:triang.
               vertex_neighbor_vertices[0][pindex + 1]]

    def add_edges_neighbrs(self, trap, graph):
        ''' 
        Setup the graph for the entire trap collection by adding edges.
        
        Parameters
        ----------
        trap: list, array
            The trap data that we want to add edges.
        graph: nx.graph object
            The graph object that we want to store the graph information.

        Note
        ----
        No diagonal edges
        '''

        for idx_initial, initial in enumerate(trap):
            neighbor_indices = self.find_neighbors(idx_initial, Delaunay(np.asarray(trap)))
            for idx_final in neighbor_indices:
                initial = tuple(initial)
                final = tuple(trap[idx_final])
                
                ## We don't want diagonal moves
                ## We favor more column moves in reserviors
                if (initial[0] - final[0] == 0):
                    graph.add_edge(initial, final, weight = 0.5)
                elif (initial[1] - final[1] == 0):
                    graph.add_edge(initial, final, weight = 1)

    def lattice_graph_reservior(self):
        '''
        Setup the graph for the reservior trap collection
        '''
        graph = nx.Graph()

        ## By adding edges
        ## Left reservior
        self.add_edges_neighbrs(self.left_reservior, graph)
        ## Right reservior
        self.add_edges_neighbrs(self.right_reservior, graph)

        return graph
    
    def lattice_graph_pararell(self):
        '''
        Overall traps, only row connections
        '''
        graph = nx.Graph()

        ## Setup the graph for the entire trap collection
        ## By adding edges
        ## all traps row connections
        total_trap_grid = self.trap_grid(self.total_trap_points_list)

        for r in range(len(self.row_index)):
            for c in range(len(self.column_index) - 1):
                initial = tuple(total_trap_grid[r][c])
                final = tuple(total_trap_grid[r][c + 1])
                distance = final[0] - initial[0]
                graph.add_edge(initial, final, weight = distance)

        return graph

    #################
    # Path related  #
    #################
    def path_reservior(self, receivers_selection, donors_selection):
        pass
    
    def make_move_selection(self, graph, donors, receivers, receivers_selection, donors_selection):
        '''
        Taking the atom-target matching distance matrix and LSAP
        and making move selections that stores the path of each pairs.
        
        
        Parameters
        ----------

        graph: nx.graph object
            The graph object that we stored all the graph topologu into.
        donors: list or array
            The preselected atoms in donor rows that we want to balance the atom number in each row.
            Will move to the corresponding receiver trap location.
        receivers: list or array
            The preselected traps in reserviors in order to balance the atom number in each row.
        receivers_selection: list, array
            The receiver matching from LSAP.
        donors_selection: list, array
            The donor matching from LSAP.
        '''
        move_selection = []
        ## Generating the paths and select just one randomly
        ## The intention is to avoid the iter.product function
        for index, d_selection_index in enumerate(tqdm(donors_selection)):
            
            ## I just choose position as basis
            start = tuple(donors[d_selection_index])
            end = tuple(receivers[receivers_selection[index]])
            
            # move_selection_each_pair = []
            
            if end == start:
                move_selection.append([list(start), list(end)])

            ## For better performance, we just select the first one
            else:
                move_selection.append([list(move) for move in nx.shortest_path(graph, source=start, target=end)])

        return move_selection

    def update_reservior(self, path_reservior):
        """
        Update atom location after evoluting the reservior path.

        Parameters
        ----------
        path_reservior: list
            The path for balancing the each row.
        """
        for move in path_reservior:
            self.atom_points_list[self.atom_points_list.index(move[0])] = move[-1]
    
    def add_idel_atoms(self, path_reservior):
        """
        Find out the atoms in reserviors that didn't move.

        Parameters
        ----------
        path_reservior: list
            The path for balancing the each row.
        """

        for a in self.atom_points_list:
            for move in path_reservior:

                if a in move[1:-1]:   
                    path_reservior.append([a, a])

        return path_reservior
    
    def reordering(self, path_total):
        '''
        A cleaner version of the reordering algorithm, including reordering and spliting.

        Note
        ----

        # Case 4 and 5: The surpreme case that require splits
        
            Case 4: If the begining and the final of a move is a subset of another path
            Case 5: If the target trap of current (following) move is in the following (current) move
                    at the same time. 

        # Case 1-3: Requires reordering 

        The initial trap in folling move in current move or target in current move in following move.
        
            Case 1: If the target is occupied
            Case 2: If the move encounters a filled trap
            Case 3: If the target is the subset of another move

        
        # TODO: Reordering might not be neccessary for 1D parallel sorting
        '''

        i = 0
        #print(f"initial path: {path_total}\n")
        solvable_counter = 0
        pbar = tqdm(total=len(path_total)-1)

        ## The main control loop
        while i < len(path_total) - 1:
        
            ## The flag that indicate all cases has passed
            ## If all passed, current move +1
            all_pass = True

            ## Inner loop for next move
            for re_idx, next_move in enumerate(path_total[i+1:]):
                
                ## Identify a non-solvable path
                if solvable_counter > len(path_total[i+1:]) + 2:
                    raise RuntimeError('Unable to obtain non-collision path via LSAP (split and reordering).')

                ## Entering main reordering algorithm
                current_move = path_total[i]

                ## Splitting cases can not resolved by reordering, thus must process pior
                ## Case 4-1, 4-2, 5
                ## Split case 1
                if (current_move[0] in next_move) and (current_move[-1] in next_move):
                    all_pass = False
                    if current_move[0] == current_move[1]: # Idle atoms
                        path_total[i] = next_move[:next_move.index(current_move[0])] + [current_move[0]]
                    else:
                        path_total[i] = next_move[:next_move.index(current_move[0])] + current_move
                    path_total[i + re_idx + 1] = next_move[next_move.index(current_move[0]):]
                    break

                ## Split case 2
                elif (next_move[0] in current_move) and (next_move[-1] in current_move):
                    all_pass = False
                    if next_move[0] == next_move[1]: # Idle atoms
                        path_total[i] = current_move[:current_move.index(next_move[0])] + [next_move[0]]
                    else:
                        path_total[i] = current_move[:current_move.index(next_move[0])] + next_move
                    path_total[i + re_idx + 1] = current_move[current_move.index(next_move[0]):]
                    break
                
                ## Split case 3
                elif (current_move[-1] in next_move) and (next_move[-1] in current_move):
                    all_pass = False
                    path_total[i + re_idx + 1] = next_move[:next_move.index(current_move[-1]) + 1]
                    path_total[i] = current_move[:current_move.index(next_move[-1]) + 1]
                    break
                
                ## Reordering cases
                ## Case 1, 2, 3
                elif (next_move[0] in current_move) or (current_move[-1] in next_move):
                    path_total.append(path_total.pop(i))
                    all_pass = False
                    solvable_counter += 1
                    break
                    
            # If the current move pass all the cases, then move on
            if all_pass:
                i += 1
                solvable_counter = 0
                pbar.update(1)

        pbar.close()
        return path_total
    ###################################
    # Atom - reservior selection      #
    # Should someday move to Path.py  #
    ###################################    

    def break_reservior(self):
        """Creating left and right reservior
        """
        left_reservior = []
        right_reservior = []
        column_index = []

        for target in self.target_trap_points_list:
            if target[1] not in column_index:
                column_index.append(target[1])

        for reservior in self.reservior_trap_points_list:
            if reservior[0] < min(column_index):
                left_reservior.append(reservior)
            else:
                right_reservior.append(reservior) 
        return left_reservior, right_reservior

    def trap_indexes(self):
        """
        Record the row and column position number
        """
        ## Regroup data base on row and column
        ## First need to scan through toatal trap data to get row and column index
        column_index = [] ## axis 1
        row_index = [] ## axis 0
        for trap in self.total_trap_points_list:
            if trap[0] not in column_index:
                column_index.append(trap[0])
            if trap[1] not in row_index: 
                row_index.append(trap[1])

        ## With accending sequence
        column_index.sort()
        row_index.sort()

        return row_index, column_index

    def trap_grid(self, data_location):
        """
        Based on the total trap location, regroup the specified trap data for better accessing data
        trough row and column indeces.
        """
        ## Create trap grid
        trap_grid = np.zeros((len(self.row_index), len(self.column_index)), dtype=np.ndarray)


        for trap in data_location:
            trap_grid[self.row_index.index(trap[1])][self.column_index.index(trap[0])] = trap

        return trap_grid

    def data_count_row(self, data_grid_row):
        """
        Calculate the non-empty in a row
        """
        count = 0
        for trap in data_grid_row:
            if trap != 0:
                count += 1
        return count

    def reservior_col_index(self, reservior):
        """
        Return left and right reservior column indeces.
        """
        left_reservior_col = []
        right_reservior_col = []
        left = True
        right = False

        for col in range(reservior.shape[1]):
            if left == True and any(reservior[:, col] != 0):
                left_reservior_col.append(col)
            elif right == True and any(reservior[:, col] != 0):
                right_reservior_col.append(col)

            if left == True and any(reservior[:, col] == 0):
                right = True
                left = False

        return left_reservior_col, right_reservior_col

    def iner_reservior_col_seq(self, left_reservior_col, right_reservior_col):
        """
        Return the inner reservior sequence starting from left wing.
        """
        iner_col_seq = []
        left_reservior_col.sort(reverse=True)
        left_data = iter(left_reservior_col)
        right_data = iter(right_reservior_col)

        for i in range(len(left_reservior_col) + len(right_reservior_col)):
            if i%2 == 0:
                iner_col_seq.append(next(left_data))
            else:
                iner_col_seq.append(next(right_data))

        return iner_col_seq

    def scan_traps(self):
        """
        Scaning through the entire array and obtain the row indicies and acceptor donor info.
        """
        self.atom_grid = self.trap_grid(self.atom_points_list)
        self.target_grid = self.trap_grid(self.target_trap_points_list)
        self.reservior_grid = self.trap_grid(self.reservior_trap_points_list)

        ## Comute the demand list of each row, indicating which row need donation or not
        ## demand = atom - target, so positive means it's a donor row, while negative means it's aquiring row
        row_demand = []
        for r in range(len(self.row_index)):
            row_demand.append(self.data_count_row(self.atom_grid[r]) - self.data_count_row(self.target_grid[r]))
        
        return row_demand

    def donor_cost(self, donors, recievers, reservior_graph):
        """
        Generate the cost matrix for donor and recievers
        """
        move_length_data = dict(nx.all_pairs_dijkstra_path_length(reservior_graph))
        cost = np.zeros((len(recievers), len(donors)))
        for r, reciever in enumerate(recievers):
            for d, donor in enumerate(donors):
                try:
                    cost[r][d] = (move_length_data[tuple(reciever)][tuple(donor)])**2
                except:
                    cost[r][d] = np.Infinity
        
        return cost

    def row_cost(self, target_trap_points, atom_points, graph, r = 2):
        """
        Cost for each 1D row
        """
        ## Defining the distant matrix, this will assign an atom to a specific target trap.
        move_length_data = dict(nx.all_pairs_dijkstra_path_length(graph))
        distant_matrix = np.zeros((len(target_trap_points), len(atom_points)))
        print("Generating distance matrix...")
        for t, trap in enumerate(target_trap_points):
            for a, atom in enumerate(atom_points): 
                # TODO: Might need to exclude the already occupied traps
                ## Slower but more secure method
                distant_matrix[t][a] = (move_length_data[tuple(trap)][tuple(atom)])**r

                ## Faster but questionable method
                # distant_matrix[t][a] = abs(trap[0] - atom[0])**r + abs(trap[1] - atom[1])**r
        return distant_matrix

    def avilable_donors(self, atoms, targets, row_demand):
        """
        Atoms in reserviors and in a positive demand rows are abilable donors.
        
        """
        donor_grid = atoms.copy()
        for index, atom in np.ndenumerate(atoms):
            if (targets[index[0]][index[1]] != 0) or (row_demand[index[0]] <= 0):
                donor_grid[index[0]][index[1]] = 0

        return donor_grid

    def avilable_recievers(self, reserviors, atoms, row_demand):
        """
        Reservior traps without donors and are in a negative demand rows are receivers 
        """
        recivers_grid = reserviors.copy()
        ## Delete atom location and non-receivers rows
        for index, atom in np.ndenumerate(atoms):
            if (row_demand[index[0]] >= 0) or (atoms[index[0]][index[1]] != 0):
                recivers_grid[index[0]][index[1]] = 0

        return recivers_grid

    def donor_reservior_match(self, targets, reserviors, atoms, row_demand):
        """
        The algorithm for presorting the reservior array to make sure that we have enough atoms in
        each row.
        """
        ## First decide avilible donor atoms and empty reservior traps
        avilable_donors = self.avilable_donors(atoms, targets, row_demand)
        avilable_receivers = self.avilable_recievers(reserviors, atoms, row_demand)

        ## Then decide receivers traps with in mind that donor num and recevers num in each wings should be identical
        column_choice = []

        for col in range(avilable_donors.shape[1]):
            if any(avilable_donors[:, col] != 0) and any(avilable_receivers[:, col] != 0):
                column_choice.append(col)


        left_res_col, right_res_col = self.reservior_col_index(reserviors)
        iner_reservior_seq = self.iner_reservior_col_seq(left_res_col, right_res_col)

        
        donors = []
        receivers = []
        i = 0
        run_limit = abs(sum([e for e in row_demand if e<0]))
        ## Main algo
        while any(ele < 0 for ele in row_demand):
            if i >= run_limit:
                raise RuntimeError("Need LSAP")
            else:
                ## Start from the most demand row
                selected_FLAG = False
                reciver_row = row_demand.index(min(row_demand))

                ## Searching from the most inner reservuir columns
                for col in iner_reservior_seq:

                    if selected_FLAG == True:
                        break
                    else:
                        if avilable_receivers[reciver_row][col] !=0:
                            ## Choosing from the nearby donor rows
                            donor_data = list(range(avilable_donors.shape[0]))
                            donor_data.sort(key = lambda x: abs(x - reciver_row))
                            for donor_row in donor_data:
                                if avilable_donors[donor_row][col] !=0:
                                    ## Append selected donor and recievers
                                    donors.append(avilable_donors[donor_row][col])
                                    receivers.append(avilable_receivers[reciver_row][col])
                                    ## Delete the selected donor and receivers from avilible data
                                    avilable_donors[donor_row][col] = 0
                                    avilable_receivers[reciver_row][col] = 0
                                    ## Update row demand data
                                    row_demand[donor_row] = row_demand[donor_row] - 1
                                    row_demand[reciver_row] = row_demand[reciver_row] + 1
                                    selected_FLAG  = True
                                    break
                                
            i += 1
        
        return donors, receivers

