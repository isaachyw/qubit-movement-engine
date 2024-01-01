"""
Geometry.py

Creating the trap geometry graph and related tools from simulation or real atom flouresence images.

Contents
--------
GeometeryGenerator: Geometry template
AtomGenerators: Generate atom positions
TargetTrapGenerators: Generate target trap positions
ReserviorTrapGenerators: Generate reservior trap positions
LoadingSaved: Loading geometry template

Contributors
------------
Chun-Wei Liu (cl3762)
"""

## General import
import os
import time
import math
import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d, Delaunay
from sklearn import preprocessing

## Opensource packages
from shapely.geometry import Point, LineString
from shapely.geometry.polygon import Polygon
import networkx as nx

# import metis
# import nxmetis
from tqdm import tqdm

## Local imports
from lib.TrapGeometry.penrose import PenroseP3, BtileS
from lib.RearrangingAlgo import OutputData


class GeometeryGenerator(object):
    """
    A template painted by atoms, target traps, reservior traps. 

    Attrubute
    ---------
    num_atoms :  int
        Total atom number.
    freespace_dim : int
        Indicate the dimension of lattice points.
    d_m : float
        The required minimum physical distance.
    graph : graph
        Creating a graph template

    Notes
    -----
    a. The position data generated here is not meshed, its not in the format of row and column for bravais lattice.
    They are in cartesian coordinates. If one wish to implement some parallel sorting algorithm, please take a look at the "bravais_mesh" method.

    """
    def __init__(self, num_targets):
        self.num_targets = num_targets
        self.freespace_dim = 2
        self.d_m = 0.001
        ## Undirectional Graph
        self.graph = nx.Graph()
        

    def arbitrary_lattice(self):
        """Create an arbitraray lattice, trap, atom locations, and graph object.
        """
        target_trap_points, vor = TargetTrapGenerators(self.num_targets).generate_arbitrary()
        print('Target trap position assigned.\n')

        reservior_trap_points = ReserviorTrapGenerators(self.num_targets, target_trap_points, vor, self.d_m).generate_arbitrary()
        print('Reservior trap position assigned.\n')

        total_trap_points = self.total_trap_points(target_trap_points, reservior_trap_points)
        self.num_atoms = math.ceil(0.5 * len(total_trap_points))

        atom_points = AtomGenerators(self.num_atoms,
                                    target_trap_points,
                                    reservior_trap_points).generate()
        print('Atom distributed.\n')
        
        
        
        ## Create graph
        self.add_edges(self.graph, target_trap_points, reservior_trap_points, total_trap_points, total_trap_points)

        ## Obatain length between all traps
        move_length_data = dict(nx.all_pairs_dijkstra_path_length(self.graph))
        
        ## Distance matrix
        cost_matrix = self.distance_matrix(move_length_data, target_trap_points, atom_points, r = 2)

        return atom_points, target_trap_points, reservior_trap_points, total_trap_points, self.graph, cost_matrix

    def generate_chain_lattice(self):
        """Create an 1D chain lattice, trap, atom locations, and graph object.
        Atom points are now preset test data.
        """
        
        target_trap_points, vor = TargetTrapGenerators(self.num_targets).generate_chain_lattice()
        print('Target trap position assigned.\n')

        reservior_trap_points = ReserviorTrapGenerators(self.num_targets, target_trap_points, vor, self.d_m).generate_chain_lattice()
        print('Reservior trap position assigned.\n')

    
        total_trap_points = self.total_trap_points(target_trap_points, reservior_trap_points)
        
        ## Number of atoms
        self.num_atoms = math.ceil(0.5 * len(total_trap_points))

        ## Get atom location
        # atom_points = AtomGenerators(self.num_atoms,
        #                             target_trap_points,
        #                             reservior_trap_points).generate()

        ## For demo
        atom_points = np.array([[0, 0], [0.1, 0], [0.5, 0], [0.7, 0], [0.8, 0]])
        print('Atom distributed.\n')

        ## Graph
        self.add_edges_one_d(self.graph, target_trap_points, reservior_trap_points, total_trap_points)

        ## Obatain length between all traps
        move_length_data = dict(nx.all_pairs_dijkstra_path_length(self.graph))
        
        ## Distance matrix
        cost_matrix = self.distance_matrix(move_length_data, target_trap_points, atom_points, r = 2)

        return atom_points, target_trap_points, reservior_trap_points, total_trap_points, self.graph, cost_matrix
        
    def rectangular_lattice(self):
        """
        Create an rectangular lattice, trap, atom locations, and graph object. With compact target-reservior scheme. Target traps are surrounded by reservior traps.
        """

        target_trap_points, vor = TargetTrapGenerators(self.num_targets).generate_rectangular_lattice()
        print('Target trap position assigned.\n')

        reservior_trap_points = ReserviorTrapGenerators(self.num_targets, target_trap_points, vor, self.d_m).generate_rectangular_lattice()
        print('Reservior trap position assigned.\n')

        ## Obtain overall trap location
        total_trap_points = self.total_trap_points(target_trap_points, reservior_trap_points)
        
        ## Number of atoms
        self.num_atoms = math.ceil(0.5 * len(total_trap_points))

        ## Rescle data to better fit the AWG card calculation                                   67, 117
        scale, center_data, center_target = self.rescale_array_parameters(total_trap_points, target_min = 70, target_max = 90)
        target_trap_points = self.rescale_array(target_trap_points, scale, center_data, center_target)
        reservior_trap_points = self.rescale_array(reservior_trap_points, scale, center_data, center_target)
        total_trap_points = self.total_trap_points(target_trap_points, reservior_trap_points)
        
        ## Get atom location
        atom_points = AtomGenerators(self.num_atoms,
                                    target_trap_points,
                                    reservior_trap_points).generate()
        print('Atom distributed.\n')

        ## Create graph 
        self.add_edges_nrst_neighbrs(self.graph, target_trap_points, reservior_trap_points, total_trap_points)

        ## Obatain length between all traps
        move_length_data = dict(nx.all_pairs_dijkstra_path_length(self.graph))
        
        ## Distance matrix
        cost_matrix = self.distance_matrix(move_length_data, target_trap_points, atom_points, r = 2)

        return atom_points, target_trap_points, reservior_trap_points, total_trap_points, self.graph, cost_matrix

    def rectangular_lattice_from_image(self, target_trap_points, reservior_trap_points, total_trap_points):
        """Create an rectangular lattice, trap, atom locations, and graph object from real image.
        The major difference was that rectangular lattice requires nearest neighbor hopping.
        """

        ## Number of atoms
        self.num_atoms = math.ceil(0.5 * len(total_trap_points))

        atom_points = AtomGenerators(self.num_atoms, target_trap_points, reservior_trap_points).generate()
        print('Atom distributed.\n')

        ## Without diagonal moves
        self.add_edges_nrst_neighbrs(self.graph, target_trap_points, reservior_trap_points, total_trap_points)
        
        ## Obatain length between all traps
        move_length_data = dict(nx.all_pairs_dijkstra_path_length(self.graph))
        
        ## Distance matrix
        cost_matrix = self.distance_matrix(move_length_data, target_trap_points, atom_points, r = 2)

        return atom_points, target_trap_points, reservior_trap_points, total_trap_points, self.graph, cost_matrix

    def quasi_lattice(self):
        """Create an quasi lattice, trap, atom locations, and graph object.
        """
        target_trap_points, vor = TargetTrapGenerators(self.num_targets).generate_quasi_lattice()
        print('Target trap position assigned.\n')

        reservior_trap_points = ReserviorTrapGenerators(self.num_targets, target_trap_points,vor, self.d_m).generate_quasi_lattice()
        print('Reservior trap position assigned.\n')

        total_trap_points = self.total_trap_points(target_trap_points, reservior_trap_points)

        ## Number of atoms
        self.num_atoms = math.ceil(0.5 * len(total_trap_points))

        atom_points = AtomGenerators(self.num_atoms,
                                    target_trap_points,
                                    reservior_trap_points).generate()
        print('Atom distributed.\n')

        
        # Create graph
        self.add_edges(self.graph, target_trap_points, reservior_trap_points, total_trap_points)

        # Obatain length between all traps
        move_length_data = dict(nx.all_pairs_dijkstra_path_length(self.graph))
        
        # Distance matrix
        cost_matrix = self.distance_matrix(move_length_data, target_trap_points, atom_points, r = 2)

        return atom_points, target_trap_points, reservior_trap_points, total_trap_points, self.graph, cost_matrix
    
    # Implementing parallel algo inspired by QuEra
    def rectangular_lattice_parallel_flank(self):
        """Create an rectangular lattice, trap, atom locations, and graph object.
        Parrallel lattice requires different reservior patterns, ie. RTR.

        Note
        ----
        Knobs to play with:
        a. Total atom number, the first argument of the AtomGenerators
        b. reservior_volume in the TargetTrapGenerators.generate_rectangular_lattice_parallel
        c. reservior_volume in the ReservuirTraoGeberators.generate_rectangular_lattice_parallel
        reservior_volume should be the same in those two modules

        Warnings
        --------
        The Graph processing part is in algorithm module, which is not ideal.
        """
        target_trap_points, vor = TargetTrapGenerators(self.num_targets).generate_rectangular_lattice_parallel()
        print(f'Target trap position assigned: {target_trap_points.shape[0]}\n')

        reservior_trap_points = ReserviorTrapGenerators(self.num_targets, target_trap_points, vor, self.d_m).generate_rectangular_lattice_parallel()
        print(f'Reservior trap position assigned: {reservior_trap_points.shape[0]}.\n')

        total_trap_num = target_trap_points.shape[0] + reservior_trap_points.shape[0]
        
        ## Number of atoms
        self.num_atoms = math.ceil(total_trap_num/3)
        
        atom_points = AtomGenerators(self.num_atoms,
                                    target_trap_points,
                                    reservior_trap_points).generate()
        print('Atom distributed.\n')


        return atom_points, target_trap_points, reservior_trap_points, self.total_trap_points(target_trap_points, reservior_trap_points), 0, 0

    def rectangular_lattice_parallel_compact(self):
        """
        Just compact array configuration just as the rectangular_lattice one.
        In this version,
        # TODO: Will need to integrate this into rectangular_lattice after everything is tested.

        Notes
        -----
        a. It will be ideal if we can leave graph infromation inside the Geometry module.
        b. It will change how we are dealing with the algorithm, not sure which method (mesh, cartesian)is actually better.

        """
        target_trap_points, vor = TargetTrapGenerators(self.num_targets).generate_rectangular_lattice()
        print('Target trap position assigned.\n')

        reservior_trap_points = ReserviorTrapGenerators(self.num_targets, target_trap_points, vor, self.d_m).generate_rectangular_lattice()
        print('Reservior trap position assigned.\n')

        ## Obtain overall trap location
        total_trap_points = self.total_trap_points(target_trap_points, reservior_trap_points)
        
        ## Rescle data to better fit the AWG card calculation                                   67, 117
        scale, center_data, center_target = self.rescale_array_parameters(total_trap_points, target_min = 72, target_max = 118)
        target_trap_points = self.rescale_array(target_trap_points, scale, center_data, center_target)
        reservior_trap_points = self.rescale_array(reservior_trap_points, scale, center_data, center_target)
        total_trap_points = self.total_trap_points(target_trap_points, reservior_trap_points)

        ## Number of atoms
        self.num_atoms = math.ceil(0.5 * len(total_trap_points))

        ## Get atom location
        atom_points = AtomGenerators(self.num_atoms,
                                    target_trap_points,
                                    reservior_trap_points).generate()
        print('Atom distributed.\n')

        ## For the sake of pallelism, we use row-column basis data
        total_trap_mesh, target_trap_mesh, reservior_trap_mesh, atom_mesh, row_index, column_index = self.bravais_mesh(target_trap_points, reservior_trap_points, total_trap_points, atom_points)

        
        ## Graph for parallel
        graph_row, graph_column = self.lattice_graph_pararell(total_trap_mesh, row_index, column_index)

        return atom_points, target_trap_points, reservior_trap_points, total_trap_points, atom_mesh, target_trap_mesh, reservior_trap_mesh, total_trap_mesh, graph_row, graph_column, row_index, column_index

    def rectangular_lattice_partition(self, num_agent = 2):
        target_trap_points, vor = TargetTrapGenerators(self.num_targets).generate_rectangular_lattice()
        print('Target trap position assigned.\n')

        reservior_trap_points = ReserviorTrapGenerators(self.num_targets, target_trap_points, vor, self.d_m).generate_rectangular_lattice()
        print('Reservior trap position assigned.\n')

        ## Obtain overall trap location
        total_trap_points = self.total_trap_points(target_trap_points, reservior_trap_points)
        
        ## Number of atoms
        self.num_atoms = math.ceil(0.5 * len(total_trap_points))

        ## Rescle data to better fit the AWG card calculation                                   67, 117
        scale, center_data, center_target = self.rescale_array_parameters(total_trap_points, target_min = 76, target_max = 82)
        target_trap_points = self.rescale_array(target_trap_points, scale, center_data, center_target)
        reservior_trap_points = self.rescale_array(reservior_trap_points, scale, center_data, center_target)
        total_trap_points = self.total_trap_points(target_trap_points, reservior_trap_points)
        
        ## Get atom location
        atom_points = AtomGenerators(self.num_atoms,
                                    target_trap_points,
                                    reservior_trap_points).generate()
        print('Atom distributed.\n')

        ## Create graph 
        self.add_edges_nrst_neighbrs(self.graph, target_trap_points, reservior_trap_points, total_trap_points)
        
        ## Graph partition
        graph_partition_dict = self.graph_partition(self.graph, num_agent)

        ## Obatain cost(distance, R1 or R2) materix for asssignment problem
        atom_site_part_dict = {}
        target_site_part_dict = {}
        reservior_site_part_dict = {}
        total_site_part_dict = {}
        cost_matrix_dict = {}
        
        for n in tqdm(range(num_agent)):
            part_graph = graph_partition_dict[n]

            ## Gather sub-graph information
            ## Atom
            atom_sites_part = [atom for atom in atom_points if tuple(atom) in list(part_graph.nodes())]
            atom_site_part_dict[n] = np.asarray(atom_sites_part)

            ## Target
            target_sites_part = [node[0] for node in list(part_graph.nodes(data=True)) if node[1]['trap_attr']=='target']
            target_site_part_dict[n] = np.asarray(target_sites_part)
            
            ## Reservior
            reservior_sites_part = [node[0] for node in list(part_graph.nodes(data=True)) if node[1]['trap_attr']=='reservior']
            reservior_site_part_dict[n] = np.asarray(reservior_sites_part)

            ## Overall trap
            total_site_part_dict[n] = np.asarray(target_sites_part + reservior_sites_part)

            ## Calculate all pair shortest path (APAS)
            move_length_data = dict(nx.all_pairs_dijkstra_path_length(part_graph))

            ## Distance matrix for asssignment problem
            cost_matrix_dict[n] = self.distance_matrix(move_length_data, target_sites_part, atom_sites_part, r = 2)
        print(f'Done.\n')

        return atom_points, target_trap_points, reservior_trap_points, total_trap_points, atom_site_part_dict, target_site_part_dict, reservior_site_part_dict, total_site_part_dict, graph_partition_dict, cost_matrix_dict

    def quasi_lattice_partition(self, num_agent = 2):
        """Create an quasi lattice, trap, atom locations, and graph object.
        """
        target_trap_points, vor = TargetTrapGenerators(self.num_targets).generate_quasi_lattice()
        print('Target trap position assigned.\n')

        reservior_trap_points = ReserviorTrapGenerators(self.num_targets, target_trap_points,vor, self.d_m).generate_quasi_lattice()
        print('Reservior trap position assigned.\n')

        total_trap_points = self.total_trap_points(target_trap_points, reservior_trap_points)

        ## Number of atoms
        self.num_atoms = math.ceil(0.5 * len(total_trap_points))

        atom_points = AtomGenerators(self.num_atoms,
                                    target_trap_points,
                                    reservior_trap_points).generate()
        print('Atom distributed.\n')

        # Create graph
        self.add_edges(self.graph, target_trap_points, reservior_trap_points, total_trap_points)
        
        ## Graph partition
        graph_partition_dict = self.graph_partition(self.graph, num_agent)

        ## Obatain cost(distance, R1 or R2) materix for asssignment problem
        atom_site_part_dict = {}
        target_site_part_dict = {}
        reservior_site_part_dict = {}
        total_site_part_dict = {}
        cost_matrix_dict = {}
        
        for n in tqdm(range(num_agent)):
            part_graph = graph_partition_dict[n]

            ## Gather sub-graph information
            ## Atom
            atom_sites_part = [atom for atom in atom_points if tuple(atom) in list(part_graph.nodes())]
            atom_site_part_dict[n] = np.asarray(atom_sites_part)

            ## Target
            target_sites_part = [node[0] for node in list(part_graph.nodes(data=True)) if node[1]['trap_attr']=='target']
            target_site_part_dict[n] = np.asarray(target_sites_part)
            
            ## Reservior
            reservior_sites_part = [node[0] for node in list(part_graph.nodes(data=True)) if node[1]['trap_attr']=='reservior']
            reservior_site_part_dict[n] = np.asarray(reservior_sites_part)

            ## Overall trap
            total_site_part_dict[n] = np.asarray(target_sites_part + reservior_sites_part)

            ## Calculate all pair shortest path (APAS)
            move_length_data = dict(nx.all_pairs_dijkstra_path_length(part_graph))

            ## Distance matrix for asssignment problem
            cost_matrix_dict[n] = self.distance_matrix(move_length_data, target_sites_part, atom_sites_part, r = 2)
        print(f'Done.\n')

        return atom_points, target_trap_points, reservior_trap_points, total_trap_points, atom_site_part_dict, target_site_part_dict, reservior_site_part_dict, total_site_part_dict, graph_partition_dict, cost_matrix_dict


    # TODO: under construction
    def triangular_lattice(self):
        """Create an rectangular lattice, trap, atom locations, and graph object.
        Not using it so far.
        """

        target_trap_points, vor = TargetTrapGenerators(self.num_atoms).generate_triangular_lattice()
        print('Target trap position assigned.\n')

        ## Commented out since not useful
        ## Distrubute atoms into target and reservior traps
        # atom_points = AtomGenerators(self.num_atoms,
        #                             target_trap_points,
        #                             reservior_trap_points).generate()
        # print('Atom distributed.')

        return target_trap_points#, atom_points

    ## Useful definitions
    def total_trap_points(self, target_trap_points, reservior_trap_points):
        """
        To create total traping location, one can just use np.append.
        """
        return np.append(target_trap_points, reservior_trap_points, axis=0)

    ## Graph part
    def find_neighbors(self, pindex, triang):
        '''
        Finding neighboring points in Delaunay triangulation, but not
        neccessarly nearst neighbors, for setting up the possible path.
        '''
        return triang.vertex_neighbor_vertices[
            1][triang.vertex_neighbor_vertices[0][pindex]:triang.
               vertex_neighbor_vertices[0][pindex + 1]]

    def add_edges_nrst_neighbrs(self, graph, target_trap_points, reservior_trap_points, total_trap_points):
        ''' 
        ONLY FOR RECTANGULAR LATTICE
        Setup the graph for the entire trap collection
        By a. adding edges, b. No diagonal edges

        Note
        ----
        Might have some issues, it doesn't always output the right hoppings number. The problem is, 
        when we are loading data from real images, we might not get the right returned value.

        One of the solution is to increase the rel_tol value, but is this safe?
        '''
        #print("total trap points:", self.total_trap_points)
        
        tri_total = Delaunay(np.array(total_trap_points))
        ## larger rel_tol makes the code unstable, for example: 5e-3
        tolerence = 5e-5
        for idx_initial, initial in enumerate(total_trap_points):
            ## Add node trap attrubutes, 
            if initial.tolist() in target_trap_points.tolist():
                graph.add_node(tuple(initial), **{'trap_attr': 'target'})

            elif initial.tolist() in reservior_trap_points.tolist():
                graph.add_node(tuple(initial), **{'trap_attr': 'reservior'})

            neighbor_indices = self.find_neighbors(idx_initial, tri_total)
            
            for idx_final in neighbor_indices:
                initial = tuple(initial)
                final = tuple(total_trap_points[idx_final])
                
                ## We don't want diagonal moves
                ## TODO: The way of filtering out diagonal trap sites is unstable for now.
                if math.isclose(initial[0], final[0], rel_tol = tolerence) or math.isclose(initial[1], final[1], rel_tol = 5e-5):
                    distance = abs(initial[0] - final[0])**2 + abs(initial[1] - final[1])**2
                    ## Networks-METIS can only accept interger edge weight
                    graph.add_edge(initial, final, weight = int(distance/tolerence))

    def add_edges(self, graph, target_trap_points, reservior_trap_points, total_trap_points):
        ''' 
        FOR ARB LATTICE
        Setup the graph for the entire trap collection
        By adding edges
        '''
        tri_total = Delaunay(np.array(total_trap_points))
        for idx_initial, initial in enumerate(total_trap_points):
            ## Add node trap attrubutes, 
            if initial in target_trap_points:
                graph.add_node(tuple(initial), **{'trap_attr': 'target'})
            elif initial in reservior_trap_points:
                graph.add_node(tuple(initial), **{'trap_attr': 'reservior'})

            ## Add edges to nodes
            neighbor_indices = self.find_neighbors(idx_initial, tri_total)
            for idx_final in neighbor_indices:
                initial = tuple(initial)
                final = tuple(total_trap_points[idx_final])
                distance = abs(initial[0] - final[0])**2 + abs(initial[1] - final[1])**2
                graph.add_edge(initial, final, weight = distance)

    def add_edges_one_d(self, graph, target_trap_points, reservior_trap_points, total_trap_points):
            """Setup the graph for 1D toy model.

            Note
            ----
            Refer to the LSAP paper.
            """
            
            ## Add node trap attrubutes, 
            for initial in total_trap_points:
                ## Add node trap attrubutes, 
                if initial in target_trap_points:
                    graph.add_node(tuple(initial), **{'trap_attr': 'target'})
                elif initial in reservior_trap_points:
                    graph.add_node(tuple(initial), **{'trap_attr': 'reservior'})

            ## Add edges
            trap_point_sorted = sorted(total_trap_points, key = lambda x: x[0])
            i = 0
            while i < len(trap_point_sorted) - 1:
                initial = tuple(trap_point_sorted[i])
                final = tuple(trap_point_sorted[i+1])
                distance = final[0] - initial[0]
                graph.add_edge(initial, final, weight = distance)
                i += 1

    # def graph_partition(self, graph, num_agent = 2):
    #     """
    #     Considering partiting with METIS algorithm a larger rearranging graph into several smaller
    #     composnents. Each sub-graph is managed by an AOD tweezer.
    #     """
    #     print(f'Partiting graph for {num_agent} agents...')
    #     ## Add node weights to graph
    #     for node in list(graph.nodes()):
    #        graph.nodes[node]['node_value'] = 1

    #     # tell METIS which node attribute to use for 
    #     graph.graph['node_weight_attr'] = 'node_value' 

    #     (cut, parts) = nxmetis.partition(graph, num_agent) 
        
    #     graph_partition_dict = {}

    #     for i in range(num_agent):
    #         ## And store the graph in dictionary
    #         print(f'Partition Graph {i} with {len(parts[i])} nodes.')
    #         graph_partition_dict[i] = graph.subgraph(parts[i])

    #     return graph_partition_dict

    def lattice_graph_pararell(self, total_trap_mesh, row_index, column_index):
        '''
        Overall traps, row and column connections
        '''
        graph_row = nx.Graph()
        graph_column = nx.Graph()

        ## Row graph
        for r in range(len(row_index)):
            for c in range(len(column_index) - 1):
                initial_row = tuple(total_trap_mesh[r][c])
                final_row = tuple(total_trap_mesh[r][c + 1])
                distance_row = final_row[0] - initial_row[0]
                graph_row.add_edge(initial_row, final_row, weight = distance_row)

        # Column graph
        for c in range(len(column_index)):
            for r in range(len(row_index) - 1):
                initial_col = tuple(total_trap_mesh[r][c])
                final_col = tuple(total_trap_mesh[r + 1][c])
                distance_col = final_col[0] - initial_col[0]
                graph_column.add_edge(initial_col, final_col, weight = distance_col)

        return graph_row, graph_column
    
    def distance_matrix(self, move_length_data, target_trap_points, atom_points, r = 2):
        ## Defining the distant matrix, this will assign an atom to a specific target trap.
        distant_matrix = np.zeros((len(target_trap_points), len(atom_points)))
        # print("Generating distance matrix...")
        for t, trap in enumerate(target_trap_points):
            for a, atom in enumerate(atom_points): 
                # TODO: Might need to exclude the already occupied traps
                ## Slower but more secure method
                distant_matrix[t][a] = (move_length_data[tuple(trap)][tuple(atom)])**r

                ## Faster but questionable method
                # distant_matrix[t][a] = abs(trap[0] - atom[0])**r + abs(trap[1] - atom[1])**r
        # print("Done\n")

        return distant_matrix

    ## Post Process
    def rescale_array_parameters(self, data, target_min = 75, target_max = 85):
        """
        Obtaining the resclae parameters, scale and center of the new data.

        Note:
        -----
        For rectangular lattice, apply this function on overall_trap.
        For general lattices, apply this on target traps.
        """
        ## The target center of mass based on given min and max range
        ## Assume our lattice is symetrical
        center_target = np.ones(data.shape[1]) * (target_max + target_min)/2

        ## Obtain the scale of the target data
        span0 = (np.amax(data, axis = 0) - np.amin(data, axis = 0))[0]  ## Max span of axis 0
        span1 = (np.amax(data, axis = 0) - np.amin(data, axis = 0))[1]  ## Max apsn of axis 1
        scale = (target_max - target_min) / max(span0, span1)

        ## Identify the center of mass of the scaled but original data
        center_data = np.sum(scale * data, axis=0)/data.shape[0]

        return scale, center_data, center_target

    def rescale_array(self, data, scale, center_data, center_target):
        """
        The actual rescale mapping function.
        """
        ## Transform the data into target data
        return np.round(data * scale + np.broadcast_to((center_target - center_data), data.shape), 5)

    ## Germetry meshing
    def bravais_mesh(self, target_position, reservior_position, total_trap_position, atom_position):
        """
        Transform the bravias trap cartesian data into row-column meshing.

        Returns
        -------
        row_index, column_index: list
            The sorted row and column coordinates for the entire MxN array, for example:
            row_index = [r1, r2, r3, ..., rM]
            column_index = [c1, c2, c3, ..., cN]
        
        trap_meshes: np.array
            The trap positions that can be access through column/row_indexes.
        """
        
        row_index, column_index = self.trap_indexes(total_trap_position)
        total_trap_mesh = self.trap_grid(total_trap_position, row_index, column_index)
        target_trap_mesh = self.trap_grid(target_position, row_index, column_index)
        reservior_trap_mesh = self.trap_grid(reservior_position, row_index, column_index)
        atom_mesh = self.trap_grid(atom_position, row_index, column_index)

        return total_trap_mesh, target_trap_mesh, reservior_trap_mesh, atom_mesh, row_index, column_index

    def trap_indexes(self, trap_position):
        """
        Record the row and column position catesian coordinate of a bravias lattice. 
        The input should be the total array in this case.

        Returns
        -------
        row_index: list
            List of row coordinates

        column_index: list
            List of column index
        """
        ## Regroup data base on row and column
        ## First need to scan through toatal trap data to get row and column index
        column_index = [] ## axis 1
        row_index = [] ## axis 0

        for trap in trap_position:
            if trap[0] not in column_index:
                column_index.append(trap[0])
            if trap[1] not in row_index: 
                row_index.append(trap[1])

        ## With accending sequence
        column_index.sort()
        row_index.sort()

        return row_index, column_index

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
            trap_grid[row_index.index(trap[1])][column_index.index(trap[0])] = tuple(trap)#.tolist()
            
        return trap_grid

class AtomGenerators(object):
    def __init__(self, num_atoms, target_trap_points,
                 reservior_trap_points):
        self.num_atoms = num_atoms
        self.target_trap_points = target_trap_points
        self.reservior_trap_points = reservior_trap_points

    def generate(self):
        # Atoms can be placed in both target and reservior traps
        avilible_positions = np.append(self.target_trap_points,
                                       self.reservior_trap_points,
                                       axis=0)
        
        ## Since np.random.choice can only apply on 1D array, we can only use index instead
        ## Distributing atom positions from target and reservior traps
        ## replace = False means that the atom position is non-repetable
        np.random.seed(153423423) # For debugging # 1 -> no need selection; -> require LSAP,  2, 9836346 -> ok
        index = np.random.choice(avilible_positions.shape[0],
                                 self.num_atoms,
                                 replace=False)
        
        ## Initialize the array
        atom_points = np.array([[0, 0]])
        for i in index:
            atom_points = np.append(atom_points,
                                    np.array([avilible_positions[i]]),
                                    axis=0)

        ## Delete the initialized data point
        atom_points = np.delete(atom_points, 0, axis=0)

        return atom_points

class TargetTrapGenerators(object):
    def __init__(self, num_targets):
        self.num_targets = num_targets
        self.freespace_dim = 2
    ## Arbitrary graphs is a good test for the voronoi
    def generate_arbitrary(self):
        ## Assigning target positions
        rng = np.random.default_rng(seed=42)  #seed=42
        target_trap_points = rng.random((self.num_targets, self.freespace_dim))

        ## Voronoi process
        vor = Voronoi(target_trap_points)

        return target_trap_points, vor

    ## 1D chain implementation
    def generate_chain_lattice(self):
        target_trap_points = np.array([[0.2, 0], [0.3, 0], [0.4, 0], [0.5, 0], [0.6, 0]])
        vor = 0
        return target_trap_points, vor

    ## Triangular lattice might not be useful at this point
    def generate_triangular_lattice(self):
        num_trap_one_side = math.ceil(math.sqrt(2 * self.num_targets))
        if (num_trap_one_side - math.sqrt(self.num_targets)) % 2 != 0:
            num_trap_one_side += 1

        num_target_one_side = round(math.sqrt(self.num_targets))
        stride = (1 / (num_trap_one_side + 0.5))
        target_trap_points = np.array([[0, 0]])
        counter = 0
        for i in np.linspace(0, 1 - stride, num=num_target_one_side):
            for j in np.linspace(0, 1, num=num_target_one_side):
                counter += 1
                if counter % 2 != 0:
                    target_trap_points = np.append(target_trap_points,
                                                   np.array([[i + stride, j]]),
                                                   axis=0)
                else:
                    target_trap_points = np.append(target_trap_points,
                                                   np.array([[i, j]]),
                                                   axis=0)
        target_trap_points = np.delete(target_trap_points, 0, axis=0)
        vor = Voronoi(target_trap_points)
        return target_trap_points, vor

    ## Generating rectangular lattice
    def generate_rectangular_lattice(self):
        num_trap_one_side = math.ceil(math.sqrt(2 * self.num_targets))
        if (num_trap_one_side - math.sqrt(self.num_targets)) % 2 != 0:
            num_trap_one_side += 1

        num_target_one_side = round(math.sqrt(self.num_targets))
        ## The gap here represents the large gap that skip half of the reservior trap
        gap = (1 / (num_trap_one_side - 1)) * (num_trap_one_side -
                                               num_target_one_side) / 2
        target_trap_points = np.array([[0, 0]])
        for i in np.linspace(0 + gap, 1 - gap, num=num_target_one_side):
            for j in np.linspace(0 + gap, 1 - gap, num=num_target_one_side):
                target_trap_points = np.append(target_trap_points,
                                               np.array([[i, j]]),
                                               axis=0)
        target_trap_points = np.delete(target_trap_points, 0, axis=0)

        vor = 0
        return target_trap_points, vor

    # TODO: Parallel algo inspired by QuEra
    def generate_rectangular_lattice_parallel(self):
        """
        Generate target trap coordinates with array basis for parallel sorting version 1 (QuEra)

        Note
        ----
        Knobs to play with:
        a. Total atom number, the first argument of the AtomGenerators
        b. reservior_volume in the TargetTrapGenerators.generate_rectangular_lattice_parallel
        c. reservior_volume in the ReservuirTraoGeberators.generate_rectangular_lattice_parallel
        """
        num_target_one_side = round(math.sqrt(self.num_targets))
        ## Add more reserviors to prevent LSAP solution
        reservior_volume = 2
        ## 2 reservior sectors * reservior_volume * half of target number + target number
        num_trap_one_side = 2 * reservior_volume * math.ceil(num_target_one_side/2) + num_target_one_side 
        atom_nominal_spacing = 1/(num_trap_one_side - 1)
        gap = atom_nominal_spacing * reservior_volume * math.ceil(num_target_one_side/2)

        target_trap_points = np.array([[0, 0]])
        for i in np.linspace(0 + gap, 1 - gap, num=num_target_one_side):
            for j in np.linspace(0, 1, num=num_target_one_side):
                target_trap_points = np.append(target_trap_points,
                                               np.array([[i, j]]),
                                               axis=0)
        target_trap_points = np.delete(target_trap_points, 0, axis=0)

        vor = 0
        return target_trap_points, vor

    ## Generating quasi lattice
    def generate_quasi_lattice(self):
        ## Refered to https://scipython.com/blog/penrose-tiling-2/
        ## A "sun"

        scale = 100

        # Input handle
        #num_gen = int(input('Number of generation:'))
        if self.num_targets == 16:
            num_gen = 1
        elif self.num_targets == 41:
            num_gen = 2
        elif self.num_targets == 86:
            num_gen = 3
        elif self.num_targets == 211:
            num_gen = 4
        elif self.num_targets == 506:
            num_gen = 5
        else:
            raise ValueError("Gen1: 16; Gen2: 41; Gen3 = 86; Gen4: 211; Gen5: 506 or ask Chun-Wei to add more.")

        config = {
            'tile-opacity': 0.9,
            'stroke-colour': '#800',
            'Stile-colour': '#f00',
            'Ltile-colour': '#ff0'
        }
        tiling = PenroseP3(scale * 1.1, ngen=num_gen, config=config)

        # What coordinate are they using?
        # The base class, RobinsonTriangle defines a generic Robinson triangle with its vertices
        # given as complex numbers, x+iy
        theta = math.pi / 5
        alpha = math.cos(theta)
        rot = math.cos(theta) + 1j * math.sin(theta)
        A1 = scale + 0.j
        B = 0 + 0j

        # Setting the verticies
        # Example 2: A "sun" initialized by a semicircular arrangement of BS tiles.

        # Intial points
        # Those points are the seeds for generating sun-like penrose tiles.
        C1 = C2 = A1 * rot
        A2 = A3 = C1 * rot
        C3 = C4 = A3 * rot
        A4 = A5 = C4 * rot
        C5 = -A1

        # Once we have the points, we can first form initial tiles.
        # Example:
        # BtileS line: 129 creates a robinson triangle formed by A, B, C.
        # set_initial_tiles is basicly just set a basic element for tiling.
        tiling.set_initial_tiles([
            BtileS(A1, B, C1),
            BtileS(A2, B, C2),
            BtileS(A3, B, C3),
            BtileS(A4, B, C4),
            BtileS(A5, B, C5)
        ])

        # Then generating the sun-like pattern through rotating the seed tiles.
        tiling.make_tiling()
        target_points = tiling.make_target_trap()
        #print(target_points)
        #a_target = np.asarray(target_points)
        # Make the pattern valued between 0 and 1
        min_max_scaler = preprocessing.MinMaxScaler()
        a_target = min_max_scaler.fit_transform(np.asarray(target_points))

        ## Rescale the array to better fit the AWG simulation
        a_target = self.rescale_array(a_target, target_min = 64, target_max = 100)
        
        #print('tr_a_target:', tr_a_target)
        tri = Delaunay(a_target)
        #plot_target_traps(tr_a_target)
        distance_among_targets = []
        for ta_idx, ta in enumerate(a_target.tolist()):
            for tb_idx, tb in enumerate(a_target.tolist()):
                if ta_idx != tb_idx:
                    distance_among_targets.append(math.sqrt((ta[0] - tb[0])**2 + (ta[1] - tb[1])**2))
        return a_target, tri

    def rescale_array(self, data, target_min = 78, target_max = 80):
        """
        Obtaining the resclae parameters, scale and center of the new data.

        Note:
        -----
        For rectangular lattice, apply this function on overall_trap.
        For general lattices, apply this on target traps.
        """
        ## Identify the center of mass of the original data
        center_data = np.sum(data, axis=0)/data.shape[0]
        ## The target center of mass based on given min and max range
        center_target = np.ones(center_data.shape) * (target_max + target_min)/2
        ## Obtain the scale of the target data
        span0 = (np.amax(data, axis = 0) - np.amin(data, axis = 0))[0]  ## Max span of axis 0
        span1 = (np.amax(data, axis = 0) - np.amin(data, axis = 0))[1]  ## Max apsn of axis 1
        scale = (target_max - target_min) / max(span0, span1)

        return data * scale + np.broadcast_to((center_target - center_data), data.shape)

## ReserviorTrapGenerators for Reservior Trap positions in 1D, 2D, 3D discrete or continuous background spaces
class ReserviorTrapGenerators(object):
    def __init__(self, num_targets, target_trap_points,
                 vor, d_m):
        self.num_targets = num_targets
        self.target_trap_points = target_trap_points
        self.vor = vor
        self.d_m = d_m
        self.freespace_dim = 2

    def generate_arbitrary(self):

        ## Defining the avilible range for reservior traps for each target trap
        ## reservior traps should locate on the circle
        ## with radius d_m and inside the cell of corresponding target.

        ## First define assign a dummy array
        reservior_trap_points = np.array([[0, 0]])

        ## plotting reservior trap locations
        target_density = 0
        if target_density > 1 / (self.d_m**2):
            print('ReserviorTrap: Too densed, do something else.')
        else:
            index = 0
            while index < len(self.target_trap_points.tolist()):
                position = self.target_trap_points.tolist()[index]
                theta = 2 * math.pi * np.random.rand()
                x = position[0] + self.d_m * math.sin(theta)
                y = position[1] + self.d_m * math.cos(theta)
                index += 1

                # TODO: need to be more efficient in choosing theta
                if x >= 0 and y >= 0:  # and (polygon.contains(point)==True)
                    reservior_trap_points = np.append(reservior_trap_points,
                                                      [[x, y]],
                                                      axis=0)
                else:
                    index -= 1

        reservior_trap_points = np.delete(reservior_trap_points, 0, axis=0)
        return reservior_trap_points

    def generate_chain_lattice(self):
        reservior_trap_points = np.array([[0.0, 0], [0.1, 0], [0.7, 0], [0.8, 0]])

        return reservior_trap_points
    
    def generate_triangular_lattice(self):
        a = 1

    def generate_rectangular_lattice(self):

        num_trap_one_side = math.ceil(math.sqrt(2 * self.num_targets))

        if (num_trap_one_side - math.sqrt(self.num_targets)) % 2 != 0:
            num_trap_one_side += 1
            
        num_target_one_side = round(math.sqrt(self.num_targets))
        gap = (1 / (num_trap_one_side - 1)) * (num_trap_one_side -
                                               num_target_one_side) / 2
        reservior_trap_points = np.array([[0, 0]])
        
        for i in np.linspace(0, 1, num=num_trap_one_side):
            for j in np.linspace(0, 1, num=num_trap_one_side):
                reservior_trap_points = np.append(reservior_trap_points,
                                                  np.array([[i, j]]),
                                                  axis=0)

        ## Delete the initial matrix.
        reservior_trap_points = np.delete(reservior_trap_points, 0, axis=0)

        ## Delete the position overlapping target traps.
        delete_list = []

        for index_r, element_r in enumerate(reservior_trap_points):
            for index_t, element_t in enumerate(self.target_trap_points):
                ## element[0,0] is the x component, element[0,1] is the y component
                if abs((element_r - element_t)[0]) <= 0.0001 and abs(
                    (element_r - element_t)[1]) <= 0.0001:
                    delete_list.append(index_r)

        reservior_trap_points = np.delete(reservior_trap_points,
                                          delete_list,
                                          axis=0)
        return reservior_trap_points

    # TODO: Parallel algo inspired by QuEra
    def generate_rectangular_lattice_parallel(self):
        """
        Generate reservior trap coordinates with array basis for parallel sorting version 1 (QuEra)

        Note
        ----
        Knobs to play with:
        a. Total atom number, the first argument of the AtomGenerators
        b. reservior_volume in the TargetTrapGenerators.generate_rectangular_lattice_parallel
        c. reservior_volume in the ReservuirTraoGeberators.generate_rectangular_lattice_parallel
        """
        num_target_one_side = round(math.sqrt(self.num_targets))
        reservior_volume = 2
        num_trap_one_side = 2 * reservior_volume * math.ceil(num_target_one_side/2) + num_target_one_side 
        #atom_nominal_spacing = 1/(num_trap_one_side - 1)
        #gap = atom_nominal_spacing * math.ceil(num_target_one_side/2)

        reservior_trap_points = np.array([[0, 0]])

        ## Create a grid
        for i in np.linspace(0, 1, num=num_trap_one_side):
            for j in np.linspace(0, 1, num=num_target_one_side):
                reservior_trap_points = np.append(reservior_trap_points,
                                                  np.array([[i, j]]),
                                                  axis=0)

        ## Delete the initial matrix.
        reservior_trap_points = np.delete(reservior_trap_points, 0, axis=0)

        ## Delete the position overlapping target traps.
        delete_list = []

        for index_r, element_r in enumerate(reservior_trap_points):
            for index_t, element_t in enumerate(self.target_trap_points):
                # element[0,0] is the x component, element[0,1] is the y component
                if math.isclose(element_r[0], element_t[0]) and math.isclose(element_r[1], element_t[1]):
                    delete_list.append(index_r)

        reservior_trap_points = np.delete(reservior_trap_points,
                                          delete_list,
                                          axis=0)

        return reservior_trap_points

    def generate_quasi_lattice(self):

        ## Defining the avilible range for reservior traps for each target trap
        ## reservior traps should locate on the circle
        ## with radius d_m and inside the cell of corresponding target.

        ## First define assign a dummy array
        reservior_trap_points = np.array([[0, 0]])

        ## plotting reservior trap locations
        target_density = 0

        if target_density > 1 / (self.d_m**2):
            print('ReserviorTrap: Too densed, do something else.')

        else:
            index = 0
            for region in self.target_trap_points[self.vor.simplices]:
                centroid = np.sum(region, axis=0) / region.shape[0]
                too_close_to_target = False

                for ver in region:
                    if math.sqrt((centroid[0] - ver[0])**2+(centroid[1] - ver[1])**2) < self.d_m:
                        too_close_to_target = True
                        break

                if too_close_to_target == True:
                    continue

                reservior_trap_points = np.append(
                    reservior_trap_points,
                    np.array([[centroid[0], centroid[1]]]),
                    axis=0)

        ## Since the first element is dummy, we just delete it.
        reservior_trap_points = np.delete(reservior_trap_points, 0, axis=0)

        print("number of reservior_trap_points",
              reservior_trap_points.shape[0])

        return reservior_trap_points

class LoadingSaved():
    def __init__(self, parent=None):
        """
        MainWindow for wavemeter display

        :param app: parent MainApplication
        :param parent: [optional] parent class
        """
        super().__init__(parent)
    
    
def main():
    
    target_trap_points, reservior_trap_points, atom_points, total_trap_points = GeometeryGenerator(num_atoms = 50).rectangular_lattice()
    OutputData.plot_config(atom_points, target_trap_points, reservior_trap_points, save_path = './save/', attribute = 'test')

if __name__ == "__main__":
    main()