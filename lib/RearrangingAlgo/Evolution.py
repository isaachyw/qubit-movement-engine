"""
Simulations for generated path. 
Simulation keep us aware of any potential bugs when developing algorithms.
And of course good for demonstration.

Contributors
------------
Chun-Wei Liu (cl3762)
"""

import numpy as np
import os

# Open source imports
from tqdm import tqdm
from lib.RearrangingAlgo import OutputData
from utils import check_folder

# List manipulation
from itertools import chain

class Simulation():
    def __init__(self, atom_position, target_trap_points, reservior_trap_points, total_trap_points, path_total, save_path, plot_mode, tweezer_mode, parent=None):
        """
        Simulate assembling process

        Inputs
            :param atom_points: target trap points (np.array)
            :param target_trap_points: target trap points (np.array)
            :param reservior_trap_points: reservior trap points (np.array)
            :param total_trap_points: total trap points (np.array)
            :param path_total: Non-collision path (list)
            :param save_path: save folder for the process (str)

        Outputs
            :param distance_of_moves: total travelling sites (int)
            :param num_of_moves: total number of moves (int)
        """
        self.atom_position_list = atom_position.tolist()
        self.atom_position = atom_position
        self.target_trap_points_list = target_trap_points.tolist()
        self.target_trap_points = target_trap_points
        self.reservior_trap_points_list = reservior_trap_points.tolist()
        self.reservior_trap_points = reservior_trap_points
        self.total_trap_points_list = total_trap_points.tolist()
        self.path_total = path_total
        self.save_path = save_path
        self.plot_mode = plot_mode
        self.tweezer_mode = tweezer_mode

    def single_tweezer_simulator(self):
        """
        A path simulator for single tweezer that competible with path structure
            path = [[v, m1, m2, ..., u], ...,[...]]
        
        """
        iter_num = 0
        staying_num = 0
        nominal_traveling_distance = 0

        print(f'Plotting single-agent simulation...')
        OutputData.plot_config(self.atom_position, self.target_trap_points,
                                self.reservior_trap_points, self.save_path,
                                f'img/StepInitial.png')

        for move_idx, move in enumerate(tqdm(self.path_total)):
            ## The atom that is staying at the same position
            if len(move) == 2 and move[0] == move[1]:
                staying_num += 1
            
            else:
                for step_index in range(len(move)):
                    if step_index < len(move) - 1:
                        if step_index == 0:
                            OutputData.plot_step_config(np.asarray(self.atom_position_list), np.asarray([move[0]]), move, self.target_trap_points,
                                    self.reservior_trap_points, self.save_path,
                                    f'img/Step{iter_num}.png')
                            iter_num += 1
                        ## The current moving atom
                        moving_atom_points = move[step_index + 1]
                        self.atom_position_list[self.atom_position_list.index(move[step_index])] = moving_atom_points
                        ## The overall atom location
                        atom_position_step = np.asarray(self.atom_position_list)
                        
                        ## Plot targer, reservior, atoms
                        if self.plot_mode == True: 
                            OutputData.plot_step_config(atom_position_step, np.asarray([moving_atom_points]), move, self.target_trap_points,
                                    self.reservior_trap_points, self.save_path,
                                    f'img/Step{iter_num}.png')

                        ## Accumulating the nominal totoal traveling distance
                        ## (without scaled)
                        nominal_traveling_distance += np.linalg.norm(np.asarray(move[step_index + 1]) - np.asarray(move[step_index]))
                        iter_num += 1
        OutputData.plot_config(atom_position_step, self.target_trap_points,
                                self.reservior_trap_points, self.save_path,
                                f'img/StepFinal.png')
        print(f'Done.\n')
        return iter_num, staying_num, nominal_traveling_distance

    def multi_tweezer_simulator(self):
        """
        A path simulator for mult-tweezer that competible with path structure
            path = [[[v], t, 'g'], [[m1], t, 'm'], ..., [[u], t, 'd']], ...,[agent_n]]
        
        First unzip the path for all agents, and then regroup by time.
        """
        staying_num = 0
        nominal_traveling_distance = 0

        ## Construct a time table
        time_collection = [data[1] for data in list(chain.from_iterable(self.path_total))]
        time_table = sorted(set(time_collection))

        ## Simulator will iterate through time table for each time point
        ## Then get into the original data of each agent and update atom position
        print(f'Plotting multi-agent simulation...')
        OutputData.plot_config(self.atom_position, self.target_trap_points,
                                self.reservior_trap_points, self.save_path,
                                f'img/StepInitial.png')

        ## Move table store which move the current AOD is playing
        move_table = {}
        for time in tqdm(time_table):
            ## Get the current moving atom
            moving_atom_points = []
            ## For each time, each agent look for current site
            for agent_index, agent_path in enumerate(self.path_total):
                ## Update atom position, and move
                for site_idx_agent, site in enumerate(agent_path):
                    if site[1] == time:
                        ## Update move
                        if site[2] == 'g':
                            ## If grab then looking for the entire move
                            move_agent = []
                            for move_site in agent_path[site_idx_agent:]:
                                move_agent.append(move_site[0])
                                if move_site[2] =='d':
                                    break
                                else:
                                    pass
                            move_table[agent_index] = move_agent
                        else:
                            self.atom_position_list[self.atom_position_list.index(agent_path[site_idx_agent - 1][0])] = site[0]
                            nominal_traveling_distance += np.linalg.norm(np.asarray(site[0]) - np.asarray(agent_path[site_idx_agent - 1][0]))
                        moving_atom_points.append(site[0])
                        break

            #print(f'Time {time}; self.atom_position_list: {self.atom_position_list}') 
            atom_position_step = np.asarray(self.atom_position_list)
            ## Plot atom configuration
            ## Plot target, reservior, atoms
            if self.plot_mode == True: 
                OutputData.plot_step_config(atom_position_step, np.asarray(moving_atom_points), move_table, self.target_trap_points,
                                    self.reservior_trap_points, self.save_path,
                                    f'img/Time{time}.png', num_agents = len(self.path_total))

        OutputData.plot_config(atom_position_step, self.target_trap_points,
                                self.reservior_trap_points, self.save_path,
                                f'img/StepFinal.png')
        print(f'Done.\n')
        return len(time_table), staying_num, nominal_traveling_distance

    def text_outputs(self):
        ## Plots
        if self.plot_mode == True:
            check_folder(os.path.join(self.save_path, 'img/') )
            OutputData.write_txt(self.save_path, 'target_trap_points',self.target_trap_points_list)
            OutputData.write_txt(self.save_path, 'reservior_trap_points', self.reservior_trap_points_list)
            OutputData.write_txt(self.save_path, 'total_trap_points', self.total_trap_points_list)
            OutputData.write_txt(self.save_path, 'path_total', self.path_total)
            OutputData.write_txt(self.save_path, 'atom_position', self.atom_position_list)

    def start(self):

        ## Some text output for bookeeping
        self.text_outputs()

        if self.tweezer_mode == 'single':
            iter_num, staying_num, nominal_traveling_distance = self.single_tweezer_simulator()
            
        elif self.tweezer_mode == 'multi':
            iter_num, staying_num, nominal_traveling_distance = self.multi_tweezer_simulator()

        ## Call the estimations
        est = Estimations(self.path_total, iter_num, staying_num, nominal_traveling_distance, len(self.atom_position_list))
        
        print(f"Data saved in {self.save_path}.")
        
        return est.num_site_moved(), est.num_of_moves(), est.total_sorting_time()

class Estimations():
    def __init__(self, path, iter_num, staying_num, nominal_traveling_distance, atom_number, parent=None):
        self.path = path
        self.iter_num = iter_num
        self.staying_num = staying_num
        self.nominal_traveling_distance = nominal_traveling_distance
        self.moving_rate_per_distance = 13.33 # microsec/micrometer
        self.grabing = 300 # microsec
        self.dropping = 300 # microsec
        self.atom_number = atom_number
        self.trap_spacing = 1.5 # micrometer
        self.map_scale = (np.sqrt(2 * self.atom_number) - 1) * self.trap_spacing # microsec # scaling respected to real measurements

    def num_site_moved(self):
        return self.iter_num - self.staying_num
    
    def num_of_moves(self):
        return len(self.path) - self.staying_num
    
    def total_traveling_distance(self):
        return self.map_scale * self.nominal_traveling_distance

    def total_sorting_time(self):
        return self.num_of_moves() * (self.grabing + self.dropping) + self.moving_rate_per_distance * self.total_traveling_distance()