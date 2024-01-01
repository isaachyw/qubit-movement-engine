"""
OutputData.py

Visialization functions, plot trap, atom, graph with nodes and edges.
And produce some text based data for bookeeping.

Contributors
------------
Chun-Wei Liu (cl3762)
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import collections as mc
import os

from utils import check_folder


def plot_atoms(atom_points):
    plt.plot(atom_points[:, 0], atom_points[:, 1],'.', color='white')

def plot_transfering_atoms_step(atom_points):
    '''
    Slightly larget site indicating the moving atoms.
    '''
    plt.plot(atom_points[:, 0], atom_points[:, 1],'.', color='white', markersize=15)

def plot_target_traps(target_trap_points):
    plt.plot(target_trap_points[:, 0], target_trap_points[:, 1], 'o', fillstyle='none', color='green')

def plot_reservior_traps(reservior_trap_points):
    plt.plot(reservior_trap_points[:, 0], reservior_trap_points[:, 1],'o', fillstyle='none', color='red')

def plot_config(atom_points, target_trap_points, reservior_trap_points, save_path, attribute):
    ## Plot initial image
    plt.style.use('dark_background')
    path = os.path.join(save_path, attribute)
    check_folder(save_path)
    plot_atoms(atom_points)
    plot_target_traps(target_trap_points)
    plot_reservior_traps(reservior_trap_points)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(path, dpi=300)
    plt.close()

def plot_step_config(atom_points, moving_atom_points, trajectory_plan, target_trap_points, reservior_trap_points, save_path, attribute, num_agents = 1):
    """
    Plotting the simulated rearranging process, moving atoms are mark as red, while spectator atoms are white.
    """
    plt.style.use('dark_background')
    path = os.path.join(save_path, attribute)
    check_folder(save_path)
    plot_atoms(atom_points)
    plot_transfering_atoms_step(moving_atom_points)
    plot_target_traps(target_trap_points)
    plot_reservior_traps(reservior_trap_points)
    plot_trajectories(trajectory_plan, num_agents)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(path, dpi=300)
    plt.close()

def plot_traps(target_trap_points, reservior_trap_points, hoppings, save_path, attribute):
    ## Plot initial image
    plt.style.use('dark_background')
    path = os.path.join(save_path, attribute)
    check_folder(save_path)
    plot_target_traps(target_trap_points)
    plot_reservior_traps(reservior_trap_points)
    plot_hoppings(hoppings)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(path, dpi=300)
    plt.close()

def plot_trajectories(moves, num_agents):
    """
    Draw the trajectory and destination of the current playing move.
    """
    if num_agents == 1:
        ## Trajectory
        plt.plot(np.asarray(moves)[:, 0], np.asarray(moves)[:, 1], color='white', linewidth=0.5, alpha=0.5)
        ## Destination
        plt.plot(np.asarray(moves)[:,0][-1], np.asarray(moves)[:, 1][-1],'o', color='white', fillstyle='none', markersize=8, linewidth=0.5, alpha=0.5)
    else:
        ## Moves is a dictionary in this case
        for n in range(num_agents):
            if len(moves[n])!=0:
                ## Trajectory
                plt.plot(np.asarray(moves[n])[:, 0], np.asarray(moves[n])[:, 1], color='white', linewidth=0.5, alpha=0.5)
                ## Destination
                plt.plot(np.asarray(moves[n])[:,0][-1], np.asarray(moves[n])[:, 1][-1],'o', color='white', fillstyle='none', markersize=8, linewidth=0.5, alpha=0.5)

def plot_hoppings(hoppings):
    if len(hoppings) != 0:
        for hopping in hoppings:
            hop_data = list(zip(*hopping))
            plt.plot(hop_data[0], hop_data[1], color='white', linewidth=0.5, alpha=0.5)


def write_txt(save_path, file_name, position_list):
    textfile = open(save_path + file_name + ".txt", "w")
    for element in position_list:
        textfile.write(str(element))
    textfile.close()

def record_path():
    NotImplementedError()

def record_num_moves():
    NotImplementedError()

def record_total_travelling_distance():
    NotImplementedError()

def record_total_rearranging_time():
    NotImplementedError()