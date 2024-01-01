"""
Misc functions
"""
import numpy as np
import os
from collections import defaultdict

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

def load_micro(dataset_filepath):
    np.random.seed(999)

    return 0

def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

def filter_quadratic(data,condition):
    result = []
    for element in data:
        if all(condition(element,other) for other in result):
            result.append(element)
    return result

def the_condition(xs,ys):
    # working with squares, 2.5e-05 is 0.005*0.005 
    return sum((x-y)*(x-y) for x,y in zip(xs,ys)) > 1e-06

def reformat_path(agent_path):
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

# Plot configs
def plot_configs(ax):
    plt.rcParams['text.usetex'] = True

    # Set bourder line
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1.5)

    # Tick parameters
    ax.tick_params(axis = 'both', which = 'major', length = 6, width = 1, direction = 'in', labelsize = 16)
    ax.tick_params(axis = 'both', which = 'minor', length = 4, width = 1, direction = 'in', labelsize = 16)
    ax.tick_params(labelbottom = True, labeltop = False, labelleft = True, labelright = False,
                     bottom = True, top = True, left = True, right = True)
    ax.yaxis.set_tick_params(right='on',which='both')
    
    return ax

def filled_marker_style(color):
    if color == 'blue':
        filled_marker_style_blue = dict(marker='o', linestyle='None', markersize=5,
                           color='darkgrey',
                           markerfacecolor='lightsteelblue',
                           markerfacecoloralt='lightsteelblue',
                           markeredgecolor='tab:blue')
        return filled_marker_style_blue
    
    elif 'red':
        filled_marker_style_red = dict(marker='o', linestyle='None', markersize=5,
                                color='darkgrey',
                                markerfacecolor='lightcoral',
                                markerfacecoloralt='lightcoral',
                                markeredgecolor='tab:red')      
        return filled_marker_style_red
    

def color_settings(n_colors):
    return iter(cm.Blues(np.linspace(0.3, 1, n_colors)))

   
       


