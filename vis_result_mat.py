import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import main_annemarie
import os
import logging
import time


def create_heatmap(csv_name, png_name, folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    df = pd.read_csv(csv_name, header=None).transpose()
    plt.figure(figsize=(8, 6))  # Optional: Set the figure size
    sns.heatmap(df, cmap='binary', annot=False, fmt='', linewidths=0.0)
    plt.title('Aktivierung der Neuronen im Sensory Layer')
    plt.xlabel('Iteration')
    plt.ylabel('Neuron im Sensory Layer')    
    save_path = os.path.join(folder_name,png_name)
    plt.savefig(save_path)
    #plt.show()

def line_plot(data, run_names, y_label, title, folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    plt.figure(figsize=(8, 6))  # Optional: Set the figure size
    num_lines = len(data)
    color_map = plt.get_cmap('viridis')
    colors = [color_map(i) for i in np.linspace(0, 1, num_lines)]

    for i in range(num_lines):
        if run_names[i] == "sens_512_rand_1024":
            plt.plot(data[i], label=run_names[i], marker='o', linestyle='--', color="red")
        else:
            plt.plot(data[i], label=run_names[i], marker='o', linestyle='-', color=colors[i])
    plt.xlabel('Iteration')
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    png_name = title + ".png"
    save_path = os.path.join(folder_name,png_name)
    plt.savefig(save_path)

def calc_mean_for_each_iteration(result_matrix):
    return np.mean(result_matrix, axis=1)

def calc_var_for_each_iteration(result_matrix):
    return np.var(result_matrix, axis=1)

def calc_diff_for_every_two_iterations(result_matrix):
    diff = []
    for i in range(len(result_matrix)-1):
        diff_arr = np.abs(np.array(result_matrix[i]) - np.array(result_matrix[i+1]))
        diff.append(np.mean(diff_arr))
    return diff



def one_run(n_sens, n_rand, foldername, means, vars, diffs, run_names):
    main_annemarie.N_RAND = n_rand
    main_annemarie.N_SENS = n_sens
    result_matrix = main_annemarie.run_simulation()
    means.append(calc_mean_for_each_iteration(result_matrix))
    vars.append(calc_var_for_each_iteration(result_matrix))
    diffs.append(calc_diff_for_every_two_iterations(result_matrix))

    # Convert the NumPy array to a Pandas DataFrame
    df = pd.DataFrame(result_matrix)

    # Specify the file path where you want to save the CSV file
    csv_file_path = "matrix_data.csv"

    # Define a format string to display numbers without scientific notation
    format_str = "%.6f"

    df.to_csv(csv_file_path, header=False, index=False, float_format=format_str)
    run_name = "sens_{n_sens}_rand_{n_rand}"
    run_names.append(run_name.format(n_sens=n_sens, n_rand=n_rand))
    filename_template = run_name + ".png"
    if CREATE_HEATMAPS:
        create_heatmap(csv_file_path, filename_template.format(n_sens=n_sens, n_rand=n_rand), foldername)



def multiple_runs_sizes(min_sens = 8, max_sens = 1024, factor_rand_sens = 2):

    means = []
    vars = []
    diffs = []
    run_names = []

    n_sens = min_sens
    while n_sens <= max_sens:
        n_rand = n_sens * factor_rand_sens
        one_run(n_sens, n_rand, "heatmap_for_sens_rand_proportional", means, vars, diffs, run_names)
        n_sens = n_sens * 2

    return means, vars, diffs, run_names

def multiple_runs_nrand(min_rand = 8, max_rand = 2048, fixed_sens = 512):
    
    means = []
    vars = []
    diffs = []
    run_names = []

    n_rand = min_rand
    while n_rand <= max_rand:
        one_run(fixed_sens, n_rand, "heatmap_for_fixed_sens", means, vars, diffs, run_names)
        n_rand = n_rand * 2

    return means, vars, diffs, run_names

def multiple_runs_nsens(min_sens = 8, max_sens = 1024, fixed_rand = 1024):
    
    means = []
    vars = []
    diffs = []
    run_names = []

    n_sens = min_sens
    while n_sens <= max_sens:
        one_run(n_sens, fixed_rand, "heatmap_for_fixed_rand", means, vars, diffs, run_names)
        n_sens = n_sens * 2
    
    return means, vars, diffs, run_names

def one_trail(run_size = True, run_nrand = True, run_nsens = True):
    if run_size:
        n_trails_for_one_experiment(1, multiple_runs_sizes, "multiple runs {n} trails".format(n=1), "lines_for_sens_rand_proportional_{n}_trails".format(n=1))
    if run_nrand:
        n_trails_for_one_experiment(1, multiple_runs_nrand, "multiple runs with fixed nsens {n} trails".format(n=1), "lines_for_fixed_sens_{n}_trails".format(n=1))
    if run_nsens:
        n_trails_for_one_experiment(1, multiple_runs_nsens, "multiple runs with fixed nrand {n} trails".format(n=1), "lines_for_fixed_rand_{n}_trails".format(n=1))

def n_trails_for_one_experiment(n, experiment, title, foldername):
    means_for_all_trails = []
    vars_for_all_trails = []
    diffs_for_all_trails = []
    run_names = []
    for i in range(n):
        means, vars, diffs, run_names = experiment()
        means_for_all_trails.append(means)
        vars_for_all_trails.append(vars)
        diffs_for_all_trails.append(diffs)
    mean_of_means = np.mean(means_for_all_trails, axis=0)
    mean_of_vars = np.mean(vars_for_all_trails, axis=0)
    mean_of_diffs = np.mean(diffs_for_all_trails, axis=0)
    if CREATE_LINE_PLOTS:
        line_plot(mean_of_means, run_names, "mean",'mean for {title}'.format(title=title), foldername)
        line_plot(mean_of_vars, run_names, "variance",'variance for {title}'.format(title=title), foldername)
        line_plot(mean_of_diffs, run_names, "mean difference between iterations",'mean difference between iterations for {title}'.format(title=title), foldername)



def n_trails(n = 10, run_size = True, run_nrand = True, run_nsens = True):
    global CREATE_HEATMAPS
    CREATE_HEATMAPS = False
    if run_size:
        n_trails_for_one_experiment(n, multiple_runs_sizes, "multiple runs {n} trails".format(n=n), "lines_for_sens_rand_proportional_{n}_trails".format(n=n))
    if run_nrand:
        n_trails_for_one_experiment(n, multiple_runs_nrand, "multiple runs with fixed nsens {n} trails".format(n=n), "lines_for_fixed_sens_{n}_trails".format(n=n))
    if run_nsens:
        n_trails_for_one_experiment(n, multiple_runs_nsens, "multiple runs with fixed nrand {n} trails".format(n=n), "lines_for_fixed_rand_{n}_trails".format(n=n))

    
CREATE_HEATMAPS = False
CREATE_LINE_PLOTS = True


def standard_run_for_heatmaps():
    global CREATE_HEATMAPS 
    CREATE_HEATMAPS = True
    global CREATE_LINE_PLOTS 
    CREATE_LINE_PLOTS = False
    one_trail(run_size = True, run_nrand = True, run_nsens = True)

def standard_run_for_line_plots():
    global CREATE_HEATMAPS
    CREATE_HEATMAPS = False
    global CREATE_LINE_PLOTS
    CREATE_LINE_PLOTS = True
    n_trails(n = 10, run_size = True, run_nrand = True, run_nsens = True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

start_time = time.time()

# for generating heatmaps
#standard_run_for_heatmaps()

# for generating line plots
standard_run_for_line_plots()



end_time = time.time()
execution_time = end_time - start_time
logger.info(f"the function took {execution_time:.4f} seconds to run.")

