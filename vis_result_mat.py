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
    folder_name = "results/" + folder_name
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    df = pd.read_csv(csv_name, header=None).transpose()
    plt.figure(figsize=(8, 6))  # Optional: Set the figure size
    sns.heatmap(df, cmap='binary', annot=False, fmt='', linewidths=0.0)
    plt.title('Aktivierung der Neuronen im Sensory Layer')
    plt.xlabel('Iteration')
    plt.ylabel('Neuron im Sensory Layer')    
    save_path = os.path.join(folder_name,png_name)
    if JUST_SHOW:
        plt.show()
    else:
        plt.savefig(save_path)
    #plt.show()

def line_plot(data, run_names, y_label, title, folder_name):

    folder_name = "results/" + folder_name
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
    if JUST_SHOW:
        plt.show()
    else:
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



def one_run(n_sens, n_rand, foldername, means, vars, diffs, run_names, track_input_duration = False):
    main_annemarie.N_RAND = n_rand
    main_annemarie.N_SENS = n_sens
    main_annemarie.STEPS = STEPS
    main_annemarie.STEP_STOP_INIT = STEP_STOP_INIT
    main_annemarie.BELL_INPUT = INPUT_BELL
    main_annemarie.BINARY_INPUT = INPUT_BINARY
    main_annemarie.CUTTOFF_BELL_INPUT = INPUT_CUTTOFF_BELL
    main_annemarie.CUTTOFF_BELL_INPUT_FACTOR = INPUT_CUTTOFF_BELL_FACTOR
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
    if track_input_duration:
        run_name = run_name + "_with_input_duration_of_" + str(main_annemarie.INPUT_DURATION)

    run_names.append(run_name.format(n_sens=n_sens, n_rand=n_rand))
    filename_template = run_name + ".png"
    if CREATE_HEATMAPS:
        create_heatmap(csv_file_path, filename_template.format(n_sens=n_sens, n_rand=n_rand), foldername)



def multiple_runs_sizes(min_sens = 8, max_sens = 2048, factor_rand_sens = 2):

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

def multiple_runs_nrand(min_rand = 8, max_rand = 4096, fixed_sens = 512):
    
    means = []
    vars = []
    diffs = []
    run_names = []

    n_rand = min_rand
    while n_rand <= max_rand:
        one_run(fixed_sens, n_rand, "heatmap_for_fixed_sens", means, vars, diffs, run_names)
        n_rand = n_rand * 2

    return means, vars, diffs, run_names

def multiple_runs_nsens(min_sens = 8, max_sens = 2048, fixed_rand = 1024):
    
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

def plot_bell_curves(n_sens = 512):
    bell_curve = main_annemarie.bell_curve_input(n_sens, n_sens/2, n_sens/8)
    binary_curve = main_annemarie.binary_input(n_sens, n_sens/2, n_sens/8)
    cuttoff_bell_curve_2 = main_annemarie.cutoff_bell_curve_input(n_sens, n_sens/2, n_sens/8, 2.0)
    cuttoff_bell_curve_1_5 = main_annemarie.cutoff_bell_curve_input(n_sens, n_sens/2, n_sens/8, 1.5)
    cuttoff_bell_curve_1_1 = main_annemarie.cutoff_bell_curve_input(n_sens, n_sens/2, n_sens/8, 1.1)
    plt.plot(bell_curve, label="bell")
    plt.plot(cuttoff_bell_curve_2, label="cutoff bell f=2")
    plt.plot(cuttoff_bell_curve_1_5, label="cutoff bell f=1.5")
    plt.plot(cuttoff_bell_curve_1_1, label="cutoff bell f=1.1")
    plt.plot(binary_curve, label="binary")
    plt.legend()
    plt.title('Input Varianten')
    plt.xlabel('Neuron im Sensory Layer')
    plt.ylabel('Aktivierung')  
    if JUST_SHOW:
        plt.show()
    else:
        if not os.path.exists("other_results"):
            os.makedirs("other_results")
        plt.savefig("other_results/bell_curves.png")
    means = []
    means.append(np.mean(bell_curve))
    means.append(np.mean(cuttoff_bell_curve_2))
    means.append(np.mean(cuttoff_bell_curve_1_5))
    means.append(np.mean(cuttoff_bell_curve_1_1))
    means.append(np.mean(binary_curve))
    names = ["bell", "cutoff bell f=2", "cutoff bell f=1.5", "cutoff bell f=1.1", "binary"]
    colors = ["blue", "orange", "green", "red", "purple"]
    plt.figure(figsize=(8, 6))  # Optional: Set the figure size
    plt.bar(names, means, color=colors)
    plt.title('Mittelwerte der Input Varianten')
    plt.xlabel('Input Varianten')
    plt.ylabel('Mittelwert')
    if JUST_SHOW:
        plt.show()
    else:
        if not os.path.exists("other_results"):
            os.makedirs("other_results")
        plt.savefig("other_results/bell_curves_means.png")
    
CREATE_HEATMAPS = False
CREATE_LINE_PLOTS = True

INPUT_BELL = False
INPUT_BINARY = False
INPUT_CUTTOFF_BELL = True
INPUT_CUTTOFF_BELL_FACTOR = 2.0


STEPS = 10
STEP_STOP_INIT = 2

JUST_SHOW = False


def standard_run_for_heatmaps():
    global CREATE_HEATMAPS 
    CREATE_HEATMAPS = True
    global CREATE_LINE_PLOTS 
    CREATE_LINE_PLOTS = False
    one_trail(run_size = True, run_nrand = True, run_nsens = True)

def standard_run_for_line_plots(n):
    global CREATE_HEATMAPS
    CREATE_HEATMAPS = False
    global CREATE_LINE_PLOTS
    CREATE_LINE_PLOTS = True
    n_trails(n = n, run_size = True, run_nrand = True, run_nsens = True)

def standard_run_for_testing_input_duration():
    global CREATE_HEATMAPS
    CREATE_HEATMAPS = True
    global CREATE_LINE_PLOTS
    CREATE_LINE_PLOTS = False
    for i in range(1, 10):
        main_annemarie.CUTTOFF_BELL_INPUT = i
        one_run(512, 1024, 'heatmap_for_input_duration', [], [], [], [], True)

def just_one_run():
    global CREATE_HEATMAPS
    global CREATE_LINE_PLOTS
    global INPUT_BELL 
    global INPUT_BINARY
    global INPUT_CUTTOFF_BELL
    global INPUT_CUTTOFF_BELL_FACTOR
    global STEPS
    global STEP_STOP_INIT
    global JUST_SHOW
    CREATE_HEATMAPS = True
    CREATE_LINE_PLOTS = True
    INPUT_BELL = False
    INPUT_BINARY = False
    INPUT_CUTTOFF_BELL = True
    INPUT_CUTTOFF_BELL_FACTOR = 2.0
    STEPS = 100
    STEP_STOP_INIT = 20
    JUST_SHOW = False
    one_run(512, 1024, 'single_runs', [], [], [], [], False)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

start_time = time.time()

# for generating heatmaps
#standard_run_for_heatmaps()

# for generating line plots
# standard_run_for_line_plots(100)

# for testing input duration
# standard_run_for_testing_input_duration()

# for just one run
# just_one_run()

# for Plotting all bell curves
plot_bell_curves()

end_time = time.time()
execution_time = end_time - start_time
logger.info(f"the function took {execution_time:.4f} seconds to run.")

