from function_library import *

for u in np.arange(-2,0.6,0.5):
    count, sim_data_maxlist, sim_data_sum_util= RunSimulation(3,10,u)
    plot_simulation_results_diff(sim_data_sum_util,"all both",u)
    plt.savefig("utility-diff-("+str(u)+")-"+"all both.pdf", format="pdf", bbox_inches="tight")