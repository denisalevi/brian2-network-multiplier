from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
x = np.linspace(0,10,2)
openmp_y = 18.467 + 59.351*x
openmp_sep_y = 2.97 + 117.371*x
cuda_y = 72.4516 + 26.90945*x
cuda_sep_y = 12.6464 + 116.452*x
plt.plot([], [], color='k', linestyle='-', label='merged')
plt.plot([], [], color='k', linestyle='--', label='separate')
plt.plot([], [], marker='o', color='r', label='OpenMP')
plt.plot([], [], marker='o', color='g', label='CUDA')
plt.plot(x, openmp_y, marker='o', color='r')
plt.plot(x, cuda_y, marker='o', color='g')
plt.plot(x, openmp_sep_y,marker='o', color='r', linestyle='dashed')
plt.plot(x, cuda_sep_y,marker='o', color='g', linestyle='dashed')
# plt.plot(openmp40_merged_data['network_count'], openmp40_merged_data['last_run_time'], label="OPENMP 40 merged", marker='o')
# plt.yscale('log')
# plt.xticks([str(i) for i in 2 ** np.arange(9)])
plt.legend()
plt.ylabel('Compile time + computation time [seconds]')
plt.xticks(np.arange(min(x), max(x)+1, 1.0))
plt.xlabel('Biological time [seconds]')

plt.grid(which="both")
plt.savefig('fig5.png')
plt.clf()

plt.plot([], [], color='k', linestyle='-', label='merged')
plt.plot([], [], color='k', linestyle='--', label='separate')
plt.plot([], [], marker='o', color='r', label='OpenMP')
plt.plot([], [], marker='o', color='g', label='CUDA')
x = np.linspace(0,100,2)
openmp_y = 18.467 + 59.351*x + 75.486
openmp_sep_y = 2.97 + 117.371*x + 92.669
cuda_y = 72.4516 + 26.90945*x + 1440.283027
cuda_sep_y = 12.6464 + 116.452*x + 730.118
plt.plot(x, openmp_y,marker='o', color='r')
plt.plot(x, cuda_y,marker='o', color='g')
plt.plot(x, openmp_sep_y,marker='o', color='r', linestyle='dashed')
plt.plot(x, cuda_sep_y,marker='o', color='g', linestyle='dashed')
# plt.plot(openmp40_merged_data['network_count'], openmp40_merged_data['last_run_time'], label="OPENMP 40 merged", marker='o')
# plt.yscale('log')
# plt.xticks([str(i) for i in 2 ** np.arange(9)])
plt.ylabel('Compile time + computation time [seconds]')
plt.xlabel('Biological time [seconds]')
plt.xticks(np.arange(min(x), max(x)+5, 5.0))
plt.legend()
plt.grid(which="both")
plt.savefig('fig5-with-setup.png')