from os import sep
from turtle import color
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
pd.options.mode.chained_assignment = None  # default='warn'
data = pd.read_csv("Truebenchmark_new.csv")

data = data[(data['has_monitors'] == True)]
data = data.groupby(['device_name','network_count','duration','is_recompile','has_monitors','is_merged','multithreading_type'], as_index=False).mean()
data.sort_values(['network_count'], inplace=True)
sep_data = data[(data['is_merged'] == False)]
sep_data_recompile = sep_data[(sep_data['is_recompile'] == True)]
sep_data_recompile[
    ['network_count','compilation_time','last_run_time', 'binary_run_time', 'total_run_time']
  ] = sep_data_recompile[
    ['network_count','compilation_time','last_run_time', 'binary_run_time', 'total_run_time']
  ].mul(127, axis=0)
sep_data_compile = sep_data[(sep_data['is_recompile'] == False)]

openmp_sep_data = sep_data[(sep_data['is_recompile'] == False) & (sep_data['multithreading_type'] == 'openmp')]
openmp_recompile_sep_data = sep_data_recompile[(sep_data_recompile['multithreading_type'] == 'openmp')]

sep_data_total = sep_data_compile.append(sep_data_recompile)
sep_data_total = sep_data_total.groupby(['device_name','duration','has_monitors','is_merged','multithreading_type'], as_index=False).sum()


openmp_sep_data = sep_data_total[(sep_data_total['multithreading_type'] == 'openmp')]
singlethread_sep_data = sep_data_total[(sep_data_total['multithreading_type'] == 'none')]
cuda_sep_data = sep_data_total[(sep_data_total['multithreading_type'] == 'GPU')]


data = data[(data['network_count'] == 128)]


openmp_merged_data = data[(data['multithreading_type'] == 'openmp') & (data['is_merged'] == True)]
singlethread_merged_data = data[(data['multithreading_type'] == 'none') & (data['is_merged'] == True)]
cuda_merged_data = data[(data['multithreading_type'] == 'GPU') & (data['is_merged'] == True)]


# plt.yscale('log')
# plt.ylabel('seconds')
#plt.plot(singlethread_sep_data)
# Shrink current axis by 20%
# plt.tight_layout()
# plt.legend(['Compile','Setup and finalisation','Simulation loop time'],loc='center left', bbox_to_anchor=(1, 0.5))
# plt.savefig('fig4-openmp.png', bbox_inches='tight')
# plt.clf()
BIGSIZE = 20
plt.rc('font', size=BIGSIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGSIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGSIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGSIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGSIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGSIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGSIZE)


plt.bar("Single sep", singlethread_sep_data['compilation_time'], label="Compile", bottom=0)
plt.bar("Single sep", singlethread_sep_data['binary_run_time']-singlethread_sep_data['last_run_time'], label="Setup and finalisation", bottom=singlethread_sep_data['compilation_time'])
plt.bar("Single sep", singlethread_sep_data['last_run_time'],label="Simulation loop time", bottom=singlethread_sep_data['compilation_time']+singlethread_sep_data['binary_run_time']-singlethread_sep_data['last_run_time'])
plt.bar("Single sep", singlethread_sep_data['result_extraction_time'] + singlethread_sep_data['duplicating_time'],label="Overhead", bottom= singlethread_sep_data['compilation_time']+singlethread_sep_data['binary_run_time'])

plt.gca().set_prop_cycle(None)
plt.bar("Single mer", singlethread_merged_data['compilation_time'], label="Compile", bottom=0)
plt.bar("Single mer", singlethread_merged_data['binary_run_time']-singlethread_merged_data['last_run_time'], label="Setup and finalisation", bottom=singlethread_merged_data['compilation_time'])
plt.bar("Single mer", singlethread_merged_data['last_run_time'],label="Simulation loop time", bottom=singlethread_merged_data['compilation_time']+singlethread_merged_data['binary_run_time']-singlethread_merged_data['last_run_time'])
plt.bar("Single mer", singlethread_merged_data['result_extraction_time'] + singlethread_merged_data['duplicating_time'],label="Overhead", bottom=singlethread_merged_data['compilation_time']+singlethread_merged_data['binary_run_time'])


plt.gca().set_prop_cycle(None)

plt.bar("OpenMP sep", openmp_sep_data['compilation_time'], label="Compile", bottom=0)
plt.bar("OpenMP sep", openmp_sep_data['binary_run_time']-openmp_sep_data['last_run_time'], label="Setup and finalisation", bottom=openmp_sep_data['compilation_time'])
plt.bar("OpenMP sep", openmp_sep_data['last_run_time'],label="Simulation loop time", bottom=openmp_sep_data['compilation_time']+openmp_sep_data['binary_run_time']-openmp_sep_data['last_run_time'])
plt.bar("OpenMP sep", openmp_sep_data['result_extraction_time'] + openmp_sep_data['duplicating_time'],label="Overhead", bottom=openmp_sep_data['compilation_time']+openmp_sep_data['binary_run_time'])

plt.gca().set_prop_cycle(None)

print("OpenMP sep")
print(openmp_sep_data['binary_run_time']-openmp_sep_data['last_run_time'])
print(openmp_sep_data['compilation_time'])
print(openmp_sep_data['last_run_time'])

print("OpenMP mer")
print(openmp_merged_data['binary_run_time']-openmp_merged_data['last_run_time'])
print(openmp_merged_data['compilation_time'])
print(openmp_merged_data['last_run_time'])

plt.bar("OpenMP mer", openmp_merged_data['compilation_time'], label="Compile", bottom=0)
plt.bar("OpenMP mer", openmp_merged_data['binary_run_time']-openmp_merged_data['last_run_time'], label="Setup and finalisation", bottom=openmp_merged_data['compilation_time'])
plt.bar("OpenMP mer", openmp_merged_data['last_run_time'],label="Simulation loop time", bottom=openmp_merged_data['compilation_time']+openmp_merged_data['binary_run_time']-openmp_merged_data['last_run_time'])
plt.bar("OpenMP mer", openmp_merged_data['result_extraction_time'] + openmp_merged_data['duplicating_time'], label="Overhead", bottom= openmp_merged_data['compilation_time']+openmp_merged_data['binary_run_time'])

# plt.yscale('log')
# plt.ylabel('seconds')
# plt.tight_layout()
# plt.legend(['Compile','Setup and finalisation','Simulation loop time'],loc='center left', bbox_to_anchor=(1, 0.5))
# plt.savefig('fig4-single-thread.png', bbox_inches='tight')
# plt.clf()
plt.gca().set_prop_cycle(None)

print("CUDA sep")
print(cuda_sep_data['binary_run_time']-cuda_sep_data['last_run_time'])
print(cuda_sep_data['compilation_time'])
print(cuda_sep_data['last_run_time'])
plt.bar("CUDA sep", cuda_sep_data['compilation_time'], label="Compile", bottom=0)
plt.bar("CUDA sep", cuda_sep_data['binary_run_time']-cuda_sep_data['last_run_time'], label="Setup and finalisation", bottom=cuda_sep_data['compilation_time'])
plt.bar("CUDA sep", cuda_sep_data['last_run_time'],label="Simulation loop time", bottom=cuda_sep_data['compilation_time']+cuda_sep_data['binary_run_time']-cuda_sep_data['last_run_time'])
plt.bar("CUDA sep", cuda_sep_data['result_extraction_time'] + cuda_sep_data['duplicating_time'],label="Overhead", bottom=cuda_sep_data['compilation_time']+cuda_sep_data['binary_run_time'])

print("CUDA merged")
print(cuda_merged_data['binary_run_time']-cuda_merged_data['last_run_time'])
print(cuda_merged_data['compilation_time'])
print(cuda_merged_data['last_run_time'])
plt.gca().set_prop_cycle(None)
plt.bar("CUDA mer", cuda_merged_data['compilation_time'], label="Compile", bottom=0)
plt.bar("CUDA mer", cuda_merged_data['binary_run_time']-cuda_merged_data['last_run_time'], label="Setup and finalisation", bottom=cuda_merged_data['compilation_time'])
plt.bar("CUDA mer", cuda_merged_data['last_run_time'],label="Simulation loop time", bottom=cuda_merged_data['compilation_time']+cuda_merged_data['binary_run_time']-cuda_merged_data['last_run_time'])
plt.bar("CUDA mer", cuda_merged_data['result_extraction_time'] + cuda_merged_data['duplicating_time'],label="Overhead", bottom=cuda_merged_data['compilation_time']+cuda_merged_data['binary_run_time'])


# plt.yscale('log')
plt.ylabel('Time [seconds]')
plt.xticks(['OpenMP sep',"OpenMP mer","Single sep","Single mer","CUDA sep","CUDA mer"])

# plt.tight_layout()
plt.legend(['Compile','Setup and finalisation','Simulation loop time', 'Overhead'],loc='center left', bbox_to_anchor=(1, 0.5))
fig = plt.gcf()
fig.set_size_inches(18, 11)
plt.savefig('fig4-cuda_new.png', bbox_inches='tight', dpi=300)
plt.clf()