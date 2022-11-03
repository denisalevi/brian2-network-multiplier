from pprint import pprint
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

data = pd.read_csv("Falsebenchmark_new.csv")
data = data[(data['network_count'] != 256)]
data = data.groupby(['device_name','network_count','duration','is_merged','multithreading_type'], as_index=False).mean()
sep_data = data[(data['is_merged'] == False)]
len_sep_data=len(sep_data['network_count'])
sep_data = pd.DataFrame(np.repeat(sep_data.values, 8, axis=0), columns=sep_data.columns)
sep_data.sort_values(['multithreading_type'], inplace=True)

sep_data[
    ['network_count','last_run_time', 'binary_run_time']
  ] = sep_data[
    ['network_count','last_run_time', 'binary_run_time']
  ].mul([i for i in 2 ** np.arange(8)]*len_sep_data, axis=0)
sep_data['network_count']=sep_data['network_count'].astype(str)
data['network_count']=data['network_count'].astype(str)

openmp_sep_data = sep_data[(sep_data['multithreading_type'] == 'openmp')].reset_index(drop=True)
singlethread_sep_data = sep_data[(sep_data['multithreading_type'] == 'none')].reset_index(drop=True)
cuda_sep_data = sep_data[(sep_data['multithreading_type'] == 'GPU')].reset_index(drop=True)

openmp_merged_data = data[(data['multithreading_type'] == 'openmp') & (data['is_merged'] == True)].reset_index(drop=True)
singlethread_merged_data = data[(data['multithreading_type'] == 'none') & (data['is_merged'] == True)].reset_index(drop=True)
cuda_merged_data = data[(data['multithreading_type'] == 'GPU') & (data['is_merged'] == True)].reset_index(drop=True)
# openmp40_merged_data = data[(data['multithreading_type'] == 'openmp40') & (data['is_merged'] == True)]

# STDDATA
data = pd.read_csv("Falsebenchmark_new.csv")
data = data[(data['network_count'] != 256)]
min_data = data.groupby(['device_name','network_count','duration','is_merged','multithreading_type'], as_index=False).min()
min_data.sort_values(['last_run_time'], inplace=True)
cuda_merged_mindata = min_data[(min_data['multithreading_type'] == 'GPU') & (min_data['is_merged'] == True)].reset_index(drop=True)
openmp_merged_mindata = min_data[(min_data['multithreading_type'] == 'openmp') & (min_data['is_merged'] == True)].reset_index(drop=True)
singlethread_merged_mindata = min_data[(min_data['multithreading_type'] == 'none') & (min_data['is_merged'] == True)].reset_index(drop=True)
cuda_sep_mindata = min_data[(min_data['multithreading_type'] == 'GPU') & (min_data['is_merged'] == False)].reset_index(drop=True)
openmp_sep_mindata = min_data[(min_data['multithreading_type'] == 'openmp') & (min_data['is_merged'] == False)].reset_index(drop=True)
singlethread_sep_mindata = min_data[(min_data['multithreading_type'] == 'none') & (min_data['is_merged'] == False)].reset_index(drop=True)

max_data = data.groupby(['device_name','network_count','duration','is_merged','multithreading_type'], as_index=False).max()
max_data.sort_values(['last_run_time'], inplace=True)
cuda_merged_maxdata = max_data[(max_data['multithreading_type'] == 'GPU') & (max_data['is_merged'] == True)].reset_index(drop=True)
openmp_merged_maxdata = max_data[(max_data['multithreading_type'] == 'openmp') & (max_data['is_merged'] == True)].reset_index(drop=True)
singlethread_merged_maxdata = max_data[(max_data['multithreading_type'] == 'none') & (max_data['is_merged'] == True)].reset_index(drop=True)
cuda_sep_maxdata = max_data[(max_data['multithreading_type'] == 'GPU') & (max_data['is_merged'] == False)].reset_index(drop=True)
openmp_sep_maxdata = max_data[(max_data['multithreading_type'] == 'openmp') & (max_data['is_merged'] == False)].reset_index(drop=True)
singlethread_sep_maxdata = max_data[(max_data['multithreading_type'] == 'none') & (max_data['is_merged'] == False)].reset_index(drop=True)



print(openmp_sep_data)
plt.plot([], [], color='k', linestyle='-', label='merged')
plt.plot([], [], color='k', linestyle='--', label='separate')
plt.plot([], [], marker='o', color='b', label='Single-threaded')
plt.plot([], [], marker='o', color='r', label='OpenMP multi-threaded')
plt.plot([], [], marker='o', color='g', label='CUDA')
plt.plot(singlethread_sep_data['network_count'], singlethread_sep_data['last_run_time'], marker='o', color='b', linestyle='dashed')
plt.plot(singlethread_merged_data['network_count'], singlethread_merged_data['last_run_time'], marker='o', color='b')
for i, networks in singlethread_merged_data['network_count'].items():
    # plt.errorbar(singlethread_sep_data['network_count'][i], 
    #       singlethread_sep_data['last_run_time'][i], 
    #       yerr=singlethread_sep_mindata['last_run_time']*int(networks)-singlethread_sep_data['last_run_time'][i],
    #       color='darkblue')
    plt.errorbar(singlethread_merged_data['network_count'][i], 
            singlethread_merged_data['last_run_time'][i], 
            yerr=[
              [singlethread_merged_data['last_run_time'][i]-singlethread_merged_mindata['last_run_time'][i]],
              [singlethread_merged_maxdata['last_run_time'][i]-singlethread_merged_data['last_run_time'][i]],
              ],
            color='darkblue')
    plt.text(singlethread_merged_data['network_count'][i], 
             singlethread_merged_data['last_run_time'][i], 
            "$\\times {s:.1f}$".format(s=singlethread_sep_data['last_run_time'][i] / singlethread_merged_data['last_run_time'][i]))


plt.plot(openmp_sep_data['network_count'], openmp_sep_data['last_run_time'], marker='o', color='r', linestyle='dashed')
plt.plot(openmp_merged_data['network_count'], openmp_merged_data['last_run_time'], marker='o', color='r')
for i, networks in openmp_merged_data['network_count'].items():
    # plt.errorbar(openmp_sep_data['network_count'][i], 
    #     openmp_sep_data['last_run_time'][i], 
    #     yerr=openmp_sep_mindata['last_run_time']*int(networks)-openmp_sep_data['last_run_time'][i],
    #     color='k')
    plt.errorbar(openmp_merged_data['network_count'][i], 
                openmp_merged_data['last_run_time'][i], 
                yerr=[
                  [openmp_merged_data['last_run_time'][i]-openmp_merged_mindata['last_run_time'][i]],
                  [openmp_merged_maxdata['last_run_time'][i]-openmp_merged_data['last_run_time'][i]],
                ],
                color='darkred')
    plt.text(openmp_merged_data['network_count'][i], 
             openmp_merged_data['last_run_time'][i], 
            "$\\times {s:.1f}$".format(s=openmp_sep_data['last_run_time'][i] / openmp_merged_data['last_run_time'][i]))

plt.plot(cuda_sep_data['network_count'], cuda_sep_data['last_run_time'], marker='o', color='g', linestyle='dashed')
plt.plot(cuda_merged_data['network_count'], cuda_merged_data['last_run_time'], marker='o', color='g')

pprint(cuda_merged_data['last_run_time'])
pprint(cuda_merged_mindata['last_run_time'])
for i, networks in cuda_merged_data['network_count'].items():
    # plt.errorbar(cuda_sep_data['network_count'][i], 
    #   cuda_sep_data['last_run_time'][i], 
    #   yerr=cuda_sep_mindata['last_run_time']*int(networks)-cuda_sep_data['last_run_time'][i],
    #   color='k')
    plt.errorbar(cuda_merged_data['network_count'][i], 
                 cuda_merged_data['last_run_time'][i], 
                 yerr=[
                  [cuda_merged_data['last_run_time'][i]-cuda_merged_mindata['last_run_time'][i]],
                  [cuda_merged_maxdata['last_run_time'][i]-cuda_merged_data['last_run_time'][i]],
                ],
                 color='darkgreen')
    plt.text(cuda_merged_data['network_count'][i], 
             cuda_merged_data['last_run_time'][i], 
            "$\\times {s:.1f}$".format(s=cuda_sep_data['last_run_time'][i] / cuda_merged_data['last_run_time'][i]))
# plt.text(cuda_merged_data['network_count'], cuda_merged_data['last_run_time'], 
#          ,
#          va='center', ha='left',alpha=1)
# plt.plot(openmp40_merged_data['network_count'], openmp40_merged_data['last_run_time'], label="OPENMP 40 merged", marker='o')
plt.yscale('log')
plt.xticks([str(i) for i in 2 ** np.arange(8)])
plt.ylabel('Time [computation / biological]')
plt.xlabel('Number of networks (M)')

#plt.plot(singlethread_sep_data)
plt.legend()
plt.grid(which="both")
plt.savefig('fig1_new.png')