from os import sep
from turtle import color
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

for has_monitors in [True, False]:
  data = pd.read_csv("Truebenchmark_new.csv")
  data = data[(data['network_count'] != 256)]
  monitor_string = "monitors" if has_monitors else "no-monitors"
  data = data[(data['has_monitors'] == has_monitors)]
  data = data.groupby(['device_name','network_count','duration','has_monitors','is_merged','multithreading_type'], as_index=False).mean()
  data.sort_values(['network_count'], inplace=True)
  sep_data = data[(data['is_merged'] == False)]
  len_sep_data=len(sep_data['network_count'])
  sep_data = pd.DataFrame(np.repeat(sep_data.values, 8, axis=0), columns=sep_data.columns)
  sep_data.sort_values(['multithreading_type'], inplace=True)
  sep_data[
      ['total_run_time','network_count','last_run_time', 'binary_run_time',"neurongroup_stateupdater","neurongroup_thresholder",'neurongroup_resetter',"synapses_pre","synapses_pre_push_spikes","spikemonitor","statemonitor","sum_ratemonitors"]
    ] = sep_data[
      ['total_run_time','network_count','last_run_time', 'binary_run_time',"neurongroup_stateupdater","neurongroup_thresholder",'neurongroup_resetter',"synapses_pre","synapses_pre_push_spikes","spikemonitor","statemonitor","sum_ratemonitors"]
    ].mul([i for i in 2 ** np.arange(8)]*len_sep_data, axis=0)

  sep_data['network_count']=sep_data['network_count'].astype(str)
  data['network_count']=data['network_count'].astype(str)


  openmp_sep_data = sep_data[(sep_data['multithreading_type'] == 'openmp')].reset_index(drop=True)
  singlethread_sep_data = sep_data[(sep_data['multithreading_type'] == 'none')].reset_index(drop=True)
  cuda_sep_data = sep_data[(sep_data['multithreading_type'] == 'GPU')].reset_index(drop=True)

  openmp_merged_data = data[(data['multithreading_type'] == 'openmp') & (data['is_merged'] == True)].reset_index(drop=True)
  singlethread_merged_data = data[(data['multithreading_type'] == 'none') & (data['is_merged'] == True)].reset_index(drop=True)
  cuda_merged_data = data[(data['multithreading_type'] == 'GPU') & (data['is_merged'] == True)].reset_index(drop=True)

  data = pd.read_csv("Truebenchmark_new.csv")
  data = data[(data['network_count'] != 256)]
  data = data[(data['has_monitors'] == has_monitors)]
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


  datapoints = ['total_run_time','last_run_time',"neurongroup_stateupdater","neurongroup_thresholder",'neurongroup_resetter',"synapses_pre","synapses_pre_push_spikes"]
  if has_monitors:
    datapoints.append("spikemonitor")
    datapoints.append("statemonitor")
    datapoints.append("sum_ratemonitors")
  for profiling_datapoint in datapoints:

    plt.plot([], [], color='k', linestyle='-', label='merged')
    plt.plot([], [], color='k', linestyle='--', label='separate')
    plt.plot([], [], marker='o', color='b', label='Single-threaded')
    plt.plot([], [], marker='o', color='r', label='OpenMP multi-threaded')
    plt.plot([], [], marker='o', color='g', label='CUDA')
    plt.plot(singlethread_sep_data['network_count'], singlethread_sep_data[profiling_datapoint], marker='o', color='b', linestyle='dashed')
    plt.plot(singlethread_merged_data['network_count'], singlethread_merged_data[profiling_datapoint], marker='o', color='b')
    for i, networks in singlethread_merged_data['network_count'].items():
      plt.errorbar(singlethread_merged_data['network_count'][i], 
            singlethread_merged_data[profiling_datapoint][i], 
            yerr=[
              [singlethread_merged_data[profiling_datapoint][i]-singlethread_merged_mindata[profiling_datapoint][i]],
              [singlethread_merged_maxdata[profiling_datapoint][i]-singlethread_merged_data[profiling_datapoint][i]],
              ],
            color='darkblue')
      plt.text(singlethread_merged_data['network_count'][i], 
              singlethread_merged_data[profiling_datapoint][i], 
              "$\\times {s:.1f}$".format(s=singlethread_sep_data[profiling_datapoint][i] / singlethread_merged_data[profiling_datapoint][i]))
    
    plt.plot(openmp_sep_data['network_count'], openmp_sep_data[profiling_datapoint], marker='o', color='r', linestyle='dashed')
    plt.plot(openmp_merged_data['network_count'], openmp_merged_data[profiling_datapoint], marker='o', color='r')
    for i, networks in openmp_merged_data['network_count'].items():
      plt.errorbar(openmp_merged_data['network_count'][i], 
                  openmp_merged_data[profiling_datapoint][i], 
                  yerr=[
                    [openmp_merged_data[profiling_datapoint][i]-openmp_merged_mindata[profiling_datapoint][i]],
                    [openmp_merged_maxdata[profiling_datapoint][i]-openmp_merged_data[profiling_datapoint][i]],
                  ],
                  color='darkred')
      plt.text(openmp_merged_data['network_count'][i], 
              openmp_merged_data[profiling_datapoint][i], 
              "$\\times {s:.1f}$".format(s=openmp_sep_data[profiling_datapoint][i] / openmp_merged_data[profiling_datapoint][i]))
      
    plt.plot(cuda_sep_data['network_count'], cuda_sep_data[profiling_datapoint], marker='o', color='g', linestyle='dashed')
    plt.plot(cuda_merged_data['network_count'], cuda_merged_data[profiling_datapoint], marker='o', color='g')
    for i, networks in cuda_merged_data['network_count'].items():
      plt.errorbar(cuda_merged_data['network_count'][i], 
                 cuda_merged_data[profiling_datapoint][i], 
                 yerr=[
                  [cuda_merged_data[profiling_datapoint][i]-cuda_merged_mindata[profiling_datapoint][i]],
                  [cuda_merged_maxdata[profiling_datapoint][i]-cuda_merged_data[profiling_datapoint][i]],
                ],
                 color='darkgreen')
      plt.text(cuda_merged_data['network_count'][i], 
              cuda_merged_data[profiling_datapoint][i], 
              "$\\times {s:.1f}$".format(s=cuda_sep_data[profiling_datapoint][i] / cuda_merged_data[profiling_datapoint][i]))
      
    plt.yscale('log')

    plt.grid(which="both")
    plt.ylabel('Time [computation / biological]')
    plt.xlabel('Number of networks (M)')
    plt.xticks([str(i) for i in 2 ** np.arange(8)])
    #plt.plot(singlethread_sep_data)
    plt.legend()
    # plt.title(monitor_string + '-'+ profiling_datapoint)
    plt.savefig('fig2-000-new-' + monitor_string + '-'+ profiling_datapoint +'.png')
    plt.clf()