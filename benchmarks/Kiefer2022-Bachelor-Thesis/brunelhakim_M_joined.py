
import time
start = time.time()
devicename = 'cuda_standalone'
devicename = 'cpp_standalone'

# number of neurons
N = 5000

# number of networks to simulate
nNetworks = 1

# duration
duration = 1

# whether to profile run
profiling = True

# folder to store plots and profiling information
resultsfolder = 'results'

# folder for the code
codefolder = 'code'

# monitors (neede for plot generation)
monitors = True

# single precision
single_precision = True

# multi threading
openmp = False

# run multiple PRMs
run_PRMs = True

# connect Synmapses with conditional connect call
use_conditional_connect = False

# benchmark folder
benchmarkfolder = '.'

## the preferences below only apply for cuda_standalone

# number of post blocks (None is default)
num_blocks = None

# atomic operations
atomics = True

# push synapse bundles
bundle_mode = True

###############################################################################
## CONFIGURATION

params = {'devicename': devicename,
          'resultsfolder': resultsfolder,
          'codefolder': codefolder,
          'N': N,
          'M': nNetworks,
          'profiling': profiling,
          'monitors': monitors,
          'PRMs': run_PRMs,
          'single_precision': single_precision,
          'openmp': openmp,
          'duration': duration,
          'use_conditional_connect': use_conditional_connect,
          'partitions': num_blocks,
          'atomics': atomics,
          'bundle_mode': bundle_mode}

from utils import set_prefs, update_from_command_line

update_from_command_line(params)

print("Before Import all:" + str(time.time()-start))
# do the imports after parsing command line arguments (quicker --help)
import os
import matplotlib
matplotlib.use('Agg')

from matplotlib.pyplot import figure, subplot, plot, xticks, ylabel, xlim, ylim, xlabel, clf
from numpy import zeros, arange, ones
import numpy as np

print("Before Import brian2cuda:" + str(time.time()-start))

import brian2tools.baseexport
from multiply_network import NetworkMultiplier
from brian2 import *
if params['devicename'] == 'cuda_standalone':
    import brian2cuda
print("Import brian2cuda:" + str(time.time()-start))

set_device('exporter')

if params['devicename'] == 'cpp_standalone' and params['openmp']:
    params['cpp_threads'] = 20
    multi_threading_type = "openmp"
elif params['devicename'] == 'cuda_standalone':
    multi_threading_type = "GPU"
else:
    multi_threading_type = "none"

# set brian2 prefs from params dict
name = set_prefs(params, prefs)

codefolder = os.path.join(params['codefolder'], name)
print('running example {}'.format(name))
print('compiling model in {}'.format(codefolder))

###############################################################################
## SIMULATION

# set_device(params['devicename'], directory=codefolder, compile=True, run=True,
#            debug=False)


Vr = 10*mV
theta = 20*mV
tau = 20*ms
delta = 2*ms
taurefr = 2*ms
duration = params['duration']*second
C = 1000
sparseness = float(C)/params['N']
J = .1*mV

# default values from brian2 example (say 'reference' regime)
sigmaext = 1*mV
muext = 25*mV

eqs = """dV/dt = (-V+muext + sigmaext * sqrt(tau) * xi)
                 /tau : volt"""

network = Network()

group = NeuronGroup(params['N'], eqs, threshold='V>theta',
                    reset='V=Vr', refractory=taurefr)
group.V = Vr

network.add(group)

conn = Synapses(group, group, on_pre='V += -J', delay=delta)
print("STARTING connecting")
conn.connect(p=sparseness)
network.add(conn)
# subgroups = []
# for m in range(0, params['M']):
#     lower = m * params['N']
#     upper = (m+1) * params['N']
#     subgroup = group[lower:upper]
#     if params['use_conditional_connect']:
#         conn.connect(condition='lower <= i and i < upper and lower <= j and j < upper ', p=sparseness)
#         network.add(conn)
#     subgroups.append(subgroup)

# # NEW better way
# param_N = params['N']
# param_M = params['M']

# if not params['use_conditional_connect']:
#     conn.connect(j="k for k in sample(i%param_M, param_M*param_N, param_M, p=sparseness)")
#     network.add(conn)

if params['monitors']:

    spikemon = SpikeMonitor(group)
    network.add(spikemon)
    statemon = StateMonitor(group, 'V', record=[0])
    network.add(statemon)
    PRM = PopulationRateMonitor(group)
    network.add(PRM)
    # if params['PRMs']:
        # PRMs = []
        # # PopulationRateMonitor results can not be "untangled" after the simulation
        # for subgroup in subgroups:
        #     PRM = PopulationRateMonitor(subgroup)
        #     network.add(PRM)
        #     PRMs.append(PRM)


print("Befoer duplication:" + str(time.time()-start))
duplicating_start = time.time()
network.run(duration)#, report='text', profile=params['profiling'])
network_multi = NetworkMultiplier(device.runs, params["M"], params)
duplicating_time = time.time() - duplicating_start

network_multi.run(duration, report='text', profile=params['profiling'])
#
# ###############################################################################
# ## RESULTS COLLECTION

if not os.path.exists(params['resultsfolder']):
    os.mkdir(params['resultsfolder']) # for plots and profiling txt file

from write_results_csv import write_results_csv, append_total_run_time

network = network_multi.getNetwork()
if (params['profiling']):
    profiling_dict = dict(network.profiling_info)
else:
    profiling_dict = dict()
sum_ratemonitors = sum([v for (k,v) in profiling_dict.items() if 'ratemonitor' in k])
write_results_csv(
benchmarkfolder, network_count=params['M'],
device_name=params['devicename'], duration=params['duration'], has_PRMs=params['monitors'], is_merged=True,
multithreading_type=multi_threading_type, uses_conditional_connect=params['use_conditional_connect'],
last_run_time=device._last_run_time, compilation_time=device.timers['compile']['all'],
binary_run_time=device.timers['run_binary'],
neurongroup_stateupdater=sum([v for (k,v) in profiling_dict.items() if 'stateupdater' in k]),
neurongroup_thresholder=sum([v for (k,v) in profiling_dict.items() if 'thresholder' in k]),
neurongroup_resetter=sum([v for (k,v) in profiling_dict.items() if 'resetter' in k]),
synapses_pre=sum([v for (k,v) in profiling_dict.items() if 'synapses_1_pre_codeobject' in k]),
synapses_pre_push_spikes=sum([v for (k,v) in profiling_dict.items() if 'synapses_1_pre_push_spikes' in k]),
spikemonitor=sum([v for (k,v) in profiling_dict.items() if 'spikemonitor' in k]),
statemonitor=sum([v for (k,v) in profiling_dict.items() if 'statemonitor' in k]),
sum_ratemonitors=sum_ratemonitors, profiling=params['profiling']
)

if params['profiling']:
    print(profiling_summary())
    profilingpath = os.path.join(params['resultsfolder'], '{}.txt'.format(name))
    with open(profilingpath, 'w') as profiling_file:
        profiling_file.write(
            str(profiling_summary()) +
            '\n_last_run_time = ' + str(device._last_run_time) +
            '\ncompilation time = ' + str(device.timers['compile']['all']) +
            '\nbinary run time = ' + str(device.timers['run_binary'])
        )
    print('profiling information saved in {}'.format(profilingpath))

result_extraction_start = time.time()
if params['monitors']:
    spikemon_results = network_multi.getSpikeMonitorResults()
    statemon_results = network_multi.getSpikeMonitorResults()
    
    PRMs=network_multi.getPRMs()
    for m in range(0, params['M']):
        # subplot(211)
        (spikemon_results[m]['t']/ms, spikemon_results[m]['i'], '.')
        # xlim(0, duration/ms)


        # subplot(212)
        (statemon_results[m]['t'] / ms, statemon_results[m])
        # xlim(0, duration / ms)

        # subplot(212)
        (PRMs[m].t/ms, PRMs[m].smooth_rate(window='flat', width=0.5*ms)/Hz)

        # plotpath = os.path.join(params['resultsfolder'], '{}.png'.format(name + "_Network_" + str(m+1)))
        # savefig(plotpath)
        # print('plot saved in {}'.format(plotpath))
        clf()

result_extraction_time = time.time() - result_extraction_start
print('_last_run_time = ', device._last_run_time)
print('compilation time = ', device.timers['compile']['all'])
print('Binary run time: ', device.timers['run_binary'])
print('Total time: ', time.time()-start, 'seconds.')
append_total_run_time(benchmarkfolder,time.time()-start, params['profiling'], duplicating_time, result_extraction_time=result_extraction_time)
