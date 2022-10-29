import ctypes
from multiprocessing import Process, Value

from examples.write_results_csv import append_total_run_time, write_results_csv


def write_results():
    # ###############################################################################
    # ## RESULTS COLLECTION

    if not os.path.exists(params['resultsfolder']):
        os.mkdir(params['resultsfolder'])  # for plots and profiling txt file
    if True:
        multi_threading_type = 'openmp' if params['openmp'] else 'GPU' if params['devicename'] == 'cuda_standalone' else 'none'
        write_results_csv(
            benchmarkfolder, network_count=params['M'],
            device_name=params['devicename'], duration=params['duration'], has_PRMs=params['monitors'], is_merged=False,
            multithreading_type=multi_threading_type, uses_conditional_connect='N/A',
            last_run_time=last_run_time, compilation_time=compilation_time,
            binary_run_time=binary_run_time,
            neurongroup_stateupdater=neurongroup_stateupdater,
            neurongroup_thresholder=neurongroup_thresholder,
            neurongroup_resetter=neurongroup_resetter,
            synapses_pre=synapses_pre,
            synapses_pre_push_spikes=synapses_pre_push_spikes,
            spikemonitor=spikemonitor,
            statemonitor=statemonitor,
            sum_ratemonitors=sum_ratemonitors, profiling=params['profiling']
        )
    profilingpath = os.path.join(params['resultsfolder'], '{}.txt'.format(name))
    with open(profilingpath, 'w') as profiling_file:
        profiling_file.write(
#            str(profiling_summary()) +
            '\n_last_run_time = ' + str(last_run_time) +
            '\ncompilation time = ' + str(compilation_time) +
            '\nbinary run time = ' + str(binary_run_time)
        )
        print('profiling information saved in {}'.format(profilingpath))


def print_results(spikemon, PRM, statemon, m):

    #
    # ###############################################################################
    # ## RESULTS COLLECTION

    if not os.path.exists(params['resultsfolder']):
        os.mkdir(params['resultsfolder'])  # for plots and profiling txt file
    if params['monitors']:
        # subplot(211)
        (spikemon.t / ms, spikemon.i, '.')
        # xlim(0, duration / ms)

        # subplot(212)
        (statemon.t / ms, statemon[0])
        # xlim(0, duration / ms)

        if params['PRMs']:
            # subplot(212)
            (PRM.t / ms, PRM.smooth_rate(window='flat', width=0.5 * ms) / Hz)
            # xlim(0, duration / ms)
            # show()

        # plotpath = os.path.join(params['resultsfolder'], '{}.png'.format(name + "_Network_" + str(m + 1)))
        # savefig(plotpath)
        # print('plot saved in {}'.format(plotpath))
        # clf()



devicename = 'cpp_standalone'

# number of neurons
N = 5000

# number of networks to simulate
nNetworks = 10

# duration
duration = .1

# folder to store plots and profiling information
resultsfolder = 'results'

# folder for the code
codefolder = 'code'

# monitors (neede for plot generation)
monitors = True

profiling = True

# single precision
single_precision = True

# openmp
openmp = False

# benchmark folder
benchmarkfolder = '.'

# run multiple PRMs
run_PRMs = True

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
          'profiling': profiling,
          'N': N,
          'M': nNetworks,
          'monitors': monitors,
          'PRMs': run_PRMs,
          'single_precision': single_precision,
          'openmp': openmp,
          'duration': duration,
          'partitions': num_blocks,
          'atomics': atomics,
          'bundle_mode': bundle_mode}

from utils import set_prefs, update_from_command_line

# update params from command line
choices={'devicename': ['cuda_standalone', 'cpp_standalone', 'genn']}
update_from_command_line(params)

# do the imports after parsing command line arguments (quicker --help)
import os
import matplotlib
matplotlib.use('Agg')

import time
start = time.time()

from brian2 import *
if params['devicename'] == 'cuda_standalone':
    import brian2cuda

if params['openmp']:
    params['cpp_threads'] = 20
# set brian2 prefs from params dict
name = set_prefs(params, prefs)

codefolder = os.path.join(params['codefolder'], name)
print('running example {}'.format(name))
print('compiling model in {}'.format(codefolder))

###############################################################################
## SIMULATION

Vr = 10 * mV
theta = 20 * mV
tau = 20 * ms
delta = 2 * ms
taurefr = 2 * ms
duration = params['duration'] * second
C = 1000
sparseness = float(C) / params['N']
J = .1 * mV

# default values from brian2 example (say 'reference' regime)
sigmaext = 1 * mV
muext = 25 * mV

eqs = """
dV/dt = (-V+muext + sigmaext * sqrt(tau) * xi)/tau : volt
"""
set_device(params['devicename'], directory=codefolder, build_on_run=False, debug=False)

networks = {}
PRMs = {}
spikemons = {}
statemons = {}

last_run_time= 0
compilation_time= 0
binary_run_time= 0
neurongroup_stateupdater= 0
neurongroup_thresholder= 0
neurongroup_resetter= 0
synapses_pre= 0
synapses_pre_push_spikes= 0
spikemonitor= 0
statemonitor= 0
sum_ratemonitors= 0

def run_sim(m):
    set_device(params['devicename'], directory=codefolder, run=True, debug=False)
    networks[m] = Network()

    group = NeuronGroup(params['N'], eqs, threshold='V>theta',
                        reset='V=Vr', refractory=taurefr)
    group.V = Vr
    networks[m].add(group)

    conn = Synapses(group, group, on_pre='V += -J', delay=delta)
    conn.connect(p=sparseness)
    networks[m].add(conn)

    if params['monitors']:
        statemons[m] = StateMonitor(group, 'V', record=[0])
        networks[m].add(statemons[m])
        spikemons[m] = SpikeMonitor(group)
        networks[m].add(spikemons[m])
        if params['PRMs']:
            PRMs[m] = PopulationRateMonitor(group)
            networks[m].add(PRMs[m])
    networks[m].run(duration, report='text', profile=params['profiling'])

    if params['profiling']:
        profiling_dict = dict(networks[m].profiling_info)
    else:
        profiling_dict = dict()
    global last_run_time
    global compilation_time
    global binary_run_time
    global neurongroup_stateupdater
    global neurongroup_thresholder
    global neurongroup_resetter
    global synapses_pre
    global synapses_pre_push_spikes
    global spikemonitor
    global statemonitor
    global sum_ratemonitors
    last_run_time += device._last_run_time
    compilation_time += device.timers['compile']['all']
    binary_run_time += device.timers['run_binary']
    neurongroup_stateupdater += sum([v for (k,v) in profiling_dict.items() if 'stateupdater' in k])
    neurongroup_thresholder += sum([v for (k,v) in profiling_dict.items() if 'thresholder' in k])
    neurongroup_resetter += sum([v for (k,v) in profiling_dict.items() if 'resetter' in k])
    synapses_pre= sum([v for (k,v) in profiling_dict.items() if 'synapses_pre_codeobject' in k])
    synapses_pre_push_spikes= sum([v for (k,v) in profiling_dict.items() if 'synapses_pre_push_spikes' in k])
    spikemonitor += sum([v for (k,v) in profiling_dict.items() if 'spikemonitor' in k])
    statemonitor +=sum([v for (k,v) in profiling_dict.items() if 'statemonitor' in k])
    sum_ratemonitors += sum([v for (k,v) in profiling_dict.items() if 'ratemonitor' in k])
        # print_results(spikemons[m], PRMs[m], m)
    # device.reinit()
    # device.activate()

    print('the generated model in {} needs to removed manually if wanted'.format(codefolder))

if False:
    ps = []
    for m in range(0, params['M']):
        p=Process(target=run_sim, args=(m,))
        p.start()
        ps.append(p)

    for p in ps:
        p.join()

    times = counter.value
else:
    for m in range(0, params['M']):
        run_sim(m)
    #device.build(directory=codefolder, compile=True, run=True, debug=False, clean=True)
    write_results()
    # for m in range(0, params['M']):
    result_extraction_start = time.time()

    if params['monitors']:
        print_results(spikemons[m], PRMs[m], statemons[m], m)
    result_extraction_time = time.time()-result_extraction_start
    print('_last_run_time = ', last_run_time)
    print('compilation time = ', compilation_time)
    print('Binary run time: ', binary_run_time)
print('Total time: ', time.time()-start, 'seconds.')
append_total_run_time(benchmarkfolder,time.time()-start, profiling=params['profiling'], result_extraction_time=result_extraction_time)
