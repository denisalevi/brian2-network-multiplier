#devicename = 'cuda_standalone'
devicename = 'cpp_standalone'

from pprint import pprint
from brian2.groups import neurongroup
from utils import set_prefs, update_from_command_line

update_from_command_line(params)

# do the imports after parsing command line arguments (quicker --help)
import os
import matplotlib
matplotlib.use('Agg')

from matplotlib.pyplot import figure, subplot, plot, xticks, ylabel, xlim, ylim, xlabel, clf

from brian2 import *
if params['devicename'] == 'cuda_standalone':
    import brian2cuda


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

#set_device(params['devicename'], directory=codefolder, compile=True, run=True,
        #   debug=False)

import brian2tools.baseexport
set_device('exporter')
from multiply_network import NetworkMultiplier


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

eqs = """
dV/dt = (-V+muext + sigmaext * sqrt(tau) * xi)/tau : volt
"""
# set_device(params['devicename'], directory=codefolder, build_on_run=False, debug=False)
network_single = Network()

group = NeuronGroup(params['N'], eqs, threshold='V>theta',
                    reset='V=Vr', refractory=taurefr)
group.V = Vr
network_single.add(group)

conn = Synapses(group, group, on_pre='V += -J', delay=delta)
conn.connect(p=sparseness)
network_single.add(conn)

if params['monitors']:
    statemon = StateMonitor(group, 'V', record=[0])
    network_single.add(statemon)
    spikemon = SpikeMonitor(group)
    network_single.add(spikemon)
    if params['PRMs']:
        PRM = PopulationRateMonitor(group)
        network_single.add(PRM)

network_single.run(1*ms)
network_multi = NetworkMultiplier(device.runs, nNetworks, params)


network_multi.run(duration)

pprint(network_multi.getSpikeMonitorResults())
pprint(network_multi.getStateMonitorResults())
pprint(network_multi.getPRMs()[0].smooth_rate(window='flat', width=0.5*ms))
PRMs=network_multi.getPRMs()
spikemon_results = network_multi.getSpikeMonitorResults()
for m in range(0, params['M']):
    subplot(211)
    plot(spikemon_results[m]['t']/ms, spikemon_results[m]['i'], '.')
    xlim(0, duration/ms)

    subplot(212)
    plot(PRMs[m].t/ms, PRMs[m].smooth_rate(window='flat', width=0.5*ms)/Hz)
    xlim(0, duration/ms)

    plotpath = os.path.join(params['resultsfolder'], '{}.png'.format(name + "_Network_" + str(m+1)))
    savefig(plotpath)
    print('plot saved in {}'.format(plotpath))
    clf()