from pprint import pprint
from brian2.groups import neurongroup
from utils import set_prefs, update_from_command_line
import os
import matplotlib
matplotlib.use('Agg')
from matplotlib.pyplot import figure, subplot, plot, xticks, ylabel, xlim, ylim, xlabel, clf

#devicename = 'cuda_standalone'
devicename = 'cpp_standalone'
from brian2 import *
if devicename == 'cuda_standalone':
    import brian2cuda

###############################################################################
## SIMULATION
import brian2tools.baseexport
set_device('exporter')
from network_multiplier import NetworkMultiplier

nNeurons = 5000
mNetworks = 10
Vr = 10*mV
theta = 20*mV
tau = 20*ms
delta = 2*ms
taurefr = 2*ms
duration = 1*second
C = 1000
sparseness = float(C)/nNeurons
J = .1*mV

# default values from brian2 example (say 'reference' regime)
sigmaext = 1*mV
muext = 25*mV

eqs = """
dV/dt = (-V+muext + sigmaext * sqrt(tau) * xi)/tau : volt
"""
network_single = Network()

group = NeuronGroup(nNeurons, eqs, threshold='V>theta',
                    reset='V=Vr', refractory=taurefr)
group.V = Vr
network_single.add(group)

conn = Synapses(group, group, on_pre='V += -J', delay=delta)
conn.connect(p=sparseness)
network_single.add(conn)

statemon = StateMonitor(group, 'V', record=[0])
network_single.add(statemon)
spikemon = SpikeMonitor(group)
network_single.add(spikemon)
PRM = PopulationRateMonitor(group)
network_single.add(PRM)

network_single.run(1*ms)
network_multi = NetworkMultiplier(device.runs, mNetworks, devicename)

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