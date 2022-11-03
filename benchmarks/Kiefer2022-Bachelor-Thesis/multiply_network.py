import pprint
from brian2 import *
import os

from brian2.equations.equations import FLOAT, SingleEquation
from utils import set_prefs


class NetworkMultiplier():

    def __init__(self, runs, M, params):
        name = set_prefs(params, prefs)

        codefolder = os.path.join(params['codefolder'], name)
        set_device(params['devicename'], directory=codefolder,
                   compile=True, run=True, debug=False)

        namespace = {}
        export = runs[0]
        pprint.pprint(export)
        network = Network()
        neurongroup_export = export['components']['neurongroup'][0]
        namespace['param_N'] = param_N = neurongroup_export['N']
        namespace['param_M'] = param_M = M

        for key in (neurongroup_export['identifiers']):
            namespace[key] = neurongroup_export['identifiers'][key]

        equation_array = []
        for variable, equation_params in neurongroup_export['equations'].items():
            expr_string = equation_params.get('expr')
            expr = Expression(expr_string) if expr_string else None
            equation_array.append(
                SingleEquation(
                    equation_params['type'], variable, equation_params['unit'],
                    equation_params.get('var_type') or FLOAT,
                    expr,
                    equation_params.get('flags') or None)
            )
        eqs = Equations(equation_array)

        neurongroup_parameters = neurongroup_export['events']['spike']
        group = NeuronGroup(param_N * param_M, eqs, threshold=neurongroup_parameters['threshold']['code'],
                            reset=neurongroup_parameters['reset']['code'], refractory=neurongroup_parameters['refractory'],)
        for initializer in export['initializers_connectors']:
            if initializer['source'] == 'neurongroup' and initializer['type'] == 'initializer':
                group.__setattr__(
                    initializer['variable'], initializer['value'])

        network.add(group)

        synapses_export = export['components']['synapses'][0]
        synapses_parameters = synapses_export['pathways'][0]

        for key in (synapses_export['identifiers']):
            namespace[key] = synapses_export['identifiers'][key]
        conn = Synapses(
            group, group,
            on_pre=synapses_parameters['code'],
            delay=synapses_parameters['delay']
        )
        for initializer in export['initializers_connectors']:
            if initializer['source'] == 'neurongroup' and initializer['type'] == 'connect':
                namespace['probability'] = probability = initializer.get('probability', 1)
                namespace['condition'] = condition = initializer.get('condition')
        print("BEFORE CONNECT")
        # for m in range(0, M):
        #     lower = m * param_N
        #     upper = (m+1) * param_N
        #     subgroup = group[lower:upper]
            
        #     conn = Synapses(subgroup, subgroup, on_pre='V += -J', delay=2*msecond)
        #     conn.connect(p=0.2)

        #     network.add(conn)
        conn.connect(
            j="k for k in sample(i%param_M, param_M*param_N, param_M, p=probability)" +
              (("if " + condition) if condition else "")
        )
        network.add(conn)
        print("AFTER CONNECT")

        # Population Rate Monitor
        if export['components'].get('populationratemonitor'):
            subgroups = []
            for m in range(0, param_M):
                lower = m * param_N
                upper = (m+1) * param_N
                subgroup = group[lower:upper]
                subgroups.append(subgroup)

            self.PRMs = []
            for subgroup in subgroups:
                PRM = PopulationRateMonitor(subgroup)
                self.PRMs.append(PRM)
                network.add(PRM)

        # Spike Monitor
        if export['components'].get('spikemonitor'):
            spikemon_export = export['components'].get('spikemonitor')
            spikemon = SpikeMonitor(group, spikemon_export[0]['variables'])
            self.spikemon = spikemon
            network.add(spikemon)

        # State Monitor
        if export['components'].get('statemonitor'):
            record_array = export['components']['statemonitor'][0]['record']
            multiple_record = []

            for m in range(0, param_M):
                for neuron_index in record_array:
                    multiple_record.append(m*param_N + neuron_index)
            statemon = StateMonitor(
                group,
                export['components']['statemonitor'][0]['variables'],
                record=multiple_record
            )
            network.add(statemon)
            self.statemon = statemon

        self.network = network
        self.namespace = namespace
        self.M = param_M
        self.N = param_N

    def run(self, duration, report=None, report_period=10*second, profile=False, level=0):
        self.network.run(duration, report, report_period,
                         self.namespace, profile, level)

    def getSpikeMonitorResults(self):
        # TODO: Expand filtering to other variables.
        # variables is a set containing additional variables
        variables = self.spikemon.record_variables
        points = np.c_[self.spikemon.t, self.spikemon.i]
        result = []
        for m in range(0, self.M):
            lower = m * self.N
            upper = (m+1) * self.N
            curPoints = points[np.where((lower <= points[:,1]) & (points[:,1] < upper))].T
            result.append({
                't': Quantity(curPoints[0], dim=ms.dim),
                'i': curPoints[1]
            })
        return result

    def getStateMonitorResults(self):
        # TODO: Expand filtering to other variables.
        # variables is an array containing additional variables
        variables = self.statemon.needed_variables
        neuron_idxs = self.statemon.record
        result = []
        for m in range(0, self.M):
            lower = m * self.N
            upper = (m+1) * self.N

            result.append({
                't': self.statemon.t,
                'V': []
            })
            for i in range(len(self.statemon.V)):
                if neuron_idxs[i] < lower or neuron_idxs[i] > upper:
                    continue
                result[m]['V'].append(self.statemon.V[i])
        return result

    def getNetwork(self):
        return self.network

    def getPRMs(self):
        return self.PRMs