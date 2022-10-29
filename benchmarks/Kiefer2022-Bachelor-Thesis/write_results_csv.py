# ###############################################################################
# ## RESULTS COLLECTION
import os

def getProfilingString(isProfiled):
    return "profiling" if isProfiled else "no-profiling"

def write_results_csv(folder, device_name, network_count, duration, has_PRMs, is_merged, multithreading_type, uses_conditional_connect,
                      last_run_time, compilation_time, binary_run_time,
                      neurongroup_stateupdater, neurongroup_thresholder, neurongroup_resetter,
                      synapses_pre, synapses_pre_push_spikes,
                      spikemonitor, statemonitor, sum_ratemonitors, profiling=True):
    # This str(profiling) is why the csvs are called False and Truebenchmark.
    # TODO: Use ternary expression to rename output.
    file = os.path.join(folder, str(profiling)+"benchmark_new.csv")
    exists = os.path.exists(file)
    with open(file, "a", newline='') as f:
        if not exists:
            # Create header
            f.write("device_name,network_count,duration,has_monitors,is_merged,multithreading_type,uses_conditional_connect,")
            f.write("last_run_time,compilation_time,binary_run_time,")
            if (profiling):
                f.write("neurongroup_stateupdater,neurongroup_thresholder,neurongroup_resetter,")
                f.write("synapses_pre,synapses_pre_push_spikes,")
                f.write("spikemonitor,statemonitor,sum_ratemonitors,")
            f.write("total_run_time,")
            f.write("duplicating_time,")
            f.write("result_extraction_time")
            f.write("\n")
        f.write(f'{device_name},{network_count},{duration},{has_PRMs},{is_merged},{multithreading_type},{uses_conditional_connect},')
        f.write(f'{last_run_time},{compilation_time},{binary_run_time},')
        if (profiling):
            f.write(f'{neurongroup_stateupdater},{neurongroup_thresholder},{neurongroup_resetter},')
            f.write(f'{synapses_pre},{synapses_pre_push_spikes},')
            f.write(f'{spikemonitor},{statemonitor},{sum_ratemonitors},')

def append_total_run_time(folder, total_run_time, profiling=True, duplicating_time=0, result_extraction_time=0):
    file = os.path.join(folder, str(profiling)+"benchmark_new.csv")
    with open(file, "a", newline='') as f:
        f.write(f'{total_run_time},')
        f.write(f'{duplicating_time},')
        f.write(f'{result_extraction_time:.20f}')
        f.write("\n")