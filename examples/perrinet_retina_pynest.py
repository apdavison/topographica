"""
retina.py from Laurent Perrinet (laurent.perrinet [at] incm.cnrs-mrs.fr).

This simulation was developed as part of the FACETS project
(http://facets.kip.uni-heidelberg.de) and was used in a
large-scale spiking model of cortical columns in V1:

  Kremkow, J., Perrinet, L., Kumar, A., Aertsen, A., and Masson, G. (2007).
  Synchrony in thalamic inputs enhances propagation of
  activity through cortical layers.
  BMC Neuroscience, vol. 8(Suppl 2), P206.
  http://dx.doi.org/10.1186/1471-2202-8-S2-P180

It requires the PyNN simulator interface
(http://neuralensemble.org/PyNN), and is currently set up to
use the NEST simulation engine (http://www.nest-initiative.org).

The model retina consists of 2 one-to-one connected layers of neurons
on a rectangular grid, modelling ON and OFF magnocellular retinal
ganglion cells in the primate retina.

Changes
-------
- February 2014: Updated to PyNN 0.8 syntax and data formats by Andrew Davison.

"""

import datetime
try:
    import pyNN.nest as pyNN
    from pyNN.utility import Timer
except:
    print "Warning -- could not import pyNN; continuing anyway..."
import numpy


def spikelist2spikematrix(DATA, N, N_time, dt):
    """
    Returns a matrix of the number of spikes during simulation.

    Is of the shape of N x N.

    The spike list is a list of tuples (rel time, neuron_id) where the
    location of each neuron_id is given by the NEURON_ID matrix, which
    is in a standardized way [[ (N*i+j) ]] where i and j are line and
    column respectively.

    For instance, for a 4x4 population:

      [[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11],
       [12, 13, 14, 15]]
    """

    spikematrix = numpy.zeros((N, N))
    if DATA.size > 0:
        for i_id, spiketrain in enumerate(DATA.spiketrains):
            column = numpy.mod(i_id, N)
            line = numpy.floor(i_id / N)
            spikematrix[line, column] = spiketrain.size
    return spikematrix / N_time * dt


def retina_default():
    """
    Return a dictionary with default parameters for the retina.
    """
    params = {}
    N = 100
    params = {
        'description': 'default retina',
        'N'          : N,    # integer;  total number of Ganglion Cells
        'N_ret'      : 2.0,  # float;  diameter of Ganglion Cell's RF
        'K_ret'      : 4.0,  # float; ratio of center vs. surround in DOG
        'dt'         : 0.1,  # discretization step in simulations (ms)
        'simtime'    : 40000*0.1,  # float; (ms)
        'syn_delay'  : 1.0,  # float; (ms)
        'noise_std'  : 2.0,  # (nA??) standard deviation of the internal noise
        'snr'        : 2.0,
        'weight'     : 1.0,
        'threads'    : 1,  #2,
        'kernelseeds': [43210987, ],  #394780234], # array with one element per thread
        # seed for random generator used when building connections
        'connectseed': 12345789,  # seed for random generator(s) used during simulation
        'now': datetime.datetime.now().isoformat()  # the date in ISO 8601 format to avoid overriding old simulations
    }

    # retinal neurons' parameters
    params['parameters_gc'] = {
        'V_th': -57.0,
        'V_reset': -70.0,
        'Vinit': -63.0,
        't_ref': 0.5,
        'g_L': 28.95,
        'C_m': 289.5,
        'E_ex': 0.0,
        'E_in': -75.0,
        'tau_syn_ex': 1.5,
        'tau_syn_in': 10.0,
        'E_sfa': -70.0,
        'q_sfa': 0.0,  #14.48,
        'tau_sfa': 110.0,
        'E_rr': -70.0,
        'q_rr': 3214.0,
        'tau_rr': 1.97 }

    params['u'] = params['parameters_gc']['Vinit'] + \
                  numpy.random.rand(N, N)* (params['parameters_gc']['V_th'] - \
                                            params['parameters_gc']['V_reset'])
    return params


def retina_debug():
    """
    Return a dictionary with parameters for the retina suitable for debugging.
    """
    params = retina_default()
    params.update({'description': 'debug retina', 'N': 8})
    return params


def run_retina(params):
    """Run the retina using the specified parameters."""

    print "Setting up simulation"
    timer = Timer()
    timer.start()  # start timer on construction
    pyNN.setup(timestep=params['dt'], max_delay=params['syn_delay'], threads=params['threads'], rng_seeds=params['kernelseeds'])

    N = params['N']
    phr_ON = pyNN.Population((N, N), pyNN.native_cell_type('dc_generator')())
    phr_OFF = pyNN.Population((N, N), pyNN.native_cell_type('dc_generator')())
    noise_ON = pyNN.Population((N, N), pyNN.native_cell_type('noise_generator')(mean=0.0, std=params['noise_std']))
    noise_OFF = pyNN.Population((N, N), pyNN.native_cell_type('noise_generator')(mean=0.0, std=params['noise_std']))

    phr_ON.set(start=params['simtime']/4, stop=params['simtime']/4*3,
               amplitude=params['amplitude'] * params['snr'])
    phr_OFF.set(start=params['simtime']/4, stop=params['simtime']/4*3,
                amplitude=-params['amplitude'] * params['snr'])

    # target ON and OFF populations
    v_init = params['parameters_gc'].pop('Vinit')
    out_ON = pyNN.Population((N, N), pyNN.native_cell_type('iaf_cond_exp_sfa_rr')(**params['parameters_gc']))
    out_OFF = pyNN.Population((N, N), pyNN.native_cell_type('iaf_cond_exp_sfa_rr')(**params['parameters_gc']))
    out_ON.initialize(v=v_init)
    out_OFF.initialize(v=v_init)

    #print "Connecting the network"

    retina_proj_ON = pyNN.Projection(phr_ON, out_ON, pyNN.OneToOneConnector())
    retina_proj_ON.set(weight=params['weight'])
    retina_proj_OFF = pyNN.Projection(phr_OFF, out_OFF, pyNN.OneToOneConnector())
    retina_proj_OFF.set(weight=params['weight'])

    noise_proj_ON = pyNN.Projection(noise_ON, out_ON, pyNN.OneToOneConnector())
    noise_proj_ON.set(weight=params['weight'])
    noise_proj_OFF = pyNN.Projection(noise_OFF, out_OFF, pyNN.OneToOneConnector())
    noise_proj_OFF.set(weight=params['weight'])

    out_ON.record('spikes')
    out_OFF.record('spikes')

    # reads out time used for building
    buildCPUTime = timer.elapsedTime()

    print "Running simulation"

    timer.start()  # start timer on construction
    pyNN.run(params['simtime'])
    simCPUTime = timer.elapsedTime()

    out_ON_DATA = out_ON.get_data().segments[0]
    out_OFF_DATA = out_OFF.get_data().segments[0]

    print "\nRetina Network Simulation:"
    print(params['description'])
    print "Number of Neurons : ", N**2
    print "Output rate  (ON) : ", out_ON.mean_spike_count(), \
        "spikes/neuron in ", params['simtime'], "ms"
    print "Output rate (OFF) : ", out_OFF.mean_spike_count(), \
        "spikes/neuron in ", params['simtime'], "ms"
    print "Build time        : ", buildCPUTime, "s"
    print "Simulation time   : ", simCPUTime, "s"

    return out_ON_DATA, out_OFF_DATA


if __name__ == '__main__':
    params = retina_debug()
    params.update({'amplitude': 0.1*numpy.ones((params['N']**2,))})
    out_ON_DATA, out_OFF_DATA = run_retina(params)

    import pylab
    spike_time = []
    neuron_id = []
    for spiketrain in out_ON_DATA.spiketrains:
        spike_time.append(spiketrain)
        neuron_id.append(numpy.ones_like(spiketrain) * spiketrain.annotations['source_index'])
    spike_time = numpy.hstack(spike_time)
    neuron_id = numpy.hstack(neuron_id)
    pylab.plot(spike_time, neuron_id, '.r')
    pylab.axis([0, params['simtime'], 0, params['N']**2 - 1])
    pylab.savefig("perrinet_retina_pynest_ON.png")
    pylab.clf()

    spike_time = []
    neuron_id = []
    for spiketrain in out_OFF_DATA.spiketrains:
        spike_time.append(spiketrain)
        neuron_id.append(numpy.ones_like(spiketrain) * spiketrain.annotations['source_index'])
    spike_time = numpy.hstack(spike_time)
    neuron_id = numpy.hstack(neuron_id)
    pylab.plot(spike_time, neuron_id, '.b')
    pylab.axis('tight')
    pylab.savefig("perrinet_retina_pynest_OFF.png")

    print spikelist2spikematrix(out_ON_DATA, params['N'],
                                params['simtime']/params['dt'], params['dt'])
