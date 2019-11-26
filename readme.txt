This package contains the forward model and CVAE network for ultrafast streaking trace retrieval problem. The purposes of the files are as follows:

./:
CVAE training:
main_phase_params_mixed_noise_johnathon_sample4: CVAE training with updated forward model (see ./Pulse_retrieval below for the description of forward model) for sample4
main_phase_params_mixed_noise_johnathon_sample3: CVAE training with updated forward model for sample3
These two files load a pre-trained model in ./models when executed. To begin a new training session, set load_model=False

Plotting:
plot_pulse_johnathon: survey the erros of streaking trace and pulse FWHM of reconstructions from simulated trace. Corresponding to the output from "generate and save reconstruction examples" session in CVAE training scripts.
plot_pulse_johnathon_experiment: Error statistics of the experimental reconstruction instances. Corresponding to the output from "load experiment trace" session in CVAE training scripts.
visualize_z: visualize the latent space distribution at various photon count levels. Must run CVAE training script first.

Direct retrieval of phase parameters:
tf_MAP_FROG-PROOF_polynomial_model: uses TensorFlow optimizer to directly find the phase parameters, given an input trace.

The following two files are for test purpose only:
main_phase_params_mixed_noise.py: CVAE training with old forward model
main_phase_params.py: CVAE training with old forward model without mixed noise levels

./Pulse retrieval:
forward model_phase: my old streaking trace forward model, with FFT paddings removed and no dipole phase considered.
forward_model_phase_params_johnathon: new streaking trace model controbuted by Johnathon. Same FFT padding as [White & Chang 2018]. Dipole phase considered.
The input to both forward models are pulse parameters instead of concatenated real and imag of XUV spectrum. The 9 parameters of the input are respectively:
[XUV Taylor 1st, 2nd, 3rd, 4th, 5th, IR CEP, IR central wavelength, IR duration, IR intensity]

layes_phase_params_johnathon_sample3: recognition and conditional prior encoder (301X38 input shape) for streaking trace sample3.
layes_phase_params_johnathon_sample4: recognition and conditional prior encoder (301X98 input shape) for streaking trace sample4.
By default sample4, its parameters and its known XUV spectrum are selected. To change the selection of trace in training data generation, modify the trace_num in the file ./Pulse_retrieval/measured_trace/get_trace.py

trace_generation_phase_params_Possion_johnathon: generate training and testing dataset based on the updated forward model.
trace_generation_phase_params_Possion: generate training and testing dataset based on my old forward model.