This folder contains most of the work achieved in the context of this miniproject.
It is there as a trace of the developement and experiments I did.
As a disclaimer I would like to state that the files contained have not been entirely reviewed, cleaned or checked.
Some files might not run but contain code snippets that were used for further developements.
It is possible that some lines of code are copypasted from the pytorch documentation, or at least strongly inspired.
The same applies to the GAN trials part, which is based on the pix2pix paper and code
Code : https://github.com/phillipi/pix2pix
Paper : https://arxiv.org/abs/1611.07004

In details : 
- The documentation folder contains the noise2noise paper and the model layers descriptions in csv format.
- The gan_architecture folder contains a model_tuning.py file where a GAN is implemented 
and a hyperparams_optim.py file to train it (same location as the test.py file to run it)
- The hyperparameters_tuning folder contains a model_tuning.py file where the final model is implemented
and a hyperparams_optim.py file to train it (same location as the test.py file to run it)
- The results_examples folder contains predictions of the 20 first validation images in png format
and a result.py file to produce them (same location as the test.py file to run it)

Nicola Antonio Santacroce
27.05.2022