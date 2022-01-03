# Modelling time-dependent temperatures of EV batteries by Physics-informed Neural Networks

This repository contains the code used in the research project on "Modelling time-dependent temperatures of EV batteries by Physics-informed Neural Networks".


# How to run
1) Either simply run the jupyter notebook of the corresponding model to train it from scratch
2) OR use the following command to load already pre-trained models found in 'models' folder for each model:
```
model.load_state_dict(torch.load(PATH))
```

INN models use FrEIA framework that can be found here:
```
https://github.com/VLL-HD/FrEIA.git
```

PINN models use PINN framework that can be found here:
```
https://github.com/ComputationalRadiationPhysics/NeuralSolvers.git
```
