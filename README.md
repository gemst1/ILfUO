# ILfUO
Imitation Learning from Unpaired Observation

## Requirements
- TensorFlow higher than version 2.0
- OpenAI Gym 0.14.0 with extra environments. Please refer ["gym"](./gym)
- MuJoCo 1.50
- mujoco-py 1.50
- RAY rllib
- OpenAI Baselines tf2 branch with extra edit. Please refer my [repository](https://github.com/gemst1/baselines/tree/tf2).

## Usage
### [dataset](./dataset/README.md)
- Training expert and collect demonstration data.
### [IfO](./IfO/README.md)
- TensorFlow 2.0 Implementation of Imitation from Observation.
### [ILfUO](./ILfUO/README.md)
- Imitation learning from unpaired observation using motion prediction.