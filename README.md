![alt text](https://i.imgur.com/kXEbZ8J.png)

### Quick Links

- [Setup](#wrench-setting-up)

- [Examples](#mag-sample-scripts)

- [Agents and performances](#bar_chart-agents--performances)

# :runner: Running The Code

To start a train session, once installed:

```
python Run.py
```

Defaults:

```Agent=Agents.DQNAgent```

```task=atari/pong```

Plots, logs, generated images, and videos are automatically stored in: ```./Benchmarking```.

![alt text](https://i.imgur.com/2jhOPib.gif)

Welcome ye, weary Traveller.

>Stop here and rest at our local tavern,
>
> Where all your reinforcements and supervisions be served, à la carte!

Drink up! :beers:

# :pen: Paper & Citing

For detailed documentation, [see our :scroll:](https://arxiv.com).

```
@inproceedings{cool,
  title={bla},
  author={Sam Lerman and Chenliang Xu},
  booktitle={bla},
  year={2022},
  url={https://openreview.net}
}
```

If you use any part of this code, **be sure to cite the above!**

An acknowledgment to [Denis Yarats](https://github.com/denisyarats), whose excellent [DrQV2 repo](https://github.com/facebookresearch/drqv2) inspired much of this library and its design.

# :open_umbrella: Unified Learning?

Yes.

All agents support discrete and continuous control, classification, and generative modeling.

See example scripts of various configurations [below](#mag-sample-scripts).

# :wrench: Setting Up 

Let's get to business.

## 1. Clone The Repo

```
git clone git@github.com:agi-init/UnifiedML.git
cd UnifiedML
```

## 2. Gemme Some Dependencies

```
conda env create --name ML --file=Conda.yml
```

## 3. Activate Your Conda Env.

```
conda activate ML
```

Optionally, for GPU support, install Pytorch with CUDA from https://pytorch.org/get-started/locally/.

# :joystick: Installing The Suites

## 1. Classify

[comment]: <> (<details>)

[comment]: <> (<summary><i>Click to open :open_book: </i></summary>)

[comment]: <> (<br>)

Comes preinstalled.

[comment]: <> (</details>)

## 2. Atari Arcade

[comment]: <> (<details>)

[comment]: <> (<summary><i>Click to open :open_book: </i></summary>)

[comment]: <> (<br>)

You can use ```AutoROM``` if you accept the license.

```
pip install autorom
AutoROM --accept-license
```
Then:
```
mkdir ./Datasets/Suites/Atari_ROMS
AutoROM --install-dir ./Datasets/Suites/Atari_ROMS
ale-import-roms ./Datasets/Suites/Atari_ROMS
```

[comment]: <> (</details>)

## 3. DeepMind Control

[comment]: <> (<details>)

[comment]: <> (<summary><i>Click to open :open_book: </i></summary>)

[comment]: <> (<br>)

Download MuJoCo from here: https://mujoco.org/download.

Make a ```.mujoco``` folder in your home directory:

```
mkdir ~/.mujoco
```

Extract and move downloaded MuJoCo folder into ```~/.mujoco```. For a linux x86_64 architecture, this looks like:

```
tar -xf mujoco210-linux-x86_64.tar.gz
mv mujoco210/ ~/.mujoco/ 
```

And run:

```
pip install --user dm_control
```

to install DeepMind Control. For any issues, consult the [DMC repo](https://github.com/deepmind/dm_control).

[comment]: <> (</details>)

# :file_cabinet: Key files

```Run.py``` handles training and evaluation loops, saving, distributed training, logging, plotting.

```Environment.py``` handles rollouts.

```./Agents``` contains self-contained agents.

# :mag: Sample scripts

### RL

[comment]: <> (<details>)

[comment]: <> (<summary><i>Click to open :open_book: </i></summary>)

[comment]: <> (<br>)

Humanoid example: 
```
python Run.py task=dmc/humanoid_run
```

DrQV2 Agent in Atari:
```
python Run.py Agent=Agents.DrQV2Agent task=atari/battlezone
```

SPR Agent in DeepMind Control:
```
python Run.py Agent=Agents.SPRAgent task=dmc/humanoid_walk
```

[comment]: <> (</details>)

### Classification

[comment]: <> (<details>)

[comment]: <> (<summary><i>Click to open :open_book: </i></summary>)

[comment]: <> (<br>)

DQN Agent on MNIST:

```
python Run.py Agent=Agents.DQNAgent task=classify/mnist RL=false
```

*Note:* ```RL=false``` sets training to standard supervised-only classification. Without ```RL=false```, an additional RL phase joins the supervised learning phase s.t. ```reward = -error```. Alternatively, and interestingly, ```supervise=false``` will *only* supervise via RL ```reward = -error``` (**experimental**).

[comment]: <> (The latent optimization could also be done over a learned parameter space as in POPLIN &#40;Wang and Ba, 2019&#41;, which lifts the domain of the optimization problem eq. &#40;1&#41; from Y to the parameter space of a fully-amortized neural network. This leverages the insight that the parameter space of over-parameterized neural networks can induce easier non-convex optimization problems than in the original space, which is also studied in Hoyer et al. &#40;2019&#41;.)

Train accuracies can be printed with ```agent.log=true```.

Evaluation with exponential moving average (EMA) of params can be toggled with the ```ema=true``` flag.

[comment]: <> (Rollouts fill up data in an online fashion, piecemeal, until depletion &#40;all data is processed&#41; and gather metadata like past predictions, which may be useful for curriculum learning.)

[comment]: <> (</details>)

### Generative Modeling

[comment]: <> (<details>)

[comment]: <> (<summary><i>Click to open :open_book: </i></summary>)

[comment]: <> (<br>)

Via the ```generate=true``` flag:
```
python Run.py task=classify/mnist generate=true
```
Implicitly treats as offline, and assumes a replay [is saved](#saving) that can be loaded.

Can also work with RL (due to frame stack, the generated images are technically multi-frame videos), but make sure to change some of the default settings to speed up training, as per below:

```
python Run.py task=atari/breakout generate=true evaluate_episodes=1 action_repeat=1 
```

[comment]: <> (ensemble could help this:)

[comment]: <> (Extensions. Analyzing and extending the amortization components has been a key development in AVI methods. Cremer et al. &#40;2018&#41; investigate suboptimality in these models are categorize it as coming from an amortization gap where the amortized model for eq. &#40;30&#41; does not properly solve it, or the approximation gap where the variational posterior is incapable of approximating the true distribution. Semi-amortization plays a crucial role in addressing the amortization gap and is explored in the semi-amortized VAE &#40;SAVAE&#41; by)

[comment]: <> (Kim et al. &#40;2018&#41; and iterative VAE &#40;IVAE&#41; by Marino et al. &#40;2018&#41;.)

[comment]: <> (</details>)

### Offline RL

[comment]: <> (<details>)

[comment]: <> (<summary><i>Click to open :open_book: </i></summary>)

[comment]: <> (<br>)

From a saved experience replay, sans additional rollouts:

```
python Run.py task=atari/breakout offline=true
```

Assumes a replay [is saved](#saving).

Implicitly treats ```replay.load=true``` and ```replay.save=true```, and only does evaluation rollouts.

[comment]: <> (</details>)

### Saving

[comment]: <> (<details>)

[comment]: <> (<summary><i>Click to open :open_book: </i></summary>)

[comment]: <> (<br>)

Agents can be saved periodically or loaded with the ```save_per_steps=``` or ```load=true``` flags, and are automatically saved at end of training with ```save=true``` by default.

```
python Run.py save_per_steps=100000 load=true
```

An experience replay can be saved or loaded with the ```replay.save=true``` or ```replay.load=true``` flags.

```
python Run.py replay.save=true replay.load=true
```

Agents and replays save to ```./Checkpoints``` and ```./Datasets/ReplayBuffer``` respectively per a unique experiment.

Careful, without ```replay.save=true``` a replay, whether new or loaded, will be deleted upon terminate.

Replays also save uniquely w.r.t. a date-time. In case of multiple saved replays per a unique experiment, the most recent is loaded.

[comment]: <> (</details>)

### Custom Architectures

[comment]: <> (<details>)

[comment]: <> (<summary><i>Click to open :open_book: </i></summary>)

[comment]: <> (<br>)

One can also optionally pass in custom architectures such as those defined in ```./Blocks/Architectures```.

Atari with ViT:

```
python Run.py recipes.Encoder.Eyes=Blocks.Architectures.ViT 
```

ResNet18 on CIFAR-10:

```
python Run.py task=classify/cifar10 RL=false recipes.Encoder.Eyes=Blocks.Architectures.ResNet18 
```

<details>
<summary><i>See more examples :open_book: </i></summary>
<br>

To train, for example MNIST, using a Vision Transformer as the Encoder:

```
python Run.py task=classify/mnist RL=false recipes.Encoder.Eyes=Blocks.Architectures.ViT
```

A GAN with a CNN Discriminator:

```
python Run.py generate=True recipes.Critic.Q_head=Blocks.Architectures.CNN recipes.critic.q_head.input_shape='${obs_shape}' 
```

Here is a more complex example, disabling the Encoder's flattening of the feature map, and instead giving the Actor and Critic unique Attention Pooling operations on their trunks to pool the unflattened features. The ```Null``` architecture disables that flattening component,

```
python Run.py recipes.Critic.trunk=Blocks.Architectures.AttentionPool recipes.Actor.trunk=Blocks.Architectures.AttentionPool task=classify/mnist offline=true recipes.Encoder.pool=Blocks.Architectures.Null

```

since otherwise ```repr_shape``` is flattened to channel dim, with no features for the attention to pool.

</details>

Of course, it's always possible to just modify the code itself, which may be easier. See for example the two CNN variants in ```./Blocks/Encoders.py```.

[comment]: <> (</details>)

### Distributed

[comment]: <> (Automatically parallelizes batches across all visible GPUs. Advanced experimental features described below.)

The simplest way to do distributed training is to use the ```parallel=true``` flag,

```
python Run.py parallel=true 
```

which automatically parallelizes the Encoder's "Eyes" across all visible GPUs. The Encoder is usually the most compute-intensive architectural portion.

To share whole agents across multiple parallel instances,
<details>
<summary><i>Click to open :open_book: </i></summary>
<br>

you can use the ```load_per_steps=``` flag.

For example, a data-collector agent and an update agent,

[comment]: <> (You can share an agent across multiple parallel instances with the ```load_per_steps=``` flag. )

```
python Run.py update_per_steps=0 replay.save=true load_per_steps=1 
```

```
python Run.py offline=true save_per_steps=2
```

in concurrent processes.

Since both use the same experiment name, they will save and load from the same agent and replay, thereby emulating distributed training. **Highly experimental!**
</details>

### Experiment naming, plotting

[comment]: <> (<details>)

[comment]: <> (<summary><i>Click to open :open_book: </i></summary>)

[comment]: <> (<br>)

The ```experiment=``` flag can help differentiate a distinct experiment; you can optionally control which experiment data is automatically plotted with ```plotting.plot_experiments=```.

```
python Run.py experiment=ExpName1 plotting.plot_experiments="['ExpName1']"
```

A unique experiment for benchmarking and saving purposes, is distinguished by: ```experiment=```, ```Agent=```, ```task=```, and ```seed=``` flags.

[comment]: <> (</details>)

# :bar_chart: Agents & Performances

# :interrobang: How is this possible

We use our new Creator framework to unify RL discrete and continuous action spaces, as elaborated in our [paper](https://arxiv.com).

Then we frame actions as "predictions" in supervised learning. We can even augment supervised learning with an RL phase, treating reward as negative error.

For generative modeling, well, it turns out that the difference between a Generator-Discriminator and Actor-Critic is rather nominal.

# :mortar_board: Pedagogy and Research

All files are designed to be useful for educational and innovational purposes in their simplicity and structure.

# :handshake: Contributing

Contributers needed.

Please, donate to help with affording compute and getting Benchmarks ready:

[![Donate](https://img.shields.io/badge/Donate-PayPal-green.svg?style=flat)](https://www.paypal.com/cgi-bin/)

We are a nonprofit, single-PhD student team whose bank account is quickly hemmoraging.

To discuss anything relating to funding or adding new features collaboratively, [please contact **agi.\_\_init\_\_**](mailto:agi.init@gmail.com). Appreciated!

<details>
<summary><i>List of features in progress :open_book: </i></summary>
<br>

in progress

</details>

# Note

### If you are only interested in the RL portion,

Check out our [**UnifiedRL**](https:github.com/agi-init/UnifiedRL) library.

It does with RL to this library what PyCharm does with Python to IntelliJ, i.e., waters it down mildly and rebrands a little.~

<hr class="solid">

[MIT license Included.](MIT_LICENSE)

[comment]: <> (changes from unified rl)
[comment]: <> (changed target to ema)