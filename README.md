# gym-ultra3d

The Ultra-3D environment is a single agent domain featuring continuous state and discrete action spaces. The goal is to teach an agent to learn the shortest sequence of probe movements that would result in the capture of an optimal AP4 2D image. We use a DQN agent to explore the 3D ultrasound space, while feeding 2D cross-sections into a simple deep CNN network. The reinforcement learning makes use of the [keras-rl](https://keras-rl.readthedocs.io/en/latest/agents/overview/) python library, integrated with a custom implementation of [OpenAI's gym environment](https://gym.openai.com/docs/).

# To run the demo

```bash
cd gym-ultra3d
pip3 install -r requirements.txt
python3 dqn_ultra.py --mode=test
```
