import time
import numpy as np
from homework1 import Hw1Env

env = Hw1Env(render_mode="gui")
for ep in range(3):
    env.reset()
    for t in range(5):
        a = np.random.randint(4)
        env.step(a)
        time.sleep(0.05)