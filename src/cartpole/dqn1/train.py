import agent
import tools
import torch
from envmanage import EnvManager
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
em = EnvManager(device)
em.reset()
em.get_state()

for i in range(5):
    em.take_action(torch.tensor([1]))
screen = em.get_state()

plt.figure()
plt.imshow(screen.squeeze(0).permute(1,2,0).cpu(), interpolation='none')
plt.title("Processsed screen")
plt.show()
