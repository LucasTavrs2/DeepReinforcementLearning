import gymnasium as gym              # cria ambiente de RL
import torch                         # base do PyTorch
import torch.nn as nn                # para definir camadas de rede neural
import torch.optim as optim          # otimizadores (Adam, SGD, etc.)


env = gym.make("ALE/SpaceInvaders-v5")

#Definir arquitetura da rede neural
class Network(nn.Module):
    def __init__(self, dim_inputs, dim_outputs):
        super(Network, self).__init__()
        self.linear = nn.Linear(dim_inputs, dim_outputs)

    def forward(self, x):
        return self.linear(x)
    
#Instanciando rede

network = Network(dim_inputs, dim_outputs)

#Instanciando otimizador
optimizer = optim.Adam(network.parameters(), lr=0.0001)

for episode in range(1000):
    state, info = env.reset()
    done = False
    while not done:
        action = select_action(action, state)  # Função para selecionar ação com base na política
        next_state, reward, terminated, truncated, _ = (env.step(action))
        done = terminated or truncated
        loss = calculate_loss(network, state, action, next_state, reward, done)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        state = next_state


#Implementando o Q-Network
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
