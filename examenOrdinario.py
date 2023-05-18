import gym
import numpy as np
import time
import matplotlib.pyplot as plt

from functions import Q_Learning

env = gym.make('CartPole-v1', render_mode='human')
(state, _) = env.reset()

upperBounds = env.observation_space.high
lowerBounds = env.observation_space.low
cartVelocityMin = -3
cartVelocityMax = 3
poleAngleVelocityMin = -10
poleAngleVelocityMax = 10
upperBounds[1] = cartVelocityMax
upperBounds[3] = poleAngleVelocityMax
lowerBounds[1] = cartVelocityMin
lowerBounds[3] = poleAngleVelocityMin

numberOfBinsPosition = 30
numberOfBinsVelocity = 30
numberOfBinsAngle = 30
numberOfBinsAngleVelocity = 30
numberOfBins = [numberOfBinsPosition, numberOfBinsVelocity, numberOfBinsAngle, numberOfBinsAngleVelocity]

alpha = 0.1
gamma = 1
epsilon = 1
numberEpisodes = 3000
learnedEpisodes = 20

Q1 = Q_Learning(env, alpha, gamma, epsilon, numberEpisodes, numberOfBins, lowerBounds, upperBounds)

Q1.simulateEpisodes()

# Lista para almacenar los puntajes obtenidos en cada episodio
episodeScores = []

# Ejecutar la estrategia aprendida en 20 episodios
for episode in range(learnedEpisodes):
    (obtainedRewardsOptimal, env1) = Q1.simulateLearnedStrategy()
    totalReward = np.sum(obtainedRewardsOptimal)
    episodeScores.append(totalReward)
    print(f"Episode: {episode + 1}, Total Reward: {totalReward}")

# Graficar los puntajes obtenidos en cada episodio
episodes = np.arange(1, learnedEpisodes + 1)
plt.plot(episodes, episodeScores, color='blue', linewidth=1)
plt.xlabel('Episodio')
plt.ylabel('Puntaje')
plt.title('Puntaje Obtenido en cada Episodio')
plt.savefig('puntaje_episodios.png')
plt.show()

# Cerrar el entorno env1
env1.close()
