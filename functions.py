import numpy as np


class Q_Learning:
    # ENTRADAS:
    # env - entorno Cart Pole
    # alpha - tamaño del paso
    # gamma - tasa de descuento
    # epsilon - parámetro para el enfoque epsilon-greedy
    # numberEpisodes - número total de episodios de simulación

    # numberOfBins - esta es una lista de 4 dimensiones que define el número de puntos de la cuadrícula
    # para la discretización de los estados
    # es decir, esta lista contiene el número de compartimentos para cada entrada de estado,
    # tenemos 4 entradas, es decir,
    # discretización para la posición del carrito, la velocidad del carrito, el ángulo del poste y la velocidad angular del poste

    # lowerBounds - límites inferiores para la discretización, lista con 4 entradas:
    # límites inferiores en la posición del carrito, velocidad del carrito, ángulo del poste y velocidad angular del poste

    # upperBounds - límites superiores para la discretización, lista con 4 entradas:
    # límites superiores en la posición del carrito, velocidad del carrito, ángulo del poste y velocidad angular del poste
    def __init__(self, env, alpha, gamma, epsilon, numberEpisodes, numberOfBins, lowerBounds, upperBounds):
        import numpy as np
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.actionNumber = env.action_space.n
        self.numberEpisodes = numberEpisodes
        self.numberOfBins = numberOfBins
        self.lowerBounds = lowerBounds
        self.upperBounds = upperBounds
        self.sumRewardsEpisode = []
        self.Qmatrix = np.random.uniform(low=0, high=1, size=(
            numberOfBins[0], numberOfBins[1], numberOfBins[2], numberOfBins[3], self.actionNumber))

    def returnIndexState(self, state):
        position = state[0]
        velocity = state[1]
        angle = state[2]
        angularVelocity = state[3]

        cartPositionBin = np.linspace(self.lowerBounds[0], self.upperBounds[0], self.numberOfBins[0])
        cartVelocityBin = np.linspace(self.lowerBounds[1], self.upperBounds[1], self.numberOfBins[1])
        poleAngleBin = np.linspace(self.lowerBounds[2], self.upperBounds[2], self.numberOfBins[2])
        poleAngleVelocityBin = np.linspace(self.lowerBounds[3], self.upperBounds[3], self.numberOfBins[3])

        indexPosition = np.maximum(np.digitize(state[0], cartPositionBin) - 1, 0)
        indexVelocity = np.maximum(np.digitize(state[1], cartVelocityBin) - 1, 0)
        indexAngle = np.maximum(np.digitize(state[2], poleAngleBin) - 1, 0)
        indexAngularVelocity = np.maximum(np.digitize(state[3], poleAngleVelocityBin) - 1, 0)

        return tuple([indexPosition, indexVelocity, indexAngle, indexAngularVelocity])

    def selectAction(self, state, index):
        # en los primeros 500 episodios seleccionamos acciones completamente aleatorias para tener suficiente exploración
        if index < 500:
            return np.random.choice(self.actionNumber)
        randomNumber = np.random.random()
        # Devuelve un número real aleatorio en el intervalo semabierto [0.0, 1.0)
        # este número se utiliza para el enfoque epsilon-greedy

        # después de 2000 episodios, empezamos a disminuir lentamente el parámetro epsilon
        if index > 2000:
            self.epsilon = 0.999 * self.epsilon
            # si se cumple esta condición, estamos explorando, es decir, seleccionamos acciones aleatorias
        if randomNumber < self.epsilon:
            # devuelve una acción aleatoria seleccionada de: 0,1,...,actionNumber-1
            return np.random.choice(self.actionNumber)
        else:
            return np.random.choice(np.where(
                self.Qmatrix[self.returnIndexState(state)] == np.max(self.Qmatrix[self.returnIndexState(state)]))[0])

    def simulateEpisodes(self):
        import numpy as np
        for indexEpisode in range(self.numberEpisodes):
            rewardsEpisode = []
            (stateS, _) = self.env.reset()
            stateS = list(stateS)

            terminalState = False
            while not terminalState:
                stateSIndex = self.returnIndexState(stateS)
                actionA = self.selectAction(stateS, indexEpisode)
                (stateSprime, reward, terminalState, _, _) = self.env.step(actionA)
                rewardsEpisode.append(reward)
                stateSprime = list(stateSprime)
                stateSprimeIndex = self.returnIndexState(stateSprime)
                QmaxPrime = np.max(self.Qmatrix[stateSprimeIndex])

                if not terminalState:
                    error = reward + self.gamma * QmaxPrime - self.Qmatrix[stateSIndex + (actionA,)]
                    self.Qmatrix[stateSIndex + (actionA,)] = self.Qmatrix[stateSIndex + (actionA,)] + self.alpha * error
                else:
                    error = reward - self.Qmatrix[stateSIndex + (actionA,)]
                    self.Qmatrix[stateSIndex + (actionA,)] = self.Qmatrix[stateSIndex + (actionA,)] + self.alpha * error

                stateS = stateSprime

            print("Episode: {}, score: {}, e: {}".format(indexEpisode + 1, np.sum(rewardsEpisode),
                                                         self.epsilon))
            self.sumRewardsEpisode.append(np.sum(rewardsEpisode))

    def simulateLearnedStrategy(self):
        import gym
        import time
        env1 = gym.make('CartPole-v1', render_mode='human')
        currentState, _ = env1.reset()
        env1.render()
        timeSteps = 10000
        rewardsEpisode = []
        accumulatedScore = 0

        for timeIndex in range(timeSteps):
            actionInStateS = np.random.choice(
                np.where(self.Qmatrix[self.returnIndexState(currentState)] == np.max(
                    self.Qmatrix[self.returnIndexState(currentState)]))[0])
            currentState, reward, terminated, truncated, info = env1.step(actionInStateS)
            accumulatedScore += reward
            rewardsEpisode.append(accumulatedScore)
            time.sleep(0.05)
            if terminated:
                time.sleep(1)
                break

        episodeRewards = np.diff([0] + rewardsEpisode).tolist()

        return episodeRewards, env1
