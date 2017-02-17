class NeuralNetwork:
    def __init__(self, num_features, hidden_size, seed=None): # фич будем брать 65
        env = gym.make('Blocks-v0')
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.init_weights(seed)

    def init_weights(self, seed):
        np.random.seed(seed)
        self.W1 = np.abs(np.random.normal(size=(self.num_features, self.hidden_size), scale=0.1))
        self.b1 = np.abs(np.random.normal(size=self.hidden_size))
        self.W2 = np.abs(np.random.normal(size=(self.hidden_size, 1), scale=0.1))
        self.b2 = np.abs(np.random.normal(size=64))  # на выход подается массив из 64 элементов
        # идея в том, чтоб подавать на вход и получать матрицу ожидаемых выигрышей, которая состоин из 16*4=64 элементов
        # нумерация следующая - первые 4 элемента - ожидаемые выигрыше при каждом действии в начальной точке, следующая четверка -
        # из состояние с индексом 1 и так далее.
        # следовательно,на выходе должны получаться та же таблица но с верными ожидаемыми выигрышами.



        global p  # число сыгранных игр
        global q  # число побед
        global m  # число шагов в каждой игре
        global error  # производная ошибки
        global n_steps  # массив с числом шагов

        if d == True and r == 0:  # если проигрыш, то ошибка равна 1
            s = env.reset()
            error = 1
            n_steps.append(m)
            p += 1  # увеличиваем число игр
            m = 1
        elif d == True and r == 1:  # при выигрыше ошибка равняется 0
            s = env.reset()
            error = 0
            q += 1  # увеличиваем число побед
            n_steps.append(m)  # записываем в список число шагов до конца игры, для статистики
            m = 1
        else:
            m += 1  # число проделанных шагов
            error = 1 - m / 20  # если игра продолжается, то ошибка уменьшается с ростом числа проделанных шагов

        db2 = np.sum(error)  # обратное распространение ошибки
        dW2 = activation.T.dot(error)

        dhidden = self.relu_grad(hidden) * error * self.W2.T
        db1 = dhidden.mean(0)
        dW1 = np.array([X.T]).T.dot(dhidden)

        return dW1, db1, dW2, db2

    def gradient_descent(self, X, y, learning_rate, s):
        dW1, db1, dW2, db2 = self.loss_gradient(X, y, s)
        self.W1 -= dW1 * learning_rate  # обновляем параметры
        self.b1 -= db1 * learning_rate
        dW2 = dW2 * learning_rate
        for i in range(100):  # 100 так как сто нейроной в скрытом слое
            self.W2[i][0] -= dW2[i]
        self.b2 -= db2 * learning_rate

    def fit(self, X, y, epochs, learning_rate):
        s = env.reset()
        s, r, d, _ = env.step(int(np.random.choice(4, 1)))  # первый шаг делаем случайно
        for epoch in range(epochs):
            self.gradient_descent(X, y, learning_rate / (epoch + 1),
                                  s)  # добавил learning decay, при каждой эпохе темп
            # изменения параметром будет снижаться