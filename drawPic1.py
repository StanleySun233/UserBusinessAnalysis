import matplotlib.pyplot as plt

n = open('trainLoss-LSTM.txt', 'r', encoding='utf-8').readlines()

n1 = n[0].split(" ")[-10:-1]
n2 = n[1].split(" ")[-10:-1]

plt.plot(range(len(n1)), n1, range(len(n1)), n2)
plt.show()
