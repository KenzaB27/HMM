import sys
import math
matrices = []
observations = []
N = 0
M = 0
T = 0
for line in sys.stdin:
    chars = line.split(' ')

    if len(matrices) == 3:
        T = int(chars[0])
        observations = [int(chars[i]) for i in range(1, len(chars)-1)]
        break
    if(len(matrices)) == 0:
        N = int(chars[0])
    if(len(matrices)) == 1:
        M = int(chars[1])
    mat = []
    for j in range(int(chars[0])):
        row = chars[2 + int(chars[1])*j:2 + (int(chars[1]) * (j + 1))]
        row = [float(el) for el in row]
        mat.append(row)
    matrices.append(mat)

A = matrices[0]
B = matrices[1]
pi = matrices[2][0]
deltas = [[0 for i in range(N)] for j in range(T)]
maxIndexes = [[-1 for i in range(N)] for j in range(T)]

delta = [[0 for i in range(N)] for j in range(T)]
for i in range(N):
    delta[0][i] = B[i][observations[0]] * pi[i]

max_indexes = [[-1 for i in range(N)] for j in range(T)]
for t in range(1, T):
    for i in range(N):
        max_delta = 0
        max_index = -1
        for j in range(N):
            tmp_delta = A[j][i] * B[i][observations[t]] * delta[t - 1][j]
            if tmp_delta > max_delta:
                max_delta = tmp_delta
                max_index = j
        delta[t][i] = max_delta
        max_indexes[t][i] = max_index

path = [-1] * T
tmp_max = 0
for i in range(N):
    if delta[T - 1][i] > tmp_max:
        path[T - 1] = i
        tmp_max = delta[T - 1][i]

for t in range(T-2, -1, -1):
    path[t] = max_indexes[t + 1][path[t + 1]]

for state in path:
    print(state, end=' ')
