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

# compute delta0
for i in range(N):
    delta = pi[i]*B[i][observations[0]]
    deltas[0][i] = math.log(delta) if delta >0 else delta

# compute deltai
for t in range(1, T):
    for i in range(N):
        for j in range(N):
            delta = B[i][observations[0]] * A[j][i]
            newVal = deltas[t-1][j] + (math.log(delta) if delta > 0 else delta)
            if deltas[t][i] < newVal:
                maxIndexes[t][i] = j
                deltas[t][i] = newVal
print(deltas)
print(maxIndexes)
most_likely_states = [0]*T
for j in range(N-1):
    if deltas[T-1][j] > deltas[T-1][j+1]:
        most_likely_states[T-1] = j
    elif deltas[T-1][j] < deltas[T-1][j+1]:
        most_likely_states[T-1] = j + 1

for i in range(T-1, 1, -1):
    most_likely_states[i-1] = maxIndexes[i][most_likely_states[i]]

for i in range(len(most_likely_states)):
    print(most_likely_states[i], end=' ')
