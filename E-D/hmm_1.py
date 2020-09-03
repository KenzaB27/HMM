import sys

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
alpha0 = [0] * N

# compute alpha0
for i in range(N):
    alpha0[i] = pi[i]*B[i][observations[0]]

# compute alphati
alphat1 = alpha0.copy()
for t in range(1,T):
    alphat = [0] * N
    for i in range(N):
        for j in range(N):
            alphat[i] = alphat[i] + alphat1[j]*A[j][i]
        alphat[i] = alphat[i] * B[i][observations[t]]

    alphat1 = alphat.copy()

print(sum(alphat1))

