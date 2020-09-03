import sys


def matrix_multiplication(mat_A, mat_B):
    result = []
    for i in range(0, len(mat_A)):
        # iterate through columns of Y
        temp = []
        for j in range(0, len(mat_B[0])):
            s = 0
        # iterate through rows of Y
            for k in range(0, len(mat_A[0])):
                s += mat_A[i][k] * mat_B[k][j]
            temp.append(s)
        result.append(temp)
    return result


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
c0 = 0
alpha0 = [0] * N
# compute alpha0
for i in range(N):
    alpha0[i] = pi[i]*B[i][observations[0]]
    c0 = c0 + alpha0[i]
print('before scaling ', alpha0)    
# scale alpha0
c0 = 1/c0
for i in range(N):
    alpha0[i] = c0*alpha0[i]
print(alpha0)
# compute alphati
alphat1 = alpha0.copy()
for t in range(1,T):
    ct = 0
    alphat = [0] * N
    for i in range(N):
        for j in range(N):
            alphat[i] = alphat[i] + alphat1[j]*A[j][i]
        alphat[i] = alphat[i] * B[i][observations[t]]
        ct = ct + alphat[i]

    # scale alphat[i]
    ct = 1/ct
    for i in range(N):
        alphat[i] = alphat[i] * ct

    alphat1 = alphat.copy()

print(alphat1)
print(sum(alphat1))

