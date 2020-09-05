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

def alpha_pass ():
    alpha = [[0 for i in range(N)] for j in range(T)]
    ct = [0] * T
    # compute alpha0
    for i in range(N):
        alpha[0][i] = pi[i]*B[i][observations[0]]
        ct[0] = ct[0] + alpha[0][i]

    # scale alpha0
    ct[0] = 1/ct[0]
    for i in range(N):
        alpha[0][i] = ct[0]*alpha[0][i]

    # compute alphati
    for t in range(1, T):
        for i in range(N):
            for j in range(N):
                alpha[t][i] = alpha[t][i] + alpha[t-1][j]*A[j][i]
            alpha[t][i] = alpha[t][i] * B[i][observations[t]]
            ct[t] = ct[t] + alpha[t][i]

        # scale alphat[i]
        ct[t] = 1/ct[t]
        for i in range(N):
            alpha[t][i] = alpha[t][i] * ct[t]

    return alpha, ct

def beta_pass(): 
    beta = [[0 for i in range(N)] for j in range(T)]
    # BetaT scaled by cT
    for i in range(N):
        beta[T-1][i] = ct[T-1]

    # beta_pass
    for t in range(T-2, 0, -1):
        for i in range(N):
            for j in range(N):
                beta[t][i] = beta[t][i] + A[i][j] * \
                    B[j][observations[t+1]] * beta[t+1][j]

            # scale beta
            beta[t][i] = ct[t] * beta[t][i]

    return beta

def sigma_pass():
    sigmat3 = [[[0 for i in range(N)] for j in range(N)] for k in range(T)]
    sigmat2 = [[0 for i in range(N)] for j in range(T)]
    for t in range(T-1):
        for i in range(N):
            for j in range(N):
                sigmat3[t][i][j] = alpha[t][i] * A[i][j] * B[j][observations[t+1]] * beta[t+1][j] 
                sigmat2[t][i] = sigmat2[t][i] + sigmat3[t][i][j]

    # special case for T-1
    for i in range(N):
        sigmat2[T-1][i] = alpha[T-1][i]

    return sigmat2, sigmat3

def reestimate():
    for i in range(N):
        pi[i] = sigmat2[0][i]
    
    for i in range(N):
        denom = 0
        for t in range(T-1):
            denom = denom + sigmat2[t][i]
        
        for j in range(N):
            numer = 0
            for t in range(T-1):
                numer = numer + sigmat3[t][i][j]

            A[i][j] = numer / denom

    for i in range(N):
        denom = 0
        for t in range(T):
            denom = denom + sigmat2[t][i]

        for j in range(M):
            numer = 0
            for t in range(T):
                if observations[t] == j:
                    numer = numer + sigmat2[t][i]
            B[i][j] = numer / denom

def comp_log():
    log_prob = 0
    for i in range(T):
        log_prob = log_prob + math.log(ct[i])
    log_prob = -log_prob
    return log_prob

max_iters = 100
old_log_prob = float('-inf')

for i in range(max_iters):
    alpha, ct = alpha_pass()
    log_prob = comp_log()
    if log_prob <= old_log_prob:
        break
    old_log_prob = log_prob
    beta = beta_pass()
    sigmat2, sigmat3 = sigma_pass()
    reestimate()

print(N, N, end=' ')
for i in range(N):
    for j in range(N):
        print(A[i][j], end=' ')
print()
print(N, M, end=' ')
for i in range(N):
    for j in range(M):
        print(B[i][j], end=' ')
