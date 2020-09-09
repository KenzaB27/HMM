import sys
import math
import numpy as np
matrices = []
observations = []
N = 3
M = 4
T = 0
for line in sys.stdin:
    chars = line.split(' ')
    T = int(chars[0])
    observations = [int(chars[i]) for i in range(1, len(chars)-1)]

A = [[0.6964767011564664, 0.013355493674583295, 0.29016780516895063], [0.10146488867814683,
                                                                       0.8120130540472764, 0.0865220572745761], [0.19211555554192847, 0.3012796960234883, 0.5066047484345829]]
B = [[0.6887970375846445, 0.2251562809316793, 0.07537025643357922, 0.010676425050097376], [0.06786812278038601, 0.41206670908319215,
                                                                                           0.28139190385771823, 0.23867326427870217], [4.828340972520831e-48, 9.647453456299683e-13, 0.35330163433142786, 0.6466983656676069]]
pi = [0.9999999999999993, 0.0, 0.0]

print(A)
print()
print(B)
print()
print(pi)
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

max_iters = 5000
old_log_prob = float('-inf')

for i in range(max_iters):
    alpha, ct = alpha_pass()
    log_prob = comp_log()
    beta = beta_pass()
    if log_prob <= old_log_prob:
        print('iter', i)
        break
    old_log_prob = log_prob
    sigmat2, sigmat3 = sigma_pass()
    reestimate()

# print(N, N, end=' ')
# for i in range(N):
#     for j in range(N):
#         print(A[i][j], end=' ')
# print()
# print(N, M, end=' ')
# for i in range(N):
#     for j in range(M):
#         print(B[i][j], end=' ')

print('Convergence')
print(A)
print()
print(B)
print()
print(pi)
