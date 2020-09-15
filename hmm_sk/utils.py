import sys
import math
import numpy as np
from difflib import SequenceMatcher

SUFFICIENTLY_SMALL_NUMBER = 10**-5
SUFFICIENTLY_BIG_NUMBER = 2**30

def alpha_pass(A, B, pi, observations):
    N = len(A)
    M = len(B)
    T = len(observations)

    alpha = [[0 for i in range(N)] for j in range(T)]
    ct = [0] * T
    # compute alpha0
    for i in range(N):
        alpha[0][i] = pi[i]*B[i][observations[0]]
        ct[0] = ct[0] + alpha[0][i]

    # scale alpha0
    ct[0]= 1/ct[0]
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


def beta_pass(A, B, observations, ct):
    N = len(A)
    T= len(observations)
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


def sigma_pass(A, B, alpha, beta, observations):
    N = len(A)
    T = len(observations)
    sigmat3 = [[[0 for i in range(N)] for j in range(N)] for k in range(T)]
    sigmat2 = [[0 for i in range(N)] for j in range(T)]
    for t in range(T-1):
        for i in range(N):
            for j in range(N):
                sigmat3[t][i][j] = alpha[t][i] * A[i][j] * \
                    B[j][observations[t+1]] * beta[t+1][j]
                sigmat2[t][i] = sigmat2[t][i] + sigmat3[t][i][j]

    # special case for T-1
    for i in range(N):
        sigmat2[T-1][i] = alpha[T-1][i]

    return sigmat2, sigmat3


def reestimate(A, B, observations, pi, sigmat2, sigmat3):
    N = len(A)
    M = len(B)
    T = len(observations)
    new_A = A.copy()
    new_B = B.copy()
    new_pi = pi.copy()
    for i in range(N):
        new_pi[i] = sigmat2[0][i]

    for i in range(N):
        denom = 0
        for t in range(T-1):
            denom = denom + sigmat2[t][i]

        for j in range(N):
            numer = 0
            for t in range(T-1):
                numer = numer + sigmat3[t][i][j]
                
            new_A[i][j] = numer / (denom + SUFFICIENTLY_SMALL_NUMBER)
        
        norm = sum(A[i])
        for j in range(N):
            A[i][j] /= norm

    for i in range(N):
        denom = 0
        for t in range(T):
            denom = denom + sigmat2[t][i]

        for j in range(M):
            numer = 0
            for t in range(T):
                if observations[t] == j:
                    numer = numer + sigmat2[t][i]
                    
            new_B[i][j] = numer / (denom + SUFFICIENTLY_SMALL_NUMBER)
        
        norm = sum(B[i])
        for j in range(M):
            B[i][j] /= norm

    return new_A, new_B, new_pi


def comp_log(ct):
    T = len(ct)
    log_prob = 0
    for i in range(T):
        log_prob = log_prob + math.log(ct[i])
    log_prob = -log_prob
    return log_prob


def baum_welch(A, B, pi, observations):
    max_iters = 350
    old_log_prob = float('-inf')
    new_A = A.copy()
    new_B = B.copy()
    new_pi = pi.copy()
    for i in range(max_iters):
        alpha, ct = alpha_pass(new_A, new_B, new_pi, observations)
        log_prob = comp_log(ct)
        beta = beta_pass(new_A, new_B, observations, ct)
        if log_prob <= old_log_prob:
            print('iter', i)
            break
        old_log_prob = log_prob
        sigmat2, sigmat3 = sigma_pass(new_A, new_B, alpha, beta, observations)
        new_A, new_B, new_pi = reestimate(
            new_A, new_B, observations, new_pi, sigmat2, sigmat3)
    return new_A, new_B, new_pi


def viterbi(A, B, pi, observations): 
    N = len(A)
    T = len(observations)

    delta = [[0 for i in range(N)] for j in range(T)]
    for i in range(N):
        delta[0][i] = B[i][observations[0]] * pi[i]

    max_indexes = [[-1 for i in range(N)] for j in range(T)]
    for t in range(1, T):
        for i in range(N):
            max_delta = float('-inf')
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

    return path

def match_pattern(seq_1, seq_2, min_pattern):
    matcher = SequenceMatcher(None, seq_1, seq_2)
    matches = matcher.find_longest_match(0, len(seq_1), 0, len(seq_2))
    if matches.size >= min_pattern:
        return seq_1[matches.a:matches.a + matches.size]
    return None