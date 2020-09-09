A = [np.random.dirichlet(np.ones(N), size=1)[0] for _ in range(N)]
B = [np.random.dirichlet(np.ones(M), size=1)[0] for _ in range(N)]
pi = (np.random.dirichlet(np.ones(N), size=1))[0]
print(A)
print()
print(B)
print()
print(pi)

A = [[0.54, 0.26, 0.2], [0.19, 0.53, 0.28], [0.22, 0.18, 0.6]]
B = [[0.5, 0.2, 0.11, 0.19], [0.22, 0.28, 0.23, 0.27], [0.19, 0.21, 0.15, 0.54]]
pi = [0.3, 0.2, 0.5]

# uniform distribution
A = [[1/N for i in range(N)] for j in range(N)]
B = [[1/M for i in range(M)] for j in range(N)]
pi = [1/N for i in range(N)]

# diagonal matrix
A = np.identity(N)
B = [np.random.dirichlet(np.ones(M), size=1)[0] for _ in range(N)]
pi = [0, 0, 1]

# close to the solution
A = [[0.6964767011564664, 0.013355493674583295, 0.29016780516895063], [0.10146488867814683,
                                                                       0.8120130540472764, 0.0865220572745761], [0.19211555554192847, 0.3012796960234883, 0.5066047484345829]]
B = [[0.6887970375846445, 0.2251562809316793, 0.07537025643357922, 0.010676425050097376], [0.06786812278038601, 0.41206670908319215,
                                                                                           0.28139190385771823, 0.23867326427870217], [4.828340972520831e-48, 9.647453456299683e-13, 0.35330163433142786, 0.6466983656676069]]
pi = [0.9999999999999993, 0.0, 0.0]
