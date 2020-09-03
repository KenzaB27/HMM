def matrix_multiplication (mat_A, mat_B):
    result = []
    for i in range(0,len(mat_A)):
    # iterate through columns of Y
        temp = []
        for j in range(0,len(mat_B[0])):
            s = 0
        # iterate through rows of Y
            for k in range(0,len(mat_A[0])):
                s += mat_A[i][k] * mat_B[k][j]
            temp.append(s)
        result.append(temp)
    return result
