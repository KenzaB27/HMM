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
for line in sys.stdin:
    chars = line.split(' ')

    mat = []
    for j in range(int(chars[0])):
        row = chars[2 + int(chars[1])*j:2 + (int(chars[1]) * (j + 1))]
        row = [float(el) for el in row]
        mat.append(row)
    matrices.append(mat)

result = matrix_multiplication(
    matrix_multiplication(matrices[2], matrices[0]), matrices[1])[0]
print(1, len(result), end=' ')
for i in range (len(result)):
    print(result[i], end=' ')
