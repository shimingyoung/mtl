def norm_11(a):
    m, n = a.shape
    for i in range(n):
        x = np.sum(a[:,i])
    y = np.sum(x)
    return(y)

def norm_1_infinity(a):
    print(a)
    y =[]
    m,n = a.shape
    for i in range(n):
        x = np.max(a[:, i])
        y.append(x)
    print(y)
    y = np.sum(x)
    return(y)

def proximal_norm_11(a, lambda1, t, gradient):
    matrix_norm = a - t*gradient
    m,n = matrix_norm.shape
    matrix_norm_1 = np.zeros(m,n)
    for i in range(m):
        for j in range(n):
            if matrix_norm[i][j] > lambda1*t:
                matrix_norm_1[i][j] = matrix_norm[i][j] + lambda1*t
            else:
                matrix_norm_1[i][j] = matrix_norm_1[i][j] - lambda1*t
    return(matrix_norm_1)

def proximal_norm_1_infinity(a, lambda1, t, gradient):
    matrix_norm = a - t*gradient
    m,n = matrix_norm.shape
    matrix_norm_1 = np.zeros(m,n)
    for i in range(m):
        for j in range(n):
            if matrix_norm[i][j] > lambda1*t:
                matrix_norm_1[i][j] = matrix_norm[i][j] + lambda1*t
            else:
                matrix_norm_1[i][j] = matrix_norm_1[i][j] - lambda1*t
    return(matrix_norm_1)
 