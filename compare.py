from time import sleep
import numpy as np
import random
import os


def compare(file1, file2):
    open_file1 = open(file1, "r")
    open_file2 = open(file2, "r")
    for line1, line2 in zip(open_file1, open_file2):
        if line1 != line2:
            return False
    return True


for i in range(1):
    M = random.randint(100, 1000)
    N = random.randint(100, 1000)
    P = random.randint(100, 1000)

    matrix1 = np.random.randint(0, 100, (M, N))
    # sparse
    matrix2 = np.random.randint(0, 100, (N, P))
    for i in range(N):
        for j in range(P):
            if random.random() < 0.99:
                matrix2[i][j] = 0
    K = 0
    for i in range(N):
        for j in range(P):
            if matrix2[i][j] != 0:
                K += 1

    mul = np.dot(matrix1, matrix2)

    with open("mul.txt", "w") as f:
        for i in range(M):
            for j in range(P):
                f.write(str(mul[i][j]) + " ")
            f.write("\n")

    # 保存矩阵到文件中，第一行保存M N P K
    with open("matrix.txt", "w") as f:
        f.write(str(M) + " " + str(N) + " " + str(P) + " " + str(K) + "\n")
        for i in range(M):
            for j in range(N):
                f.write(str(matrix1[i][j]) + " ")
            f.write("\n")
        for i in range(N):
            for j in range(P):
                if matrix2[i][j] != 0:
                    f.write(str(i) + " " + str(j) + " " + str(matrix2[i][j]) + "\n")

    # 调用./sparse_matrix_multiplication
    os.system("./sparse_matrix_multiplication < matrix.txt > result.txt")
    # os.system("./sparse_matrix_multiplication < case.in > result.txt")

    if compare("mul.txt", "result.txt"):
        print("Correct")
    else:
        print("Wrong")
