import numpy as np
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
#import matplotlib.colors as mcolors
from matplotlib.ticker import AutoMinorLocator
import math as mth

fg = plt.figure(figsize=(15, 12), constrained_layout=True)
gs = gridspec.GridSpec(ncols=3, nrows=10, figure=fg)
ax_1 = fg.add_subplot(gs[0:5, 0])
ax_2 = fg.add_subplot(gs[5: , 0])

ax_3 = fg.add_subplot(gs[0:5, 1])
ax_4 = fg.add_subplot(gs[5: , 1])

ax_5 = fg.add_subplot(gs[0:2, 2])
ax_6 = fg.add_subplot(gs[2:6, 2]) 
ax_7 = fg.add_subplot(gs[6: , 2])


def my_set_axis(AX) :
    AX.grid(which="both", axis='both', linestyle="--", color="gray", linewidth=0.5)
    AX.xaxis.set_minor_locator(AutoMinorLocator())
    AX.yaxis.set_minor_locator(AutoMinorLocator())
    AX.set_ylim(ymin=-1, ymax=1)

def chess_template(Q, N, h) :
    #if N % 2 == 0:
    #    print("excess layer")
    U = np.zeros((N, N), dtype=np.float)
    V = np.zeros((N, N), dtype=np.float)
    P = np.zeros((N, N), dtype=np.float)
    R = np.zeros((N, N), dtype=np.float)
    sled_U = [1.0]
    sled_V = [0.0]
    U[0][0] = 1.0
    U[1][1] = 1.0
    U[1][0] = 1.0
    for i in range(2, N) :
        U[i][i] = 1.0
        for j in range(0 + i%2, i, 2) :
            if j == 0 :
                U[i][0] = U[i - 2][0] + 2 * h * P[i - 2][0]
                V[i][0] = V[i - 2][0] + 2 * h * R[i - 2][0]
                sled_U.append(U[i][0])
                sled_V.append(V[i][0])
            else :
                U[i][j] = U[i - 1][j - 1] + h * P[i - 1][j - 1]
                V[i][j] = V[i - 1][j - 1] + h * R[i - 1][j - 1]
            P[i][j] = P[i - 1][j + 1] - Q[j + 1] * (U[i - 1][j + 1] - V[i - 1][j + 1])
            R[i][j] = R[i - 1][j + 1] - Q[j + 1] * (V[i - 1][j + 1] - U[i - 1][j + 1])
        for j in range(0 + i%2, i + i%2 , 2) :
            if i % 2 == 0 :
                U[i][j + 1] = U[i][j + 2]
                V[i][j + 1] = V[i][j + 2]
                P[i][j + 1] = P[i][j + 2]
                R[i][j + 1] = R[i][j + 2]
            else :
                U[i][j - 1] = U[i][j]
                V[i][j - 1] = V[i][j]
                P[i][j - 1] = P[i][j]
                R[i][j - 1] = R[i][j]
    return U, V, P, R, sled_U, sled_V

N = 200
h = 0.02
fg.suptitle('N = ' + str(N) + ', h = ' + str(h))

if N % 2 != 0 :
    print('uncorrect N')

Q = np.zeros(N, dtype=np.float)

ampl_sin = 0.2
k_sin =  0.05  #1.0 / 45.0
for i in range(0, N) :
    value = (ampl_sin * mth.sin(mth.pi * i * k_sin)) + ((ampl_sin / 2.0) *  mth.sin(mth.pi * i * k_sin * 2.0) ) + ((ampl_sin / 4.0) *  mth.sin(mth.pi * i * k_sin * 4.0) )
    if value < 0 :
        Q[i] = 0
    else :
        Q[i] = value 

U, V, P, R, sled_U, sled_V = chess_template(Q, N, h)
U = U.tolist()
V = V.tolist()
P = P.tolist()
R = R.tolist()
U.reverse()
V.reverse()
P.reverse()
R.reverse()

num_correct, num_uncorrect = 0, 0
for i in range(0, N) :
    for j in range(0, N - i) :
        if abs(U[i][j]) > 1.0 :
            print('Error', end=' ')
        if abs(V[i][j]) > 1.0 :
            print('Error', end=' ')
        val = U[i][j] + V[i][j]
        if val <= 1.000000001 and val >= 0.999999999 :
            num_correct += 1
        if R[i][j] + P[i][j] == 0:
            num_uncorrect += 1
print(num_correct, num_uncorrect)

print(len(sled_U))

def solve_inverse(N, sled, diagW, h):
    print('N: ' + str(N) + ' diag: ' + str(diagW) + ' h: ' + str(h))
    q = np.zeros(N)
    W = np.zeros((2 * N, N))
    R = np.zeros((2 * N, N))
    for i in range(0, N) :
        W[i][i] = diagW
    for i in range(0, len(sled)) :
        W[2*i][0] = sled[i]
    print(W)

    for i in range(0, N) :
        sum_tmp = 0
        for n in range(0, i + 1) :  #sum 0-i
            a = 2*(i + 1) - n
            print(i, n, a)
            sum_tmp += q[n] * W[a][n] * h
        q[i] = (-1 * sled[i] - sum_tmp) / (W[i + 2][i] * h)  # sled[m]   W[m+1][m-1]
        for k in range(0, i + 1) :
            a = 2*(i + 1) - k
            R[a][k] = R[a - 1][k + 1] - q[k] * W[a][k] * h
        for k in range(1, i + 1 + 1) :
            a = 2*(i + 2) - k
            W[a][k] = W[a - 1][k - 1] + R[a - 1][k - 1] * h
    return q


#solve_inverse(N//2, sled_U, 1, h)
print(solve_inverse(10, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 1, h))



#plt.show()
