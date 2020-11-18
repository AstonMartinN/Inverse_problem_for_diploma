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

def wave_step(U, V, P, R, Q, N, h, step) :
    U2 = np.zeros(N, dtype=np.float)
    V2 = np.zeros(N, dtype=np.float)
    P2 = np.zeros(N, dtype=np.float)
    R2 = np.zeros(N, dtype=np.float)
    U2[0] = U[0] + (h * P[0])
    V2[0] = V[0] + (h * R[0])
    for i in range(1, step) :
        U2[i] = U[i - 1] + (h * P[i - 1])
        V2[i] = V[i - 1] + (h * R[i - 1])
    U2[step] = 1 #V2[step] = 0 P2[step] = 0 R2[step] = 0
    for i in range(0, step) :
        P2[i] = P[i + 1] - Q[i] * (U[i] - V[i])
        R2[i] = R[i + 1] - Q[i] * (V[i] - U[i])
    return U2, V2, P2, R2

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

Q = np.zeros(N, dtype=np.float)

ampl_sin = 0.2
k_sin =  0.03  #1.0 / 45.0
for i in range(0, N) :
    #value = (ampl_sin * mth.sin(mth.pi * i * k_sin)) + ((ampl_sin / 2.0) *  mth.sin(mth.pi * i * k_sin * 2.0) ) + ((ampl_sin / 4.0) *  mth.sin(mth.pi * i * k_sin * 4.0) )
    value = ampl_sin * mth.sin(mth.pi * i * k_sin)
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

ax_1.set_title('U(x, t)')
ax_1.set_xlabel('Узлы')
ax_1.set_ylabel('Слои')
img_1 = ax_1.imshow(U, cmap='hsv', vmin=-1, vmax=1)#, interpolation='bilinear')
fg.colorbar(img_1, ax=ax_1)

ax_2.set_title('V(x, t)')
ax_2.set_xlabel('Узлы')
ax_2.set_ylabel('Слои')
img_2 = ax_2.imshow(V, cmap='gist_rainbow', vmin=-1, vmax=1)#, interpolation='bilinear')
fg.colorbar(img_2, ax=ax_2)

ax_3.set_title('P(x, t)')
ax_3.set_xlabel('Узлы')
ax_3.set_ylabel('Слои')
img_3 = ax_3.imshow(P, cmap='hsv', vmin=-1, vmax=1)#, interpolation='bilinear')
fg.colorbar(img_3, ax=ax_3)

ax_4.set_title('R(x, t)')
ax_4.set_xlabel('Узлы')
ax_4.set_ylabel('Слои')
img_4 = ax_4.imshow(R, cmap='gist_rainbow', vmin=-1, vmax=1)#, interpolation='bilinear')
fg.colorbar(img_4, ax=ax_4)

ax_5.set_ylim(ymin=-0.05, ymax=1)
ax_5.plot(Q, color='red')
ax_5.set_xlabel('x')
#ax_5.set_title('q(x) = A * (sin(pi * k * x) + sin(2 * pi * k * x)), A = ' + str(ampl_sin) + ' k = ' + str(k_sin))
ax_5.set_title('q(x) = A * sin(pi * k * x), A = ' + str(ampl_sin) + ' k = ' + str(k_sin))
ax_5.plot([0, len(Q)], [0, 0], color='lime')



ax_6.set_ylim(ymin=-1, ymax=1.2)
ax_6.set_title('След U(0, t)')
ax_6.set_xlabel('t')
ax_6.plot(sled_U, color='black')
ax_7.set_ylim(ymin=-1, ymax=1)
ax_7.set_title('След V(0, t)')
ax_7.set_xlabel('t')
ax_7.plot(sled_V, color='black')
null_y_axis = ([0, len(sled_U)],[0, 0])
ax_6.plot(null_y_axis[0], null_y_axis[1], color='lime')
ax_7.plot(null_y_axis[0], null_y_axis[1], color='lime')

plt.show()
