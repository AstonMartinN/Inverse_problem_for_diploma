import numpy as np
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
#import matplotlib.colors as mcolors
from matplotlib.ticker import AutoMinorLocator
import math as mth
import random

alpha = 0.5
fg = plt.figure(figsize=(16, 8), constrained_layout=True)
gs = gridspec.GridSpec(ncols=5, nrows=10, figure=fg)
ax_1 = fg.add_subplot(gs[0:5, 0])
ax_2 = fg.add_subplot(gs[5: , 0])

ax_3 = fg.add_subplot(gs[0:5, 1:3])
ax_4 = fg.add_subplot(gs[5: , 1:3])

ax_5 = fg.add_subplot(gs[0:2, 3:])
ax_6 = fg.add_subplot(gs[2:6, 3:]) 
ax_7 = fg.add_subplot(gs[6: , 3:])



#gs = gridspec.GridSpec(ncols=3, nrows=10, figure=fg)
#ax_1 = fg.add_subplot(gs[0:5, 0])
#ax_2 = fg.add_subplot(gs[5: , 0])

#ax_3 = fg.add_subplot(gs[0:5, 1])
#ax_4 = fg.add_subplot(gs[5: , 1])

#ax_5 = fg.add_subplot(gs[0:2, 2])
#ax_6 = fg.add_subplot(gs[2:6, 2]) 
#ax_7 = fg.add_subplot(gs[6: , 2])


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
    if N % 2 != 0:
        print("excess layer")
    U = np.zeros((2*N, N), dtype=np.float)
    V = np.zeros((2*N, N), dtype=np.float)
    P = np.zeros((2*N, N), dtype=np.float)
    R = np.zeros((2*N, N), dtype=np.float)
    U[0][0] = 1
    U[1][1] = 1
    for i in range(2, N) :
        U[i][i] = 1.0
        for j in range(0 + i%2, i, 2) :
            if j == 0 :
                U[i][0] = U[i - 2][0] + 2 * h * P[i - 2][0]
                V[i][0] = V[i - 2][0] + 2 * h * R[i - 2][0]
            else :
                U[i][j] = U[i - 1][j - 1] + h * P[i - 1][j - 1]
                V[i][j] = V[i - 1][j - 1] + h * R[i - 1][j - 1]
            P[i][j] = P[i - 1][j + 1] - h * Q[j + 1] * (U[i - 1][j + 1] - V[i - 1][j + 1])
            R[i][j] = R[i - 1][j + 1] - h * Q[j + 1] * (V[i - 1][j + 1] - U[i - 1][j + 1])
    for i in range(N, 2*N) :
        for j in range(0 + i%2, 2*N - i, 2) :
            if j == 0 :
                U[i][0] = U[i - 2][0] + 2 * h * P[i - 2][0]
                V[i][0] = V[i - 2][0] + 2 * h * R[i - 2][0]
            else :
                U[i][j] = U[i - 1][j - 1] + h * P[i - 1][j - 1]
                V[i][j] = V[i - 1][j - 1] + h * R[i - 1][j - 1]
            P[i][j] = P[i - 1][j + 1] - h * Q[j + 1] * (U[i - 1][j + 1] - V[i - 1][j + 1])
            R[i][j] = R[i - 1][j + 1] - h * Q[j + 1] * (V[i - 1][j + 1] - U[i - 1][j + 1])
    return U, V, P, R

N = 200
h = 0.02
fg.suptitle('N = ' + str(N) + ', h = ' + str(h))

Q = np.zeros(N, dtype=np.float)


#******FIRST PROBLEM******************
#ampl_sin = 0.1 #0.0002
#k_sin =  0.03  #1.0 / 45.0
#for i in range(0, N) :
#    value = 0.2 + ampl_sin * mth.cos(mth.pi * i * k_sin - 0.5)
#    if value < 0 :
#        Q[i] = 0
#    else :
#        Q[i] = value 

ampl_sin = 0.05 #0.0002
k_sin =  0.02  #1.0 / 45.0
for i in range(0, N) :
    value = 0.08 + ampl_sin * mth.cos(mth.pi * i * k_sin - 0.5)
    if value < 0 :
        Q[i] = 0
    else :
        Q[i] = value 

#**********SECOND PROBLEM************************
#ampl_sin = 0.15 #0.0002
#k_sin =  0.017  #0.3
#for i in range(0, N) :
#    value = ampl_sin * mth.sin(mth.pi * i * k_sin - 0.5)
#    if value < 0 :
#        Q[i] = 0
#    else :
#        Q[i] = value 

#*****************************THIRD PROBLEM********************
#ampl_sin = 0.15 #0.0002
#k_sin =  0.03  #1.0 / 45.0
#shift = 0.2
#for i in range(0, N) :
#    value = shift * (1.0 - i/N)  + ampl_sin *(1 - i/N) * mth.sin(mth.pi * i * k_sin - 0.5)
#    if value < 0.0 :
#        Q[i] = 0.0
#    else : 
#        Q[i] = value 

#********************FOURTH PROBLEM************************
#ampl_sin = 0.2 #0.15
#k_sin =  0.03  #1.0 / 45.0
#for i in range(0, N) :
#    if i < 50 :
#        Q[i] = 0.0 + i*((ampl_sin)/(50))
#    elif i < 100 :
#        Q[i] = ampl_sin + (i - 50)*((0.0 - ampl_sin)/(50))
#    elif i < 150 :
#        Q[i] = 0.0 + (i - 100)*((ampl_sin)/(50))
#    else :
#        Q[i] = ampl_sin + (i - 150)*((0.0 - ampl_sin)/(50))


U, V, P, R = chess_template(Q, N, h)
U = U.tolist()
V = V.tolist()
P = P.tolist()
R = R.tolist()

sled_R = []
sled_P = []
for i in range(0, N):
    sled_P.append(P[2*i][0])
    sled_R.append(R[2*i][0])

sled_U, sled_V = [], []
for i in range(0, N):
    sled_U.append(U[2*i][0])
    sled_V.append(V[2*i][0])


print(sled_P)
#print('\n ########################################## \n ########################################## \n')
#new{
#ampl = 0.01 * max([abs(i) for i in sled_P])
#for i in range(0, len(sled_P)):
##    sled_P[i] = sled_P[i] + ampl * mth.sin(i*i + 10.33333 + i)
#new}
#print(sled_P)



for i in range(0, N):
    for j in range(int(i%2), 2*N -2, 2):
        U[j + 1][i] = U[j][i]
        V[j + 1][i] = V[j][i]

U.reverse()
V.reverse()
P.reverse()
R.reverse()




ax_1.set_title('U(x, t)')
ax_1.set_xlabel('Узлы')
ax_1.set_ylabel('Слои')
img_1 = ax_1.imshow(U, cmap='gist_rainbow', vmin=-1, vmax=1)
fg.colorbar(img_1, ax=ax_1)

ax_2.set_title('V(x, t)')
ax_2.set_xlabel('Узлы')
ax_2.set_ylabel('Слои')
img_2 = ax_2.imshow(V, cmap='gist_rainbow', vmin=-1, vmax=1)#, interpolation='bilinear')
fg.colorbar(img_2, ax=ax_2)

#ax_3.set_title('P(x, t)')
#ax_3.set_xlabel('Узлы')
#ax_3.set_ylabel('Слои')
#img_3 = ax_3.imshow(np.array(P), cmap='gist_rainbow', vmin=-1, vmax=1)#, interpolation='bilinear')
#fg.colorbar(img_3, ax=ax_3)



#ax_4.set_title('R(x, t)')
#ax_4.set_xlabel('Узлы')
#ax_4.set_ylabel('Слои')
#img_4 = ax_4.imshow(R, cmap='gist_rainbow', vmin=-1, vmax=1)#, interpolation='bilinear')
#fg.colorbar(img_4, ax=ax_4)




null_y_axis = ([0, len(sled_P)],[0, 0])
ax_3.plot(null_y_axis[0], null_y_axis[1], color='lime')
ax_4.plot(null_y_axis[0], null_y_axis[1], color='lime')
#ax_3.set_ylim(ymin=-1, ymax=1.2)
ax_3.set_title('P(0, t)')#U
ax_3.set_xlabel('t')
ax_3.plot(sled_P, color='black')
#ax_4.set_ylim(ymin=-1, ymax=1)
ax_4.set_title('R(0, t)')#V
ax_4.set_xlabel('t')
ax_4.plot(sled_R, color='black')



ax_5.set_ylim(ymin=-0.05, ymax=0.5)
ax_5.set_xlabel('i')
ax_5.plot([0, len(Q)], [0, 0], color='lime')
ax_5.plot(Q, color='red')
ax_5.set_title('q(x) = 0.08 + A * cos(pi * k * i - 0.5), A = ' + str(ampl_sin) + ' k = ' + str(k_sin))
#ax_5.set_title('q(x) = abs(A * sin(pi * k * i - 0.5)), A = ' + str(ampl_sin) + ' k = ' + str(k_sin))



null_y_axis = ([0, len(sled_P)],[0, 0])
ax_6.plot(null_y_axis[0], null_y_axis[1], color='lime')
ax_7.plot(null_y_axis[0], null_y_axis[1], color='lime')

#ax_6.set_ylim(ymin=-0.2, ymax=1.2)
ax_6.set_title('U(0, t)')#U
ax_6.set_xlabel('t')
ax_6.plot(sled_U, color='black')

#ax_7.set_ylim(ymin=-0.2, ymax=1.2)
ax_7.set_title('V(0, t)')#V
ax_7.set_xlabel('t')
ax_7.plot(sled_V, color='black')

plt.show()
