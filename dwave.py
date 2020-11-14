import numpy as np
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
#import matplotlib.colors as mcolors
from matplotlib.ticker import AutoMinorLocator
import math as mth

fg = plt.figure(figsize=(15, 12), constrained_layout=True)
gs = gridspec.GridSpec(ncols=3, nrows=2, figure=fg)
ax_1 = fg.add_subplot(gs[0, 0])
ax_2 = fg.add_subplot(gs[0, 1])
ax_3 = fg.add_subplot(gs[1, 0])
ax_4 = fg.add_subplot(gs[1, 1])
ax_5 = fg.add_subplot(gs[0, 2])
ax_6 = fg.add_subplot(gs[1, 2]) 
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
    U2[step] = 1
    #V2[step] = 0
    #P2[step] = 0
    #R2[step] = 0
    for i in range(0, step) :
        P2[i] = P[i + 1] - Q[i] * (U[i] - V[i])
        R2[i] = R[i + 1] - Q[i] * (V[i] - U[i])
    return U2, V2, P2, R2


N = 200
h = 0.02
fg.suptitle('N = ' + str(N) + ', h = ' + str(h))
U = np.zeros(N, dtype=np.float)
V = np.zeros(N, dtype=np.float)
P = np.zeros(N, dtype=np.float)
R = np.zeros(N, dtype=np.float)
Q = np.zeros(N, dtype=np.float)
U[0] = 1.0
#pik1 = 0.5
#pik2 = 0.25
#Q[49] = pik2
#Q[50] = pik1
#Q[51] = pik2
#Q[99] = pik2
#Q[100] = pik1
#Q[101] = pik2
#Q[149] = pik2
#Q[150] = pik1
#Q[151] = pik2
k_sin = 0.2
for i in range(0, N) :
    value = k_sin * mth.sin((mth.pi * i) / 30.0)
    if value < 0 :
        Q[i] = 0
    else :
        Q[i] = value

x_plot_ax = np.arange(0, N*h, h)


#for i in range(1, N) :
#    U, V, P, R = wave_step(U, V, P, R, Q, N, h, i)
#    ax_1.clear()
#    my_set_axis(ax_1)
#    ax_1.set_title("Плоскость колебания U")
#    ax_1.plot(x_plot_ax, Q, color='red', linestyle='--')
#    ax_1.plot(x_plot_ax, U, color='green')
#    ax_2.clear()
#    my_set_axis(ax_2)
#    ax_2.set_title("Плоскость колебания V")
#    ax_2.plot(x_plot_ax, Q, color='red', linestyle='--')
#    ax_2.plot(x_plot_ax, V, color='green')
#    plt.savefig('./frames/frame_' + str(i) + '.png')
#plt.show()

my_set_axis(ax_5)
ax_5.set_title("Функция Q")
ax_5.plot(x_plot_ax, Q, color='red')
data_U = []
data_V = []
sled_U = []
sled_V = []
for i in range(1, N) :
    U, V, P, R = wave_step(U, V, P, R, Q, N, h, i)
    data_U.append(U.view())
    data_V.append(V.view())
    sled_U.append(U[0])
    sled_V.append(V[0])
    if i == 100 :
        ax_6.set_title("U на 100 итерации")
        ax_6.plot(U)
data_U.reverse()
data_V.reverse()
ax_1.set_axis_off()
ax_1.set_title("Плоскость U")
ax_2.set_axis_off()
ax_2.set_title("Плоскость V")
ax_1.imshow(data_U, cmap='seismic', vmin=-1, vmax=1, interpolation='none')
ax_2.imshow(data_V, cmap='seismic', vmin=-1, vmax=1, interpolation='none')

ax_3.set_title("След U")
ax_3.plot(sled_U)
ax_4.set_title("След V")
ax_4.plot(sled_V)
plt.show()


