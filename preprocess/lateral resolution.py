from scipy.optimize import curve_fit
from scipy.special import erf
from utils.utilfunc import *

plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False


def fitfun(x, a, b, c, d):
    return a * erf((x-b)/(np.sqrt(2)*c))+d

def findidx_HM(y):
    '''Find indexes correspond to FWHM of list y'''
    y_max = max(y)
    y_HM = min(y, key = lambda x: abs(x-y_max/2))
    idx = []
    for x in range(len(y)):
        if y[x] == y_HM:
            idx.append(x)
    y_new = y[:]
    y_new[idx[0]-4:idx[0]+4] = [0]*8
    y_HM_new = min(y_new, key = lambda x: abs(x-y_max/2))
    for x in range(len(y_new)):
        if y_new[x] == y_HM_new:
            idx.append(x)
    return idx



edgelen = 100
step_size = 2.5
AR = np.load("./PreProcess/blade edge/blade_AR.npy")
OR = np.load("./PreProcess/blade edge/blade_OR.npy")

fig1 = plt.figure(1)
ax1 = fig1.add_subplot(1, 1, 1)
ax1.imshow(AR)
pt = plt.ginput(1)

edge_line = np.array([AR[y, x] for y, x in zip(list(range(int(pt[0][1]),
                                                          int(pt[0][1])+edgelen+1)), [int(pt[0][0])]*edgelen)])


par, _ = curve_fit(fitfun, np.arange(1, edgelen+1),
                   edge_line, p0=[0.1, 30, 10, 0], maxfev=5000)
resn = 2*abs(par[2])*np.sqrt(2*np.log(2))*step_size


fig = plt.figure()
ax1 = fig.add_subplot(111)

xdata = np.arange(1, edgelen+1, 0.1)
ydata = fitfun(xdata, *par)

ARline1 = (edge_line-np.amin(ydata))/(np.amax(ydata)-np.amin(ydata))
ydata1 = (ydata-np.amin(ydata))/(np.amax(ydata)-np.amin(ydata))

ax1.plot(np.arange(1, edgelen+1) * step_size, ARline1,
         'ks', markersize=4, label='-original data')
ax1.plot(xdata * step_size, ydata1, 'r', linewidth=2, label='ESF')
LSF = ydata[1:] - ydata[:-1]
idx = findidx_HM(LSF)
x_FWHM = xdata[idx]
y_FWHM = ydata1[idx]

ax1.plot(xdata[1:] * step_size, abs(LSF) /
         np.amax(abs(LSF)), 'g', linewidth=2, label='LSF')
ax1.plot(x_FWHM, y_FWHM, 'k--')
ax1.annotate('%.4fμm'%resn, xy=(xdata[idx][0], ydata1[idx][0]), 
                xytext=(xdata[idx][1], ydata1[idx][1]),
                arrowprops=dict(facecolor='black', linewidth=2, arrowstyle='<->'))

ax1.set_xlabel('Distance(μm)', fontsize=16)
ax1.legend(loc='upper right', fontsize=12)
plt.show()
print("resolution: %6.4f" % resn)


