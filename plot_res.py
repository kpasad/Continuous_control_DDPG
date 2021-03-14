import matplotlib.pyplot as plt
import pickle as pk
import glob
import sys
sys.path.insert(0, '../Value_methods')
from paramutils import *
import numpy as np

basepath = r'C:\Users\kpasad\Dropbox\ML\project\deep-reinforcement-learning-master\p2_continuous-control\LR_sweep\\'
all_pk = glob.glob(basepath+'*.pk')

ma_length = 20
for pk_file in all_pk:
    filename=pk_file.rsplit('\\')[-1]
    #legend = filename.rsplit('.')[0]
    [scores,actor_loss,critic_loss, params] = pk.load(open(pk_file,'rb'))
    legend = "LR:"+str(params.critic_nw_lr)
    #ma= np.convolve(scores, np.ones(ma_length), 'valid') / ma_length
    ma = np.convolve(scores, np.ones(ma_length), 'valid') / ma_length
    plt.plot(ma,label=legend)

plt.legend(loc="upper left")
plt.grid(b=True, which='both', color='0.65', linestyle='-')
plt.xlabel('Episodes')
plt.ylabel('Moving Average Scores (window ='+str(ma_length)+')')
plt.title('Scores')
plt.show()