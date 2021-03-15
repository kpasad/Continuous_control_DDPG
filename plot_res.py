import matplotlib.pyplot as plt
import pickle as pk
import glob
import sys
from paramutils import *
import numpy as np


basepath = r'C:\Users\kpasad\Dropbox\ML\project\deep-reinforcement-learning-master\Continuous_control_DDPG\OU_noise_sweep_opt_LR\\'
all_pk = glob.glob(basepath+'*.pk')

ma_length = 20
max_x=600
for pk_file in all_pk:
    filename=pk_file.rsplit('\\')[-1]
    print(pk_file)
    #pk_file=basepath+'Continuous_ctrl_17_43_46.pk'
    #legend = filename.rsplit('.')[0]
    [scores,actor_loss,critic_loss, params] = pk.load(open(pk_file,'rb'))
    legend = "LR:"+str(params.OU_noise_sigma)
    #ma= np.convolve(scores, np.ones(ma_length), 'valid') / ma_length
    ma = np.convolve(scores[1:max_x], np.ones(ma_length), 'valid') / ma_length
    plt.plot(ma,label=legend)
    #break

plt.legend(loc="upper left")
plt.grid(b=True, which='both', color='0.65', linestyle='-')
plt.xlabel('Episodes')
plt.ylabel('Moving Average Scores (window ='+str(ma_length)+')')
plt.title('Scores')
plt.show()