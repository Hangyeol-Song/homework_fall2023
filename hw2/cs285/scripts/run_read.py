"""
Prob. 3
Should reach out Average Return 200

python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 1000 --exp_name cartpole
python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 1000 -rtg --exp_name cartpole_rtg
python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 1000 -na --exp_name cartpole_na
python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 1000 -rtg -na --exp_name cartpole_rtg_na
python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 4000 --exp_name cartpole_lb
python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 4000 -rtg --exp_name cartpole_lb_rtg
python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 4000 -na --exp_name cartpole_lb_na
python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 4000 -rtg -na --exp_name cartpole_lb_rtg_na
"""

"""
Prob. 4
Should reach out averageReturn over 300
python cs285/scripts/run_hw2.py --env_name HalfCheetah-v4 -n 100 -b 5000 -rtg --discount 0.95 -lr 0.01 --exp_name cheetah
python cs285/scripts/run_hw2.py --env_name HalfCheetah-v4 -n 100 -b 5000 -rtg --discount 0.95 -lr 0.01 --use_baseline -blr 0.01 -bgs 5 --exp_name cheetah_baseline_blr=0.01_bgs=5
python cs285/scripts/run_hw2.py --env_name HalfCheetah-v4 -n 100 -b 5000 -rtg --discount 0.95 -lr 0.01 --use_baseline -blr 0.001 -bgs 5 --exp_name cheetah_baseline_blr=0.001_bgs=5
python cs285/scripts/run_hw2.py --env_name HalfCheetah-v4 -n 100 -b 5000 -rtg --discount 0.95 -lr 0.01 --use_baseline -blr 0.01 -bgs 10 --exp_name cheetah_baseline_blr=0.01_bgs=10
python cs285/scripts/run_hw2.py --env_name HalfCheetah-v4 -n 100 -b 5000 -rtg --discount 0.95 -lr 0.01 --use_baseline -blr 0.001 -bgs 10 --exp_name cheetah_baseline_blr=0.001_bgs=10
python cs285/scripts/run_hw2.py --env_name HalfCheetah-v4 -n 100 -b 5000 -rtg --discount 0.95 -lr 0.01 --use_baseline -blr 0.01 -bgs 1 --exp_name cheetah_baseline_blr=0.01_bgs=1
python cs285/scripts/run_hw2.py --env_name HalfCheetah-v4 -n 100 -b 5000 -rtg --discount 0.95 -lr 0.01 --use_baseline -blr 0.001 -bgs 1 --exp_name cheetah_baseline_blr=0.001_bgs=1

"""

"""
Prob. 5
Should reach out averageReture close to 200
Lamda = [0, 0.95, 0.98, 0.99, 1]
python cs285/scripts/run_hw2.py --env_name LunarLander-v2 --ep_len 1000 --discount 0.99 -n 300 -l 3 -s 128 -b 2000 -lr 0.001 --use_reward_to_go --use_baseline --gae_lambda 0 --exp_name lunar_lander_lambda_0
python cs285/scripts/run_hw2.py --env_name LunarLander-v2 --ep_len 1000 --discount 0.99 -n 300 -l 3 -s 128 -b 2000 -lr 0.001 --use_reward_to_go --use_baseline --gae_lambda 0.95 --exp_name lunar_lander_lambda_0.95
python cs285/scripts/run_hw2.py --env_name LunarLander-v2 --ep_len 1000 --discount 0.99 -n 300 -l 3 -s 128 -b 2000 -lr 0.001 --use_reward_to_go --use_baseline --gae_lambda 0.98 --exp_name lunar_lander_lambda_0.98
python cs285/scripts/run_hw2.py --env_name LunarLander-v2 --ep_len 1000 --discount 0.99 -n 300 -l 3 -s 128 -b 2000 -lr 0.001 --use_reward_to_go --use_baseline --gae_lambda 0.99 --exp_name lunar_lander_lambda_0.99
python cs285/scripts/run_hw2.py --env_name LunarLander-v2 --ep_len 1000 --discount 0.99 -n 300 -l 3 -s 128 -b 2000 -lr 0.001 --use_reward_to_go --use_baseline --gae_lambda 1 --exp_name lunar_lander_lambda_1
"""

"""
Prob. 6

for seed in $(seq 1 5); do
python cs285/scripts/run_hw2.py --env_name InvertedPendulum-v4 -n 100 \
--exp_name pendulum_default_s$seed \
-rtg --use_baseline -na \
--batch_size 5000 \
--seed $seed
done

"""


"""
Prob. 7
Reach return 300 by 20 iteration

python cs285/scripts/run_hw2.py \
--env_name Humanoid-v4 --ep_len 1000 \
--discount 0.99 -n 1000 -l 3 -s 256 -b 50000 -lr 0.001 \
--baseline_gradient_steps 50 \
-na --use_reward_to_go --use_baseline --gae_lambda 0.97 \
--exp_name humanoid --video_log_freq 5
"""

import glob, os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

def get_section_results(file):
    """
        requires tensorflow==1.12.0
    """
    AverageReturn = []
    BaselineLoss = []

    for e in tf.compat.v1.train.summary_iterator(file):
        for v in e.summary.value:
            if v.tag == 'Eval_AverageReturn':
                AverageReturn.append(v.simple_value)
            elif v.tag == 'Baseline_Loss':
                BaselineLoss.append(v.simple_value)                

    return AverageReturn, BaselineLoss

def result_plot(AverageReturns, BaselineLosses, legend_list):
    t = np.arange(0,len(AverageReturns[0]),1).tolist()
    if any(BaselineLosses):
        
        plt.subplot(2,1,1)
        for i in range(len(AverageReturns)):
            plt.plot(t,AverageReturns[i],label=legend_list[i])
        plt.ylabel('AverageReturn')
        plt.legend()
        plt.grid()
        
        plt.subplot(2,1,2)
        for i in range(len(BaselineLosses)):
            if BaselineLosses[i]:
                plt.plot(t,BaselineLosses[i],label=legend_list[i])
        plt.ylabel('BaselineLoss')
        plt.legend()
        # plt.grid()
        plt.tight_layout()
    
    else:
        for i in range(len(AverageReturns)):
            plt.plot(t,AverageReturns[i],label=legend_list[i])
        plt.ylabel('AverageReturn')

    plt.xlabel('Iteration')    
    plt.legend()
    plt.grid()

    
def result_run(logdir,legend_list):
    plt.figure()
    eventfiles = sorted(glob.glob(logdir),key=os.path.getmtime)
    if eventfiles != None:
        AverageReturns = []
        BaselineLosses = []
        for evenfile in eventfiles:
            AverageReturn, BaselineLoss = get_section_results(evenfile)
            AverageReturns.append(AverageReturn)
            BaselineLosses.append(BaselineLoss)
        result_plot(AverageReturns, BaselineLosses,legend_list)
 
if __name__ == '__main__':

    logdir_prob3 = 'data/Prob3/q2_pg_*/events*'
    logdir_prob4 = 'data/Prob4/q2_pg_*/events*'
    logdir_prob5 = 'data/Prob5/q2_pg_*/events*'


    legend_prob3 = ['-b=1000','-b=1000 -rtg','-b=1000 -na','-b=1000 -rtg -na','-b=4000','-b=4000 -rtg','-b=4000 -na','-b=4000 -rtg -na']
    legend_prob4 = ['w/o baseline','w/ baseline -blr=0.01 -bgs=5','w/ baseline -blr=0.001 -bgs=5','w/ baseline -blr=0.01 -bgs=10','w/ baseline -blr=0.001 -bgs=10','w/ baseline -blr=0.01 -bgs=1','w/ baseline -blr=0.001 -bgs=1']
    legend_prob5 = ['-lamda=0','-lamda=0.95','-lamda=0.98','-lamda=0.99','-lamda=1',]
    
    result_run(logdir_prob3, legend_prob3)
    result_run(logdir_prob4, legend_prob4)
    result_run(logdir_prob5, legend_prob5)
    
    plt.show()

