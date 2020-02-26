
import pandas as pd
import json

# adv train
monitor_path = "/Users/huawei/PycharmProjects/AdvMARL/log/ppo1_pong-06202019-144638"

# adv train 0.001
monitor_path = "/Users/huawei/PycharmProjects/AdvMARL/log/ppo1_pong-06202019-132541"

# normal train weight0
monitor_path = "/Users/huawei/PycharmProjects/AdvMARL/log/ppo1_pong-06202019-120211"

# normal train
monitor_path ="/Users/huawei/PycharmProjects/AdvMARL/log/ppo1_pong-06202019-111130"

# adv train 0.01 0
monitor_path = "/Users/huawei/PycharmProjects/AdvMARL/log/ppo1_pong-06202019-173707"

# adv train 0 0.001
monitor_path = "/Users/huawei/PycharmProjects/AdvMARL/log/ppo1_pong-06202019-190945"


monitor_path = "/Users/huawei/PycharmProjects/AdvMARL/log/ppo1_pong-06212019-124321"

def plot_monitor(monitor_path):

    data = pd.read_csv("{}/monitor.csv".format(monitor_path), skiprows=[0],header=0)

    data['score_board'] = data['score_board'].replace({'\'': '"'}, regex=True)

    data_score = pd.io.json.json_normalize(data['score_board'].apply(json.loads))

    if False: # count_miss_catch:

        data_score['total_round'] = data_score[['left.oppo_double_hit', 'left.oppo_miss_catch',
                                                'left.oppo_miss_start', 'left.oppo_slow_ball',
                                                'right.oppo_double_hit', 'right.oppo_miss_catch',
                                                'right.oppo_miss_start', 'right.oppo_slow_ball']].sum(axis=1)

        data_score_next = data_score.shift(periods=1)

        data_score_epoch = data_score - data_score_next

        # data_score_epoch = data_score_epoch[data_score_epoch['total_round']!=0]

        data_score_epoch['left_winning'] = data_score_epoch[['left.oppo_double_hit', 'left.oppo_miss_catch',
                                                             'left.oppo_miss_start', 'left.oppo_slow_ball']].abs().sum(axis=1)
    else:
        # data_score_epoch['tie_winning'] = data_score_epoch['left.not_finish'].abs()
        data_score['total_round'] = data_score[['left.oppo_double_hit', 'left.oppo_miss_catch',
                                                 'left.oppo_slow_ball',
                                                # 'left.oppo_miss_start', 'right.oppo_miss_start',
                                                'right.oppo_double_hit', 'right.oppo_miss_catch',
                                                 'right.oppo_slow_ball']].sum(axis=1)

        data_score_next = data_score.shift(periods=1)

        data_score_epoch = data_score - data_score_next

        # data_score_epoch = data_score_epoch[data_score_epoch['total_round']!=0]

        data_score_epoch['left_winning'] = data_score_epoch[['left.oppo_double_hit', 'left.oppo_miss_catch', # 'left.oppo_miss_start',
                                                              'left.oppo_slow_ball']].abs().sum(axis=1)

    wining_rate_sum = data_score_epoch['left_winning'].rolling(100).sum()
    total_round_sum = data_score_epoch['total_round'].rolling(100).sum()

    wining_rate = wining_rate_sum/total_round_sum
    # wining_rate.fillna(0, inplace=True)

    # wining_rate.plot()

    # tie_rate = data_score_epoch['tie_winning']/data_score_epoch['total_round']
    # tie_rate.fillna(1, inplace=True)
    # b = tie_rate.rolling(100).mean()
    # b.plot()
    #
    # c = (tie_rate+wining_rate).rolling(100).mean()
    # c.plot()

    # result = pd.concat([a,b,c],names=['winning_rate','tie_rate','total'],axis=1)
    result = pd.concat([wining_rate],names=['winning_rate'],axis=1)


    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(4,6))
    ax = result.plot()
    lines = ax.lines


    result['random'] = 0.01
    line1, =ax.plot(result.index, result['random'], color='yellow')
    line1.set_dashes([2,2,10,2])


    result['normal'] = 0.5
    line2, =ax.plot(result.index, result['normal'], color='black')
    line2.set_dashes([2,2,10,2])

    ax.set_ylim(-0.1,1)

    # ax.legend((lines[0],lines[1],lines[2],line1, line2), ("winning_rate", "tie_rate", "total", "random","normal"))
    ax.legend((lines[0],lines[1],lines[2],line1, line2), ("winning_rate","random","normal"))



    fig = ax.get_figure()
    fig.savefig("{}/monitor.fig.png".format(monitor_path))
    plt.close()


# plot_monitor(monitor_path)
import os
for file_folder in os.listdir("./log/"):

    if "07232019" not in file_folder:
        continue
    monitor_path = os.path.join("./log/", file_folder)
    if os.path.isdir(monitor_path) and os.path.isfile(os.path.join(monitor_path,"monitor.csv")) \
            and os.path.exists(os.path.join(monitor_path,"monitor.csv")):
        try:
            plot_monitor(monitor_path)
        except KeyError:
            print("Key Error in {}, please check the header of the file".format(monitor_path))
