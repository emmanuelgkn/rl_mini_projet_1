from plots import *

ddpg_config = DDPGConfig()
ddpg = run_ddpg(ddpg_config)

td3_config = TD3Config()
td3 = run_td3(td3_config)

plot_overestimated_bias(ddpg=ddpg, td3=td3)
plot_evaluations(ddpg,td3)
