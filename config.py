import configparser

config = configparser.ConfigParser()
config.read('./config.conf')

# ---------------------------------
default = 'DEFAULT'
ppo = 'PPO'
icm = 'ICM'
rnd = 'RND'
# ---------------------------------
default_config = config[default]
ppo_config = config[ppo]
icm_config = config[icm]
rnd_config = config[rnd]
