import datetime
import wandb
from trainer_def import WandbTrainer, json2dict
from dataset_def import SineData
from custom_act_lstm import CustomLSTMLayer

config_dict = json2dict('config.json')
datetime_now = datetime.datetime.now()
now_str = datetime.datetime.strftime(datetime_now,"%Y%m%d%H%M%S")
run_id = now_str
run = wandb.init(project='Sine-Gates',config=config_dict)

trainer = WandbTrainer(run)