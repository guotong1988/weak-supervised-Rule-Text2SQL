
from sqlnet.dbengine import DBEngine
from rl.train_rl import config
from rl.train_rl import get_data
import json
engine_train = DBEngine("train.db")
engine_dev = DBEngine("dev.db")
train_data, train_table, dev_data, dev_table, train_loader, dev_loader = get_data("./", config)

for line in train_data:
    if len(line["sql"]["conds"])<=0:
        print(line["sql"])