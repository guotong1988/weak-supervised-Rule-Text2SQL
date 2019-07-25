
from sqlnet.dbengine import DBEngine
from rl.train_rl import config
from train import *
import json
engine_train = DBEngine("train.db")
engine_dev = DBEngine("dev.db")

train_data, train_table, dev_data, dev_table, _, _ = load_wikisql("./", False, -1, no_w2i=True, no_hs_tok=True)
train_loader, dev_loader = get_loader_wikisql(train_data, dev_data, 32, shuffle_train=False)


def process(train_data_,name,engine_):
  for i,item in enumerate(train_data_):
    if i%100==0:
        print(i)
    # if i==15988:
    #     print()

    # sql = {'sel': 5, 'conds': [[3, 0, "26"], [6, 1, "8"]], 'agg': 1}
    # table_id = '2-10240125-1'
    # t = train_table[table_id]
    # a = engine_.execute_query_v2(table_id, sql)

    answer = engine_.execute_query_v2(item["table_id"],item["sql"])
    if answer==[None]:
      print(None)
    train_data_[i]["answer"] = answer

  f = open(name,mode="w",encoding="utf-8")

  for line in train_data_:
    json.dump(line, f)
    f.write('\n')

  f.close()

process(train_data,"train_with_answer.jsonl",engine_train)
process(dev_data,"dev_with_answer.jsonl",engine_dev)