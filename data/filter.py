f = open("gen_data1.jsonl",encoding="utf-8",mode="r")
f2 = open("gen_data2.jsonl",encoding="utf-8",mode="w")
lines = f.readlines()
import json
for line in lines:
    jdata = json.loads(line)
    if len(jdata["sql"]["conds"])<=4:
        json.dump(jdata, f2)
        f2.write('\n')
    else:
        print()

