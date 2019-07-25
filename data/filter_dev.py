f = open("dev_with_answer.jsonl",encoding="utf-8",mode="r")
f2 = open("dev_with_answer_filter.jsonl",encoding="utf-8",mode="w")
lines = f.readlines()
import json
for line in lines:
    jdata = json.loads(line)
    if len(jdata["sql"]["conds"])<=1:
        json.dump(jdata, f2)
        f2.write('\n')
    else:
        print()

