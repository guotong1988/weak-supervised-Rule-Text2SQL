# Rule-SQL 

[![LICENSE](https://img.shields.io/badge/license-Anti%20996-blue.svg)](https://github.com/996icu/996.ICU/blob/master/LICENSE)

https://arxiv.org/abs/1907.00620

(Incorporating database design info into weak supervised text-to-sql generation)

## Run

Run `data/add_answer.py` to add answer for the origin wikisql data.

Run `rl/data.py` for the SQL exploration.

## Data

https://drive.google.com/file/d/1iJvsf38f16el58H4NPINQ7uzal5-V4v4/

## Result without EG （1 condition）

| **Model**   | Dev <br />logical form <br />accuracy | Dev<br />execution<br/> accuracy | Test<br /> logical form<br /> accuracy | Test<br /> execution<br /> accuracy |
| ----------- | ------------------------------------- | -------------------------------- | -------------------------------------- | ----------------------------------- |
| Neural Symbolic Machines | -                  | 74.9               | -                   | 74.8     |
| Our Methods | 68.7                   | 81.2               | 68.5                    | 81.0     |


## Reference

https://github.com/naver/sqlova
