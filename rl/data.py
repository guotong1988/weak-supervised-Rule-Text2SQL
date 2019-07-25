import torch
import bert.tokenization as tokenization
from bert.modeling import BertConfig, BertModel
from sqlova.utils.utils_wikisql import *
from sqlova.model.nl2sql.wikisql_models import *
from sqlnet.dbengine import DBEngine
from train import test
import random
from decimal import Decimal

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = {}
config["batch_size"] = 8
config["data_path"] = "../data/"
config["num_target_layers"] = 2
config["dropout"] = 0.3
config["max_seq_length"] = 222
config["toy_model"] = False
config["toy_size"] = 12
config["accumulate_gradients"] = 2
config["EG"] = False



def get_opt(model, model_bert):
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=1e-3, weight_decay=0)

    opt_bert = torch.optim.Adam(filter(lambda p: p.requires_grad, model_bert.parameters()),
                                lr=1e-5, weight_decay=0)
    return opt, opt_bert

def get_bert(BERT_PATH):
    bert_config_file = BERT_PATH + "/bert_config_uncased_L-12_H-768_A-12.json"
    vocab_file = BERT_PATH +  "/vocab_uncased_L-12_H-768_A-12.txt"
    init_checkpoint = BERT_PATH + "/pytorch_model_uncased_L-12_H-768_A-12.bin"

    bert_config = BertConfig.from_json_file(bert_config_file)
    tokenizer = tokenization.FullTokenizer(
        vocab_file=vocab_file, do_lower_case=True)
    bert_config.print_status()

    model_bert = BertModel(bert_config)
    model_bert.load_state_dict(torch.load(init_checkpoint, map_location='cpu'))
    print("Load pre-trained BERT parameters.")
    model_bert.to(device)
    return model_bert, tokenizer, bert_config

def train(train_loader, train_table, model, model_bert, opt, bert_config, tokenizer,
          max_seq_length, num_target_layers, accumulate_gradients=1, check_grad=True,
          st_pos=0, opt_bert=None, path_db=None, dset_name='train'):

    ave_loss = 0
    count = 0 # count the # of examples
    count_sc = 0 # count the # of correct predictions of select column
    count_sa = 0 # of selectd aggregation
    count_wn = 0 # of where number
    count_wc = 0 # of where column
    count_wo = 0 # of where operator
    count_wv = 0 # of where-value
    count_wvi = 0 # of where-value index (on question tokens)
    count_logic_form_acc = 0  # of logical form acc
    count_execute_acc = 0   # of execution acc

    # Engine for SQL querying.
    engine = DBEngine(os.path.join(path_db, f"{dset_name}.db"))

    explored_data_list = []

    for batch_index, batch_data in enumerate(train_loader):

        count += len(batch_data)

        if count < st_pos:
            continue
        # Get fields
        question, question_token, sql, sql_text, sql_t, table, header_token, header \
            = get_fields(batch_data, train_table, no_hs_t=True, no_sql_t=True)

        len_question_bert, len_header_token, number_header, \
        question_token_bert, token_to_berttoken_index, berttoken_to_token_index \
            = get_wemb_bert_v2(bert_config, model_bert, tokenizer, question_token, header, max_seq_length,
                               num_out_layers_n=num_target_layers, num_out_layers_h=num_target_layers)


        # select column
        def equal_in(cell_, pred_answer_column):
            for cell in pred_answer_column:
                if cell == cell_:
                    return True
            return False
        # RL
        # where number

        def list_in_list(small,big):
            for cell in big:
                try:
                    cell_ = int(cell)
                    if cell_ in small:
                        return True
                    cell_ = float(cell)
                    if cell_ in small:
                        return True
                except:
                    cell_ = str(cell)
                    if cell_.lower() in small:
                        return True

            for cell in small:
                try:
                    cell_ = int(cell)
                    if cell_ in big:
                        return True
                    cell_ = float(cell)
                    if cell_ in big:
                        return True
                except:
                    cell_ = str(cell)
                    if cell_.lower() in big:
                        return True

            return False

        def list_exact_match(input1,input2):
            tmp1 = [str(item) for item in input1]
            tmp2 = [str(item) for item in input2]
            if sorted(tmp1)==sorted(tmp2):
                return True
            return False

        def contains(big_list,small_list):
            return set(small_list).issubset(set(big_list))


        for i in range(len(batch_data)):
            print(sql[i])
            explored_data = {}
            explore_count = 0
            breakall = False

            reward_where = False
            reward_where_cond1 = 0
            reward_where_cond2 = 0
            reward_where_cond3 = 0
            reward_where_cond4 = 0

            len_question = len(question_token[i])

            gt_answer_list = batch_data[i]["answer"]

            where_number_random = 0
            select_column_random = -1
            select_agg_random = 0
            col = 0
            op = 0
            start = 0
            end = 0

            col2 = 0
            op2 = 0
            start2 = 0
            end2 = 0

            col3 = 0
            op3 = 0
            start3 = 0
            end3 = 0

            col4 = 0
            op4 = 0
            start4 = 0
            end4 = 0

            saved_where_number = -1
            saved_col1 = -1
            saved_op1 = -1
            saved_start1 = -1
            saved_end1 = -1
            saved = False

            # select_column_random = 3
            # where_number_random = 1
            while True:

                final_select_agg = None
                final_select_column = None
                final_conds = []

                tmp_conds = []
                if where_number_random == 0:
                    select_column_random += 1
                    if select_column_random==len(header[i]):
                        select_agg_random+=1
                        select_column_random=0
                    if select_agg_random==6:
                        where_number_random+=1
                        select_agg_random=0

                if where_number_random == 1:

                    end+=1
                    if end>=len_question+1:
                        start+=1
                        end = start + 1
                    if start>=len_question:
                        op+=1
                        start=0
                    if op==3:
                        col+=1
                        op=0
                    if col==len(header[i]):
                        select_column_random+=1
                        col=0
                    if select_column_random==len(header[i]):
                        select_agg_random+=1
                        select_column_random=0
                    if select_agg_random==6:
                        where_number_random+=1
                        select_agg_random=0

                if saved == True and where_number_random==2:
                    col = saved_col1
                    start = saved_start1
                    end = saved_end1
                    op = saved_op1
                    end2 += 1
                    if end2 >= len_question+1:
                        start2 += 1
                        end2 = start2 + 1
                    if start2 >= len_question:
                        op2 += 1
                        start2 = 0
                    if op2 == 3:
                        col2 += 1
                        op2 = 0
                    if col2 == len(header[i]):
                        select_column_random += 1
                        col2 = 0
                    if select_column_random == len(header[i]):
                        select_agg_random += 1
                        select_column_random = 0
                    if select_agg_random == 6:
                        where_number_random += 1
                        select_agg_random = 0

                if saved==False and where_number_random==2:
                    end += 1
                    if end >= len_question+1:
                        start += 1
                        end = start + 1
                    if start >= len_question:
                        op += 1
                        start = 0
                    if op == 3:
                        col += 1
                        op = 0
                    if col == len(header[i]):
                        end2 += 1
                        col = 0
                    if end2 >= len_question+1:
                        start2 += 1
                        end2 = start2 + 1
                    if start2 >= len_question:
                        op2 += 1
                        start2 = 0
                    if op2 == 3:
                        col2 += 1
                        op2 = 0
                    if col2 == len(header[i]):
                        select_column_random += 1
                        col2 = 0
                    if select_column_random==len(header[i]):
                        select_agg_random+=1
                        select_column_random=0
                    if select_agg_random==6:
                        where_number_random+=1
                        select_agg_random=0

                if where_number_random==3:
                    # break #TODO
                    end += 1
                    if end >= len_question+1:
                        start += 1
                        end = start + 1
                    if start >= len_question:
                        op += 1
                        start = 0
                    if op == 3:
                        col += 1
                        op = 0
                    if col == len(header[i]):
                        end2 += 1
                        col = 0
                    if end2 >= len_question+1:
                        start2 += 1
                        end2 = start2 + 1
                    if start2 >= len_question:
                        op2 += 1
                        start2 = 0
                    if op2 == 3:
                        col2 += 1
                        op2 = 0
                    if col2 == len(header[i]):
                        end3 += 1
                        col2 = 0
                    if end3 >= len_question+1:
                        start3 += 1
                        end3 = start3 + 1
                    if start3 >= len_question:
                        op3 += 1
                        start3 = 0
                    if op3 == 3:
                        col3 += 1
                        op3 = 0
                    if col3 == len(header[i]) :
                        select_column_random += 1
                        col3 = 0
                    if select_column_random==len(header[i]):
                        select_agg_random+=1
                        select_column_random=0
                    if select_agg_random==6:
                        where_number_random+=1
                        select_agg_random=0

                if where_number_random == 4:
                    end += 1
                    if end >= len_question+1:
                        start += 1
                        end = start + 1
                    if start >= len_question:
                        op += 1
                        start = 0
                    if op == 3:
                        col += 1
                        op = 0
                    if col == len(header[i]) :
                        end2 += 1
                        col = 0
                    if end2 >= len_question+1:
                        start2 += 1
                        end2 = start2 + 1
                    if start2 >= len_question:
                        op2 += 1
                        start2 = 0
                    if op2 == 3:
                        col2 += 1
                        op2 = 0
                    if col2 == len(header[i]):
                        end3 += 1
                        col2 = 0
                    if end3 >= len_question+1:
                        start3 += 1
                        end3 = start3 + 1
                    if start3 >= len_question:
                        op3 += 1
                        start3 = 0
                    if op3 == 3:
                        col3 += 1
                        op3 = 0
                    if col3 == len(header[i]):
                        end4 += 1
                        col3 = 0
                    if end4 >= len_question+1:
                        start4 += 1
                        end4 = start4+1
                    if start4 >= len_question:
                        op4 += 1
                        start4 = 0
                    if op4 == 3:
                        col4 += 1
                        op4 = 0
                    if col4 == len(header[i]):
                        select_column_random += 1
                        col4 = 0
                    if select_column_random == len(header[i]):
                        select_agg_random += 1
                        select_column_random = 0
                    if select_agg_random == 6:
                        where_number_random += 1
                        select_agg_random = 0

                if where_number_random == 1:
                    cond = []
                    cond.append(col)
                    cond.append(op)

                    pr_wv_str = question_token[i][start:end]
                    cond_value = merge_wv_t1_eng(pr_wv_str, question[i])
                    try:
                        cond_value_ = float(cond_value)
                    except:
                        cond_value_ = cond_value
                    # if type(cond_value_) == str:  # and random.randint(1,2)==1:
                    #     op = 0
                    cond.append(cond_value_)
                    tmp_conds.append(cond)

                if where_number_random == 2:
                    cond = []
                    cond.append(col)
                    cond.append(op)
                    pr_wv_str = question_token[i][start:end]
                    cond_value = merge_wv_t1_eng(pr_wv_str, question[i])
                    try:
                        cond_value_ =  float(cond_value)
                    except:
                        cond_value_ = cond_value
                    cond.append(cond_value_)
                    tmp_conds.append(cond)

                    cond = []
                    cond.append(col2)
                    cond.append(op2)
                    pr_wv_str = question_token[i][start2:end2]
                    cond_value = merge_wv_t1_eng(pr_wv_str, question[i])
                    try:
                        cond_value_ =  float(cond_value)
                    except:
                        cond_value_ = cond_value
                    cond.append(cond_value_)
                    tmp_conds.append(cond)

                if where_number_random == 3:
                    cond = []
                    cond.append(col)
                    cond.append(op)
                    pr_wv_str = question_token[i][start:end]
                    cond_value = merge_wv_t1_eng(pr_wv_str, question[i])
                    try:
                        cond_value_ =  float(cond_value)
                    except:
                        cond_value_ = cond_value
                    cond.append(cond_value_)
                    tmp_conds.append(cond)

                    cond = []
                    cond.append(col2)
                    cond.append(op2)
                    pr_wv_str = question_token[i][start2:end2]
                    cond_value = merge_wv_t1_eng(pr_wv_str, question[i])
                    try:
                        cond_value_ =  float(cond_value)
                    except:
                        cond_value_ = cond_value
                    cond.append(cond_value_)
                    tmp_conds.append(cond)

                    cond = []
                    cond.append(col3)
                    cond.append(op3)
                    pr_wv_str = question_token[i][start3:end3]
                    cond_value = merge_wv_t1_eng(pr_wv_str, question[i])
                    try:
                        cond_value_ = float(cond_value)
                    except:
                        cond_value_ = cond_value
                    cond.append(cond_value_)
                    tmp_conds.append(cond)

                if where_number_random == 4:
                    cond = []
                    cond.append(col)
                    cond.append(op)
                    pr_wv_str = question_token[i][start:end]
                    cond_value = merge_wv_t1_eng(pr_wv_str, question[i])
                    try:
                        cond_value_ = float(cond_value)
                    except:
                        cond_value_ = cond_value
                    cond.append(cond_value_)
                    tmp_conds.append(cond)

                    cond = []
                    cond.append(col2)
                    cond.append(op2)
                    pr_wv_str = question_token[i][start2:end2]
                    cond_value = merge_wv_t1_eng(pr_wv_str, question[i])
                    try:
                        cond_value_ =  float(cond_value)
                    except:
                        cond_value_ = cond_value
                    cond.append(cond_value_)
                    tmp_conds.append(cond)

                    cond = []
                    cond.append(col3)
                    cond.append(op3)
                    pr_wv_str = question_token[i][start3:end3]
                    cond_value = merge_wv_t1_eng(pr_wv_str, question[i])
                    try:
                        cond_value_ = float(cond_value)
                    except:
                        cond_value_ = cond_value
                    cond.append(cond_value_)
                    tmp_conds.append(cond)

                    cond = []
                    cond.append(col4)
                    cond.append(op4)
                    pr_wv_str = question_token[i][start4:end4]
                    cond_value = merge_wv_t1_eng(pr_wv_str, question[i])
                    try:
                        cond_value_ = float(cond_value)
                    except:
                        cond_value_ = cond_value
                    cond.append(cond_value_)
                    tmp_conds.append(cond)

                # print(select_column_random, select_agg_random, tmp_conds)
                pred_answer_column = engine.execute(table[i]['id'], select_column_random, select_agg_random, tmp_conds)

                explore_count += 1
                if explore_count % 100000 == 0:
                    print(explore_count)
                if explore_count > 500000:
                    break

                exact_match = list_exact_match(gt_answer_list, pred_answer_column)
                if where_number_random==1 and not exact_match and contains(pred_answer_column,gt_answer_list)\
                        and saved_where_number==-1 and op==0: # 可能会导致1condition的错过
                    # where_number_random = 2 # 可能会导致1condition的错过
                    saved_start1 = start
                    saved_end1 = end
                    saved_col1 = col
                    saved_op1 = op
                    saved = True

                # answer in
                if exact_match:
                    if pred_answer_column==[None]:
                        break

                    print("explore sql", select_column_random, select_agg_random, tmp_conds)

                    if type(gt_answer_list[0]) == str and select_agg_random!=0:
                        print("fake sql")
                    elif where_number_random == 1 and type(tmp_conds[0][2])==str and tmp_conds[0][1]!=0 or\
                        where_number_random == 2 and type(tmp_conds[1][2]) == str and tmp_conds[1][1] != 0 or\
                        where_number_random == 3 and type(tmp_conds[2][2]) == str and tmp_conds[2][1] != 0 or \
                        where_number_random == 4 and type(tmp_conds[3][2]) == str and tmp_conds[3][1] != 0:
                        print("fake sql")
                    else:
                        # print("explore answer", pred_answer_column)
                        if type(pred_answer_column[0])==int or type(pred_answer_column[0])==float:
                            final_select_agg = select_agg_random
                        else:
                            final_select_agg = 0

                        if final_select_agg == 0:
                            pred_answer_column2 = engine.execute(table[i]['id'], select_column_random, 0, [])
                            for cell in gt_answer_list:
                                if cell in pred_answer_column2 or equal_in(cell, pred_answer_column2):
                                    final_select_column = select_column_random
                                    break
                        else:
                            final_select_column = select_column_random

                        if final_select_agg == 0:
                            pred_answer_column3 = engine.execute(table[i]['id'], "*", 0, tmp_conds)
                            # answer in
                            for cell in gt_answer_list:
                                if cell in pred_answer_column3 or equal_in(cell, pred_answer_column3):
                                    reward_where = True
                                    break
                        else:
                            reward_where = True

                        # same column: word in question and where column
                        if where_number_random >= 1:
                            pred_answer_column4 = engine.execute(table[i]['id'], tmp_conds[0][0], 0, [])
                            for cell in pred_answer_column4:
                                try:
                                    cell_ = str(float(cell))
                                    if cell_ in question[i].lower():
                                        reward_where_cond1 += 0.1
                                        break
                                    cell_ = str(int(cell))
                                    if cell_ in question[i].lower():
                                        reward_where_cond1 += 0.1
                                        break
                                except:
                                    cell = str(cell)
                                    if cell in question[i].lower():
                                        reward_where_cond1 += 0.1
                                        break
                                # same column: where value and where column
                            value = tmp_conds[0][2]
                            if value in pred_answer_column4:
                                reward_where_cond1 += 0.1
                            try:
                                value = float(tmp_conds[0][2])
                                if value in pred_answer_column4:
                                    reward_where_cond1 += 0.1
                            except:
                                pass
                            try:
                                value = int(tmp_conds[0][2])
                                if value in pred_answer_column4:
                                    reward_where_cond1 += 0.1
                            except:
                                pass
                            try:
                                value = str(int(tmp_conds[0][2]))
                                if value in pred_answer_column4:
                                    reward_where_cond1 += 0.1
                            except:
                                pass
                            try:
                                value = str(float(tmp_conds[0][2]))
                                if value in pred_answer_column4:
                                    reward_where_cond1 += 0.1
                            except:
                                pass


                        # same column: word in question and where column
                        if where_number_random >= 2:
                            pred_answer_column4 = engine.execute(table[i]['id'], tmp_conds[1][0], 0, [])
                            for cell in pred_answer_column4:
                                try:
                                    cell_ = str(float(cell))
                                    if cell_ in question[i].lower():
                                        reward_where_cond2 += 0.1
                                        break
                                    cell_ = str(int(cell))
                                    if cell_ in question[i].lower():
                                        reward_where_cond2 += 0.1
                                        break
                                except:
                                    cell = str(cell)
                                    if cell in question[i].lower():
                                        reward_where_cond2 += 0.1
                                        break
                                # same column: where value and where column
                            value = tmp_conds[1][2]
                            if value in pred_answer_column4:
                                reward_where_cond2 += 0.1
                            try:
                                value = float(tmp_conds[1][2])
                                if value in pred_answer_column4:
                                    reward_where_cond2 += 0.1
                            except:
                                pass
                            try:
                                value = int(tmp_conds[1][2])
                                if value in pred_answer_column4:
                                    reward_where_cond2 += 0.1
                            except:
                                pass
                            try:
                                value = str(int(tmp_conds[1][2]))
                                if value in pred_answer_column4:
                                    reward_where_cond2 += 0.1
                            except:
                                pass
                            try:
                                value = str(float(tmp_conds[1][2]))
                                if value in pred_answer_column4:
                                    reward_where_cond2 += 0.1
                            except:
                                pass

                        # same column: word in question and where column
                        if where_number_random >= 3:
                            pred_answer_column4 = engine.execute(table[i]['id'], tmp_conds[2][0], 0, [])
                            for cell in pred_answer_column4:
                                try:
                                    cell_ = str(float(cell))
                                    if cell_ in question[i].lower():
                                        reward_where_cond3 += 0.1
                                        break
                                    cell_ = str(int(cell))
                                    if cell_ in question[i].lower():
                                        reward_where_cond3 += 0.1
                                        break
                                except:
                                    cell = str(cell)
                                    if cell in question[i].lower():
                                        reward_where_cond3 += 0.1
                                        break
                                # same column: where value and where column
                            value = tmp_conds[2][2]
                            if value in pred_answer_column4:
                                reward_where_cond3 += 0.1
                            try:
                                value = float(tmp_conds[2][2])
                                if value in pred_answer_column4:
                                    reward_where_cond3 += 0.1
                            except:
                                pass
                            try:
                                value = int(tmp_conds[2][2])
                                if value in pred_answer_column4:
                                    reward_where_cond3 += 0.1
                            except:
                                pass
                            try:
                                value = str(int(tmp_conds[2][2]))
                                if value in pred_answer_column4:
                                    reward_where_cond3 += 0.1
                            except:
                                pass
                            try:
                                value = str(float(tmp_conds[2][2]))
                                if value in pred_answer_column4:
                                    reward_where_cond3 += 0.1
                            except:
                                pass

                        # same column: word in question and where column
                        if where_number_random >= 4:
                            pred_answer_column4 = engine.execute(table[i]['id'], tmp_conds[3][0], 0, [])
                            for cell in pred_answer_column4:
                                try:
                                    cell_ = str(float(cell))
                                    if cell_ in question[i].lower():
                                        reward_where_cond4 += 0.1
                                        break
                                    cell_ = str(int(cell))
                                    if cell_ in question[i].lower():
                                        reward_where_cond4 += 0.1
                                        break
                                except:
                                    cell = str(cell)
                                    if cell in question[i].lower():
                                        reward_where_cond4 += 0.1
                                        break
                                # same column: where value and where column
                            value = tmp_conds[3][2]
                            if value in pred_answer_column4:
                                reward_where_cond4 += 0.1
                            try:
                                value = float(tmp_conds[3][2])
                                if value in pred_answer_column4:
                                    reward_where_cond4 += 0.1
                            except:
                                pass
                            try:
                                value = int(tmp_conds[3][2])
                                if value in pred_answer_column4:
                                    reward_where_cond4 += 0.1
                            except:
                                pass
                            try:
                                value = str(int(tmp_conds[3][2]))
                                if value in pred_answer_column4:
                                    reward_where_cond4 += 0.1
                            except:
                                pass
                            try:
                                value = str(float(tmp_conds[3][2]))
                                if value in pred_answer_column4:
                                    reward_where_cond4 += 0.1
                            except:
                                pass

                        """ 有问题，cond op 只能强制为 = 因为 > 或 < 不在一行
                        if where_number_random >= 1 and final_select_agg==0:
                            tmp_conds2 = tmp_conds
                            tmp_conds2[0][1] = 0  # EQUAL
                            pred_answer_column5 = engine.execute(table[i]['id'], tmp_conds2[0][0], 0, tmp_conds2)
                            # same row: the answer and this cell
                            for row in table[i]["rows"]:
                                if list_in_list(pred_answer_column5, row) and list_in_list(gt_answer_list, row):
                                    reward_where_cond1 += 0.1
                                    break
    
                        if where_number_random >= 2 and final_select_agg==0:
                            tmp_conds2 = tmp_conds
                            tmp_conds2[0][1] = 0  # EQUAL
                            tmp_conds2[1][1] = 0  # EQUAL
                            pred_answer_column5 = engine.execute(table[i]['id'], tmp_conds2[1][0], 0, tmp_conds2)
                            # same row: the answer and this cell
                            for row in table[i]["rows"]:
                                if list_in_list(pred_answer_column5, row) and list_in_list(gt_answer_list, row):
                                    reward_where_cond2 += 0.1
                                    break
    
                        if where_number_random >= 3 and final_select_agg==0:
                            tmp_conds2 = tmp_conds
                            tmp_conds2[0][1] = 0  # EQUAL
                            tmp_conds2[1][1] = 0  # EQUAL
                            tmp_conds2[2][1] = 0  # EQUAL
                            pred_answer_column5 = engine.execute(table[i]['id'], tmp_conds2[2][0], 0, tmp_conds2)
                            # same row: the answer and this cell
                            for row in table[i]["rows"]:
                                if list_in_list(pred_answer_column5, row) and list_in_list(gt_answer_list, row):
                                    reward_where_cond3 += 0.1
                                    break
    
                        if where_number_random >= 4 and final_select_agg==0:
                            tmp_conds2 = tmp_conds
                            tmp_conds2[0][1] = 0  # EQUAL
                            tmp_conds2[1][1] = 0  # EQUAL
                            tmp_conds2[2][1] = 0  # EQUAL
                            tmp_conds2[3][1] = 0  # EQUAL
                            pred_answer_column5 = engine.execute(table[i]['id'], tmp_conds2[3][0], 0, tmp_conds2)
                            # same row: the answer and this cell
                            for row in table[i]["rows"]:
                                if list_in_list(pred_answer_column5, row) and list_in_list(gt_answer_list, row):
                                    reward_where_cond4 += 0.1
                                    break
                        """

                        if  reward_where_cond1>=0.2 and reward_where==True and where_number_random>=1:
                            final_conds.append(tmp_conds[0])
                        if  reward_where_cond2 >= 0.2 and reward_where == True and where_number_random >= 2:
                            final_conds.append(tmp_conds[1])
                        if  reward_where_cond3 >= 0.2 and reward_where == True and where_number_random >= 3:
                            final_conds.append(tmp_conds[2])
                        if  reward_where_cond4 >= 0.2 and reward_where == True and where_number_random >= 4:
                            final_conds.append(tmp_conds[3])
                        if final_select_agg!=None and final_select_column!=None and (
                                where_number_random == 1 and len(final_conds) == 1 or
                                where_number_random == 2 and len(final_conds) == 2 or
                                where_number_random == 3 and len(final_conds) == 3 or
                                where_number_random == 4 and len(final_conds) == 4):
                            break
                        if  final_select_agg!=None and final_select_column!=None and where_number_random==0:
                            break

            if final_select_column!=None:
                explored_data["sel"] = final_select_column
                explored_data["agg"] = final_select_agg
                explored_data["conds"] = final_conds
                explored_data_list.append(explored_data)
                print(len(explored_data_list))
                one_data = batch_data[i]
                one_data["sql"] = explored_data
                one_data["query"] = explored_data
                f = open("gen_data.jsonl", mode="a", encoding="utf-8")
                json.dump(one_data, f)
                f.write('\n')
                f.close()

    print("Done")
    ave_loss /= count
    acc_sc = count_sc / count
    acc_sa = count_sa / count
    acc_wn = count_wn / count
    acc_wc = count_wc / count
    acc_wo = count_wo / count
    acc_wvi = count_wv / count
    acc_wv = count_wv / count
    acc_lx = count_logic_form_acc / count
    acc_x = count_execute_acc / count

    acc = [ave_loss, acc_sc, acc_sa, acc_wn, acc_wc, acc_wo, acc_wvi, acc_wv, acc_lx, acc_x]

    aux_out = 1

    return acc, aux_out


def get_data(path_wikisql, args):
    train_data, train_table, dev_data, dev_table, _, _ = load_wikisql_v2(path_wikisql,
                                                                         args["toy_model"],
                                                                         args["toy_size"],
                                                                         no_w2i=True,
                                                                         no_hs_tok=True)
    train_loader, dev_loader = get_loader_wikisql(train_data, dev_data, args["batch_size"], shuffle_train=True)

    return train_data, train_table, dev_data, dev_table, train_loader, dev_loader

def get_models(config, BERT_PATH, trained=False, path_model_bert=None, path_model=None):
    # some constants
    agg_ops = ['', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']
    cond_ops = ['=', '>', '<', 'OP']  # do not know why 'OP' required. Hence,

    # Get BERT
    model_bert, tokenizer, bert_config = get_bert(BERT_PATH)
    input_size = bert_config.hidden_size * config["num_target_layers"]  # Seq-to-SQL input vector dimenstion

    return tokenizer, bert_config

def print_result(epoch, acc, dname):
    ave_loss, acc_sc, acc_sa, acc_wn, acc_wc, acc_wo, acc_wvi, acc_wv, acc_lx, acc_x = acc

    print(f'{dname} results ------------')
    print(
        f" Epoch: {epoch}, ave loss: {ave_loss}, acc_sc: {acc_sc:.3f}, acc_sa: {acc_sa:.3f}, acc_wn: {acc_wn:.3f}, \
        acc_wc: {acc_wc:.3f}, acc_wo: {acc_wo:.3f}, acc_wvi: {acc_wvi:.3f}, acc_wv: {acc_wv:.3f}, acc_lx: {acc_lx:.3f}, acc_x: {acc_x:.3f}"
    )


if __name__ == '__main__':

    ## 2. Paths
    path_h = '..'
    path_wikisql = os.path.join(path_h, 'data')
    BERT_PATH = path_wikisql

    path_save_for_evaluation = './'

    ## 3. Load data
    train_data, train_table, dev_data, dev_table, train_loader, dev_loader = get_data(path_wikisql, config)

    tokenizer, bert_config = get_models(config, BERT_PATH)

    # opt, opt_bert = get_opt(model, model_bert)

    ## 6. Train
    acc_lx_t_best = -1
    epoch_best = -1
    epoch = 1
    for epoch in range(epoch):
        # train
        acc_train, aux_out_train = train(train_loader,
                                         train_table,
                                         None,
                                         None,
                                         None,
                                         None,
                                         tokenizer,
                                         config["max_seq_length"],
                                         config["num_target_layers"],
                                         config["accumulate_gradients"],
                                         opt_bert=None,
                                         st_pos=0,
                                         path_db=path_wikisql,
                                         dset_name='train')


        print_result(epoch, acc_train, 'train')