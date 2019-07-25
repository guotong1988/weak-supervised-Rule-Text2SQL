import torch
import bert.tokenization as tokenization
from bert.modeling import BertConfig, BertModel
from sqlova.utils.utils_wikisql import *
from sqlova.model.nl2sql.wikisql_models import *
from sqlnet.dbengine import DBEngine
from train import test

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
    model.train()
    model_bert.train()

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

    for batch_index, batch_data in enumerate(train_loader):
        count += len(batch_data)

        if count < st_pos:
            continue
        # Get fields
        question, question_token, sql, sql_text, sql_t, table, header_token, header \
            = get_fields(batch_data, train_table, no_hs_t=True, no_sql_t=True)

        gt_select_column, gt_select_agg, gt_wherenumber, gt_wherecolumn, g_wo, g_wv = get_gt(sql)
        # get ground truth where-value index under CoreNLP tokenization scheme. It's done already on trainset.
        gt_wherevalueindex_corenlp = get_gt_wherevalueindex_corenlp(batch_data)

        emb_question, emb_header, len_question, len_header_token, number_header, \
        question_token_bert, token_to_berttoken_index, berttoken_to_token_index \
            = get_wemb_bert(bert_config, model_bert, tokenizer, question_token, header, max_seq_length,
                            num_out_layers_n=num_target_layers, num_out_layers_h=num_target_layers)

        try:
            #
            gt_wherevalueindex = get_gt_wherevalueindex_bert_from_gt_wherevalueindex_corenlp(token_to_berttoken_index, gt_wherevalueindex_corenlp)
        except:
            # Exception happens when where-condition is not found in nlu_tt.
            # In this case, that train example is not used.
            # During test, that example considered as wrongly answered.
            # e.g. train: 32.
            continue

        # score
        # gt_wherevalueindex = start_index, end_index
        # score_where_value: [batch,4,question_len,2]
        score_select_column, score_select_agg, score_where_number, score_where_column, score_whereop, score_where_value, \
            score_select_column_softmax, score_select_agg_softmax, score_where_number_softmax,\
            score_where_column_softmax, score_whereop_softmax, score_where_value_softmax \
                    = model(emb_question, len_question, emb_header, len_header_token, number_header,
                           g_sc=gt_select_column, g_sa=gt_select_agg, g_wn=gt_wherenumber,
                           g_wc=gt_wherecolumn, g_wvi=gt_wherevalueindex)

        # Calculate loss & step
        loss = Loss_selectwhere_startend_v2(score_select_column, score_select_agg, score_where_number, score_where_column,
                          score_whereop, score_where_value, gt_select_column, gt_select_agg, gt_wherenumber, gt_wherecolumn, g_wo, gt_wherevalueindex)

        # RL

        # Random explore
        pred_selectcolumn_random, pred_selectagg_random, pred_wherenumber_random, pred_wherecolumn_random, pred_whereop_random, pred_wherevalueindex_random = \
                  pred_selectwhere_startend_random(score_select_column_softmax, score_select_agg_softmax, score_where_number_softmax,
                                                               score_where_column_softmax, score_whereop_softmax, score_where_value_softmax,)
        pred_wherevalue_str_random, pred_wherevalue_str_bert_random = convert_pred_wvi_to_string(pred_wherevalueindex_random, question_token,
                                                                                   question_token_bert,
                                                                                   berttoken_to_token_index, question)

        # Sort pr_wc:
        #   Sort pr_wc when training the model as pr_wo and pr_wvi are predicted using ground-truth where-column (g_wc)
        #   In case of 'dev' or 'test', it is not necessary as the ground-truth is not used during inference.
        pred_wherecolumn_sorted_random = sort_pred_wherecolumn(pred_wherecolumn_random, gt_wherecolumn)
        random_sql_int = generate_sql_i_v2(pred_selectcolumn_random, pred_selectagg_random, pred_wherenumber_random,
                                        pred_wherecolumn_sorted_random, pred_whereop_random, pred_wherevalue_str_random, question)




            # Prediction
        pr_sc, pr_sa, pr_wn, pr_wc, pr_wo, pr_wvi = pred_selectwhere_startend(score_select_column, score_select_agg,
                                                                                  score_where_number,
                                                                                  score_where_column, score_whereop,
                                                                                  score_where_value, )
        pred_wherevalue_str, pred_wherevalue_str_bert_random = convert_pred_wvi_to_string(pr_wvi, question_token,
                                                                                       question_token_bert,
                                                                                       berttoken_to_token_index,
                                                                                       question)

            # Sort pr_wc:
            #   Sort pr_wc when training the model as pr_wo and pr_wvi are predicted using ground-truth where-column (g_wc)
            #   In case of 'dev' or 'test', it is not necessary as the ground-truth is not used during inference.
        pr_wc_sorted = sort_pred_wherecolumn(pr_wc, gt_wherecolumn)
        pred_sql_int = generate_sql_i(pr_sc, pr_sa, pr_wn, pr_wc_sorted, pr_wo, pred_wherevalue_str, question)





        # select column
        def equal_in(cell_, pred_answer_column):
            for cell in pred_answer_column:
                if cell == cell_:
                    return True
            return False

        # select_column_random_int = torch.squeeze(torch.softmax(score_select_column,dim=-1).multinomial(1),dim=-1)
        #select_column_int = torch.squeeze(torch.argmax(score_select_column,dim=-1),dim=-1)
        """
        batch_reward_select_column = []
        for i in range(len(batch_data)):
            gt_answer_list = batch_data[i]["answer"]
            pred_answer_column = engine.execute(table[i]['id'], random_sql_int[i]["sel"], 0, [])
            reward = -1
            for cell in gt_answer_list:
                if cell in pred_answer_column or equal_in(cell,pred_answer_column):
                    reward = 1
            batch_reward_select_column.append(reward)

        onehot_action_batch_selectcolumn = []
        for sql1 in random_sql_int:
            tmp = [0] * score_select_column.shape[1]
            tmp[sql1["sel"]] = 1
            onehot_action_batch_selectcolumn.append(tmp)

        RL_loss_select_column = torch.mean(-torch.log(torch.sum(
            torch.softmax(score_select_column,dim=-1) * torch.tensor(onehot_action_batch_selectcolumn, dtype=torch.float).to(device),dim=-1)
            )* torch.tensor(batch_reward_select_column,dtype=torch.float).to(device))
        """
        # if batch_index%100==0:
        #     print("RL_loss_select_column",RL_loss_select_column.data.cpu().numpy())

        # RL
        # where number

        def list_in_list(small,big):
            for cell in big:
                try:
                    cell = str(int(cell))
                except:
                    cell = str(cell)
                if cell.lower() in small:
                    return True

        batch_reward_where = []
        for i in range(len(batch_data)):
            gt_answer_list = batch_data[i]["answer"]
            reward = 0
            if len(random_sql_int[i]["conds"])>0 and len(pred_sql_int[i]["conds"])>0:
                tmp_conds = []
                tmp_cond = pred_sql_int[i]["conds"][0]
                tmp_cond[0] = random_sql_int[i]["conds"][0][0]
                tmp_cond[2] = random_sql_int[i]["conds"][0][2]
                tmp_conds.append(tmp_cond)
                if tmp_conds[0][2]!="":
                    pred_answer_column = engine.execute(table[i]['id'], "*", 0, tmp_conds)
                    # answer in
                    for cell in gt_answer_list:
                        if cell in pred_answer_column or equal_in(cell, pred_answer_column):
                            reward = 1

                    pred_answer_column4 = engine.execute(table[i]['id'], pred_sql_int[i]["sel"], 0, tmp_conds)
                    # answer absolute in
                    for cell in gt_answer_list:
                        if cell in pred_answer_column4 or equal_in(cell, pred_answer_column4):
                            reward = 1

                if len(tmp_conds)>=1:
                    # same column: word in question and where column
                    pred_answer_column2 = engine.execute(table[i]['id'], tmp_conds[0][0], 0, [])
                    for cell in pred_answer_column2:
                        try:
                            cell = str(int(cell))
                        except:
                            cell = str(cell)
                        if cell in question[i].lower():
                            reward = 1
                    # same column: where value and where column
                        if cell == tmp_conds[0][2]:
                            reward = 1


                tmp_conds2 = []
                tmp_cond2 = pred_sql_int[i]["conds"][0]
                tmp_cond2[0] = random_sql_int[i]["conds"][0][0]
                tmp_cond2[2] = random_sql_int[i]["conds"][0][2]
                tmp_cond2[1] = 0 # EQUAL
                tmp_conds2.append(tmp_cond2)
                if len(tmp_conds2) >= 1:
                    pred_answer_column3 = engine.execute(table[i]['id'], tmp_conds2[0][0], 0, tmp_conds2)
                    # same row: the answer and this cell
                    for row in table[i]["rows"]:
                        if list_in_list(pred_answer_column3,row) and list_in_list(gt_answer_list,row):
                            reward = 1
                            reward = 1

            batch_reward_where.append(reward)

        """
        onehot_action_batch_wherenumber = []
        # where_number_int = torch.squeeze(torch.argmax(score_where_number, dim=-1), dim=-1)
        # where_number_int = torch.squeeze(torch.softmax(score_where_number, dim=-1).multinomial(1), dim=-1)
        where_number_int = []
        for tmp_sql in random_sql_int:
            where_number_int.append(len(tmp_sql["conds"]))
        for action_int in where_number_int:
            tmp = [0] * score_where_number.shape[1]
            tmp[action_int] = 1
            onehot_action_batch_wherenumber.append(tmp)

        RL_loss_where_number = torch.mean(-torch.log(torch.sum(
            torch.softmax(score_where_number, dim=-1) * torch.tensor(onehot_action_batch_wherenumber, dtype=torch.float).to(
                device), dim=-1)
        ) * torch.tensor(batch_reward_where, dtype=torch.float).to(device))
        """

        # RL
        # where column
        # where_column_int = torch.squeeze(torch.argmax(score_where_column, dim=-1), dim=-1)
        # where_column_int = torch.squeeze(torch.softmax(score_where_column, dim=-1).multinomial(1), dim=-1)

        where_column_int = []
        for tmp_sql in random_sql_int:
            if len(tmp_sql["conds"])==0:
                where_column_int.append(-1)
            else:
                where_column_int.append(tmp_sql["conds"][0][0])
        onehot_action_batch_wherecolumn = []
        for action_int in where_column_int:
            tmp = [0] * score_where_column.shape[1]
            if action_int!=-1: # all 0 will runtime error
                tmp[action_int] = 1
            else: # this is must
                tmp = [0.01] * score_where_column.shape[1]
            onehot_action_batch_wherecolumn.append(tmp)

        RL_loss_where_column = torch.mean(-torch.log(torch.sum(
            torch.softmax(score_where_column, dim=-1) * torch.tensor(onehot_action_batch_wherecolumn,
                                                                     dtype=torch.float).to(
                device), dim=-1)
        ) * torch.tensor(batch_reward_where, dtype=torch.float).to(device))


        # RL
        # where value
        # pred_wherevalueindex_random [8,4,2]
        action_wherevalue = []
        for one in pred_wherevalueindex_random:
            if len(one) > 0:
                action = []
                for i in range(1):
                    if len(one)>i:
                        start = one[i][0]
                        tmp_start = [0] * score_where_value.shape[2]
                        tmp_start[start] = 1
                        end = one[i][1]
                        tmp_end = [0] * score_where_value.shape[2]
                        tmp_end[end] = 1
                        action.append([tmp_start,tmp_end])
                    else:
                        action.append([[0.01] * score_where_value.shape[2],[0.01] * score_where_value.shape[2]])
                action_wherevalue.append(action)
            else:
                action = []
                for i in range(1):
                    action.append([[0.01] * score_where_value.shape[2],[0.01] * score_where_value.shape[2]])
                action_wherevalue.append(action)

        tmp_action = torch.squeeze(torch.tensor(action_wherevalue, dtype=torch.float).to(device),dim=1)
        tmp_score_where_value = torch.transpose(score_where_value[:,0,:,:],1,2)
        RL_loss_where_value = torch.mean(
            -torch.log(torch.sum(
            torch.softmax(tmp_score_where_value, dim=-1) * tmp_action, dim=-1)
        ) * torch.unsqueeze(torch.tensor(batch_reward_where, dtype=torch.float),
                        dim=-1).to(device))


        # loss += RL_loss_select_column
        # loss += RL_loss_where_number
        loss += RL_loss_where_column
        loss += RL_loss_where_value
        # Calculate gradient
        if batch_index % accumulate_gradients == 0: # mode
            # at start, perform zero_grad
            opt.zero_grad()
            if opt_bert:
                opt_bert.zero_grad()
            loss.backward()
            if accumulate_gradients == 1:
                opt.step()
                if opt_bert:
                    opt_bert.step()
        elif batch_index % accumulate_gradients == (accumulate_gradients-1):
            # at the final, take step with accumulated graident
            loss.backward()
            opt.step()
            if opt_bert:
                opt_bert.step()
        else:
            # at intermediate stage, just accumulates the gradients
            loss.backward()


        # Cacluate accuracy
        count_sc1_list, count_sa1_list, count_wn1_list, \
        count_wc1_list, count_wo1_list, \
        count_wvi1_list, count_wv1_list = get_count_sw_list(gt_select_column, gt_select_agg, gt_wherenumber, gt_wherecolumn, g_wo, gt_wherevalueindex,
                                                                   pr_sc, pr_sa, pr_wn, pr_wc, pr_wo, pr_wvi,
                                                                   sql, pred_sql_int,
                                                                   mode='train')

        count_lx1_list = get_count_lx_list(count_sc1_list, count_sa1_list, count_wn1_list, count_wc1_list,
                                       count_wo1_list, count_wv1_list)
        # lx stands for logical form accuracy

        # Execution accuracy test.
        count_x1_list, g_ans, pr_ans = get_count_x_list(engine, table, gt_select_column, gt_select_agg, sql, pr_sc, pr_sa, pred_sql_int)

        # statistics
        ave_loss += loss.item()

        # count
        count_sc += sum(count_sc1_list)
        count_sa += sum(count_sa1_list)
        count_wn += sum(count_wn1_list)
        count_wc += sum(count_wc1_list)
        count_wo += sum(count_wo1_list)
        count_wvi += sum(count_wvi1_list)
        count_wv += sum(count_wv1_list)
        count_logic_form_acc += sum(count_lx1_list)
        count_execute_acc += sum(count_x1_list)

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

    # Get Seq-to-SQL

    number_cond_ops = len(cond_ops)
    number_agg_ops = len(agg_ops)
    model = Seq2SQL_v1(input_size=input_size,
                       hidden_size=100,
                       num_layer=2,
                       dropout=config["dropout"],
                       number_cond_ops=number_cond_ops,
                       number_agg_ops=number_agg_ops)
    model = model.to(device)

    if trained:
        assert path_model_bert != None
        assert path_model != None

        if torch.cuda.is_available():
            res = torch.load(path_model_bert)
        else:
            res = torch.load(path_model_bert, map_location='cpu')
        model_bert.load_state_dict(res['model_bert'])
        model_bert.to(device)

        if torch.cuda_is_available():
            res = torch.load(path_model)
        else:
            res = torch.load(path_model, map_location='cpu')

        model.load_state_dict(res['model'])

    return model, model_bert, tokenizer, bert_config

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
    # test_data, test_table = load_wikisql_data(path_wikisql, mode='test', toy_model=args.toy_model, toy_size=args.toy_size, no_hs_tok=True)
    # test_loader = torch.utils.data.DataLoader(
    #     batch_size=args.bS,
    #     dataset=test_data,
    #     shuffle=False,
    #     num_workers=4,
    #     collate_fn=lambda x: x  # now dictionary values are not merged!
    # )
    ## 4. Build & Load models
    model, model_bert, tokenizer, bert_config = get_models(config, BERT_PATH)

    ## 4.1.
    # To start from the pre-trained models, un-comment following lines.
    # path_model_bert =
    # path_model =
    # model, model_bert, tokenizer, bert_config = get_models(args, BERT_PT_PATH, trained=True, path_model_bert=path_model_bert, path_model=path_model)

    ## 5. Get optimizers
    opt, opt_bert = get_opt(model, model_bert)

    ## 6. Train
    acc_lx_t_best = -1
    epoch_best = -1
    epoch = 200
    for epoch in range(epoch):
        # train
        acc_train, aux_out_train = train(train_loader,
                                         train_table,
                                         model,
                                         model_bert,
                                         opt,
                                         bert_config,
                                         tokenizer,
                                         config["max_seq_length"],
                                         config["num_target_layers"],
                                         config["accumulate_gradients"],
                                         opt_bert=opt_bert,
                                         st_pos=0,
                                         path_db=path_wikisql,
                                         dset_name='train')

        # check DEV
        with torch.no_grad():
            acc_dev, results_dev, count_list = test(dev_loader,
                                                    dev_table,
                                                    model,
                                                    model_bert,
                                                    bert_config,
                                                    tokenizer,
                                                    config["max_seq_length"],
                                                    config["num_target_layers"],
                                                    detail=False,
                                                    path_db=path_wikisql,
                                                    st_pos=0,
                                                    dset_name='dev', EG=config["EG"])

        print_result(epoch, acc_train, 'train')
        print_result(epoch, acc_dev, 'dev')