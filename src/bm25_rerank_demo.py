import os
import numpy as np
from rank_bm25 import BM25Okapi

# key为id,value为text
query_dict = dict()
passage_dict = dict()
qp_pair_dict = dict()

inverse_query_dict = dict()
inverse_passage_dict = dict()


def init_data():
    path = "data/validation/msmarco-passagetest2019-43-top1000.tsv"
    count = 0
    with open(path, "r") as f:
        lines = f.readlines()
        for line in lines:
            #qid, pid, query, passage
            data = line.split('\t')
            for i, _ in enumerate(data):
                data[i] = data[i].strip()

            if data[0] not in query_dict:
                query_dict[data[0]] = data[2]
            if data[1] not in passage_dict:
                passage_dict[data[1]] = data[3]
            if data[0] not in qp_pair_dict:
                qp_pair_dict[data[0]] = []
            qp_pair_dict[data[0]].append(data[1])

            # count += 1
            # if count == 1000:
            #     break
    # dict inverse 以便输出数据
    # for k, v in query_dict.items():
    #     inverse_query_dict[v] = k
    # for k, v in passage_dict.items():
    #     inverse_passage_dict[v] = k
    # pass


def get_query_corpus(query_id):
    res = []
    for pid in qp_pair_dict[query_id]:
        res.append(passage_dict[pid])
    return res


if __name__ == "__main__":
    init_data()
    # print(query_dict)
    # print(qp_pair_dict)
    output_list = []
    for query_id, query_text in query_dict.items():

        corpus = get_query_corpus(query_id)
        tokenized_corpus = [passage.split(' ') for passage in corpus]

        bm25 = BM25Okapi(tokenized_corpus, k1=8, b=0.68)
        tokenized_query = query_text.split(' ')
        # ranks = bm25.get_top_n(tokenized_query, corpus, n=len(corpus))
        passage_ids = qp_pair_dict[query_id]
        scores = bm25.get_scores(tokenized_query)  # np.array
        scores_sort_list = np.argsort(-scores)

        # <查询ID> Q0 <文档ID> <文档排序> <文档评分> <系统ID>
        for i, passage_id in enumerate(passage_ids):
            temp_res = []
            # passage_id = inverse_passage_dict[passage]
            temp_res.append(str(query_id))
            temp_res.append("Q0")
            temp_res.append(str(passage_id))
            temp_res.append(str(scores_sort_list[i]+1))

            temp_res.append(str(round(scores[i], 4)))
            temp_res.append("durant")
            res_line = "\t".join(temp_res)
            res_line += "\n"
            output_list.append(res_line)

    output_path = "result/bm25_demo_8_0.68.trec"
    with open(output_path, "w") as f:
        for line in output_list:
            f.write(line)
