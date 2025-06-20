import os
import json
import ast

datasets = ["14lap", "14res", "15res", "16res"]

for dataset in datasets:
    test_file = os.path.join("mydataset/ASTE_G", dataset, "gpt4o_test.jsonl")
    predict_file = os.path.join("mydataset/ASTE_G", dataset, "predict_label.json")
    with open(test_file, 'r', encoding='utf8') as f:
        test_datas = json.load(f)
    with open(predict_file, 'r', encoding='utf8') as f:
        predict_datas = json.load(f)

    all_corrects = 0

    pseudo_corrects = 0
    predict_corrects = 0
    
    pseudo_correct_predict_correct = 0
    pseudo_correct_predict_uncorrect = 0
    pseudo_uncorrect_predict_correct = 0

    for i in range(len(test_datas)):
        test_data = test_datas[i]
        predict_data = predict_datas[i]
        response = test_data["response"]
        label = json.loads(test_data["label"])
        predict = predict_data["predict_final "]
        predict = set(ast.literal_eval(predict))

        try:
            pseudo_label = set()
            response = ast.literal_eval(response)
            for triplet in response:
                try:
                    pseudo_label.add(tuple(item.lower() for item in triplet))
                except:
                    continue
        except:
            pseudo_label = set()


        label = [[item.lower() for item in t] for t in label]
        label = set(tuple(t) for t in label)

        for t in pseudo_label:
            if t in label:
                pseudo_corrects += 1
            if t in label and t in predict:
                pseudo_correct_predict_correct += 1
            if t in label and t not in predict:
                pseudo_correct_predict_uncorrect += 1


        for t in predict:
            if t in label:
                predict_corrects += 1
            if t in label and t not in pseudo_label:
                pseudo_uncorrect_predict_correct += 1

        all_corrects += len(label)

    print("数据集{}: 总元组个数{}，大模型预测正确的元组个数为{}，对齐模型预测正确的元组个数为{}，\n大模型正确且对齐模型正确的元组个数为{}，大模型正确但对齐模型不正确的元组个数为{}，大模型不正确但对齐模型正确的元组个数为{}"
          .format(dataset, all_corrects, pseudo_corrects, predict_corrects, pseudo_correct_predict_correct,pseudo_correct_predict_uncorrect, pseudo_uncorrect_predict_correct))
    print("*"*50)
