import torch

def scoring(pred, target, topk):
    pred = torch.argsort(pred, dim=1, descending=True)
    pred = pred.cpu().detach().numpy()  # [batch_size, hashtag_vocab_size]
    # print('pred')
    # print(pred)
    target = target  # [batch_size, hashtag_vocab_size]
    # print('target')
    # print(target)
    tag_label = []

    for this_data in target:
        tag_label.append([])
        for idx, each_tag in enumerate(this_data):
            if each_tag != 0:
                tag_label[-1].append(idx)
    precision = []
    recall = []
    f1 = []
    print(pred)
    for i in range(len(pred)):
        this_precision = 0
        this_recall = 0
        this_f1 = 0
        if (len(tag_label[i]) != 0):
            for j in range(topk):
                if pred[i][j] in tag_label[i]:
                    this_precision += 1
            for j in range(len(tag_label[i])):
                if tag_label[i][j] in pred[i][:topk]:
                    this_recall += 1
            this_precision /= topk
            this_recall /= len(tag_label[i])
            if this_precision != 0 and this_recall != 0:
                this_f1 = 2 * (this_precision * this_recall) / (this_precision + this_recall)
        precision.append(this_precision)
        recall.append(this_recall)
        f1.append(this_f1)
    return precision, recall, f1