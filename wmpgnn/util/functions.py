import torch

def acc_four_class(pred, label):
    #     print("pred", pred)
    correct = 0
    correct_class1 = 0
    correct_class2 = 0
    correct_class3 = 0
    correct_class4 = 0

    pred_argmax = torch.argmax(pred, dim=1)
    #     print("pred_argmax ", pred_argmax)
    #     label_argmax = torch.argmax(data.y,dim=1)

    pred_class1 = (pred_argmax == 0).sum()
    pred_class2 = (pred_argmax == 1).sum()
    pred_class3 = (pred_argmax == 2).sum()
    pred_class4 = (pred_argmax == 3).sum()

    true_class1 = (label == 0).sum()
    true_class2 = (label == 1).sum()
    true_class3 = (label == 2).sum()
    true_class4 = (label == 3).sum()

    if len(pred) != len(label):
        print("something goes wrong in acc_four_class")
        print(len(pred), len(label))

    else:
        correct_class1 = torch.sum(pred_argmax[label == 0] == label[label == 0])
        correct_class2 = torch.sum(pred_argmax[label == 1] == label[label == 1])
        correct_class3 = torch.sum(pred_argmax[label == 2] == label[label == 2])
        correct_class4 = torch.sum(pred_argmax[label == 3] == label[label == 3])

    correct_preds = torch.Tensor([correct_class1, correct_class2, correct_class3, correct_class4])
    all_preds = torch.Tensor((pred_class1, pred_class2, pred_class3, pred_class4))
    all_label = torch.Tensor((true_class1, true_class2, true_class3, true_class4))

    acc = torch.div(correct_preds, all_label)

    return acc


def weight_four_class(dataset,hetero=False):
    true_class1 = 0
    true_class2 = 0
    true_class3 = 0
    true_class4 = 0
    num_sample = 0

    for tdata in dataset:
        if hetero:
            y = tdata[('tracks','to','tracks')].y
        else:
            y = tdata.y
        true_class1 += (y.argmax(dim=1) == 0).sum()
        true_class2 += (y.argmax(dim=1) == 1).sum()
        true_class3 += (y.argmax(dim=1) == 2).sum()
        true_class4 += (y.argmax(dim=1) == 3).sum()
        num_sample += len(y)


    weight_class1 = num_sample / (4 * true_class1)
    weight_class2 = num_sample / (4 * true_class2)
    weight_class3 = num_sample / (4 * true_class3)
    weight_class4 = num_sample / (4 * true_class4)
    weight = torch.stack((weight_class1, weight_class2, weight_class3, weight_class4))

    print(weight)
    return weight