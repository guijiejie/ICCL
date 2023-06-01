import numpy as np

def _knn(similarity, target, topk=1):
    datasize = similarity.shape[0]

    correct = (target == target.transpose())

    assert correct.shape == similarity.shape
    select_index = np.arange(datasize).reshape(datasize, -1)

    correct[select_index, select_index] = False

    sim_index = np.argpartition(similarity, -topk, axis=1)[:, -topk:]

    result = correct[select_index, sim_index].sum() / float(topk) / datasize

    return result


def KNN(pred, target, topk=500, l2norm=True):
    assert len(pred.shape) == 2
    assert len(target.shape) == 2
    print(pred.shape)
    print(target.shape)

    if l2norm:
        pred = pred / np.linalg.norm(pred, axis=1, keepdims=True)

    out = np.matmul(pred, pred.transpose())

    return _knn(out, target, topk)

if __name__ == "__main__":
    a = np.random.rand(16, 512)
    b = np.random.randint(0, 16, size=(16, 1))
    print(b)
    print(KNN(a, b, topk=10))
