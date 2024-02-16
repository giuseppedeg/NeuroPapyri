import numpy as np

def prATk(actual, predicted, k=10):
    pr=[]
    for actual, predicted in zip(actual, predicted):
        if len(predicted)>k:
            predicted = predicted[:k]

        score = 0.0
        num_hits = 0.0

        for i,p in enumerate(predicted):
            if p in actual:
                num_hits += 1.0
                score += num_hits / (i+1.0)

        if not actual:
            num_hits = 0.0
        
        pr.append(num_hits/k)

    return pr

def rcATk(actual, predicted, k=10):
    pass

def apk(actual, predicted, k=10):
    apr=[]
    for actual, predicted in zip(actual, predicted):
        if len(predicted)>k:
            predicted = predicted[:k]

        score = 0.0
        num_hits = 0.0

        for i,p in enumerate(predicted):
            if p in actual:
                num_hits += 1.0
                score += num_hits / (i+1.0)

        if not actual:
            score = 0.0
        if score == 0:
            apr.append(0)
        else:
            apr.append(score / num_hits)

    return apr

def mapk(actual, predicted, k=10):
    """
    Computes the mean average precision at k.
    """
    return np.mean(apk(actual,predicted,k))

def topAccuracy(actual, predicted, k=10):
    num_hits = 0
    for act, pred in zip(actual, predicted):
        if len(pred)>k:
            pred = pred[:k]

        for a in act:
            if a in pred:
                num_hits += 1
                continue
    
    return num_hits / len(actual)

if __name__ == "__main__":
    act = [[6], [1]]
    pred = [[6,6,0,0,0,6,0,6,0,0], [0,0,0,1,1,1,0,0,0,0]]

    pAT1 = prATk(act,pred, 5)
    print(pAT1)

    apk = apk(act,pred, 6)
    print(apk)

    map = mapk(act,pred,6)
    print(map)

    to1A = topAccuracy(act,pred,4)
    print(to1A)

    