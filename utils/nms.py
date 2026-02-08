import torch


def nms(boxes, scores, overlap=0.5, top_k=200):
    keep = scores.new(scores.size(0)).zero_().long()
    if boxes.numel() == 0:
        return keep, 0

    x1, y1, x2, y2 = boxes.t()
    area = (x2 - x1) * (y2 - y1)

    _, idx = scores.sort()
    idx = idx[-top_k:]

    count = 0
    while idx.numel() > 0:
        i = idx[-1]
        keep[count] = i
        count += 1
        if idx.size(0) == 1:
            break

        idx = idx[:-1]
        xx1 = torch.clamp(x1[idx], min=x1[i])
        yy1 = torch.clamp(y1[idx], min=y1[i])
        xx2 = torch.clamp(x2[idx], max=x2[i])
        yy2 = torch.clamp(y2[idx], max=y2[i])

        w = torch.clamp(xx2 - xx1, min=0)
        h = torch.clamp(yy2 - yy1, min=0)
        inter = w * h

        union = area[idx] + area[i] - inter
        iou = inter / union
        idx = idx[iou <= overlap]

    return keep, count
