class AverageMeter:
    
    def __init__(self, name):

        self.name = name
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """Update average value.
        
        Params:
            val: Value being appended.
            n: Number of items that were used to compute val.
        """

        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class AvgDictMeter:

    def __init__(self, name, keys):
        self.name = name
        self.keys = keys
        self.meters = {key: AverageMeter(key) for key in keys}
        
    def update(self, vals, n=1):
        for key in self.keys:
            self.meters.update(vals[key], n=n)