from collections import OrderedDict
from networks.context.debug import DebugKeys
from networks.context.processing.bags.bag import Bag
from networks.context.processing.batch import MiniBatch


class MultiInstanceBatch(MiniBatch):

    def __init__(self, bags, batch_id=None):
        super(MultiInstanceBatch, self).__init__(bags, batch_id)

    def to_network_input(self):
        result = OrderedDict()

        for bag_index, bag in enumerate(self.iter_by_bags()):
            assert(isinstance(bag, Bag))
            for sample_index, sample in enumerate(bag):
                for arg, value in sample:
                    if arg not in result:
                        result[arg] = [[None] * len(bag) for _ in range(len(self.bags))]
                    result[arg][bag_index][sample_index] = value

        for bag in self.iter_by_bags():
            if self.I_LABELS not in result:
                result[self.I_LABELS] = []
            result[self.I_LABELS].append(bag.BagLabel.to_uint())

        if DebugKeys.MiniBatchShow:
            MiniBatch.debug_output(result)

        return result
