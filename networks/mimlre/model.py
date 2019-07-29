from networks.context.model import ContextLevelTensorflowModel
from networks.mimlre.helper.initialization import MIMLREModelInitHelper
from networks.mimlre.processing.batch import MultiInstanceBatch


class MIMLRETensorflowModel(ContextLevelTensorflowModel):

    def create_batch_by_bags_group(self, bags_group):
        return MultiInstanceBatch(bags_group)

    def create_model_init_helper(self):
        return MIMLREModelInitHelper(io=self.IO, settings=self.Settings)
