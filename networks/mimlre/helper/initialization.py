from networks.context.helpers.initialization import ContextModelInitHelper
from networks.context.processing.sample import Sample
from networks.mimlre.processing.bags import MultiInstanceBagsCollection


class MIMLREModelInitHelper(ContextModelInitHelper):

    def __init__(self, io, settings):
        super(MIMLREModelInitHelper, self).__init__(io=io, settings=settings)

    @staticmethod
    def create_bags_collection(relations_collection, news_terms_collection, data_type, settings):
        return MultiInstanceBagsCollection.from_linked_relations(
            relations_collection,
            max_bag_size=settings.BagSize,
            data_type=data_type,
            shuffle=True,
            create_empty_sample_func=lambda: Sample.create_empty(settings),
            create_sample_func=lambda r: MIMLREModelInitHelper.create_sample_from_relation_and_ntc(
                relation=r, ntc=news_terms_collection, settings=settings))
