import numpy as np

from networks.context.processing.bags.bag import Bag
from networks.context.processing.bags.collection import BagsCollection
from networks.context.processing.relations.collection import ExtractedRelationsCollection, ExtractedRelation
from networks.context.processing.sample import Sample


class MultiInstanceBagsCollection(BagsCollection):
    """
    Has a different algo of bags completion.
    May contain a various amount of instances (samples) within a bag.
    """

    @classmethod
    def from_linked_relations(
            cls,
            relation_collection,
            data_type,
            max_bag_size,
            create_sample_func,
            create_empty_sample_func,
            shuffle):
        assert(isinstance(relation_collection, ExtractedRelationsCollection))
        assert(isinstance(data_type, unicode))
        assert(isinstance(max_bag_size, int) and max_bag_size > 0)
        assert(callable(create_sample_func))
        assert(callable(create_empty_sample_func))
        assert(isinstance(shuffle, bool))

        def last_bag():
            return bags[-1]

        def complete_last_bag():
            bag = last_bag()
            while len(bag) < max_bag_size:
                bag.add_sample(create_empty_sample_func())

        def is_empty_last_bag():
            return len(last_bag())

        def is_context_continued(c_rel, p_rel):
            return p_rel.TextPosition.LeftEntityPosition.SentenceIndex + 1 == \
                   c_rel.TextPosition.LeftEntityPosition.SentenceIndex

        bags = []

        for relations in relation_collection.iter_by_linked_relations():

            bags.append(Bag(label=relations[0].Label))
            for r_ind, relation in enumerate(relations):
                assert(isinstance(relation, ExtractedRelation))

                if len(last_bag()) == max_bag_size:
                    bags.append(Bag(label=relation.Label))

                s = create_sample_func(relation)

                prior_rel = relations[r_ind-1] if r_ind > 0 else None
                if prior_rel is not None and not is_empty_last_bag():
                    if not is_context_continued(c_rel=relation, p_rel=prior_rel):
                        complete_last_bag()
                        bags.append(Bag(label=relation.Label))

                assert(isinstance(s, Sample))
                last_bag().add_sample(s)

            if is_empty_last_bag():
                bags = bags[:-1]
                continue

            complete_last_bag()

        if shuffle:
            np.random.shuffle(bags)

        return cls(bags)

