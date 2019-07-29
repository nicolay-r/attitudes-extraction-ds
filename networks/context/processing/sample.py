import collections

import numpy as np
from collections import OrderedDict
from core.source.entity import Entity
from networks.context.processing.relations.collection import ExtractedRelation
from networks.context.configurations.base import CommonModelSettings
import utils


class Sample(object):
    """
    Base sample which is a part of a Bag
    It provides a to_network_input method which
    generates an input info in an appropriate way
    """

    I_X_INDS = "x_indices"
    I_SUBJ_IND = "subj_inds"
    I_OBJ_IND = "obj_inds"
    I_SUBJ_DISTS = "subj_dist"
    I_OBJ_DISTS = "obj_dist"
    I_POS_INDS = "pos_inds"
    I_TERM_TYPE = "term_type"
    I_POSITION_IN_TEXT = "pos_in_text"
    I_NLP_FEATURES = "nlp_features"

    def __init__(self, X,
                 subj_ind,
                 obj_ind,
                 dist_from_subj,
                 dist_from_obj,
                 pos_indices,
                 term_type,
                 relation_id):
        """
            X: np.ndarray
                x indices for embedding
            y: int
            subj_ind: int
                subject index positions
            obj_ind: int
                object index positions
            dist_from_subj: np.ndarray
            dist_from_obj: np.ndarray
            pos_indices: np.ndarray
        """
        assert(isinstance(X, np.ndarray))
        assert(isinstance(subj_ind, int))
        assert(isinstance(obj_ind, int))
        assert(isinstance(dist_from_subj, np.ndarray))
        assert(isinstance(dist_from_obj, np.ndarray))
        assert(isinstance(pos_indices, np.ndarray))
        assert(isinstance(term_type, np.ndarray))
        assert(isinstance(relation_id, int))

        self.__relation_id = relation_id

        self.values = OrderedDict(
            [(Sample.I_X_INDS, X),
             (Sample.I_SUBJ_IND, subj_ind),
             (Sample.I_OBJ_IND, obj_ind),
             (Sample.I_SUBJ_DISTS, dist_from_subj),
             (Sample.I_OBJ_DISTS, dist_from_obj),
             (Sample.I_POS_INDS, pos_indices),
             (Sample.I_TERM_TYPE, term_type)])

    @property
    def RelationID(self):
        return self.__relation_id

    @classmethod
    def create_empty(cls, settings):
        assert(isinstance(settings, CommonModelSettings))
        blank = np.zeros(settings.TermsPerContext)
        return cls(X=blank,
                   subj_ind=0,
                   obj_ind=1,
                   dist_from_subj=blank,
                   dist_from_obj=blank,
                   pos_indices=blank,
                   term_type=blank,
                   relation_id=-1)

    @classmethod
    def from_relation(cls,
                      relation,
                      terms,
                      term_index_in_sentence_func,
                      settings):
        assert(isinstance(relation, ExtractedRelation))
        assert(isinstance(terms, list))
        assert(isinstance(settings, CommonModelSettings))

        subj_ind = term_index_in_sentence_func(relation.TextPosition.LeftEntityPosition.TermIndex)
        obj_ind = term_index_in_sentence_func(relation.TextPosition.RightEntityPosition.TermIndex)

        pos_indices = utils.calculate_pos_indices_for_terms(
            terms=terms,
            pos_tagger=settings.PosTagger)

        x_indices = utils.calculate_embedding_indices_for_terms(
            terms=terms,
            term_embedding_matrix=settings.TermEmbeddingMatrix,
            word_embedding=settings.WordEmbedding,
            static_embedding=settings.StaticWordEmbedding)

        term_type = Sample.__create_term_types(terms)

        sentence_len = len(x_indices)

        pad_size = settings.TermsPerContext
        pad_value = 0

        if sentence_len < pad_size:
            cls.__pad_right_inplace(pos_indices, pad_size=pad_size, filler=pad_value)
            cls.__pad_right_inplace(x_indices, pad_size=pad_size, filler=pad_value)
            cls.__pad_right_inplace(term_type, pad_size=pad_size, filler=pad_value)
        else:
            b, e, subj_ind, obj_ind = cls.__crop_bounds(
                sentence_len=sentence_len,
                window_size=settings.TermsPerContext,
                e1=subj_ind,
                e2=obj_ind)
            cls.__crop_inplace([x_indices, pos_indices, term_type], begin=b, end=e)

        assert(len(pos_indices) ==
               len(x_indices) ==
               len(term_type) ==
               settings.TermsPerContext)

        # Fast hot fix
        x_indices[obj_ind] = 0
        x_indices[subj_ind] = 0

        dist_from_subj = Sample.__dist(subj_ind, settings.TermsPerContext)
        dist_from_obj = Sample.__dist(obj_ind, settings.TermsPerContext)

        return cls(X=np.array(x_indices),
                   subj_ind=subj_ind,
                   obj_ind=obj_ind,
                   dist_from_subj=dist_from_subj,
                   dist_from_obj=dist_from_obj,
                   pos_indices=np.array(pos_indices),
                   term_type=np.array(term_type),
                   relation_id=relation.RelationID)

    @staticmethod
    def __dist(pos, size):
        result = np.zeros(size)
        for i in range(len(result)):
            result[i] = i-pos if i-pos >= 0 else i-pos+size
        return result

    @staticmethod
    def __create_term_types(terms):
        assert(isinstance(terms, collections.Iterable))
        feature = []
        for term in terms:
            if isinstance(term, unicode):
                feature.append(0)
            elif isinstance(term, Entity):
                feature.append(1)
            else:
                feature.append(-1)

        return feature

    @staticmethod
    def __crop_inplace(lists, begin, end):
        for i, lst in enumerate(lists):
            if end < len(lst):
                del lst[end:]
            del lst[:begin]

    @staticmethod
    def check_ability_to_create_sample(window_size, relation):
        subj_ind = relation.TextPosition.LeftEntityPosition.TermIndex
        obj_ind = relation.TextPosition.RightEntityPosition.TermIndex
        return abs(subj_ind - obj_ind) < window_size

    @staticmethod
    def __crop_bounds(sentence_len, window_size, e1, e2):
        assert(isinstance(sentence_len, int))
        assert(isinstance(window_size, int) and window_size > 0)
        assert(isinstance(e1, int) and isinstance(e2, int))
        assert(e1 >= 0 and e2 >= 0)
        assert(e1 < sentence_len and e2 < sentence_len)
        w_begin = 0
        w_end = window_size
        while not (Sample.__in_window(w_b=w_begin, w_e=w_end, i=e1) and
                   Sample.__in_window(w_b=w_begin, w_e=w_end, i=e2)):
            w_begin += 1
            w_end += 1

        return w_begin, w_end, e1 - w_begin, e2 - w_begin

    @staticmethod
    def __in_window(w_b, w_e, i):
        return i >= w_b and i < w_e

    @staticmethod
    def __pad_right_inplace(lst, pad_size, filler):
        """
        Pad list ('lst') with additional elements (filler)

        lst: list
        pad_size: int
            result size
        filler: int
        returns: None
            inplace
        """
        assert(pad_size - len(lst) > 0)
        lst.extend([filler] * (pad_size - len(lst)))

    def save(self, filepath):
        pass

    def load(self, filepath):
        pass

    def __iter__(self):
        for key, value in self.values.iteritems():
            yield key, value
