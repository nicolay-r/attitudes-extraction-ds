# -*- coding: utf-8 -*-
from networks.context.processing.terms.news_terms import NewsTerms


class NewsTermsCollection:

    def __init__(self):
        self.__by_id = {}

    def get_by_news_id(self, news_ID):
        assert(isinstance(news_ID, int))
        return self.__by_id[news_ID]

    def add_news_terms(self, news_terms):
        assert(isinstance(news_terms, NewsTerms))
        assert(news_terms.RelatedNewsID not in self.__by_id)
        self.__by_id[news_terms.RelatedNewsID] = news_terms

    def iter_news_terms(self, news_ID):
        assert(isinstance(news_ID, int))
        for term in self.__by_id[news_ID].iter_terms():
            yield term

    def iter_news_ids(self):
        for news_id in self.__by_id.iterkeys():
            yield news_id

    def calculate_min_terms_per_context(self):
        if len(self.__by_id) == 0:
            return None

        return min([len(news_terms) for news_terms in self.__by_id.itervalues()])
