from core.source.entity import Entity
from core.source.tokens import Token
from networks.context.processing.terms.news_terms import NewsTerms


class NewsTermsHelper(object):

    def __init__(self, news_terms):
        assert(isinstance(news_terms, NewsTerms))
        self.__news_terms = news_terms

    def debug_show_terms(self):
        for term in self.__news_terms.iter_terms():
            if isinstance(term, unicode):
                print "Word:    '{}'".format(term.encode('utf-8'))
            elif isinstance(term, Token):
                print "Token:   '{}' ('{}')".format(term.get_token_value().encode('utf-8'),
                                                    term.get_original_value().encode('utf-8'))
            elif isinstance(term, Entity):
                print "Entity:  '{}'".format(term.value.encode('utf-8'))
            else:
                raise Exception("unsuported type {}".format(term))

    def debug_statistics(self):
        terms = list(self.__news_terms.iter_terms())
        words = filter(lambda term: isinstance(term, unicode), terms)
        tokens = filter(lambda term: isinstance(term, Token), terms)
        entities = filter(lambda term: isinstance(term, Entity), terms)
        total = len(words) + len(tokens) + len(entities)

        print "Extracted news_words info, NEWS_ID: {}".format(self.__news_terms.RelatedNewsID)
        print "\tWords: {} ({}%)".format(len(words), 100.0 * len(words) / total)
        print "\tTokens: {} ({}%)".format(len(tokens), 100.0 * len(tokens) / total)
        print "\tEntities: {} ({}%)".format(len(entities), 100.0 * len(entities) / total)


