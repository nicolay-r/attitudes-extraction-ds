"""
(C) IINemo
Original: https://github.com/IINemo/syntaxnet_wrapper
"""


class Span(object):
    def __init__(self, begin=-1, end=-1):
        self.begin = begin
        self.end = end

    def left_overlap(self, other):
        return (self.begin <= other.begin and self.end <= other.end and self.end > other.begin
                or
                self.begin >= other.begin and self.end <= other.end)

    def overlap(self, other):
        return self.left_overlap(other) or other.left_overlap(self)

    def __unicode__(self):
        return u'{} {}'.format(self.begin, self.end)

    def __str__(self):
        return self.__unicode__().encode('utf-8')

    def equal(self, other):
        return self.begin == other.begin and self.end == other.end


class Word(Span):
    def __init__(self, pos_tag=u'', morph=u'', word_form=u'',
                 parent=-1, link_name=u'', *args, **kwargs):
        assert(isinstance(pos_tag, unicode))
        assert(isinstance(morph, unicode))
        assert(isinstance(word_form, unicode))
        assert(isinstance(pos_tag, unicode))
        assert(isinstance(parent, int))
        assert(isinstance(link_name, unicode))

        super(Word, self).__init__(*args, **kwargs)
        self.pos_tag = pos_tag
        self.morph = morph
        self.word_form = word_form
        self.parent = parent
        self.link_name = link_name

    def __unicode__(self):
        return u'{} | word_form: {} pos_tag: {} morph: {} parent {} link_name: {}'.format(
            super(Word, self).__unicode__(),
            self.word_form,
            self.pos_tag,
            self.morph,
            self.parent,
            self.link_name)

    def __str__(self):
        return self.__unicode__().encode('utf8')
