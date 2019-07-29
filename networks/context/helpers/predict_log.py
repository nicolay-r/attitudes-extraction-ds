

class PredictVariables:

    def __init__(self):
        self.__relations_ids = []
        self.__by_param_names = {}

    def has_variable(self, var_name):
        return var_name in self.__by_param_names

    def add(self, names, results, relation_ids):
        assert(isinstance(names, list))
        assert(isinstance(results, list))
        assert(isinstance(relation_ids, list))

        for i, name in enumerate(names):
            if name not in self.__by_param_names:
                self.__by_param_names[name] = []
            self.__by_param_names[name].append(results[i])
            self.__relations_ids.append(relation_ids[i])

    def iter_by_value(self, value):
        for i, value in enumerate(self.__by_param_names[value]):
            yield self.__relations_ids[i], value
