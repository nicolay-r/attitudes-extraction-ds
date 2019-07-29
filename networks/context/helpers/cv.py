import random


def items_to_cv_pairs(cv, items_list, shuffle=True, seed=1):
    """
    Splits array of indices into list of pairs (train_indices_list,
    test_indices_list)
    """

    def chunk_it(sequence, num):
        avg = len(sequence) / float(num)
        out = []
        last = 0.0

        while last < len(sequence):
            out.append(sequence[int(last):int(last + avg)])
            last += avg

        return out

    if shuffle:
        random.Random(seed).shuffle(items_list)

    chunks = chunk_it(items_list, cv)

    for test_index, chunk in enumerate(chunks):
        train_indices = range(len(chunks))
        train_indices.remove(test_index)

        train = [v for train_index in train_indices for v in chunks[train_index]]
        test = chunk

        yield train, test
