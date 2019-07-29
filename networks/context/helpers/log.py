
def display_log(log_names, log_values):
    assert(len(log_names) == len(log_values))
    print '==========================================='
    for index, log_value in enumerate(log_values):
        print "{}: {}".format(log_names[index], log_value)
    print '==========================================='