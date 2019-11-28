import gc
import random
from os import path, makedirs
from os.path import join
from networks.context.configurations.base import CommonModelSettings
from networks.ranlp.io_dev import RaNLPConfTaskDevIO
from networks.ranlp.io_rsr import RaNLPConfTaskRuSentRelIO
from networks.ranlp.io_rsr_dev import RaNLPConfTaskRuSentRelWithDevIO
from predict_nn.ranlp.callback import RaNLPTaskCallback
import io_utils


def run_cv_testing(model_name,
                   create_config,
                   create_network,
                   create_model,
                   create_io,
                   test_on_epochs,
                   modify_settings_callback=None,
                   cancel_training_by_cost=True):
    assert(isinstance(model_name, unicode))
    assert(callable(create_config))
    assert(callable(create_network))
    assert(callable(create_model))
    assert(isinstance(test_on_epochs, list))
    assert(callable(modify_settings_callback) or modify_settings_callback is None)
    assert(isinstance(cancel_training_by_cost, bool))

    io, callback = create_io_and_callback(create_io=create_io,
                                          model_name=model_name,
                                          test_on_epochs=test_on_epochs,
                                          cancel_training_by_cost=cancel_training_by_cost)

    if isinstance(io, RaNLPConfTaskRuSentRelWithDevIO):
        __save_separated_collection(io.RuSentRelIO)
        __save_separated_collection(io.DevIO)
    else:
        __save_separated_collection(io)

    with callback:
        for _ in range(io.CVCount):
            settings = create_config()
            network = create_network()

            __modify_settings(settings=settings,
                              test_on_epochs=test_on_epochs)

            if modify_settings_callback is not None:
                modify_settings_callback(settings)

            model = create_model(io=io,
                                 network=network,
                                 settings=settings,
                                 callback=callback)

            model.run(load_model=False)

            del settings
            del network
            del model

            io.inc_cv_index()
            gc.collect()


def __modify_settings(settings, test_on_epochs):
    assert(isinstance(settings, CommonModelSettings))
    assert(isinstance(test_on_epochs, list))

    settings.modify_test_on_epochs(test_on_epochs)
    settings.modify_classes_count(3)
    settings.modify_learning_rate(0.01)
    settings.modify_use_class_weights(True)
    settings.modify_dropout_keep_prob(0.9)
    settings.modify_bags_per_minibatch(8)


def create_io_and_callback(create_io,
                           model_name,
                           test_on_epochs,
                           cancel_training_by_cost):
    assert(callable(create_io))
    assert(isinstance(model_name, unicode))
    assert(isinstance(test_on_epochs, list))
    assert(isinstance(cancel_training_by_cost, bool))

    io = create_io(model_to_pretrain_name=model_name)

    callback = RaNLPTaskCallback(
        test_on_epochs=test_on_epochs,
        log_dir=io_utils.get_log_root(),
        io=io,
        cv_filepath=u"{}-cv.txt".format(model_name),
        cancel_by_cost=cancel_training_by_cost)

    callback.PredictVerbosePerFileStatistic = False

    return io, callback


def __save_separated_collection(io):
    assert(isinstance(io, RaNLPConfTaskRuSentRelIO) or
           isinstance(io, RaNLPConfTaskDevIO))

    def __create_folder(dirpath):
        if not path.exists(dirpath):
            makedirs(dirpath)

    source_file = io.SourceFile
    target_data_folder = io.SplittedDataFolder

    __create_folder(target_data_folder)

    sample = None
    samples = []
    with open(source_file, 'r') as f:
        for line in f.readlines():
            if '---' in line:
                if sample is not None:
                    samples.append(sample)
                sample = [line]
            else:
                sample.append(line)

    __write_samples(samples=samples,
                    io=io,
                    target_data_folder=target_data_folder,
                    check_rule=__check_rule if io.CVCount > 2 else __check_rule_fixed)


def __write_samples(samples, io, target_data_folder, check_rule):
    """
    check_rule: func
        (sample_index, part_size, part_index)
    """
    assert(callable(check_rule))

    parts_to_split = io.CVCount
    prefix = u"part_{}.txt"

    # CV mode
    random.Random(1).shuffle(samples)

    print "Separate samples into {} files".format(io.CVCount)
    print "Texts count: {}".format(len(samples))

    part_size = len(samples) / parts_to_split
    for part_index in range(parts_to_split):
        part = str(part_index).zfill(2)
        filepath = join(target_data_folder, prefix.format(part))

        print "Writing: {}".format(filepath)

        with open(filepath, "w") as out:

            written = 0
            for s_index, sample in enumerate(samples):
                if check_rule(s_index, part_size, part_index):
                    for line in sample:
                        out.write(line)
                    written += 1

            print "News written: {}".format(written)


def __check_rule(s_index, part_size, part_index):
    """
    CV.
    """
    return s_index / part_size == part_index


def __check_rule_fixed(s_index, part_size, part_index):
    """
    Fixed, rsr separation.
    """
    if part_index == 0:
        return s_index < 43
    else:
        return s_index >= 43
