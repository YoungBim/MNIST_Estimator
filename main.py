# This is code can be used with tf-1.6.0 that needs cuda 9

# The three lines below do not work but you can test them if you like.
# import os
# os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda-9.0/lib64/"
# print("LD_LIBRARY_PATH:", os.environ["LD_LIBRARY_PATH"])
# Suggestion is to directly do the following command from terminal :
# export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64/
# And then from that one terminal run your favorite IDE / simply run "python main.py"

import tensorflow as tf
import os
print("Working with tensorflow", tf.__version__)


if __name__ == "__main__":

    ############################
    # Hardware/Verbosity Setup #
    ############################

    # See https://stackoverflow.com/questions/37893755/tensorflow-set-cuda-visible-devices-within-jupyter
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # See https://github.com/tensorflow/tensorflow/issues/7778
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
    tf.logging.set_verbosity(tf.logging.INFO)

    ##############
    # Data Setup #
    ##############

    # Get the MNIST dataset
    from tensorflow.examples.tutorials.mnist import input_data
    raw_data_directory = os.path.dirname("/media/bassam/Datasets1/MNIST_dataset/")
    mnist = input_data.read_data_sets(raw_data_directory, reshape=False)
    print("MNIST dataset - train", mnist.train._labels.shape[0], "- valid", mnist.validation._labels.shape[0], "- test",
          mnist.test._labels.shape[0])

    # From the dataset extract train/val/test parts in a dictionary
    # the dictionary is a list of tuples (image, label) each element of the tuple is a numpy array
    filename_suffix = ["train", "valid", "test"]
    mnist_dic = {
        filename_suffix[0]: list(zip(mnist.train._images, mnist.train._labels)),
        filename_suffix[1]: list(zip(mnist.validation._images, mnist.validation._labels)),
        filename_suffix[2]: list(zip(mnist.test._images, mnist.test._labels))
    }

    ###################
    # Parameter Setup #
    ###################

    # Define the parameters necessary to use the TFRecParser/Writer API
    dataset_splits = ["train", "valid", "test"]
    num_shards = [5, 1, 1]
    dataset_name = "mnist"
    tfrecords_path = os.path.dirname("/media/bassam/Datasets1/tfreccords/")
    compression_type = tf.python_io.TFRecordCompressionType.NONE

    # Hyper parameters definition
    hyper_parameters = tf.contrib.training.HParams(
        multi_gpu=False,
        num_epochs=5,
        learning_rate=1e-3,
        batch_size=100,
        batch_size_test=1000,
        nb_class=10,
    )

    ###################
    # Input functions #
    ###################

    # Convert the train/val/test parts of the mnist dataset into tf.Records using one TFRecManager per mode
    from tfrec_manager import write_tfrecords
    tfrec_manager = {}
    for split, nshards in list(zip(dataset_splits, num_shards)):

        tfrec_manager[split] = write_tfrecords(raw_dataset=mnist_dic[split], dataset_name=dataset_name,
                                               tfrecords_directory=tfrecords_path, split=split,
                                               num_shards=nshards, compression_type=compression_type)

    # Create the functions that generates batches from the tfrecords files (train/val/test)
    from tfrec_manager import data_input_fn
    input_fn = {}
    for split in dataset_splits:
        if split == "train" or split == "eval":
            input_fn[split] = data_input_fn(dataset_name=dataset_name, tfrecords_directory=tfrecords_path,
                                            split=split, batch_size=hyper_parameters.batch_size,
                                            prefetch=10, shuffle_tfrec=True, buffer_size=20
                                            )
        else:
            input_fn[split] = data_input_fn(dataset_name=dataset_name, tfrecords_directory=tfrecords_path,
                                            split=split, batch_size=hyper_parameters.batch_size,
                                            prefetch=1, shuffle_tfrec=False
                                            )  # TODO : Check that these parameters work @test

    #############
    # RunConfig #
    #############

    # Compute the number of steps for one epoch
    steps_per_epoch = len(mnist_dic["train"]) // hyper_parameters.batch_size  # len dataset // batch size
    num_train_steps = hyper_parameters.num_epochs * steps_per_epoch  # Total number of steps of the training
    save_ckpt_step = steps_per_epoch//2  # Number of steps

    # Checkpoints and summaries parameter setup
    import datetime
    logdir = os.path.join("/home/bassam/MNIST_Estimator/",
                                          datetime.datetime.now().strftime("%Y_%m_%d_%H%M%S"))
    run_config = tf.contrib.learn.RunConfig(
        model_dir=logdir,
        keep_checkpoint_max=3,
        save_checkpoints_steps=save_ckpt_step,
        save_summary_steps=50)

    #############
    # Estimator #
    #############

    # Estimator definition
    from model import mnist_net_model_fn
    model_fn = mnist_net_model_fn

    # Batch size should be divisible in number of GPU's
    # See https://github.com/tensorflow/models/blob/master/official/mnist/mnist.py#L184
    if hyper_parameters.multi_gpu:
        from multi_gpu import validate_batch_size_for_multi_gpu
        validate_batch_size_for_multi_gpu(hyper_parameters.batch_size)

        model_fn = tf.contrib.estimator.replicate_model_fn(
            model_fn, loss_reduction=tf.losses.Reduction.MEAN)

    mnist_classifier = tf.estimator.Estimator(
        model_fn=model_fn,
        config=run_config,
        params=hyper_parameters
    )

    ##############
    # Experiment #
    ##############

    # Define the experiment object
    experiment = tf.contrib.learn.Experiment(
        mnist_classifier,
        train_input_fn=input_fn["train"],
        eval_input_fn=input_fn["valid"],
        train_steps=num_train_steps
    )

    # Full train and evaluation every time a checkpoint is built (if no use of "min_eval_frequency" in the experiment)
    experiment.train_and_evaluate()

    # Full train : experiment.train()
    # One evaluation step until the input is exhausted : experiment.evaluate()
    # Singe step evaluation experiment.test()
