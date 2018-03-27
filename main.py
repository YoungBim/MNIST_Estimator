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

    # See https://stackoverflow.com/questions/37893755/tensorflow-set-cuda-visible-devices-within-jupyter
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # See https://github.com/tensorflow/tensorflow/issues/7778
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
    tf.logging.set_verbosity(tf.logging.INFO)

    # Get the MNIST dataset
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("/home/bassam/MNIST_Estimator/MNIST_rawdata", reshape=False)
    print("MNIST dataset - train", mnist.train._labels.shape[0], "- valid", mnist.validation._labels.shape[0], "- test",
          mnist.test._labels.shape[0])

    # From the dataset the train/val/test parts are extracted in a dictionary
    # this dictionary is a list of tuples (image, label) each element of the tuple is a numpy array
    filename_suffix = ["train", "valid", "test"]
    mnist_dic = {
        filename_suffix[0]: list(zip(mnist.train._images, mnist.train._labels)),
        filename_suffix[1]: list(zip(mnist.validation._images, mnist.validation._labels)),
        filename_suffix[2]: list(zip(mnist.test._images, mnist.test._labels))
    }

    # Define the parameters necessary to use the TFRecManager API
    filename_suffix = ["train", "valid", "test"]
    num_shards = [5, 1, 1]
    filename = "mnist"
    directory = os.path.dirname("/home/bassam/MNIST_Estimator/TFRec/")
    compression_type = tf.python_io.TFRecordCompressionType.NONE

    # Hyper parameters definition
    hyper_parameters = tf.contrib.training.HParams(
        multi_gpu=False,
        num_epochs=5,
        learning_rate=1e-3,
        batch_size=100,
        batch_size_test=1000,
        nb_class=10,
        data_directory=directory)

    # Compute the number of steps for one epoch
    steps_per_epoch = len(mnist_dic["train"]) // hyper_parameters.batch_size  # len dataset // batch size
    num_train_steps = hyper_parameters.num_epochs * steps_per_epoch  # Total number of steps of the training
    save_ckpt_step = steps_per_epoch//2  # Number of steps

    # Checkpoints and summaries parameter setup
    run_config = tf.contrib.learn.RunConfig(
        model_dir="/home/bassam/MNIST_Estimator/logdir/",
        keep_checkpoint_max=3,
        save_checkpoints_steps=save_ckpt_step,
        save_summary_steps=50)

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

    # Convert the train/val/test parts of the mnist dataset into tf.Records using one TFRecManager per mode
    from tfrec_manager import TFRecManager
    tfrec_manager = {}
    for suffix, nshards in list(zip(filename_suffix, num_shards)):
        file = filename + "_" + suffix
        tfrec_manager[suffix] = TFRecManager(directory, file, nshards, compression_type=compression_type)
        tfrec_manager[suffix].generate_tfrec(mnist_dic[suffix])

    # Create the functions that generates batches from the tfrecords files (train/val/test)
    input_fn = {}
    for suffix in filename_suffix:
        if suffix == "train":
            input_fn[suffix] = tfrec_manager[suffix].data_input_fn(batch_size=hyper_parameters.batch_size,
                                                                   shuffle_tfrec=True)
        else:
            input_fn[suffix] = tfrec_manager[suffix].data_input_fn(batch_size=hyper_parameters.batch_size_test,
                                                                   shuffle_tfrec=True)

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

    # Test the model
    experiment.test()
