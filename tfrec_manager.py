import os
from random import shuffle
from glob import glob
import numpy as np
import tensorflow as tf

# Datasets that the API handles
from datasets import mnist

datasets_map = {
    'mnist': mnist
}

###############
# WRITER PART #
###############


def write_tfrecords(raw_dataset, dataset_name, tfrecords_directory, split,
                    num_shards=1, compression_type=tf.python_io.TFRecordCompressionType.NONE):
    """
    write_tfrecords generates one/multiple .tfrecords files representing the entire dataset

    Args:
        raw_dataset         : List of tuple of np.arrays [(image, label), [...], (image, label)] corresponding
                              to the entire dataset
        dataset_name        : Name of the dataset that should match one of the keys in datasets_map
        tfrecords_directory : Path of the folders containing one folder per dataset.
                              Each folder contains tfrecords file(s)
        split               : One of 'train', 'val', 'test'
        num_shards          : number of shards the dataset should be split
        compression_type    : can be any instance of tf.python_io.TFRecordCompressionType (GZIP, NONE, ZLIB)

    Returns:
        None
    """

    # Filenames are made of the dataset name and split part
    filename = dataset_name + "_" + split

    # Compression type TODO : Workaround the crash for GZIP
    options = tf.python_io.TFRecordOptions(compression_type=compression_type)

    if not os.path.exists(tfrecords_directory):
        os.makedirs(tfrecords_directory)
    if not os.path.exists(os.path.join(tfrecords_directory, dataset_name)):
        os.makedirs(os.path.join(tfrecords_directory, dataset_name))

    if num_shards == 1:
        print("Writing tfrec ", filename)
        dataset_shard_to_tfrec(raw_dataset, os.path.join(tfrecords_directory, dataset_name, filename), options)
    else:
        shard_dataset = np.array_split(raw_dataset, num_shards)
        for shard_num, shard in enumerate(shard_dataset):
            print("Writing tfrec ", filename, str(shard_num), "/", len(shard_dataset))
            dataset_shard_to_tfrec(shard, os.path.join(tfrecords_directory, dataset_name,
                                                       filename + '_' + (str(shard_num+1))), options)


def dataset_shard_to_tfrec(dataset_shard, tfrec_filename, options):
    """
    dataset_shard_to_tfrec builds one tf.Records corresponding to a shard of the entire dataset

    Args:
        dataset_shard  : list of tuple of np.arrays [(image, label), ..., (image, label)] corresponding to the shard
        tfrec_filename : path of the .tfrecords file that is to write
        options        : options for the tf.python_io.TFRecordWriter (see tf.python_io.TFRecordOptions)

    Returns:
        None
    """

    tfrec_filename = tfrec_filename + '.tfrecords'
    if os.path.exists(tfrec_filename):
        print("tfrecords ", tfrec_filename, " already generated")
        return
    else:
        mnist.writer(dataset_shard, tfrec_filename, options)


###############
# PARSER PART #
###############


def data_input_fn(dataset_name, tfrecords_directory, split, batch_size,
                  prefetch, shuffle_tfrec=False, buffer_size=None):
    """
    data_input_fn provides the input function

    Args:
        dataset_name        : Name of the dataset that should match one of the keys in datasets_map
        tfrecords_directory : Path of the folders containing one folder per dataset.
                              Each folder contains tfrecords file(s)
        split               : One of 'train', 'val', 'test'
        batch_size          : batch size at which sample will be picked in the dataset
        prefetch            : number of batches to prefetch
        shuffle_tfrec       : When set to True, the .tfrecords files and examples are shuffled
        buffer_size         : Size of the buffer allocated to shuffle the samples (see tf.data.TFRecordDataset.shuffle)

    Returns:
        _input_fn           : The input function required by the tf.contrib.learn.Experiment
    """

    directory = os.path.join(tfrecords_directory, dataset_name)

    def _input_fn():
        filenames = glob(os.path.join(directory, dataset_name + '_' + split + '*' + '.tfrecords'))
        if shuffle_tfrec:                             # If activated, the filename are processed at random
            shuffle(filenames)
        dataset = tf.data.TFRecordDataset(filenames)  # TODO : compression_type=compression_type (crashes)
        dataset = dataset.repeat(None)                # Infinite iterations: let experiment determine num_epochs
        if shuffle_tfrec:                             # If activated, locally shuffles the data in the .tfrecords
            dataset = dataset.shuffle(buffer_size=buffer_size * batch_size)
        dataset = dataset.batch(batch_size)           # Batch the examples
        dataset = dataset.map(mnist.parser)           # Parse the examples using the parser function
        dataset = dataset.prefetch(prefetch)          # Make sure the GPU doesn't starves and preload 10 batches

        iterator = dataset.make_one_shot_iterator()
        features, labels = iterator.get_next()

        return features, labels

    return _input_fn
