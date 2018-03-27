import os
from random import shuffle
from glob import glob
import numpy as np
import tensorflow as tf
from tqdm import tqdm


class TFRecManager:
    """
        TODO : Add the generic file header
        TFRecBuilder Class dedicated to write tf.Records from raw (numpy arrays) data
    """

    def __init__(self, directory, filename, num_shards=1, compression_type=tf.python_io.TFRecordCompressionType.NONE):
        """
        __init__ : Constructor of the TFRecBuilder class

        directory         : path of the tfrecords file(s)
        filename          : base filename of the .tfrecords file(s) (without the ''tfrecords'' extension)
        num_shards        : number of shards the dataset should be split
        compression_type  : can be any instance of tf.python_io.TFRecordCompressionType (GZIP, NONE, ZLIB)
        """
        self.directory = directory
        self.filename = filename
        self.num_shards = num_shards
        self.compression_type = compression_type
        self.options = tf.python_io.TFRecordOptions(compression_type=compression_type)

    # TODO : create the function mnist_example(self): so that the user just has to re-implement this part :D

    def dataset_shard_to_tfrec(self, tfrec_filename, dataset_shard):
        """
        dataset_shard_to_tfrec : builds one tf.Records corresponding to a shard of the entire dataset

        tfrec_filename : path of the .tfrecords file that is to write
        dataset_shard  : list of tuple of np.arrays [(image, label), ..., (image, label)] corresponding to the shard
        """

        def _int64_feature(value):
            return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

        def _bytes_feature(value):
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

        def _example_from_dict(dic):
            return tf.train.Example(features=tf.train.Features(feature=dic))

        def mnist_writer():
            with tf.python_io.TFRecordWriter(tfrec_filename, options=self.options) as writer:
                for count, (image, label) in tqdm(enumerate(dataset_shard)):
                    height, width, depth = image.shape
                    img_str = image.tostring()
                    example = _example_from_dict({
                        'height': _int64_feature(height),
                        'width': _int64_feature(width),
                        'depth': _int64_feature(depth),
                        'label': _int64_feature(int(label)),
                        'image_raw': _bytes_feature(img_str)
                    })
                    writer.write(example.SerializeToString())

        tfrec_filename = tfrec_filename + '.tfrecords'
        if os.path.exists(tfrec_filename):
            print("TFRecords ", tfrec_filename, " already generated")
            return
        else:
            mnist_writer()

    def generate_tfrec(self, dataset):
        """
        generate_tfrec : generates one/multiple .tfrecords files representing the entire dataset

        dataset    : list of tuple of np.arrays [(image, label), [...], (image, label)] corresponding to the entire data
        """

        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

        if self.num_shards == 1:
            print("Writing tfrec ", self.filename)
            self.dataset_shard_to_tfrec(os.path.join(self.directory, self.filename), dataset)
        else:
            shard_dataset = np.array_split(dataset, self.num_shards)
            for shard_num, shard in enumerate(shard_dataset):
                print("Writing tfrec ", self.filename, str(shard_num), "/", len(shard_dataset))
                self.dataset_shard_to_tfrec(os.path.join(self.directory,
                                                         self.filename + '_' + (str(shard_num+1))), shard)

    def data_input_fn(self, batch_size, shuffle_tfrec=False):
        """
        data_input_fn : provides access to the input function

        batch_size        : batch size at which sample will be picked in the dataset
        """

        def mnist_parser(record):
            """
            mnist_parser : core function to parse a .tfrecords generated for mnist

            """
            features = {
                'height': tf.FixedLenFeature([], tf.int64),
                'width': tf.FixedLenFeature([], tf.int64),
                'depth': tf.FixedLenFeature([], tf.int64),
                'label': tf.FixedLenFeature([], tf.int64),
                'image_raw': tf.FixedLenFeature([], tf.string)
            }
            parsed_example = tf.parse_example(record, features)

            height_raw = tf.cast(parsed_example['height'], tf.int32)
            height_raw.set_shape(shape=batch_size)
            width_raw = tf.cast(parsed_example['width'], tf.int32)
            width_raw.set_shape(shape=batch_size)
            depth_raw = tf.cast(parsed_example['depth'], tf.int32)
            depth_raw.set_shape(shape=batch_size)

            image = tf.decode_raw(parsed_example['image_raw'], tf.float32)
            # image_shape = tf.stack([tf.constant(batch_size), height_raw[0], width_raw[0], depth_raw[0]],
            #                        name='img_shape')
            # image = tf.reshape(image, image_shape)
            label = tf.cast(parsed_example['label'], tf.int32)

            return image, label

        def _input_fn():
            filenames = glob(os.path.join(self.directory, self.filename + '*' + '.tfrecords'))
            if shuffle_tfrec:                    # If activated, the filename are processed at random
                shuffle(filenames)
            dataset = tf.data.TFRecordDataset(filenames)  # compression_type=self.compression_type makes it crash
            dataset = dataset.repeat(None)       # Infinite iterations: let experiment determine num_epochs
            if shuffle_tfrec:                    # If activated, localy shuffles the data in the tf.records
                dataset = dataset.shuffle(buffer_size=20 * batch_size)
            dataset = dataset.batch(batch_size)  # Batch the examples
            dataset = dataset.map(mnist_parser)  # Parse the examples using the parser function
            dataset = dataset.prefetch(10)       # Make sure the GPU doesn't starves and preload 10 batches

            iterator = dataset.make_one_shot_iterator()
            features, labels = iterator.get_next()

            return features, labels

        return _input_fn
