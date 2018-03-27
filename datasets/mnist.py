import tensorflow as tf
from datasets import common
from tqdm import tqdm


def writer(dataset_shard, tfrec_filename, options):
    """
    writer core function to parse a .tfrecords for mnist dataset

    Args:
        dataset_shard  : list of tuple of np.arrays [(image, label), ..., (image, label)] corresponding to the shard
        tfrec_filename : path of the .tfrecords file that is to write
        options        : options for the tf.python_io.TFRecordWriter (see tf.python_io.TFRecordOptions)

    Returns:
        image  : a tf.Tensor corresponding to the image
        label  : a tf.Tensor corresponding to the label
    """
    with tf.python_io.TFRecordWriter(tfrec_filename, options=options) as record_writer:
        for count, (image, label) in tqdm(enumerate(dataset_shard)):
            height, width, depth = image.shape
            img_str = image.tostring()
            example = common.example_from_dict({
                'height': common.int64_feature(height),
                'width': common.int64_feature(width),
                'depth': common.int64_feature(depth),
                'label': common.int64_feature(int(label)),
                'image': common.bytes_feature(img_str)
            })
            record_writer.write(example.SerializeToString())


def parser(record):
    """
    parser core function to parse a .tfrecords generated for mnist dataset

    Args:
        record : Path of the .tfrecord file to parse

    Returns:
        image  : a tf.Tensor corresponding to the image
        label  : a tf.Tensor corresponding to the label
    """
    features = {
        'height': tf.FixedLenFeature([], tf.int64),
        'width': tf.FixedLenFeature([], tf.int64),
        'depth': tf.FixedLenFeature([], tf.int64),
        'label': tf.FixedLenFeature([], tf.int64),
        'image': tf.FixedLenFeature([], tf.string)
    }
    parsed_example = tf.parse_example(record, features)

    height_raw = tf.cast(parsed_example['height'], tf.int32)
    width_raw = tf.cast(parsed_example['width'], tf.int32)
    depth_raw = tf.cast(parsed_example['depth'], tf.int32)

    image = tf.decode_raw(parsed_example['image'], tf.float32)
    label = tf.cast(parsed_example['label'], tf.int32)

    return image, label
