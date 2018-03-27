import tensorflow as tf


def simple_model(features, nb_class=10):
    """
    A simple network for the MNIST classification problem
    This function will be called to create the model
    """
    # Reshape the input
    features = tf.reshape(features, [-1, 28, 28, 1], name='input_reshape')
    tf.summary.image('images', features)
    features = tf.layers.flatten(features)

    # Describe the network
    model = tf.layers.dense(features, units=100, activation=tf.nn.relu)
    model = tf.layers.dense(model, units=200, activation=tf.nn.relu)
    model = tf.layers.dense(model, units=nb_class)

    return model


def mnist_net_model_fn(features, labels, mode, params):
    """
    A model_fn to create an Estimator instance
    This function will be called to create a new graph each time an estimator method is called
    """
    # Pure Keras-related
    tf.keras.backend.set_learning_phase(mode == tf.estimator.ModeKeys.TRAIN)

    # Pass through the network
    with tf.name_scope("simple_model"):
        logits = simple_model(features, nb_class=params.nb_class)
    predictions = {'class': tf.argmax(logits, axis=1), 'image': features}

    # In the case of tf.estimator.ModeKeys.PREDICT
    if mode == tf.estimator.ModeKeys.PREDICT:
        with tf.name_scope('PREDICT'):
            return tf.estimator.EstimatorSpec(mode=tf.estimator.ModeKeys.PREDICT, predictions=predictions)

    labels = tf.one_hot(labels, depth=10)
    one_hot_encoded_label = tf.reshape(labels, [-1, params.nb_class])
    gt_label = tf.argmax(tf.cast(one_hot_encoded_label, tf.int32), axis=1)

    # Compute the loss
    entropy = tf.nn.softmax_cross_entropy_with_logits(labels=tf.cast(labels, tf.float32), logits=logits)

    # In the case tf.estimator.ModeKeys.EVAL
    if mode == tf.estimator.ModeKeys.EVAL:
        with tf.name_scope('EVAL'):
            loss = tf.reduce_mean(entropy, name='loss')  # Computes the mean over all the examples in the batch
            # Metrics & Summary
            eval_metric_ops = get_eval_metric_ops(gt_label, predictions['class'])

            return tf.estimator.EstimatorSpec(predictions=predictions,
                                              loss=loss,
                                              mode=mode,
                                              eval_metric_ops=eval_metric_ops)

    # In the case of tf.estimator.ModeKeys.TRAIN
    if mode == tf.estimator.ModeKeys.TRAIN:
        # Add a summary for accuracy
        eval_metric_ops = get_eval_metric_ops(gt_label, predictions['class'])
        tf.summary.scalar('Accuracy', eval_metric_ops["Accuracy"][1])

        with tf.name_scope('TRAIN'):
            loss = tf.reduce_mean(entropy, name='loss')  # Computes the mean over all the examples in the batch
            tf.identity(loss, 'cross_entropy')
            # Optimizer
            train_op = tf.train.AdamOptimizer(learning_rate=params.learning_rate)

            # See https://github.com/tensorflow/models/blob/master/official/mnist/mnist.py#L113
            if params.multi_gpu:
                train_op = tf.contrib.estimator.TowerOptimizer(train_op)

            train_op = train_op.minimize(loss=loss, global_step=tf.train.get_global_step())

            # Return the EstimatorSpec instance
            return tf.estimator.EstimatorSpec(predictions=predictions,
                                              loss=loss,
                                              train_op=train_op,
                                              mode=mode)


def get_eval_metric_ops(labels, predictions):
    """Return a dict of the evaluation Ops.
    Args:
        labels (Tensor): Labels tensor for training and evaluation.
        predictions (Tensor): Predictions Tensor.
    Returns:
        Dict of metric results keyed by name.
    """
    return {
        'Accuracy': tf.metrics.accuracy(
            labels=labels,
            predictions=predictions,
            name='accuracy')
    }
