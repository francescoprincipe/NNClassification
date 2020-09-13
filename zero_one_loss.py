import tensorflow as tf


def zero_one_loss(y_true, y_pred):

    equals_array = tf.keras.backend.equal(y_true, tf.keras.backend.argmax(y_pred, 1))

    loss = tf.keras.backend.get_value(1.0 - tf.reduce_sum(tf.keras.backend.cast(equals_array, tf.int32)) / tf.size(equals_array))

    return loss
