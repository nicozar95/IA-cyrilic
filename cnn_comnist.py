from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf
import os
import sys
from PIL import Image

tf.logging.set_verbosity(tf.logging.INFO)

def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  input_layer = tf.reshape(features["x"], [-1, 278, 278, 1])

  # Convolutional Layer #1
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[70, 70],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #1
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  # 139 Image size

  # Convolutional Layer #2
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[70, 70],
      padding="same",
      activation=tf.nn.relu)
  # Pooling layer #2
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  # 70 Image Size

  # Convolutional Layer #3
  conv3 = tf.layers.conv2d(
      inputs=pool2,
      filters=128,
      kernel_size=[70, 70],
      padding="same",
      activation=tf.nn.relu)
  # Pooling layer #3
  pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)

  # 45 Image Size


  # Dense Layer #1 and only
  pool3_flat = tf.reshape(pool3, [-1, 45 * 45 * 128])
  dense = tf.layers.dense(inputs=pool3_flat, units=3072, activation=tf.nn.relu)
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits Layer
  logits = tf.layers.dense(inputs=dropout, units=33)

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def main(unused_argv):


  # Load training and eval data
  class_names = os.listdir("./train")
  comnist = convert_to_tfrecord("comnist","./train",class_names,)
  train_data = comnist.train.images # Returns np.array
  train_labels = np.asarray(comnist.train.labels, dtype=np.int32)
  eval_data = comnist.test.images # Returns np.array
  eval_labels = np.asarray(comnist.test.labels, dtype=np.int32)

  # Create the Estimator
  mnist_classifier = tf.estimator.Estimator(
    model_fn=cnn_model_fn, model_dir="./tmp/comnist_convnet_model")

  # Set up logging for predictions
  tensors_to_log = {"probabilities": "softmax_tensor"}
  logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=50)

  # Train the model
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": train_data},
    y=train_labels,
    batch_size=100,
    num_epochs=None,
    shuffle=True)
  mnist_classifier.train(
    input_fn=train_input_fn,
    steps=20000,
    hooks=[logging_hook])

    # Evaluate the model and print results
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": eval_data},
    y=eval_labels,
    num_epochs=1,
    shuffle=False)
  eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
  print(eval_results)


  def convert_to_tfrecord(dataset_name, data_directory, class_map, segments=1, directories_as_labels=True, files='**/*.png'):

    # Create a dataset of file path and class tuples for each file
    filenames = glob.glob(os.path.join(data_directory, files))
    classes = (os.path.basename(os.path.dirname(name)) for name in filenames) if directories_as_labels else [None] * len(filenames)
    dataset = list(zip(filenames, classes))

    # If sharding the dataset, find how many records per file
    num_examples = len(filenames)
    samples_per_segment = num_examples // segments

    print(f"Have {samples_per_segment} per record file")

    for segment_index in range(segments):
        start_index = segment_index * samples_per_segment
        end_index = (segment_index + 1) * samples_per_segment

        sub_dataset = dataset[start_index:end_index]
        record_filename = os.path.join(data_directory, f"{dataset_name}-{segment_index}.tfrecords")

        with tf.python_io.TFRecordWriter(record_filename) as writer:
            print(f"Writing {record_filename}")

            for index, sample in enumerate(sub_dataset):
                sys.stdout.write(f"\rProcessing sample {start_index+index+1} of {num_examples}")
                sys.stdout.flush()

                file_path, label = sample
                image = Image.open(file_path)
                image = image.resize((224, 224))
                image_raw = np.array(image).tostring()

                features = {
                    'label': _int64_feature(class_map[label]),
                    'text_label': _bytes_feature(label),
                    'image': _bytes_feature(image_raw)
                }
                example = tf.train.Example(features=tf.train.Features(feature=features))
                writer.write(example.SerializeToString())

if __name__ == "__main__":
  tf.app.run()
