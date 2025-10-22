import tensorflow as tf
from keras import backend

from .VerifyNet import verify_net_model, utils


class VerifyNet:
    def __init__(self, precision, verify_net_path):
        backend.clear_session()

        self.__verify_net = verify_net_model.get_verify_net_model(precision, verify_net_path)
        self._verify_inference = tf.function(
            lambda anchor_batch, sample_batch: self.__verify_net(
                [anchor_batch, sample_batch], training=False
            ),
            reduce_retracing=True,
        )

    def _prepare_pair(self, anchor, sample):
        anchor_batch, sample_batch = utils.preprocess_predict_input(anchor, sample)
        anchor_tensor = tf.convert_to_tensor(anchor_batch, dtype=tf.float32)
        sample_tensor = tf.convert_to_tensor(sample_batch, dtype=tf.float32)
        return anchor_tensor, sample_tensor

    def verify_fingerprints(self, anchor, sample):
        anchor_tensor, sample_tensor = self._prepare_pair(anchor, sample)
        prediction = self._verify_inference(anchor_tensor, sample_tensor)
        return float(prediction.numpy().reshape(-1)[0])

    def verify_fingerprints_batch(self, pairs):
        if not pairs:
            return []

        anchor_tensors = []
        sample_tensors = []
        for anchor, sample in pairs:
            anchor_tensor, sample_tensor = self._prepare_pair(anchor, sample)
            anchor_tensors.append(anchor_tensor)
            sample_tensors.append(sample_tensor)

        anchor_batch = tf.concat(anchor_tensors, axis=0)
        sample_batch = tf.concat(sample_tensors, axis=0)
        predictions = self._verify_inference(anchor_batch, sample_batch)
        return predictions.numpy().reshape(-1).tolist()

    def plot_model(self, file_path):
        tf.keras.utils.plot_model(self.__verify_net, to_file=file_path,
                                  show_shapes=True, expand_nested=True)
