# import tensorflow as tf
# import tensorflow_addons as tfa
# loss_fn = tfa.losses.NpairsLoss()

# from tensorflow.python.ops.numpy_ops import np_config
# np_config.enable_numpy_behavior()

# # from https://github.com/tensorflow/addons/blob/v0.17.0/tensorflow_addons/losses/npairs.py#L23-L73 and https://keras.io/examples/vision/supervised-contrastive-learning/
# @tf.keras.utils.register_keras_serializable(package="Addons")
# @tf.function
# def npairs_loss(y_true: TensorLike, y_pred: TensorLike) -> tf.Tensor:
#     """Computes the npairs loss between `y_true` and `y_pred`.
#     Npairs loss expects paired data where a pair is composed of samples from
#     the same labels and each pairs in the minibatch have different labels.
#     The loss takes each row of the pair-wise similarity matrix, `y_pred`,
#     as logits and the remapped multi-class labels, `y_true`, as labels.
#     The similarity matrix `y_pred` between two embedding matrices `a` and `b`
#     with shape `[batch_size, hidden_size]` can be computed as follows:
#     >>> a = tf.constant([[1, 2],
#     ...                 [3, 4],
#     ...                 [5, 6]], dtype=tf.float16)
#     >>> b = tf.constant([[5, 9],
#     ...                 [3, 6],
#     ...                 [1, 8]], dtype=tf.float16)
#     >>> y_pred = tf.matmul(a, b, transpose_a=False, transpose_b=True)
#     >>> y_pred
#     <tf.Tensor: shape=(3, 3), dtype=float16, numpy=
#     array([[23., 15., 17.],
#        [51., 33., 35.],
#        [79., 51., 53.]], dtype=float16)>
#     <... Note: constants a & b have been used purely for
#     example purposes and have no significant value ...>
#     See: http://www.nec-labs.com/uploads/images/Department-Images/MediaAnalytics/papers/nips16_npairmetriclearning.pdf
#     Args:
#       y_true: 1-D integer `Tensor` with shape `[batch_size]` of
#         multi-class labels.
#       y_pred: 2-D float `Tensor` with shape `[batch_size, batch_size]` of
#         similarity matrix between embedding matrices.
#     Returns:
#       npairs_loss: float scalar.
#     """
#     y_pred = tf.convert_to_tensor(y_pred)
#     y_true = tf.cast(y_true, y_pred.dtype)

#     # Expand to [batch_size, 1]
#     y_true = tf.expand_dims(y_true, -1)
#     y_true = tf.cast(tf.equal(y_true, tf.transpose(y_true)), y_pred.dtype)
#     y_true /= tf.math.reduce_sum(y_true, 1, keepdims=True)

#     loss = tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true)

#     return tf.math.reduce_mean(loss)


# #test with tensorflow 
# y_true = np.array([0,1,2,3,0,1,2,3]).reshape(-1,8)
# y_true_for_torch = np.array([0,1,2,3,0,1,2,3]).reshape(-1,8)
# feature_vectors = np.random.random((8,1024))
# feature_vectors_normalized = tf.math.l2_normalize(feature_vectors, axis=1)
# temperature=1
# logits = tf.divide(
#     tf.matmul(
#         feature_vectors_normalized, tf.transpose(feature_vectors_normalized)
#     ),
#     temperature)

# y_pred = logits
# y_true = tf.squeeze(y_true)
# y_true = tf.expand_dims(y_true, -1)
# y_true = tf.cast(tf.equal(y_true, tf.transpose(y_true)), y_pred.dtype)
# y_true /= tf.math.reduce_sum(y_true, 1, keepdims=True) #横向标准化啦
# loss = tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true)
# tf.math.reduce_mean(loss) # npair loss =1.9861075433921989


class Npairloss(nn.Module):

  def __init__(self,temperature=1.0):
    super(Npairloss,self).__init__()
    self.temperature = temperature
      
  def forward(self,y_true,feature_vectors):
    # y_true = [01,2,3,4]... label encoding format
    # feature_vectors is the embedding matrix


    feature_vectors_normalized = torch.nn.functional.normalize(feature_vectors,dim=1,p=2.0)
    logits = torch.matmul(feature_vectors_normalized,feature_vectors_normalized.T)/self.temperature
    y_true = y_true.view(-1,1)




    y_true = (y_true==y_true.T).to(feature_vectors.dtype)

    y_true = y_true/y_true.sum(1,keepdim=True)

    

    

    loss = torch.nn.functional.cross_entropy(logits,y_true) ## logsoftmax + nllloss equal to softmax_with_logits

    return loss
# Npairloss(1.0)(torch.from_numpy(y_true_for_torch),torch.from_numpy(feature_vectors)) # loss = 1.9861,the same as tf version of npair loss


