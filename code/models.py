###list of models:

import numpy as np
import os
#os.environ["CUDA_VISIBLE_DEVICES"]="1"
#
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import add_arg_scope
from layers import wide_resnet, wide_resnet_snorm, valid_resnet
from layers import  wide_resnet_snorm
from tfops import specnormconv3d, dynamic_deconv3d
import tensorflow_hub as hub
import tensorflow_probability
import tensorflow_probability as tfp
tfd = tensorflow_probability.distributions
tfd = tfp.distributions
tfb = tfp.bijectors


### Model
def _mdn_model_fn(features, labels, nchannels, n_y, n_mixture, dropout, optimizer, mode, pad, distribution='logistic'):

    # Check for training mode
    is_training = mode == tf.estimator.ModeKeys.TRAIN
        
    def _module_fn():
        """
        Function building the module
        """
    
        feature_layer = tf.placeholder(tf.float32, shape=[None, None, None, None, nchannels], name='input')
        obs_layer = tf.placeholder(tf.float32, shape=[None, None, None, None, n_y], name='observations')

        # Builds the neural network
        if pad == 0:
            net = slim.conv3d(feature_layer, 16, 5, activation_fn=tf.nn.leaky_relu, padding='same')
        elif pad == 2:
            net = slim.conv3d(feature_layer, 16, 5, activation_fn=tf.nn.leaky_relu, padding='valid')
        if pad == 4:
            net = slim.conv3d(feature_layer, 16, 5, activation_fn=tf.nn.leaky_relu, padding='valid')
            net = slim.conv3d(net, 16, 5, activation_fn=tf.nn.leaky_relu, padding='valid')
        #net = wide_resnet(feature_layer, 8, activation_fn=tf.nn.leaky_relu, is_training=is_training)
        #net = wide_resnet(net, 16, activation_fn=tf.nn.leaky_relu, keep_prob=dropout, is_training=is_training)
        net = wide_resnet(net, 16, activation_fn=tf.nn.leaky_relu, keep_prob=dropout, is_training=is_training)
        net = wide_resnet(net, 16, activation_fn=tf.nn.leaky_relu, keep_prob=dropout, is_training=is_training)
        if distribution == 'logistic': net = slim.conv3d(net, 32, 3, activation_fn=tf.nn.tanh)
        else: net = slim.conv3d(net, 32, 3, activation_fn=tf.nn.leaky_relu)

        # Define the probabilistic layer 
        net = slim.conv3d(net, n_mixture*3*n_y, 1, activation_fn=None)
        cube_size = tf.shape(obs_layer)[1]
        net = tf.reshape(net, [-1, cube_size, cube_size, cube_size, n_y, n_mixture*3])
#         net = tf.reshape(net, [None, None, None, None, n_y, n_mixture*3])
        loc, unconstrained_scale, logits = tf.split(net,
                                                    num_or_size_splits=3,
                                                    axis=-1)
        scale = tf.nn.softplus(unconstrained_scale)

        # Form mixture of discretized logistic distributions. Note we shift the
        # logistic distribution by -0.5. This lets the quantization capture "rounding"
        # intervals, `(x-0.5, x+0.5]`, and not "ceiling" intervals, `(x-1, x]`.
        if distribution == 'logistic':
            discretized_logistic_dist = tfd.QuantizedDistribution(
                distribution=tfd.TransformedDistribution(
                    distribution=tfd.Logistic(loc=loc, scale=scale),
                    bijector=tfb.AffineScalar(shift=-0.5)),
                low=0.,
                high=2.**3-1)

            mixture_dist = tfd.MixtureSameFamily(
                mixture_distribution=tfd.Categorical(logits=logits),
                components_distribution=discretized_logistic_dist)

        elif distribution == 'normal':

            mixture_dist = tfd.MixtureSameFamily(
                mixture_distribution=tfd.Categorical(logits=logits),
                components_distribution=tfd.Normal(loc=loc, scale=scale))
            

        # Define a function for sampling, and a function for estimating the log likelihood
        #sample = tf.squeeze(mixture_dist.sample())
        sample = mixture_dist.sample()
        loglik = mixture_dist.log_prob(obs_layer)
        hub.add_signature(inputs={'features':feature_layer, 'labels':obs_layer}, 
                          outputs={'sample':sample, 'loglikelihood':loglik,
                                   'loc':loc, 'scale':scale, 'logits':logits})
    

    # Create model and register module if necessary
    spec = hub.create_module_spec(_module_fn)
    module = hub.Module(spec, trainable=True)
    if isinstance(features,dict):
        predictions = module(features, as_dict=True)
    else:
        predictions = module({'features':features, 'labels':labels}, as_dict=True)
    
    if mode == tf.estimator.ModeKeys.PREDICT:    
        hub.register_module_for_export(module, "likelihood")
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    
    loglik = predictions['loglikelihood']
    # Compute and register loss function
    neg_log_likelihood = -tf.reduce_sum(loglik, axis=-1)
    neg_log_likelihood = tf.reduce_mean(neg_log_likelihood)
    
    tf.losses.add_loss(neg_log_likelihood)
    total_loss = tf.losses.get_total_loss(add_regularization_losses=True)

    train_op = None
    eval_metric_ops = None

    # Define optimizer
    if mode == tf.estimator.ModeKeys.TRAIN:
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            global_step=tf.train.get_global_step()
            boundaries = list(np.array([1e4, 2e4, 4e4, 5e4, 6e4]).astype(int))
            values = [1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 1e-6]
            learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)
            train_op = optimizer(learning_rate=learning_rate).minimize(loss=total_loss, global_step=global_step)
            tf.summary.scalar('rate', learning_rate)                            
        tf.summary.scalar('loss', neg_log_likelihood)
    elif mode == tf.estimator.ModeKeys.EVAL:
        
        eval_metric_ops = { "log_p": neg_log_likelihood}

    return tf.estimator.EstimatorSpec(mode=mode,
                                      predictions=predictions,
                                      loss=total_loss,
                                      train_op=train_op,
                                      eval_metric_ops=eval_metric_ops)







### Model
def _mdn_mask_model_fn(features, labels, nchannels, n_y, n_mixture, dropout, optimizer, mode, pad, lr0=1e-3, distribution='logistic', masktype='constant'):

    # Check for training mode
    is_training = mode == tf.estimator.ModeKeys.TRAIN
        
    def _module_fn():
        """
        Function building the module
        """
    
        feature_layer = tf.placeholder(tf.float32, shape=[None, None, None, None, nchannels], name='input')
        obs_layer = tf.placeholder(tf.float32, shape=[None, None, None, None, n_y], name='observations')
        mask_layer = tf.clip_by_value(obs_layer, 0, 0.001)*1000
        #       
        # Builds the neural network
        if pad == 0:
            net = slim.conv3d(feature_layer, 16, 5, activation_fn=tf.nn.leaky_relu, padding='same')
        elif pad == 2:
            net = slim.conv3d(feature_layer, 16, 5, activation_fn=tf.nn.leaky_relu, padding='valid')
        #net = wide_resnet(feature_layer, 8, activation_fn=tf.nn.leaky_relu, is_training=is_training)
        #net = wide_resnet(net, 16, activation_fn=tf.nn.leaky_relu, keep_prob=dropout, is_training=is_training)
        net = wide_resnet(net, 16, activation_fn=tf.nn.leaky_relu, keep_prob=dropout, is_training=is_training)
        net = wide_resnet(net, 16, activation_fn=tf.nn.leaky_relu, keep_prob=dropout, is_training=is_training)
        if distribution == 'logistic': net = slim.conv3d(net, 32, 3, activation_fn=tf.nn.tanh)
        else: net = slim.conv3d(net, 32, 3, activation_fn=tf.nn.leaky_relu)

        #Predicted mask
        masknet = slim.conv3d(net, 8, 1, activation_fn=tf.nn.leaky_relu)
        out_mask = slim.conv3d(masknet, 1, 1, activation_fn=None)
        pred_mask = tf.nn.sigmoid(out_mask)

        # Define the probabilistic layer 
        likenet = slim.conv3d(net, 64, 1, activation_fn=tf.nn.leaky_relu)
        net = slim.conv3d(likenet, n_mixture*3*n_y, 1, activation_fn=None)
        cube_size = tf.shape(obs_layer)[1]
        net = tf.reshape(net, [-1, cube_size, cube_size, cube_size, n_y, n_mixture*3])
#         net = tf.reshape(net, [None, None, None, None, n_y, n_mixture*3])
        loc, unconstrained_scale, logits = tf.split(net,
                                                    num_or_size_splits=3,
                                                    axis=-1)
        scale = tf.nn.softplus(unconstrained_scale) + 1e-3

        # Form mixture of discretized logistic distributions. Note we shift the
        # logistic distribution by -0.5. This lets the quantization capture "rounding"
        # intervals, `(x-0.5, x+0.5]`, and not "ceiling" intervals, `(x-1, x]`.
        if distribution == 'logistic':
            discretized_logistic_dist = tfd.QuantizedDistribution(
                distribution=tfd.TransformedDistribution(
                    distribution=tfd.Logistic(loc=loc, scale=scale),
                    bijector=tfb.AffineScalar(shift=-0.5)),
                low=0.,
                high=2.**3-1)

            mixture_dist = tfd.MixtureSameFamily(
                mixture_distribution=tfd.Categorical(logits=logits),
                components_distribution=discretized_logistic_dist)

        elif distribution == 'normal':

            mixture_dist = tfd.MixtureSameFamily(
                mixture_distribution=tfd.Categorical(logits=logits),
                components_distribution=tfd.Normal(loc=loc, scale=scale))

        # Define a function for sampling, and a function for estimating the log likelihood
        #sample = tf.squeeze(mixture_dist.sample())
        rawsample = mixture_dist.sample()
        sample = rawsample*pred_mask
        rawloglik = mixture_dist.log_prob(obs_layer)
        print(rawloglik)
        print(out_mask)
        print(mask_layer)
        
        #loss1 = - rawloglik* out_mask #This can be constant mask as well if we use mask_layer instead
        if masktype == 'constant': loss1 = - rawloglik* mask_layer 
        elif masktype == 'vary': loss1 = - rawloglik* pred_mask 
        loss2 = tf.nn.sigmoid_cross_entropy_with_logits(logits=out_mask,
                                                labels=mask_layer) 
        loglik = - (loss1 + loss2)

        hub.add_signature(inputs={'features':feature_layer, 'labels':obs_layer}, 
                          outputs={'sample':sample, 'loglikelihood':loglik,
                                   'loc':loc, 'scale':scale, 'logits':logits,
                                   'rawsample':rawsample, 'pred_mask':pred_mask, 'out_mask':out_mask,
                                   'rawloglik':rawloglik, 'loss1':loss1, 'loss2':loss2})


    # Create model and register module if necessary
    spec = hub.create_module_spec(_module_fn)
    module = hub.Module(spec, trainable=True)
    if isinstance(features,dict):
        predictions = module(features, as_dict=True)
    else:
        predictions = module({'features':features, 'labels':labels}, as_dict=True)
    
    if mode == tf.estimator.ModeKeys.PREDICT:    
        hub.register_module_for_export(module, "likelihood")
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    
    rawloglik = tf.reduce_mean(predictions['rawloglik'])    
    loss1 = tf.reduce_mean(predictions['loss1'])    
    loss2 = tf.reduce_mean(predictions['loss2'])    

    # Compute and register loss function
    loglik = predictions['loglikelihood']
    # Compute and register loss function
    neg_log_likelihood = -tf.reduce_sum(loglik, axis=-1)
    neg_log_likelihood = tf.reduce_mean(neg_log_likelihood)
    
    tf.losses.add_loss(neg_log_likelihood)
    total_loss = tf.losses.get_total_loss(add_regularization_losses=True)

    #loss = tf.reduce_mean(loss)    
    #tf.losses.add_loss(loss)
    #total_loss = tf.losses.get_total_loss(add_regularization_losses=True)

    train_op = None
    eval_metric_ops = None

    # Define optimizer
    if mode == tf.estimator.ModeKeys.TRAIN:
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            global_step=tf.train.get_global_step()
            boundaries = list(np.array([1e4, 2e4, 4e4, 5e4, 6e4]).astype(int))
            #values = [1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 1e-6]
            values = [lr0, lr0/2, lr0/10, lr0/20, lr0/100, lr0/1000]
            learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)
            train_op = optimizer(learning_rate=learning_rate).minimize(loss=total_loss, global_step=global_step)
            logging_hook = tf.train.LoggingTensorHook({"iter":global_step, "negloglik" : neg_log_likelihood, 
                                                       "rawloglik" : rawloglik, "loss1" : loss1, "loss2" : loss2 }, every_n_iter=50)

            tf.summary.scalar('rate', learning_rate)                            
        tf.summary.scalar('loglik', neg_log_likelihood)
    elif mode == tf.estimator.ModeKeys.EVAL:
        
        eval_metric_ops = { "log_p": neg_log_likelihood}

    return tf.estimator.EstimatorSpec(mode=mode,
                                      predictions=predictions,
                                      loss=total_loss,
                                      train_op=train_op,
                                      eval_metric_ops=eval_metric_ops, training_hooks = [logging_hook])
