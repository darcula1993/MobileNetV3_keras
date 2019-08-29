"""MobileNet v3 models for Keras.
"""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
import warnings
import numpy as np

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import backend
from tensorflow.keras import models
from tensorflow.keras import utils as keras_utils
from keras_applications import imagenet_utils
from keras_applications.imagenet_utils import decode_predictions
from keras_applications.imagenet_utils import _obtain_input_shape


BASE_WEIGHT_PATH = None




def correct_pad(backend, inputs, kernel_size):
	"""Returns a tuple for zero-padding for 2D convolution with downsampling.

	# Arguments
		input_size: An integer or tuple/list of 2 integers.
		kernel_size: An integer or tuple/list of 2 integers.

	# Returns
		A tuple.
	"""
	img_dim = 2 if backend.image_data_format() == 'channels_first' else 1
	input_size = backend.int_shape(inputs)[img_dim:(img_dim + 2)]

	if isinstance(kernel_size, int):
		kernel_size = (kernel_size, kernel_size)

	if input_size[0] is None:
		adjust = (1, 1)
	else:
		adjust = (1 - input_size[0] % 2, 1 - input_size[1] % 2)

	correct = (kernel_size[0] // 2, kernel_size[1] // 2)

	return ((correct[0] - adjust[0], correct[0]),
			(correct[1] - adjust[1], correct[1]))


def preprocess_input(x, **kwargs):
	"""Preprocesses a numpy array encoding a batch of images.

	# Arguments
		x: a 4D numpy array consists of RGB values within [0, 255].

	# Returns
		Preprocessed array.
	"""
	return imagenet_utils.preprocess_input(x, mode='tf', **kwargs)


# This function is taken from the original tf repo.
# It ensures that all layers have a channel number that is divisible by 8
# It can be seen here:
# https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py


def _make_divisible(v, divisor, min_value=None):
	if min_value is None:
		min_value = divisor
	new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
	# Make sure that round down does not go down by more than 10%.
	if new_v < 0.9 * v:
		new_v += divisor
	return new_v
def MobileNetV3_small(input_shape=None,
				alpha=1.0,
				include_top=True,
				weights='imagenet',
				input_tensor=None,
				pooling=None,
				classes=1000,
				**kwargs):
	"""Instantiates the MobileNetV2 architecture.

	# Arguments
		input_shape: optional shape tuple, to be specified if you would
			like to use a model with an input img resolution that is not
			(224, 224, 3).
			It should have exactly 3 inputs channels (224, 224, 3).
			You can also omit this option if you would like
			to infer input_shape from an input_tensor.
			If you choose to include both input_tensor and input_shape then
			input_shape will be used if they match, if the shapes
			do not match then we will throw an error.
			E.g. `(160, 160, 3)` would be one valid value.
		alpha: controls the width of the network. This is known as the
		width multiplier in the MobileNetV2 paper, but the name is kept for
		consistency with MobileNetV1 in Keras.
			- If `alpha` < 1.0, proportionally decreases the number
				of filters in each layer.
			- If `alpha` > 1.0, proportionally increases the number
				of filters in each layer.
			- If `alpha` = 1, default number of filters from the paper
				 are used at each layer.
		include_top: whether to include the fully-connected
			layer at the top of the network.
		weights: one of `None` (random initialization),
			  'imagenet' (pre-training on ImageNet),
			  or the path to the weights file to be loaded.
		input_tensor: optional Keras tensor (i.e. output of
			`layers.Input()`)
			to use as image input for the model.
		pooling: Optional pooling mode for feature extraction
			when `include_top` is `False`.
			- `None` means that the output of the model
				will be the 4D tensor output of the
				last convolutional block.
			- `avg` means that global average pooling
				will be applied to the output of the
				last convolutional block, and thus
				the output of the model will be a
				2D tensor.
			- `max` means that global max pooling will
				be applied.
		classes: optional number of classes to classify images
			into, only to be specified if `include_top` is True, and
			if no `weights` argument is specified.

	# Returns
		A Keras model instance.

	# Raises
		ValueError: in case of invalid argument for `weights`,
			or invalid input shape or invalid alpha, rows when
			weights='imagenet'
	"""
	if not (weights in {'imagenet', None} or os.path.exists(weights)):
		raise ValueError('The `weights` argument should be either '
						 '`None` (random initialization), `imagenet` '
						 '(pre-training on ImageNet), '
						 'or the path to the weights file to be loaded.')

	if weights == 'imagenet' and include_top and classes != 1000:
		raise ValueError('If using `weights` as `"imagenet"` with `include_top` '
						 'as true, `classes` should be 1000')

	# Determine proper input shape and default size.
	# If both input_shape and input_tensor are used, they should match
	if input_shape is not None and input_tensor is not None:
		try:
			is_input_t_tensor = backend.is_keras_tensor(input_tensor)
		except ValueError:
			try:
				is_input_t_tensor = backend.is_keras_tensor(
					keras_utils.get_source_inputs(input_tensor))
			except ValueError:
				raise ValueError('input_tensor: ', input_tensor,
								 'is not type input_tensor')
		if is_input_t_tensor:
			if backend.image_data_format == 'channels_first':
				if backend.int_shape(input_tensor)[1] != input_shape[1]:
					raise ValueError('input_shape: ', input_shape,
									 'and input_tensor: ', input_tensor,
									 'do not meet the same shape requirements')
			else:
				if backend.int_shape(input_tensor)[2] != input_shape[1]:
					raise ValueError('input_shape: ', input_shape,
									 'and input_tensor: ', input_tensor,
									 'do not meet the same shape requirements')
		else:
			raise ValueError('input_tensor specified: ', input_tensor,
							 'is not a keras tensor')

	# If input_shape is None, infer shape from input_tensor
	if input_shape is None and input_tensor is not None:

		try:
			backend.is_keras_tensor(input_tensor)
		except ValueError:
			raise ValueError('input_tensor: ', input_tensor,
							 'is type: ', type(input_tensor),
							 'which is not a valid type')

		if input_shape is None and not backend.is_keras_tensor(input_tensor):
			default_size = 224
		elif input_shape is None and backend.is_keras_tensor(input_tensor):
			if backend.image_data_format() == 'channels_first':
				rows = backend.int_shape(input_tensor)[2]
				cols = backend.int_shape(input_tensor)[3]
			else:
				rows = backend.int_shape(input_tensor)[1]
				cols = backend.int_shape(input_tensor)[2]

			if rows == cols and rows in [96, 128, 160, 192, 224]:
				default_size = rows
			else:
				default_size = 224

	# If input_shape is None and no input_tensor
	elif input_shape is None:
		default_size = 224

	# If input_shape is not None, assume default size
	else:
		if backend.image_data_format() == 'channels_first':
			rows = input_shape[1]
			cols = input_shape[2]
		else:
			rows = input_shape[0]
			cols = input_shape[1]

		if rows == cols and rows in [96, 128, 160, 192, 224]:
			default_size = rows
		else:
			default_size = 224

	input_shape = _obtain_input_shape(input_shape,
									  default_size=default_size,
									  min_size=32,
									  data_format=backend.image_data_format(),
									  require_flatten=include_top,
									  weights=weights)

	if backend.image_data_format() == 'channels_last':
		row_axis, col_axis = (0, 1)
	else:
		row_axis, col_axis = (1, 2)
	rows = input_shape[row_axis]
	cols = input_shape[col_axis]

	if weights == 'imagenet':
		if alpha not in [0.35, 0.50, 0.75, 1.0, 1.3, 1.4]:
			raise ValueError('If imagenet weights are being loaded, '
							 'alpha can be one of `0.35`, `0.50`, `0.75`, '
							 '`1.0`, `1.3` or `1.4` only.')

		if rows != cols or rows not in [96, 128, 160, 192, 224]:
			rows = 224
			warnings.warn('`input_shape` is undefined or non-square, '
						  'or `rows` is not in [96, 128, 160, 192, 224].'
						  ' Weights for input shape (224, 224) will be'
						  ' loaded as the default.')

	if input_tensor is None:
		img_input = layers.Input(shape=input_shape)
	else:
		if not backend.is_keras_tensor(input_tensor):
			img_input = layers.Input(tensor=input_tensor, shape=input_shape)
		else:
			img_input = input_tensor

	channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1

	first_block_filters = _make_divisible(16 * alpha, 8)

	x = layers.ZeroPadding2D(padding=correct_pad(backend, img_input, 3),
							 name='Conv1_pad')(img_input)

	x = layers.Conv2D(first_block_filters,
					  kernel_size=3,
					  strides=(2, 2),
					  padding='valid',
					  use_bias=False,
					  name='Conv1')(x)
	x = layers.BatchNormalization(axis=channel_axis,
								  epsilon=1e-3,
								  momentum=0.999,
								  name='bn_Conv1')(x)



	x = layers.Activation(HardSwish,name='Conv_1_hs')(x)

	x = _bneck_block(x, kernel = 3, filters=24, alpha=alpha, stride=2,activation = "relu",se = True,
							expansion=16, block_id=0)

	x = _bneck_block(x, kernel = 3, filters=24, alpha=alpha, stride=2,activation = "relu",se = False,
							expansion=72, block_id=1)

	x = _bneck_block(x, kernel = 5, filters=40, alpha=alpha, stride=1,activation = "relu",se = False,
							expansion=88, block_id=2)

	x = _bneck_block(x, kernel = 5, filters=40, alpha=alpha, stride=2,activation = "hswish",se = True,
							expansion=96, block_id=3)

	x = _bneck_block(x, kernel = 3, filters=80, alpha=alpha, stride=1,activation = "hswish",se = True,
							expansion=240, block_id=4)
	x = _bneck_block(x, kernel = 3, filters=80, alpha=alpha, stride=1,activation = "hswish",se = True,
							expansion=240, block_id=5)
	x = _bneck_block(x, kernel = 3, filters=80, alpha=alpha, stride=1,activation = "hswish",se = True,
							expansion=120, block_id=6)
	x = _bneck_block(x, kernel = 3, filters=80, alpha=alpha, stride=1,activation = "hswish",se = True,
							expansion=144, block_id=7)

	x = _bneck_block(x, kernel = 3, filters=112, alpha=alpha, stride=2,activation = "hswish",se = True,
							expansion=288, block_id=8)
	x = _bneck_block(x, kernel = 3, filters=112, alpha=alpha, stride=1,activation = "hswish",se = True,
							expansion=576, block_id=9)
	x = _bneck_block(x, kernel = 5, filters=160, alpha=alpha, stride=1,activation = "hswish",se = True,
							expansion=576, block_id=10)


	if alpha > 1.0:
		filters = _make_divisible(576 * alpha, 8)
		last_block_filters = _make_divisible(1280 * alpha, 8)
	else:
		filters = 960
		last_block_filters = 1280
	x = layers.Conv2D(filters,
					kernel_size = 1,
					use_bias = False,
					name = "Conv_2")(x)

	x = layers.BatchNormalization(axis=channel_axis,
								  epsilon=1e-3,
								  momentum=0.999,
								  name='Conv_2_bn')(x)

	x = layers.Activation(HardSwish,name='Conv_2_hs')(x)

	x = layers.AveragePooling2D(pool_size = (7,7))(x)

	

	x = layers.Conv2D(last_block_filters,
					  kernel_size=1,
					  use_bias=False,
					  name='Conv_3')(x)

	x = layers.Activation(HardSwish,name='Conv_3_hs')(x)


	if include_top:
		x = layers.Conv2D(1001,
						  kernel_size=1,
						  use_bias=False,
						  name='out_conv')(x)		
		x = backend.squeeze(x, 1)
		x = backend.squeeze(x, 1)

	# Ensure that the model takes into account
	# any potential predecessors of `input_tensor`.
	if input_tensor is not None:
		inputs = keras_utils.get_source_inputs(input_tensor)
	else:
		inputs = img_input

	# Create model.
	model = models.Model(inputs, x,
						 name='mobilenetv3_%0.2f_%s' % (alpha, rows))

	# Load weights.
	if weights == 'imagenet':
		if include_top:
			model_name = ('mobilenet_v2_weights_tf_dim_ordering_tf_kernels_' +
						  str(alpha) + '_' + str(rows) + '.h5')
			weight_path = BASE_WEIGHT_PATH + model_name
			weights_path = keras_utils.get_file(
				model_name, weight_path, cache_subdir='models')
		else:
			model_name = ('mobilenet_v2_weights_tf_dim_ordering_tf_kernels_' +
						  str(alpha) + '_' + str(rows) + '_no_top' + '.h5')
			weight_path = BASE_WEIGHT_PATH + model_name
			weights_path = keras_utils.get_file(
				model_name, weight_path, cache_subdir='models')
		model.load_weights(weights_path)
	elif weights is not None:
		model.load_weights(weights)

	return model

def MobileNetV3_large(input_shape=None,
				alpha=1.0,
				include_top=True,
				weights='imagenet',
				input_tensor=None,
				pooling=None,
				classes=1000,
				**kwargs):
	"""Instantiates the MobileNetV2 architecture.

	# Arguments
		input_shape: optional shape tuple, to be specified if you would
			like to use a model with an input img resolution that is not
			(224, 224, 3).
			It should have exactly 3 inputs channels (224, 224, 3).
			You can also omit this option if you would like
			to infer input_shape from an input_tensor.
			If you choose to include both input_tensor and input_shape then
			input_shape will be used if they match, if the shapes
			do not match then we will throw an error.
			E.g. `(160, 160, 3)` would be one valid value.
		alpha: controls the width of the network. This is known as the
		width multiplier in the MobileNetV2 paper, but the name is kept for
		consistency with MobileNetV1 in Keras.
			- If `alpha` < 1.0, proportionally decreases the number
				of filters in each layer.
			- If `alpha` > 1.0, proportionally increases the number
				of filters in each layer.
			- If `alpha` = 1, default number of filters from the paper
				 are used at each layer.
		include_top: whether to include the fully-connected
			layer at the top of the network.
		weights: one of `None` (random initialization),
			  'imagenet' (pre-training on ImageNet),
			  or the path to the weights file to be loaded.
		input_tensor: optional Keras tensor (i.e. output of
			`layers.Input()`)
			to use as image input for the model.
		pooling: Optional pooling mode for feature extraction
			when `include_top` is `False`.
			- `None` means that the output of the model
				will be the 4D tensor output of the
				last convolutional block.
			- `avg` means that global average pooling
				will be applied to the output of the
				last convolutional block, and thus
				the output of the model will be a
				2D tensor.
			- `max` means that global max pooling will
				be applied.
		classes: optional number of classes to classify images
			into, only to be specified if `include_top` is True, and
			if no `weights` argument is specified.

	# Returns
		A Keras model instance.

	# Raises
		ValueError: in case of invalid argument for `weights`,
			or invalid input shape or invalid alpha, rows when
			weights='imagenet'
	"""
	if not (weights in {'imagenet', None} or os.path.exists(weights)):
		raise ValueError('The `weights` argument should be either '
						 '`None` (random initialization), `imagenet` '
						 '(pre-training on ImageNet), '
						 'or the path to the weights file to be loaded.')

	if weights == 'imagenet' and include_top and classes != 1000:
		raise ValueError('If using `weights` as `"imagenet"` with `include_top` '
						 'as true, `classes` should be 1000')

	# Determine proper input shape and default size.
	# If both input_shape and input_tensor are used, they should match
	if input_shape is not None and input_tensor is not None:
		try:
			is_input_t_tensor = backend.is_keras_tensor(input_tensor)
		except ValueError:
			try:
				is_input_t_tensor = backend.is_keras_tensor(
					keras_utils.get_source_inputs(input_tensor))
			except ValueError:
				raise ValueError('input_tensor: ', input_tensor,
								 'is not type input_tensor')
		if is_input_t_tensor:
			if backend.image_data_format == 'channels_first':
				if backend.int_shape(input_tensor)[1] != input_shape[1]:
					raise ValueError('input_shape: ', input_shape,
									 'and input_tensor: ', input_tensor,
									 'do not meet the same shape requirements')
			else:
				if backend.int_shape(input_tensor)[2] != input_shape[1]:
					raise ValueError('input_shape: ', input_shape,
									 'and input_tensor: ', input_tensor,
									 'do not meet the same shape requirements')
		else:
			raise ValueError('input_tensor specified: ', input_tensor,
							 'is not a keras tensor')

	# If input_shape is None, infer shape from input_tensor
	if input_shape is None and input_tensor is not None:

		try:
			backend.is_keras_tensor(input_tensor)
		except ValueError:
			raise ValueError('input_tensor: ', input_tensor,
							 'is type: ', type(input_tensor),
							 'which is not a valid type')

		if input_shape is None and not backend.is_keras_tensor(input_tensor):
			default_size = 224
		elif input_shape is None and backend.is_keras_tensor(input_tensor):
			if backend.image_data_format() == 'channels_first':
				rows = backend.int_shape(input_tensor)[2]
				cols = backend.int_shape(input_tensor)[3]
			else:
				rows = backend.int_shape(input_tensor)[1]
				cols = backend.int_shape(input_tensor)[2]

			if rows == cols and rows in [96, 128, 160, 192, 224]:
				default_size = rows
			else:
				default_size = 224

	# If input_shape is None and no input_tensor
	elif input_shape is None:
		default_size = 224

	# If input_shape is not None, assume default size
	else:
		if backend.image_data_format() == 'channels_first':
			rows = input_shape[1]
			cols = input_shape[2]
		else:
			rows = input_shape[0]
			cols = input_shape[1]

		if rows == cols and rows in [96, 128, 160, 192, 224]:
			default_size = rows
		else:
			default_size = 224

	input_shape = _obtain_input_shape(input_shape,
									  default_size=default_size,
									  min_size=32,
									  data_format=backend.image_data_format(),
									  require_flatten=include_top,
									  weights=weights)

	if backend.image_data_format() == 'channels_last':
		row_axis, col_axis = (0, 1)
	else:
		row_axis, col_axis = (1, 2)
	rows = input_shape[row_axis]
	cols = input_shape[col_axis]

	if weights == 'imagenet':
		if alpha not in [0.35, 0.50, 0.75, 1.0, 1.3, 1.4]:
			raise ValueError('If imagenet weights are being loaded, '
							 'alpha can be one of `0.35`, `0.50`, `0.75`, '
							 '`1.0`, `1.3` or `1.4` only.')

		if rows != cols or rows not in [96, 128, 160, 192, 224]:
			rows = 224
			warnings.warn('`input_shape` is undefined or non-square, '
						  'or `rows` is not in [96, 128, 160, 192, 224].'
						  ' Weights for input shape (224, 224) will be'
						  ' loaded as the default.')

	if input_tensor is None:
		img_input = layers.Input(shape=input_shape)
	else:
		if not backend.is_keras_tensor(input_tensor):
			img_input = layers.Input(tensor=input_tensor, shape=input_shape)
		else:
			img_input = input_tensor

	channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1

	first_block_filters = _make_divisible(16 * alpha, 8)

	x = layers.ZeroPadding2D(padding=correct_pad(backend, img_input, 3),
							 name='Conv1_pad')(img_input)

	x = layers.Conv2D(first_block_filters,
					  kernel_size=3,
					  strides=(2, 2),
					  padding='valid',
					  use_bias=False,
					  name='Conv1')(x)
	x = layers.BatchNormalization(axis=channel_axis,
								  epsilon=1e-3,
								  momentum=0.999,
								  name='bn_Conv1')(x)



	x = layers.Activation(HardSwish,name='Conv_1_hs')(x)

	x = _bneck_block(x, kernel = 3, filters=16, alpha=alpha, stride=1,activation = "relu",se = False,
							expansion=16, block_id=0)

	x = _bneck_block(x, kernel = 3, filters=24, alpha=alpha, stride=2,activation = "relu",se = False,
							expansion=64, block_id=1)
	x = _bneck_block(x, kernel = 3, filters=24, alpha=alpha, stride=1,activation = "relu",se = False,
							expansion=72, block_id=2)

	x = _bneck_block(x, kernel = 5, filters=40, alpha=alpha, stride=2,activation = "relu",se = True,
							expansion=72, block_id=3)
	x = _bneck_block(x, kernel = 5, filters=40, alpha=alpha, stride=1,activation = "relu",se = True,
							expansion=120, block_id=4)
	x = _bneck_block(x, kernel = 5, filters=40, alpha=alpha, stride=1,activation = "relu",se = True,
							expansion=120, block_id=5)

	x = _bneck_block(x, kernel = 3, filters=80, alpha=alpha, stride=2,activation = "hswish",se = False,
							expansion=240, block_id=6)
	x = _bneck_block(x, kernel = 3, filters=80, alpha=alpha, stride=1,activation = "hswish",se = False,
							expansion=200, block_id=7)
	x = _bneck_block(x, kernel = 3, filters=80, alpha=alpha, stride=1,activation = "hswish",se = False,
							expansion=184, block_id=8)
	x = _bneck_block(x, kernel = 3, filters=80, alpha=alpha, stride=1,activation = "hswish",se = False,
							expansion=184, block_id=9)

	x = _bneck_block(x, kernel = 3, filters=112, alpha=alpha, stride=1,activation = "hswish",se = True,
							expansion=480, block_id=10)
	x = _bneck_block(x, kernel = 3, filters=112, alpha=alpha, stride=1,activation = "hswish",se = True,
							expansion=672, block_id=11)
	x = _bneck_block(x, kernel = 5, filters=160, alpha=alpha, stride=2,activation = "hswish",se = True,
							expansion=672, block_id=12)

	x = _bneck_block(x, kernel = 5,filters=160, alpha=alpha, stride=1,activation = "hswish",se = True,
							expansion=960, block_id=13)
	x = _bneck_block(x, kernel = 5,filters=160, alpha=alpha, stride=1,activation = "hswish",se = True,
							expansion=960, block_id=14)


	if alpha > 1.0:
		filters = _make_divisible(960 * alpha, 8)
		last_block_filters = _make_divisible(1280 * alpha, 8)
	else:
		filters = 960
		last_block_filters = 1280
	x = layers.Conv2D(filters,
					kernel_size = 1,
					use_bias = False,
					name = "Conv_2")(x)

	x = layers.BatchNormalization(axis=channel_axis,
								  epsilon=1e-3,
								  momentum=0.999,
								  name='Conv_2_bn')(x)

	x = layers.Activation(HardSwish,name='Conv_2_hs')(x)

	x = layers.AveragePooling2D(pool_size = (7,7))(x)

	

	x = layers.Conv2D(last_block_filters,
					  kernel_size=1,
					  use_bias=False,
					  name='Conv_3')(x)

	x = layers.Activation(HardSwish,name='Conv_3_hs')(x)


	if include_top:
		x = layers.Conv2D(1001,
						  kernel_size=1,
						  use_bias=False,
						  name='out_conv')(x)		
		x = backend.squeeze(x, 1)
		x = backend.squeeze(x, 1)

	# Ensure that the model takes into account
	# any potential predecessors of `input_tensor`.
	if input_tensor is not None:
		inputs = keras_utils.get_source_inputs(input_tensor)
	else:
		inputs = img_input

	# Create model.
	model = models.Model(inputs, x,
						 name='mobilenetv3_%0.2f_%s' % (alpha, rows))

	# Load weights.
	if weights == 'imagenet':
		if include_top:
			model_name = ('mobilenet_v2_weights_tf_dim_ordering_tf_kernels_' +
						  str(alpha) + '_' + str(rows) + '.h5')
			weight_path = BASE_WEIGHT_PATH + model_name
			weights_path = keras_utils.get_file(
				model_name, weight_path, cache_subdir='models')
		else:
			model_name = ('mobilenet_v2_weights_tf_dim_ordering_tf_kernels_' +
						  str(alpha) + '_' + str(rows) + '_no_top' + '.h5')
			weight_path = BASE_WEIGHT_PATH + model_name
			weights_path = keras_utils.get_file(
				model_name, weight_path, cache_subdir='models')
		model.load_weights(weights_path)
	elif weights is not None:
		model.load_weights(weights)

	return model


def _bneck_block(inputs, kernel, expansion, stride, alpha, filters, se, activation, block_id):
	channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1

	in_channels = backend.int_shape(inputs)[channel_axis]
	pointwise_conv_filters = int(filters * alpha)
	pointwise_filters = _make_divisible(pointwise_conv_filters, 8)
	x = inputs
	prefix = 'block_{}_'.format(block_id)
	

	# Expand
	if block_id:
		
		x = layers.Conv2D(expansion,
						  kernel_size=1,
						  padding='same',
						  use_bias=False,
						  activation=None,
						  name=prefix + 'expand')(x)
		x = layers.BatchNormalization(axis=channel_axis,
									  epsilon=1e-3,
									  momentum=0.999,
									  name=prefix + 'expand_BN')(x)

		if activation == "relu": 
			x = layers.ReLU(6., name = prefix + 'expand_relu')(x)
		else:
			x = layers.Activation(HardSwish,name = prefix + 'expand_hs')(x)

	else:
		prefix = 'expanded_conv_'

	# Depthwise
	if stride == 2:
		x = layers.ZeroPadding2D(padding=correct_pad(backend, x, kernel),
								 name=prefix + 'pad')(x)

	x = layers.DepthwiseConv2D(kernel_size = kernel,
							   strides = stride,
							   activation=None,
							   use_bias=False,
							   padding='same' if stride == 1 else 'valid',
							   name=prefix + 'depthwise')(x)
	x = layers.BatchNormalization(axis=channel_axis,
								  epsilon=1e-3,
								  momentum=0.999,
								  name=prefix + 'depthwise_BN')(x)

	if se:
		x = squeeze_excite_block(x,prefix)
	


	if activation == "relu":
		x = layers.ReLU(6., name = prefix + 'depthwise_relu')(x)
	else:
		x = layers.Activation(HardSwish,name=prefix + 'depthwise_hs')(x)


	# Project
	x = layers.Conv2D(pointwise_filters,
					  kernel_size=1,
					  padding='same',
					  use_bias=False,
					  activation=None,
					  name=prefix + 'project')(x)
	x = layers.BatchNormalization(axis=channel_axis,
								  epsilon=1e-3,
								  momentum=0.999,
								  name=prefix + 'project_BN')(x)

	if in_channels == pointwise_filters and stride == 1:
		return layers.Add(name=prefix + 'add')([inputs, x])
	return x



def HardSwish(x,name = None):
	
	x = x * (tf.nn.relu6(x + 3) / 6)
	return x

def HardSigmoid(x,name = None):
	
	x = (tf.nn.relu6(x + 3) / 6)
	return x

def squeeze_excite_block(inputs, prefix,ratio=4):
	''' Create a channel-wise squeeze-excite block

	References
	-   [Squeeze and Excitation Networks](https://arxiv.org/abs/1709.01507)
	'''
	init = inputs
	channel_axis = 1 if backend.image_data_format() == "channels_first" else -1
	
	#filters = init._keras_shape[channel_axis]
	filters = init._shape_val[channel_axis]
	se_shape = (1, 1, filters)

	se = layers.GlobalAveragePooling2D()(init)
	se = layers.Reshape(se_shape)(se)
	se = layers.Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
	se = layers.Dense(filters, kernel_initializer='he_normal', use_bias=False)(se)
	se = layers.Activation(HardSigmoid,name=prefix + 'se_hm')(se)

	if backend.image_data_format() == 'channels_first':
		se = layers.Permute((3, 1, 2))(se)

	x = layers.multiply([init, se])
	return x