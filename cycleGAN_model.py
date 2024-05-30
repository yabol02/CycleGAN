from random import random
from numpy import load, zeros, ones, asarray
from numpy.random import randint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, LeakyReLU, Activation, Concatenate
import tensorflow_addons as tfa
import  matplotlib.pyplot as plt

# Discriminator Model (PatchGAN 70x70 like in the original paper)
def define_discriminator(image_shape):
	init = RandomNormal(stddev=0.02)
	in_image = Input(shape=image_shape)
	d = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(in_image)
	d = LeakyReLU(alpha=0.2)(d)
	d = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = tfa.layers.InstanceNormalization(axis=-1)(d)
	d = LeakyReLU(alpha=0.2)(d)
	d = Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = tfa.layers.InstanceNormalization(axis=-1)(d)
	d = LeakyReLU(alpha=0.2)(d)
	d = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = tfa.layers.InstanceNormalization(axis=-1)(d)
	d = LeakyReLU(alpha=0.2)(d)
	d = Conv2D(512, (4,4), padding='same', kernel_initializer=init)(d)
	d = tfa.layers.InstanceNormalization(axis=-1)(d)
	d = LeakyReLU(alpha=0.2)(d)
	patch_out = Conv2D(1, (4,4), padding='same', kernel_initializer=init)(d)
	model = Model(in_image, patch_out)
	model.compile(loss='mse', optimizer=Adam(lr=0.0002, beta_1=0.5), loss_weights=[0.5])
	return model

# ResNet block (to be used in the Generator)
def resnet_block(n_filters, input_layer):
	init = RandomNormal(stddev=0.02)
	x = Conv2D(n_filters, (3,3), padding='same', kernel_initializer=init)(input_layer)
	x = tfa.layers.InstanceNormalization(axis=-1)(x)
	x = Activation('relu')(x)
	x = Conv2D(n_filters, (3,3), padding='same', kernel_initializer=init)(x)
	x = tfa.layers.InstanceNormalization(axis=-1)(x)
	x = Concatenate()([x, input_layer])
	return x

# Generator model (encoder-decoder)
def define_generator(image_shape, n_resnet=9):
	init = RandomNormal(stddev=0.02)
	in_image = Input(shape=image_shape)
	g = Conv2D(64, (7,7), padding='same', kernel_initializer=init)(in_image)
	g = tfa.layers.InstanceNormalization(axis=-1)(g)
	g = Activation('relu')(g)
	g = Conv2D(128, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
	g = tfa.layers.InstanceNormalization(axis=-1)(g)
	g = Activation('relu')(g)
	g = Conv2D(256, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
	g = tfa.layers.InstanceNormalization(axis=-1)(g)
	g = Activation('relu')(g)
	for _ in range(n_resnet):
		g = resnet_block(256, g)
	g = Conv2DTranspose(128, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
	g = tfa.layers.InstanceNormalization(axis=-1)(g)
	g = Activation('relu')(g)
	g = Conv2DTranspose(64, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
	g = tfa.layers.InstanceNormalization(axis=-1)(g)
	g = Activation('relu')(g)
	g = Conv2D(3, (7,7), padding='same', kernel_initializer=init)(g)
	g = tfa.layers.InstanceNormalization(axis=-1)(g)
	out_image = Activation('tanh')(g)
	model = Model(in_image, out_image)
	return model


# Definition of a composite model for updating generators by adversarial and cycle loss. It will be used to train each generator separately. 
def define_composite_model(g_model_1, d_model, g_model_2, image_shape):
	# Make the generator of interest trainable as we will be updating these weights by keeping other models constant. This function is used to train both generators, one at a time. 
	g_model_1.trainable = True
	# Mark discriminator and second generator as non-trainable
	d_model.trainable = False
	g_model_2.trainable = False
    
	# Adversarial Loss
	input_gen = Input(shape=image_shape)
	gen1_out = g_model_1(input_gen)
	output_d = d_model(gen1_out)
	# Identity Loss
	input_id = Input(shape=image_shape)
	output_id = g_model_1(input_id)
	# Forward Cycle Loss
	output_f = g_model_2(gen1_out)
	# Backward Cycle Loss
	gen2_out = g_model_2(input_id)
	output_b = g_model_1(gen2_out)
    
	# Definition of the model graph
	model = Model([input_gen, input_id], [output_d, output_id, output_f, output_b])
	
    # Definition of the optimizer
	opt = Adam(lr=0.0002, beta_1=0.5)
	# Compilation of the model with weighting of least squares loss and L1 loss
	model.compile(loss=['mse', 'mse', 'mse', 'mse'], loss_weights=[1, 5, 10, 10], optimizer=opt)

	return model

# Load and prepare training images
def load_real_samples(filename):
	# Load of the dataset
	data = load(filename)
	# Unpacking the arrays
	X1, X2 = data['arr_0'], data['arr_1']
	# Scaling the arrays (from [0,255] to [-1,1])
	X1 = (X1 - 127.5) / 127.5
	X2 = (X2 - 127.5) / 127.5
	return [X1, X2]

# Selection of a batch of random samples. Returns images and target (real images' label is 1) 
def generate_real_samples(dataset, n_samples, patch_shape):
	# Choosing random instances
	ix = randint(0, dataset.shape[0], n_samples)
	# Retrieving selected images
	X = dataset[ix]
	# Generating 'real' class labels (1)
	y = ones((n_samples, patch_shape, patch_shape, 1))
	return X, y

# Generation of a batch of images. Returns images and targets (fake images' label is 0) 
def generate_fake_samples(g_model, dataset, patch_shape):
	# Generating fake images
	X = g_model.predict(dataset)
	# Creating 'fake' class labels (0)
	y = zeros((len(X), patch_shape, patch_shape, 1))
	return X, y

# Save the generator models to file
def save_models(step, g_model_AtoB, g_model_BtoA):
	# Saving the first generator model
	filename1 = f'g_model_AtoB_{step+1:08d}.h5'
	g_model_AtoB.save(filename1)
	# Saving the second generator model
	filename2 = f'g_model_BtoA_{step+1:08d}.h5'
	g_model_BtoA.save(filename2)
	print(f'>Saved: {filename1} and {filename2}')

# Generation of images using the save model and plot input and output images
def summarize_performance(step, g_model, trainX, name, n_samples=5):
	# Selecting a sample of input images
	X_in, _ = generate_real_samples(trainX, n_samples, 0)
	# Generating translated images
	X_out, _ = generate_fake_samples(g_model, X_in, 0)
	# scale all pixels from [-1,1] to [0,1]
	X_in = (X_in + 1) / 2.0
	X_out = (X_out + 1) / 2.0
	# Plotting real images
	for i in range(n_samples):
		plt.subplot(2, n_samples, 1 + i)
		plt.axis('off')
		plt.imshow(X_in[i])
	# Plotting translated image
	for i in range(n_samples):
		plt.subplot(2, n_samples, 1 + n_samples + i)
		plt.axis('off')
		plt.imshow(X_out[i])
	# Saving plot to file
	filename1 = f'{name}_generated_plot_{step+1:08d}.png'
	plt.savefig(filename1)
	plt.close()

# Update the image pool for fake images to reduce model oscillation and the discriminators using a history of generated images rather than the ones produced by the latest generators.
# Original paper recommended keeping an image buffer that stores the 50 previously created images.
def update_image_pool(pool, images, max_size=50):
	selected = list()
	for image in images:
		if len(pool) < max_size:
			# Stock the pool
			pool.append(image)
			selected.append(image)
		elif random() < 0.5:
			# Using the image, but don't adding it to the pool
			selected.append(image)
		else:
			# Replacing an existing image and using the replaced image
			ix = randint(0, len(pool))
			selected.append(pool[ix])
			pool[ix] = image
	return asarray(selected)

# Training of both CycleGAN models
def train(d_model_A, d_model_B, g_model_AtoB, g_model_BtoA, c_model_AtoB, c_model_BtoA, dataset, losses, epochs=1):
	# Defining properties of the training run
	n_epochs, n_batch, = epochs, 1  # Batch size  is fixed to 1 as suggested in the paper
	# Determining the output square shape of the discriminator
	n_patch = d_model_A.output_shape[1]
	# Unpacking the dataset
	trainA, trainB = dataset
	# Preparing the image pool for fake images
	poolA, poolB = list(), list()
	# Calculating the number of batches per training epoch
	batch_per_epoch = int(len(trainA) / n_batch)
	# Calculating the number of training iterations
	n_steps = batch_per_epoch * n_epochs
    
	# Enumerating the epochs manually
	for i in range(n_steps):
		# Selecting a batch of real samples from each domain (A and B)
		X_realA, y_realA = generate_real_samples(trainA, n_batch, n_patch)
		X_realB, y_realB = generate_real_samples(trainB, n_batch, n_patch)
		# Generating a batch of fake samples using both B to A and A to B generators
		X_fakeA, y_fakeA = generate_fake_samples(g_model_BtoA, X_realB, n_patch)
		X_fakeB, y_fakeB = generate_fake_samples(g_model_AtoB, X_realA, n_patch)
		# Updating fake images in the pool (the paper suggests a buffer of 50 images)
		X_fakeA = update_image_pool(poolA, X_fakeA)
		X_fakeB = update_image_pool(poolB, X_fakeB)
        
		# Updating generator B → A via the composite model
		g_loss2, _, _, _, _  = c_model_BtoA.train_on_batch([X_realB, X_realA], [y_realA, X_realA, X_realB, X_realA])
		# Updating the discriminator for A → [real/fake]
		dA_loss1 = d_model_A.train_on_batch(X_realA, y_realA)
		dA_loss2 = d_model_A.train_on_batch(X_fakeA, y_fakeA)
		
        # Updating the generator A → B via the composite model
		g_loss1, _, _, _, _ = c_model_AtoB.train_on_batch([X_realA, X_realB], [y_realB, X_realB, X_realA, X_realB])
		# Updating the discriminator for B → [real/fake]
		dB_loss1 = d_model_B.train_on_batch(X_realB, y_realB)
		dB_loss2 = d_model_B.train_on_batch(X_fakeB, y_fakeB)
		
		# Adding all losses to the dict with them
		losses['dA1'].append(dA_loss1)
		losses['dA2'].append(dA_loss2)
		losses['dB1'].append(dB_loss1)
		losses['dB2'].append(dB_loss2)
		losses['g1'].append(g_loss1)
		losses['g2'].append(g_loss2)

        # Performance summary
        # Since the batch size =1, the number of iterations should be the same as the size of the dataset. For example, having 100 images, an epoch will be 100 iterations.
		print(f'Iteration>{i+1}, dA[{dA_loss1:.3f},{dA_loss2:.3f}] dB[{dB_loss1:.3f},{dB_loss2:.3f}] g[{g_loss1:.3f},{g_loss2:.3f}]')
		# Evaluating the model performance periodically. For example, if batch size =100, performance will be summarized after every 75th iteration.
		if (i+1) % (batch_per_epoch * 1) == 0:
			# Plotting  A → B translation
			summarize_performance(i, g_model_AtoB, trainA, 'AtoB')
			# Plotting B → A translation
			summarize_performance(i, g_model_BtoA, trainB, 'BtoA')
		if (i+1) % (batch_per_epoch * 5) == 0:
			# Save the models. For example, if  the batch size =100, model will be saved after every 75th iteration x 5 = 375 iterations.
			save_models(i, g_model_AtoB, g_model_BtoA)
	return losses

