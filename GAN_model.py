import tensorflow as tf
import numpy as np
from generator_model import GeneratorModel
from discriminator_model import DiscriminatorModel
from GaussianScaleMixtures_model import GSM_model

from copy import deepcopy

import constants as c
from scipy.misc import imsave
from utils import  denormalize_frames_mine, normalize_frames_mine
import os
from glob import glob
import cabac
import performance_cal as pc
from timeit import default_timer as timer



class GAN_model:
    def __init__(self, session, summary_writer):

        self.sess = session
        self.summary_writer = summary_writer

        self.discrim_acc_MA = (c.LOWER + c.UPPER)/2
        self.define_graph()



    def define_graph(self):

        print(c.DIM_REDUCTION, " times dimensional reduction")
        self.Generator = GeneratorModel()

        if c.ADVERSARIAL:
            self.Discriminator = DiscriminatorModel()
        self.Entropy_model = GSM_model(c.NUMBER_OF_MIXTURE)



        #Define the data for the model
        with tf.name_scope('data'):
            self.input_image = tf.placeholder( tf.float32, shape=[c.BATCH_SIZE, c.TRAIN_HEIGHT, c.TRAIN_WIDTH, 3], name="input_image")
            self.input_image_test = tf.placeholder( tf.float32, shape=[c.BATCH_SIZE_TEST, c.FULL_HEIGHT, c.FULL_WIDTH, 3], name="input_image_test")

            self.batch_size = tf.shape(self.input_image)[0]
            self.is_train = tf.placeholder( tf.bool)

        #Calculate the predicted compressed output of model for training time
        self.prediction, self.code, self.pre_quant_code = self.Generator(self.input_image, self.is_train)

        #Same as above but for the test time input
        self.prediction_test, self.code_test, self.pre_quant_code_test = self.Generator(self.input_image_test, self.is_train, reuse=True)

        #If using an adversial loss the discriminator must be used on both the generated
        #and real data
        if c.ADVERSARIAL:
            self.d_real = self.Discriminator(self.input_image, self.is_train)
            self.d_fake = self.Discriminator(self.prediction, self.is_train, reuse=True)

        ##
        #   Calculate the loss and the optimisators
        ##

        with tf.name_scope('train'):
            ##
            #lp reconstruction loss
            ##
            self.loss_recon = tf.reduce_mean(tf.reduce_sum(tf.abs(self.input_image - self.prediction)**c.L_NUM,[1,2,3]))

            ##
            #Entropy Loss
            ##
            if c.QUANT_MODEL:
                additive_noise = tf.random_uniform(shape=tf.shape(self.code), minval = -0.5, maxval = 0.5, dtype=tf.float32, name='Additive_noise')
                self.loss_entropy = self.Entropy_model.negative_log_likelihood(self.code + additive_noise)
            else:
                self.loss_entropy = self.Entropy_model.negative_log_likelihood(self.code)
            #Draw samples from the GSM model (usful for tensorboard visualisation)
            self.GSM_sample = self.Entropy_model.sample_data(c.GSM_SAMPLE)


            ##
            #Adversial loss functions for gen and dis
            ##
            if c.ADVERSARIAL:
                dfake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_fake, labels=tf.zeros_like(self.d_fake,dtype=tf.float32)))
                dreal_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_real, labels=tf.scalar_mul(0.9,tf.ones_like(self.d_real,dtype=tf.float32))))
                gfake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_fake, labels=tf.ones_like(self.d_fake,dtype=tf.float32)))
                self.loss_adv_G = gfake_loss
                self.loss_adv_D = dfake_loss + dreal_loss
                #Addtional calculation to see discriminator accuracy
                fake_right = tf.cast(tf.nn.sigmoid(self.d_fake)<0.5, tf.float32)
                real_right = tf.cast(tf.nn.sigmoid(self.d_real)>=0.5, tf.float32)
                self.precision = tf.reduce_mean(tf.concat([fake_right, real_right],0))

                #Add the GAN generator loss to the reconstruction loss
                #self.loss_recon = c.LAM_LP*self.loss_recon
                #self.loss_recon += c.LAM_ADV*self.loss_adv_G


            #scale the losses to make more interpretable (according to .....)
            loss_recon_scaled = c.SCALE_RECON*self.loss_recon
            self.generator_loss = loss_recon_scaled
            if c.ADVERSARIAL:
                self.generator_loss += c.LAM_ADV*self.loss_adv_G
            self.loss_entropy_scaled = c.SCALE_ENTROPY*self.loss_entropy

            #Combine the recon/gan loss with the entropy loss for generator
            self.generator_loss  = (1.0-c.ALPHA)*self.generator_loss + c.ALPHA*self.loss_entropy_scaled
            #Build Discriminator loss if required
            if c.ADVERSARIAL:
                self.discriminator_loss = self.loss_adv_D  #May need to scale this

            if c.ADVERSARIAL:
            #Calcualate and output the gradient of the loss function with respect to recon and adversial component
                self.grad_adv = tf.gradients(self.loss_adv_G, self.prediction)
                self.grad_recon = tf.gradients(self.loss_recon, self.prediction)
                self.grad_entropy = tf.gradients(self.loss_entropy , self.code)
                tf.summary.histogram('Gradient_adverisal', self.grad_adv)
                tf.summary.histogram('Gradient_reconstruction', self.grad_recon)
                tf.summary.histogram('Gradient_entropy', self.grad_entropy)


            #Add the losses to summaries
            tf.summary.scalar('Loss_reconstruction_scaled', loss_recon_scaled)
            tf.summary.scalar('Loss_entropy_scaled', self.loss_entropy_scaled)
            tf.summary.scalar('Loss_total', self.generator_loss)
            if c.ADVERSARIAL:
                tf.summary.scalar('Loss_adv_discriminator_unscaled', self.discriminator_loss)
                tf.summary.scalar('Loss_adv_generator_unscaled', self.loss_adv_G)
                tf.summary.scalar('Discriminator_precision', self.precision)

            tf.summary.histogram('Code', self.code)
            tf.summary.histogram('GSM_model_sample', self.GSM_sample)
            tf.summary.histogram('Pre_Quantisation_model_code', self.pre_quant_code)

            #Set up the global variable and the increment function
            self.global_step = tf.Variable(0, trainable=False, name='global_step')
            self.increment_global_step_op = tf.assign(self.global_step, self.global_step+1)

            #Get the varaibles in the generator and discriminator
            self.g_variables = []
            self.d_variables = []
            #self.gsm_variables = []
            for x in tf.trainable_variables():
                if x.op.name.startswith('GEN'):
                    self.g_variables.append(x)
                elif x.op.name.startswith('DIS'):
                    self.d_variables.append(x)
                elif x.op.name.startswith('GSM'):
                    self.g_variables.append(x)

            #Clip the gradients in to stabilise the training and create optimizers
            self.g_grads,_ = tf.clip_by_global_norm(tf.gradients(self.generator_loss, self.g_variables), 10)
            self.optimizer_g = tf.train.AdamOptimizer(c.LRATE_G,beta1=0.5)

            if c.ADVERSARIAL:
                self.d_grads,_ = tf.clip_by_global_norm(tf.gradients(self.discriminator_loss, self.d_variables),10)
                self.optimizer_d = tf.train.GradientDescentOptimizer(c.LRATE_D)

            with tf.device('/gpu:0'):
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                #Above two lines ensure that the update op are executed before performing training step
                #required to upate the moving average in batchnorm
                    self.train_op_G = self.optimizer_g.apply_gradients(zip(self.g_grads, self.g_variables))
                    if c.ADVERSARIAL:
                        self.train_op_D = self.optimizer_d.apply_gradients(zip(self.d_grads, self.d_variables))


        #with tf.name_scope('performance measures'):



        ##Merge all the summeries for both generator and discriminator for tensorboard
        self.summaries = tf.summary.merge_all()


    def generator_train_step(self, batch):

        feed_dict = {self.input_image : batch, self.is_train : True}


        if c.ADVERSARIAL:
            _, prediction, generator_loss, loss_adv_g, loss_recon, loss_entropy, precision, summaries  = \
                self.sess.run([self.train_op_G,
                            self.prediction,
                            self.generator_loss,
                            self.loss_adv_G,
                            self.loss_recon,
                            self.loss_entropy,
                            self.precision,
                            self.summaries],
                            feed_dict=feed_dict)

            self.update_discrim_accur_MA(precision)
        else:
            _, prediction, generator_loss, loss_recon, loss_entropy, summaries = \
                self.sess.run([self.train_op_G,
                            self.prediction,
                            self.generator_loss,
                            self.loss_recon,
                            self.loss_entropy,
                            self.summaries],
                            feed_dict=feed_dict)

        global_step = self.get_global_step()

        self.most_recent_summary = summaries


        if global_step % c.STATS_FREQ == 0:
            print('GeneratorModel : Step ', global_step)
            print('                 Generator Loss    : ', generator_loss)
            print('                                     ')
            #Need to add more user output for train time
        if global_step % c.IMG_SAVE_FREQ == 0:
            print('Saving images ..')
            for image in range(len(batch)):
                pred_dir = c.get_dir(os.path.join(c.IMG_SAVE_DIR, 'Step_' +str(global_step), str(image)))

                #save the input image
                img = batch[image, :, :, :]
                img = denormalize_frames_mine(img)
                imsave(os.path.join(pred_dir, 'input.png'), img)

                #save the compressed image
                compressed_image = denormalize_frames_mine(prediction[image,:,:,:])
                imsave(os.path.join(pred_dir, 'compressed.png'), compressed_image)
            print('Saved images!')

        return global_step

    def discriminator_train_step(self, batch):

        feed_dict = {self.input_image : batch, self.is_train : True}


        _, discriminator_loss, precision, summaries = \
        self.sess.run([self.train_op_D,
                        self.discriminator_loss,
                        self.precision,
                        self.summaries],
                        feed_dict=feed_dict)

        #Update the moving average for the disciminator accuracy
        self.update_discrim_accur_MA(precision)

        self.most_recent_summary = summaries

        global_step = self.get_global_step()

        if global_step % c.STATS_FREQ == 0:
            print('DiscriminatorModel : Step ', global_step)
            print('                     Discriminator Loss : ', discriminator_loss)
            print('                     Discriminator precision :', precision)
            print('                                              ')
        return global_step

    def generator_test_batch(self, batch, global_step, save_imgs=True):
        print('Compressing test batch')

        feed_dict = {self.input_image_test : batch, self.is_train : False}

        prediction, code = self.sess.run([self.prediction_test, self.code_test], feed_dict=feed_dict)

        ##
        # Everything else in the function is currently rough and needs
        # finalising
        ##

        start = timer()
        time_test_code = self.sess.run(self.code_test, feed_dict=feed_dict)
        end = timer()
        print("Time taken for compression ", end - start)

        ##Reshape the code / quantised coefficents via rasta scan ordering into
        #A continuous integer stream to be encoded by the entropy encoder

        print("Size of coffiecnt tensor")
        print(code.shape)

        code = code.astype(np.int32)
        code = np.reshape(code, (-1, code.shape[1]*code.shape[2]*code.shape[3]))

        print("Size of reshaped coefficent array")
        print(code.shape)

        num_bits_total = 0
        for i in range(code.shape[0]):
            num_bits = cabac.encode(code[i,:])
            print("Bytes for image ", i, " : ", num_bits/8)
            print("Bits per pixel image ", i, " : ", num_bits/(c.FULL_HEIGHT*c.FULL_WIDTH*3))
            print("Effective entropy ", i, " : ", num_bits/code.shape[1])
            num_bits_total += num_bits

        av_bits_per_pixel = num_bits_total/(code.shape[0]*c.FULL_HEIGHT*c.FULL_WIDTH*3)


        entropy_total = 0
        for i in range(code.shape[0]):
            entropy = cabac.estimate_entropy(code[i,:])
            print("Entropy image " , i  ," : ", entropy)
            entropy_total += entropy
        entropy = entropy_total/code.shape[0]
        print("Average entropy")
        print(entropy)


        print("Average number of bpp")
        print(av_bits_per_pixel)

        print("PSNR calcuation")
        compressed_images = deepcopy(prediction)
        input_images = deepcopy(batch)
        compressed_images = denormalize_frames_mine(compressed_images)
        input_images = denormalize_frames_mine(input_images)
        print(pc.PSNR_error_np(compressed_images, input_images))


        if save_imgs:
            for image in range(len(batch)):
                pred_dir = c.get_dir(os.path.join( c.IMG_SAVE_DIR, 'Tests/Step_' + str(global_step), str(image)))

                #Save the input image
                img = batch[image,:,:,:]
                img = denormalize_frames_mine(img)
                imsave(os.path.join(pred_dir, 'input.png'), img)

                #Save the compressed img
                compressed_image = denormalize_frames_mine(prediction[image,:,:,:])
                imsave(os.path.join(pred_dir, 'compressed.png'), compressed_image)
        print('Saved test batch')

    def update_discrim_accur_MA(self, current_accur):
        #Use an expotential moving average approach to update the discriminator
        #Accuracy, this moving average is used to balance the training procedure
        self.discrim_acc_MA = c.SMOOTHING*current_accur + (1.0-c.SMOOTHING)*self.discrim_acc_MA

    def get_global_step(self):
        return self.sess.run(self.global_step)

    def increment_global_step(self):
        return self.sess.run(self.increment_global_step_op)

    def save_summaries(self):


        global_step = self.get_global_step()
        if global_step % c.SUMMARY_FREQ == 0:
            self.summary_writer.add_summary(self.most_recent_summary, global_step)
            print('Summary Saved!!')



    def compress(self, frame):
        frame_in = frame.reshape((1,c.FULL_HEIGHT, c.FULL_WIDTH, 3))
        frame_in = normalize_frames_mine(frame_in)
        feed_dict = {self.input_image_test : frame_in, self.is_train : False }
        compressed_output, code = self.sess.run([self.prediction_test, self.code_test], feed_dict=feed_dict)
        output =  denormalize_frames_mine(compressed_output)
        print('BPP: ' , (code.shape[0]*code.shape[1]*code.shape[2]*code.shape[3])/(c.FULL_HEIGHT * c.FULL_WIDTH * 3))
        return output.reshape((c.FULL_HEIGHT, c.FULL_WIDTH, 3))


