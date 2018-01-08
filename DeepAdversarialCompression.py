import tensorflow as tf

from GAN_model import GAN_model
import constants as c
from utils import get_train_batch, get_test_batch


class DAC_runner:
    def __init__(self, num_train_steps, model_load_path):

        #keep track of the training with external global step count
        #(Both the generator and the discriminator have their own tensorflow
        #step counts within the graph)
        self.global_step = 0
        self.num_train_steps = num_train_steps

        ##Initalise a tensorflow session and write for the GAN model to use
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        self.summary_writer = tf.summary.FileWriter(c.SUMMARY_SAVE_DIR, graph=self.sess.graph)

        print('Init GAN model')
        self.GAN_model = GAN_model(self.sess, self.summary_writer)

        print('Init variables')
        self.saver = tf.train.Saver(max_to_keep=100)
        self.sess.run(tf.global_variables_initializer())

        #If given a path load a saved model
        if model_load_path is not None:
            self.saver.restore(self.sess, model_load_path)
            print('Model restored from ' + model_load_path)

    def __del__(self):
        self.sess.close()

    def train_standard(self):

        for i in range(self.num_train_steps):

            #If using the adversial loss first update the discriminator
            if c.ADVERSARIAL:
                batch = get_train_batch()
                self.GAN_model.discriminator_train_step(batch)

            #Update the generator
            batch = get_train_batch()
            self.global_step =  self.GAN_model.generator_train_step(batch)

            self.GAN_model.save_summaries()
            self.GAN_model.increment_global_step()

            if self.global_step % c.MODEL_SAVE_FREQ == 0:
                print('Saving Model')
                self.saver.save(self.sess, c.MODEL_SAVE_DIR + 'model.ckpt',
                                global_step=self.global_step)
                print('Model Save Success')

            if self.global_step % c.TEST_FREQ == 0:
                print('Testing Model')
                self.test(self.global_step)
                print('Testing Compleated')

    def train_babysit(self):

        print("The generator and discriminator will be balanced during training")

        for i in range(self.num_train_steps):


            if self.global_step % c.SUMMARY_FREQ == 0:
                print("Discriminator accuracy moving average : ", self.GAN_model.discrim_acc_MA)

            #Update discriminator on condition the accuracy is not too high
            if (self.GAN_model.discrim_acc_MA < c.UPPER):
                batch = get_train_batch()
                self.global_step = self.GAN_model.discriminator_train_step(batch)

            #Update the generator on condition the accuracy isnt too low
            if (self.GAN_model.discrim_acc_MA > c.LOWER):
                batch = get_train_batch()
                self.global_step = self.GAN_model.generator_train_step(batch)

            self.GAN_model.save_summaries()
            self.GAN_model.increment_global_step()

            if self.global_step % c.MODEL_SAVE_FREQ == 0:
                print('Saving Model')
                self.saver.save(self.sess, c.MODEL_SAVE_DIR + 'model.ckpt',
                                global_step=self.global_step)
                print('Model Save Success')

            if self.global_step % c.TEST_FREQ == 0:
                print('Testing Model')
                self.test(self.global_step)
                print('Testing Compleated')


    def test(self, test_no=1):
        batch = get_test_batch(c.BATCH_SIZE_TEST)
        self.GAN_model.generator_test_batch(batch, test_no)

    def train(self):
        if c.BABYSIT & c.ADVERSARIAL:
            self.train_babysit()
        else:
            self.train_standard()

    def compress(self, frame):
        return self.GAN_model.compress( frame)



def main():
    load_path = None
    #load_path = './'+c.MODEL_SAVE_DIR + 'model.ckpt-60000'

 ## make all the fancy command line stuff here
    DeepAdversialCompression = DAC_runner(121000, load_path)

    DeepAdversialCompression.train()
    #DeepAdversialCompression.test()



if __name__ == '__main__':
    main()
