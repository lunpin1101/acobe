#!/usr/bin/python3.7

import argparse, json, gc, matplotlib, numpy, os, random, sys
import tensorflow, tensorly, sklearn.linear_model
import tensorflow.keras as keras
tensorflow.config.experimental.set_visible_devices (devices=[], device_type='GPU')
matplotlib.use ('Agg')
import matplotlib.pyplot
import scipy.cluster.hierarchy
import scipy.spatial.distance

def main ():
    parser = argparse.ArgumentParser (description='Find anomalies from deviation matrices')
    # inputs
    parser.add_argument ('--input-dir', help='Specify input directories', nargs='+')
    parser.add_argument ('--input-list', help='Specify lists of deviations', nargs='+')
    parser.add_argument ('--input-user', help='Specify a list of users', nargs='+')
    parser.add_argument ('--input-date', help='Specify a list of dates', nargs='+')
    # features
    parser.add_argument ('--features', help='Specify features to work on with', nargs='+')
    parser.add_argument ('--no-level-zero', help='Remove benign-level features (e.g., alexa-0', action='store_true')
    # plot heatmap
    parser.add_argument ('--image-dir', help='Specify output image directory')
    parser.add_argument ('--image-prefix', help='Specify output image filename prefix')
    parser.add_argument ('--image-sigma', help='Specify an upper bound of deviation', type=float)
    parser.add_argument ('--image-window', help='Specify a window-size for images', type=int)
    parser.add_argument ('--image-negweight', help='Specify a weight for negative pixels', type=float)
    # mode
    parser.add_argument ('--verbose', help='Verbose mode', action='store_true')
    parser.add_argument ('--test-module', help='Test Model and then exit', action='store_true')
    parser.add_argument ('--model', help='Specify a model for anomaly detection')
    parser.add_argument ('--top', help='Only output top N anomalies', type=int)
    parser.add_argument ('--synthesis', help='Synthesize images and plot anomalous pixels', action='store_true')
    parser.add_argument ('--train-on-date', help='Specify dates for training set', nargs='+')
    parser.add_argument ('--test-on-date', help='Specify dates for testing set', nargs='+')
    parser.add_argument ('--use-gpu', help='Use GPU (default off)', action='store_true')
    # outputs
    parser.add_argument ('-o', '--output', help='Write results to file')
    parser.add_argument ('--output-dir', help='Arrange results in directory')
    args = parser.parse_args ()

    # configuration
    args.input_dir = [] if args.input_dir is None else args.input_dir
    args.input_list = [] if args.input_list is None else args.input_list
    args.features = ['all'] if args.features is None else args.features
    args.image_prefix = '' if args.image_prefix is None else args.image_prefix
    args.image_sigma = 3.0 if args.image_sigma is None else args.image_sigma
    args.image_window = 15 if args.image_window is None or args.image_window < 1 else args.image_window
    args.image_negweight = 1.0 if args.image_negweight is None or args.image_negweight < 0.0 else args.image_negweight
    if args.use_gpu: 
        gpus = tensorflow.config.experimental.list_physical_devices ('GPU')
        tensorflow.config.experimental.set_visible_devices (devices=gpus, device_type='GPU')
        for i in range (0, len (gpus)): tensorflow.config.experimental.set_memory_growth (gpus [i], True)

    if args.test_module:
        if args.model == 'autoencoder1': Autoencoder1.testModule (args)
        if args.model == 'autoencoder2': Autoencoder2.testModule (args)
        if args.model == 'autoencoderR1': AutoencoderR.testModule (args, Autoencoder1)
        if args.model == 'autoencoderR2': AutoencoderR.testModule (args, Autoencoder2)
        if args.model == 'anogan': AnoGAN.testModule (args)
        if args.model == 'decomposition': TensorDecomposition.testModule (args)
        exit (0)

    # read inputs
    users = UserManager (args)
    for l in args.input_list:
        for f in open (l, 'r'):
            users.append (Deviation (f.strip (), args))
    for _d in args.input_dir:
        for user in os.listdir (_d):
            if args.input_user is None or user in args.input_user:
                userdir = os.path.join (_d, user)
                for date in os.listdir (userdir):
                    if args.input_date is None or date in args.input_date:
                        datedir = os.path.join (userdir, date)
                        users.append (Deviation (datedir, args))
   
    # get image for common behaviors (average)
    if len (users) < 1: exit (0)
    mean = users.mean () if len (users) > 1 else None
    dates = mean.getDateRange () if len (users) > 1 else users.sample ().getDateRange ()

    # apply anomaly detection
    if args.model is not None:
        evaluation = [] 
        # build a model for each individual user
        for u in users:
            # build training set 
            trainX = []
            for index in range (0, len (dates) - args.image_window + 1):
                dateLabels = dates [index: index + args.image_window]
                if args.train_on_date is not None and dateLabels [-1] not in args.train_on_date: continue
                meanmap = mean.heatmap (dates=dateLabels)
                heatmap = users [u].heatmap (dateLabels)
                if heatmap is not None:
                    trainX.append (Figure.Heatmap ([heatmap, meanmap], args=args, tags=[u, dateLabels [-1]]))
            # build testing set 
            testX = []
            for index in range (0, len (dates) - args.image_window + 1):
                dateLabels = dates [index: index + args.image_window]
                if args.test_on_date is not None and dateLabels [-1] not in args.test_on_date: continue
                meanmap = mean.heatmap (dates=dateLabels)
                heatmap = users [u].heatmap (dateLabels)
                if heatmap is not None:
                    testX.append (Figure.Heatmap ([heatmap, meanmap], args=args, tags=[u, dateLabels [-1]]))
            # get dimensions
            if len (trainX) == 0 or len (testX) == 0: exit (0)
            width = args.image_window * 2
            height = trainX [0].size / width
            # autoencoder1
            if args.model == 'autoencoder1':
                model = Autoencoder1 (height=height, width=width, args=args)
                model.train (X=list (trainX), epochs=64, batchsize=2048)
                scores, syns = model.test (X=list (testX))
            # autoencoder2
            if args.model == 'autoencoder2':
                model = Autoencoder2 (height=height, width=width, args=args)
                model.train (X=list (trainX), epochs=16, batchsize=32)
                scores, syns = model.test (X=list (testX))
            # autoencoderR1
            if args.model == 'autoencoderR1':
                model = AutoencoderR (height=height, width=width, args=args, model=Autoencoder1)
                model.train (X=list (trainX), epochs=4, batchsize=32)
                scores, syns = model.test (X=list (testX))
            # autoencoderR2
            if args.model == 'autoencoderR2':
                model = AutoencoderR (height=height, width=width, args=args, model=Autoencoder2)
                model.train (X=list (trainX), epochs=4, batchsize=32)
                scores, syns = model.test (X=list (testX))
            # anogan
            if args.model == 'anogan':
                model = AnoGAN (height=height, width=width, args=args)
                model.train (X=list (trainX), epochs=4, batchsize=32)
                scores, syns = model.test (X=list (testX))
            # decomposition
            if args.model == 'decomposition':
                model = TensorDecomposition (args=args)
                model.train (X=list (trainX), epochs=2048)
                scores, syns = model.test (X=list (testX))
            # delete model
            if model is not None: 
                keras.backend.clear_session ()
                del (model); gc.collect ()
            # attach results to evaluation
            for index, score in enumerate (list (scores)):
                fig = testX [index]
                user = fig.tags () [0]
                date = fig.tags () [1]
                evaluation.append ([user, score, date])
        # evaluate
        evaluation = sorted (evaluation, key=lambda obj: obj [1], reverse=True)
        # output evaluation
        if args.output is None and args.output_dir is None: print (json.dumps (evaluation))
        if args.output is not None:
            with open (args.output, 'w') as fout: fout.write (json.dumps (evaluation) + '\n')
        if args.output_dir is not None:
            filename = 'result.log'
            for date in args.test_on_date:
                datedir = os.path.join (args.output_dir, date)
                logfile = os.path.join (datedir, filename)
                os.makedirs (os.path.join (args.output_dir, date), exist_ok=True)
                if os.path.isfile (logfile): os.remove (logfile)
            for user, score, date in evaluation:
                with open (os.path.join (args.output_dir, date, filename), 'a') as fout:
                    fout.write (json.dumps ([user, score, date, args.model]) + '\n') 
        # plot summary
        if args.image_dir is not None: # plot summary
            title = '_'.join ([args.image_prefix, args.model, 'summary', dates [0], dates [-1]])
            figure = matplotlib.pyplot.figure (1)
            matplotlib.pyplot.yscale ('linear')
            matplotlib.pyplot.grid (True)
            matplotlib.pyplot.plot (numpy.arange (len (evaluation)), [tup [1] for tup in evaluation], '-', lw=2)
            figure.suptitle (title)
            figure.tight_layout ()
            matplotlib.pyplot.savefig (os.path.join (args.image_dir, title + '.png'), dpi=600)
            matplotlib.pyplot.close (figure)

    # plot users
    if args.image_dir:
        order = list (sorted (users.keys ()))
        # set risks
        if args.model is not None and len (evaluation) > 1:
            order = []
            for user, score, date in evaluation:
                if user not in order: order.append (user)
            for i, u in enumerate (order):
                users [u].setRisk (i)
        # plot
        for i, u in enumerate (order): 
            if args.top is None or i < args.top: 
                users [u].heatmap (plot=mean)



################################################
###           Tensor Decomposition           ###
################################################

class TensorDecomposition (object):
    def __init__ (self, args=None):
        gc.collect()
        self.args = args 
        self.k = 1
        self.lambda_u = 0.1
        self.lambda_v = 0.1
        self.lambda_w = 0.1
        self.X = None
    def standard_parafac (self, X, epochs=2048):
        A = numpy.ones ((X.shape [0], self.k))
        B = numpy.ones ((X.shape [1], self.k))
        C = numpy.ones ((X.shape [2], self.k))
        clf = sklearn.linear_model.ridge.Ridge (fit_intercept=False, alpha=0.1)
        for epoch in range (0, epochs):
            # estimate A
            mttrpA = tensorly.tenalg.khatri_rao ([B, C])
            destA = tensorly.transpose (tensorly.base.unfold (X, 0))
            clf.fit (mttrpA, destA)
            A = clf.coef_
            # estimate B
            mttrpB = tensorly.tenalg.khatri_rao ([A, C])
            destB = tensorly.transpose (tensorly.base.unfold (X, 1))
            clf.fit (mttrpB, destB)
            B = clf.coef_
            # estimate C
            mttrpC = tensorly.tenalg.khatri_rao ([A, B])
            destC = tensorly.transpose (tensorly.base.unfold (X, 2))
            clf.fit (mttrpC, destC)
            C = clf.coef_
            if self.args.verbose: 
                Y = tensorly.kruskal_tensor.kruskal_to_tensor ([A, B, C])
                print ('standard_parafac-' + str (epochs - epoch - 1) + ': ' + str (numpy.mean (self.compare (X, Y))))
        return [A, B, C]
    def regularized_parafac (self, X, constraints, epochs=2048):
        cA, cB, cC = constraints
        A = numpy.ones ((X.shape [0], self.k))
        B = numpy.ones ((X.shape [1], self.k))
        C = numpy.ones ((X.shape [2], self.k))
        for epoch in range (0, epochs):
            # estimate A
            mttrpA = numpy.dot (tensorly.base.unfold (X, 0), tensorly.tenalg.khatri_rao ([B, C]))
            destA = numpy.multiply (numpy.dot (B.T, B), numpy.dot (C.T, C))
            A = tensorly.transpose (tensorly.solve (
                    tensorly.transpose (numpy.dot (destA, destA.T) + self.lambda_u * numpy.eye (self.k)),
                    tensorly.transpose (self.lambda_u * cA + numpy.dot (mttrpA, destA.T))))
            # estimate B
            mttrpB = numpy.dot (tensorly.base.unfold (X, 1), tensorly.tenalg.khatri_rao ([A, C]))
            destB = numpy.multiply (numpy.dot (A.T, A), numpy.dot (C.T, C))
            B = tensorly.transpose (tensorly.solve (
                    tensorly.transpose (numpy.dot (destB, destB.T) + self.lambda_v * numpy.eye (self.k)),
                    tensorly.transpose (self.lambda_v * cB + numpy.dot (mttrpB, destB.T))))
            # estimate C: B*(CA)=X2, C*(BA)=X3
            mttrpC = numpy.dot (tensorly.base.unfold (X, 2), tensorly.tenalg.khatri_rao ([A, B]))
            destC = numpy.multiply (numpy.dot (A.T, A), numpy.dot (B.T, B))
            C = tensorly.transpose (tensorly.solve (
                    tensorly.transpose (numpy.dot (destC, destC.T) + self.lambda_w * numpy.eye (self.k)),
                    tensorly.transpose (self.lambda_w * cC + numpy.dot (mttrpC, destC.T))))
            if self.args.verbose: 
                Y = tensorly.kruskal_tensor.kruskal_to_tensor ([A, B, C])
                print ('regularized_parafac-' + str (epochs - epoch - 1) + ': ' + str (numpy.mean (self.compare (X, Y))))
        return [A, B, C]
    def fit (self, X, epochs=2048):
        self.X = X
        self.epochs = epochs
    def evaluate (self, X, epochs=None, synthesis=False):
        scores, syns = [], []
        numCubes = len (self.args.test_on_date)
        lenCubes = int (len (X) / numCubes)
        epochs = self.epochs if epochs is None else epochs
        # decomposition of the last training cube
        X0 = self.X if self.X is not None else X [: lenCubes]
        X0 = self.checkArray (X0 [0 - lenCubes: ])
        F0 = self.standard_parafac (X0, epochs)
        Y0 = tensorly.kruskal_tensor.kruskal_to_tensor (F0)
        # for each testing cube
        for cubeIndex in range (0, numCubes):
            # reconstruction
            X1 = self.checkArray (X [cubeIndex * lenCubes: (cubeIndex + 1) * lenCubes])
            F1 = self.regularized_parafac (X1, F0, epochs)
            Y1 = tensorly.kruskal_tensor.kruskal_to_tensor (F1)
            # scoring
            scores1 = self.compare (Y0, Y1)
            scores2 = self.compare (Y0, X1)
            scores3 = self.compare (X1, Y1)
            scores += list (numpy.max ([scores1, scores2, scores3], 0))
            # synthesis
            if synthesis: syns += list (numpy.reshape (numpy.array (Y1), X1.shape))
        return scores, syns
    def train (self, X, epochs=2048):
        return self.fit (X, epochs)
    def test (self, X, epochs=None, synthesis=False):
        return self.evaluate (X, epochs, synthesis)
    @staticmethod
    def checkArray (array):
        array = numpy.array (array)
        return array
    @staticmethod
    def compare (X, Y):
        ret = []
        for d in range (0, X.shape [0]):
            ret.append (numpy.sqrt (numpy.sum (numpy.power (X [d] - Y [d], 2))))
        return ret
    @classmethod
    def testModule (cls, args=None):
        train = [Figure.getNormalFigure (12, 14) for i in range (0, 1024*4)]
        test1 = [Figure.getNormalFigure (12, 14) for i in range (0, 8)]
        test2 = [Figure.getAbnormalFigure (12, 14) for i in range (0, 8)]
        decomposition = cls (args=args)
        decomposition.fit (train)
        scores1, syns1 = decomposition.evaluate (test1, synthesis=True, epochs=128)
        scores2, syns2 = decomposition.evaluate (test2, synthesis=True)
        print ('Normal: ' + str (numpy.mean (scores1)))
        print ('Abnormal: ' + str (numpy.mean (scores2)))
        print (syns1 [0])



################################################
###                  AnoGAN                  ###
################################################

class AnoGAN (object):
    def __init__ (self, height, width, args=None, loss='mean_squared_error'):
        gc.collect()
        # variables
        self.args = args
        self.width = (int ((width - 1) / 4) + 1) * 4
        self.height = (int ((height - 1) / 4) + 1) * 4
        self.col = int (self.width / 4)
        self.row = int (self.height / 4)
        self.channels = 1
        self.zdim = 100
        self.units = 128
        self.kernels = (5, 5)
        self.strides = (2, 2)
        self.stddev = 0.02
        self.relu = 1.0
        self.dropout = 0.3
        self.activation = 'tanh'
        self.loss = loss
        # generator
        self.generator = keras.models.Sequential ()
        self.generator.add (keras.layers.Dense (self.row * self.col * self.units, input_dim=self.zdim, 
                            kernel_initializer=keras.initializers.RandomNormal (stddev=self.stddev)))
        self.generator.add (keras.layers.LeakyReLU (self.relu))
        self.generator.add (keras.layers.BatchNormalization ())
        self.generator.add (keras.layers.Reshape ((self.row, self.col, self.units)))
        self.generator.add (keras.layers.UpSampling2D (size=self.strides))
        self.generator.add (keras.layers.Conv2D (int (self.units / 2), kernel_size=self.kernels, padding='same'))
        self.generator.add (keras.layers.LeakyReLU (self.relu))
        self.generator.add (keras.layers.BatchNormalization ())
        self.generator.add (keras.layers.UpSampling2D (size=self.strides))
        self.generator.add (keras.layers.Conv2D (self.channels, kernel_size=self.kernels, padding='same', activation='tanh'))
        self.generator.compile (loss=self.loss, optimizer='adam')
        if self.args.verbose: print ('\nGenerator: '); self.generator.summary ()
        # discriminator
        self.discriminator = keras.models.Sequential ()
        self.discriminator.add (keras.layers.Conv2D (int (self.units / 2), kernel_size=self.kernels, strides=self.strides, padding='same',
                                input_shape=(self.height, self.width, self.channels), kernel_initializer=keras.initializers.RandomNormal (stddev=self.stddev)))
        self.discriminator.add (keras.layers.LeakyReLU (self.relu))
        self.discriminator.add (keras.layers.BatchNormalization ())
        self.discriminator.add (keras.layers.Dropout (self.dropout))
        self.discriminator.add (keras.layers.Conv2D (self.units, kernel_size=self.kernels, strides=self.strides, padding='same'))  # layer [-5]
        self.discriminator.add (keras.layers.LeakyReLU (self.relu))
        self.discriminator.add (keras.layers.BatchNormalization ())
        self.discriminator.add (keras.layers.Dropout (self.dropout))
        self.discriminator.add (keras.layers.Flatten ())
        self.discriminator.add (keras.layers.Dense (1, activation=self.activation))
        self.discriminator.compile (loss=self.loss, optimizer='adam')
        if self.args.verbose: print ('\nDiscriminator: '); self.discriminator.summary ()
        # gan
        self.ganInput = keras.layers.Input (shape=(self.zdim, ))
        self.ganMidput = self.generator (self.ganInput)
        self.ganOutput = self.discriminator (self.ganMidput)
        self.GAN = keras.models.Model (inputs=self.ganInput, outputs=self.ganOutput)
        self.GAN.compile (loss=self.loss, optimizer='adam')
        if self.args.verbose: print ('\nGAN: '); self.GAN.summary ()
        # intermediate feature extractor
        self.intermediate = keras.models.Model (inputs=self.discriminator.layers [0].input, outputs=self.discriminator.layers [-5].output)
        # self.intermediate.compile (loss=self.loss, optimizer='adam')
        self.intermediate.compile (loss='binary_crossentropy', optimizer='adam')
        if self.args.verbose: print ('\nFeature Extractor: '); self.intermediate.summary ()
        # AnoGAN detector
        self.anoInput = keras.layers.Input (shape=(self.zdim, ))
        self.anoMidput = self.generator (keras.layers.Activation (self.activation) (keras.layers.Dense ((self.zdim)) (self.anoInput)))
        self.anoOutput = self.intermediate (self.anoMidput)
        self.detector = keras.models.Model (inputs=self.anoInput, outputs=[self.anoMidput, self.anoOutput])
        self.detector.compile (loss=self.sum_of_residual, loss_weights=[0.9, 0.1], optimizer='adam')
        if self.args.verbose: print ('\nAnoGAN Detector: '); self.detector.summary ()
    def fit (self, X, epochs=4, batchsize=512):
        X = self.checkArray (X)
        for epoch in range (0, epochs):
            # shuffle
            indices = [i for i in range (0, X.shape [0])]
            random.shuffle (indices)
            # train
            while len (indices) > 0:
                # prepare batches
                batch_images = [X [indices.pop ()] for i in range (0, min (batchsize, len (indices)))]
                generated_images = self.generator.predict (numpy.random.uniform (0, 1, (len (batch_images), self.zdim)), verbose=0)
                # train discriminator
                batch_X = numpy.concatenate ((batch_images, generated_images))
                batch_Y = numpy.array ([1] * len (batch_images) + [0] * len (generated_images))
                discriminator_loss = self.discriminator.train_on_batch (batch_X, batch_Y)
                # train generator
                self.discriminator.trainable = False
                batch_X = numpy.random.uniform (0, 1, (len (batch_images), self.zdim))
                batch_Y = numpy.array ([1] * len (batch_images))
                generator_loss = self.GAN.train_on_batch (batch_X, batch_Y)
                self.discriminator.trainable = True
                # verbose
                if self.args.verbose: 
                    print ('epoch ' + str (epochs - epoch - 1) + ' index ' + str (len (indices)), end='    ')
                    print ('discriminator_loss=' + str (discriminator_loss), end='    ')
                    print ('generator_loss=' + str (generator_loss))
    def evaluate (self, X, epochs=32, synthesis=False):
        scores = []
        syns = []
        for i, x in enumerate (X):
            x = self.checkArray ([x])
            z = numpy.random.uniform (0, 1, (1, self.zdim))
            dx = self.intermediate.predict (x)
            # learning for changing latent
            loss = self.detector.fit (z, [x, dx], epochs=epochs, verbose=0)
            # synthesis image
            if synthesis:
                syn, _ = self.detector.predict (z)
                syns.append (syn [0])
            # scoring
            score = loss.history ['loss'][-1]
            scores.append (score)
            if self.args.verbose: print ('Test-' + str (len (X) - i - 1) + '=' + str (score))
        shape = numpy.array (X).shape                  
        if synthesis: syns = numpy.reshape (numpy.array (syns)[:, :shape [1], :shape [2]], shape)
        return scores, syns
    def train (self, X, epochs=4, batchsize=512):
        return self.fit (X, epochs, batchsize)
    def test (self, X, epochs=32, synthesis=False):
        return self.evaluate (X, epochs, synthesis)
    def checkArray (self, array):
        # pad to fit dimensions
        for i in range (0, len (array)):
            height, width= array [i].shape
            for h in range (height, self.height):
                array [i] = numpy.append (array [i], [[0.0] * width], axis=0)
            for w in range (width, self.width):
                array [i] = numpy.append (array [i], [[0.0]] * (self.height), axis=1)
        # reshape array
        array = numpy.array ([x.reshape (self.height, self.width, self.channels) for x in array])
        return array
    @staticmethod
    def sum_of_residual (y_true, y_pred):
        # return tensorflow.reduce_sum (abs (y_true - y_pred))
        return tensorflow.reduce_sum (tensorflow.abs (y_true - y_pred))
    @classmethod
    def testModule (cls, args=None):
        train = [Figure.getNormalFigure (12, 14) for i in range (0, 1024*4)]
        test1 = [Figure.getNormalFigure (12, 14) for i in range (0, 8)]
        test2 = [Figure.getAbnormalFigure (12, 14) for i in range (0, 8)]
        anogan = cls (12, 14, args=args)
        anogan.fit (train)
        scores1, syns1 = anogan.evaluate (test1, synthesis=True, epochs=128)
        scores2, syns2 = anogan.evaluate (test2, synthesis=True)
        print ('Normal: ' + str (numpy.mean (scores1)))
        print ('Abnormal: ' + str (numpy.mean (scores2)))
        print (syns1 [0])



################################################
###               Autoencoders               ###
################################################

class Autoencoder1 (object):
    def __init__ (self, length=None, height=None, width=None, args=None, loss='mean_squared_error'):
        gc.collect()
        self.args = args
        self.length = length if length is not None else int (width * height)
        self.units = 512
        self.relu = 1.0
        self.stddev = 0.02
        self.loss = loss
        self.activation = 'tanh'
        # encoder: code-(512 / 8 =64)
        self.encoder = keras.models.Sequential ()
        self.encoder.add (keras.layers.Dense (self.units, input_dim=self.length, 
                          kernel_initializer=keras.initializers.RandomNormal (stddev=self.stddev)))
        self.encoder.add (keras.layers.LeakyReLU (self.relu))
        self.encoder.add (keras.layers.BatchNormalization ())
        self.encoder.add (keras.layers.Dense (int (self.units / 2)))
        self.encoder.add (keras.layers.LeakyReLU (self.relu))
        self.encoder.add (keras.layers.BatchNormalization ())
        self.encoder.add (keras.layers.Dense (int (self.units / 4)))
        self.encoder.add (keras.layers.LeakyReLU (self.relu))
        self.encoder.add (keras.layers.BatchNormalization ())
        self.encoder.add (keras.layers.Dense (int (self.units / 8)))
        self.encoder.add (keras.layers.LeakyReLU (self.relu))
        self.encoder.add (keras.layers.BatchNormalization ())
        self.encoder.compile (optimizer='adadelta', loss=self.loss)
        if self.args.verbose: print ('\nEncoder'); self.encoder.summary ()
        # decoder
        self.decoder = keras.models.Sequential () 
        self.decoder.add (keras.layers.Dense (int (self.units / 8), input_dim=int (self.units / 8),
                          kernel_initializer=keras.initializers.RandomNormal (stddev=self.stddev))) 
        self.decoder.add (keras.layers.LeakyReLU (self.relu))
        self.decoder.add (keras.layers.BatchNormalization ())
        self.decoder.add (keras.layers.Dense (int (self.units / 4)))
        self.decoder.add (keras.layers.LeakyReLU (self.relu))
        self.decoder.add (keras.layers.BatchNormalization ())
        self.decoder.add (keras.layers.Dense (int (self.units / 2)))
        self.decoder.add (keras.layers.LeakyReLU (self.relu))
        self.decoder.add (keras.layers.Dense (self.length, activation=self.activation))
        self.decoder.compile (optimizer='adadelta', loss=self.loss)
        if self.args.verbose: print ('\nDecoder'); self.decoder.summary ()
        # model
        self.input = keras.layers.Input (shape=(self.length,))
        self.midput = self.encoder (self.input)
        self.output = self.decoder (self.midput)
        self.autoencoder = keras.models.Model (self.input, self.output)
        self.autoencoder.compile (optimizer='adadelta', loss=self.loss)
        if self.args.verbose: print ('\nAutoencoder'); self.autoencoder.summary ()
    def fit (self, X, epochs=4, batchsize=512):
        X = self.checkArray (X)
        loss = self.autoencoder.fit (X, X, epochs=epochs, batch_size=batchsize, shuffle=True, verbose=self.args.verbose)
    def evaluate (self, X, synthesis=False):
        scores = []
        syns = []
        for i, x in enumerate (X):
            x = self.checkArray ([x])
            score = self.autoencoder.evaluate (x, x, verbose=False)
            scores.append (score)
            if synthesis: syns.append (self.autoencoder.predict (x, verbose=False) [0])
            if self.args.verbose: print ('Test-' + str (len (X) - i - 1) + '=' + str (score))
        if synthesis: syns = numpy.reshape (numpy.array (syns), numpy.array (X).shape)
        return scores, syns
    def predict (self, X):
        return numpy.reshape (self.autoencoder.predict (self.checkArray (X), verbose=self.args.verbose), X.shape)
    def train (self, X, epochs=4, batchsize=512):
        return self.fit (X, epochs, batchsize)
    def test (self, X, synthesis=False):
        return self.evaluate (X, synthesis)
    def checkArray (self, array):
        array = numpy.array (array).reshape (len (array), self.length)
        return array
    @classmethod
    def testModule (cls, args):
        train = [Figure.getNormalFigure (12, 14) for i in range (0, 1024*4)]
        test1 = [Figure.getNormalFigure (12, 14) for i in range (0, 8)]
        test2 = [Figure.getAbnormalFigure (12, 14) for i in range (0, 8)]
        autoencoder = cls (width=12, height=14, args=args)
        autoencoder.fit (train)
        scores1, syns1 = autoencoder.evaluate (test1, synthesis=True)
        scores2, syns2 = autoencoder.evaluate (test2, synthesis=True)
        print ('Normal: ' + str (numpy.mean (scores1)))
        print ('Abnormal: ' + str (numpy.mean (scores2)))
        print (syns1 [0])

class Autoencoder2 (object):
    def __init__ (self, height, width, args=None, loss='mean_squared_error', channels=1):
        gc.collect()
        self.args = args
        self.width = (int ((width - 1) / 4) + 1) * 4
        self.height = (int ((height - 1) / 4) + 1) * 4
        self.channels = channels
        self.units = 1024 
        self.col = int (self.width) / 4
        self.row = int (self.height) / 4
        self.kernels = (5, 5)
        self.strides = (2, 2)
        self.relu = 1.0
        self.stddev = 0.02
        self.loss = loss
        # encoder: code-(h/4, w/4, unit/2)
        self.encoder = keras.models.Sequential ()
        self.encoder.add (keras.layers.Conv2D (self.units, self.kernels, padding='same',
                          input_shape=(self.height, self.width, self.channels), kernel_initializer=keras.initializers.RandomNormal (stddev=self.stddev)))
        self.encoder.add (keras.layers.LeakyReLU (self.relu))
        self.encoder.add (keras.layers.BatchNormalization ())
        self.encoder.add (keras.layers.MaxPooling2D (self.strides, padding='same'))
        self.encoder.add (keras.layers.Conv2D (int (self.units / 2), self.kernels, padding='same'))
        self.encoder.add (keras.layers.LeakyReLU (self.relu))
        self.encoder.add (keras.layers.BatchNormalization ())
        self.encoder.add (keras.layers.MaxPooling2D (self.strides, padding='same'))
        self.encoder.compile (optimizer='adadelta', loss=self.loss)
        if self.args.verbose: print ('\nEncoder'); self.encoder.summary ()
        # decoder
        self.decoder = keras.models.Sequential () 
        self.decoder.add (keras.layers.Conv2D (int (self.units / 2), self.kernels, padding='same',
                          input_shape=(self.row, self.col, int (self.units / 2)), kernel_initializer=keras.initializers.RandomNormal (stddev=self.stddev)))
        self.decoder.add (keras.layers.LeakyReLU (self.relu))
        self.decoder.add (keras.layers.BatchNormalization ())
        self.decoder.add (keras.layers.UpSampling2D (self.strides))
        self.decoder.add (keras.layers.Conv2D (int (self.units / 2), self.kernels, padding='same'))
        self.decoder.add (keras.layers.LeakyReLU (self.relu))
        self.decoder.add (keras.layers.BatchNormalization ())
        self.decoder.add (keras.layers.UpSampling2D (self.strides))
        self.decoder.add (keras.layers.Conv2D (self.units, self.kernels, padding='same'))
        self.decoder.add (keras.layers.LeakyReLU (self.relu))
        self.decoder.add (keras.layers.Conv2D (self.channels, self.kernels, activation='tanh', padding='same'))
        self.decoder.compile (optimizer='adadelta', loss=self.loss)
        if self.args.verbose: print ('\nDecoder'); self.decoder.summary ()
        # model
        self.input = keras.layers.Input (shape=(self.height, self.width, self.channels))
        self.midput = self.encoder (self.input)
        self.output = self.decoder (self.midput)
        self.autoencoder = keras.models.Model (self.input, self.output)
        self.autoencoder.compile (optimizer='adadelta', loss=self.loss)
        if self.args.verbose: print ('\nAutoencoder'); self.autoencoder.summary ()
    def fit (self, X, epochs=4, batchsize=512):
        X = self.checkArray (X)
        loss = self.autoencoder.fit (X, X, epochs=epochs, batch_size=batchsize, shuffle=True, verbose=self.args.verbose)
    def evaluate (self, X, synthesis=False):
        scores = []
        syns = []
        for i, x in enumerate (X):
            x = self.checkArray ([x])
            score = self.autoencoder.evaluate (x, x, verbose=False)
            scores.append (score)
            if synthesis: syns.append (self.autoencoder.predict (x, verbose=False) [0])
            if self.args.verbose: print ('Test-' + str (len (X) - i - 1) + '=' + str (score))
        shape = numpy.array (X).shape                  
        if synthesis: syns = numpy.reshape (numpy.array (syns)[:, :shape [1], :shape [2]], shape)
        return scores, syns
    def predict (self, X):
        shape = numpy.array (X).shape                  
        syns = self.autoencoder.predict (self.checkArray (X), verbose=self.args.verbose)
        syns = numpy.reshape (numpy.array (syns)[:, :shape [1], :shape [2]], shape)
        return syns
    def train (self, X, epochs=4, batchsize=512):
        return self.fit (X, epochs, batchsize)
    def test (self, X, synthesis=False):
        return self.evaluate (X, synthesis)
    def checkArray (self, array):
        # pad to fit dimensions
        padding = [0.0] if self.channels < 2 else [[0.0] * self.channels]
        for i in range (0, len (array)):
            height, width = array [i].shape [: 2]
            array = list (array)
            for h in range (height, self.height):
                array [i] = numpy.append (array [i], [padding * width], axis=0)
            for w in range (width, self.width):
                array [i] = numpy.append (array [i], [padding] * (self.height), axis=1)
        # reshape array
        array = numpy.reshape (array, (len (array), self.height, self.width, self.channels))
        return array
    @classmethod
    def testModule (cls, args):
        train = [Figure.getNormalFigure (12, 14) for i in range (0, 1024*4)]
        test1 = [Figure.getNormalFigure (12, 14) for i in range (0, 8)]
        test2 = [Figure.getAbnormalFigure (12, 14) for i in range (0, 8)]
        autoencoder = cls (12, 14, args=args)
        autoencoder.fit (train)
        scores1, syns1 = autoencoder.evaluate (test1, synthesis=True)
        scores2, syns2 = autoencoder.evaluate (test2, synthesis=True)
        print ('Normal: ' + str (numpy.mean (scores1)))
        print ('Abnormal: ' + str (numpy.mean (scores2)))
        print (syns1 [0])
       
class AutoencoderR (object):
    def __init__ (self, height, width, args=None, model=Autoencoder2, loss='mean_squared_error'):
        gc.collect()
        self.hyperlambda = 0.0314159
        self.hyperepsilon = 0.0314159 * 4
        self.model = model
        self.maxsize = 1024 * 1024
        self.epochs = 2048
        self.index = 0
        self.trainX = []
        self.args = args
        self.loss = loss
    def fit (self, X, epochs=4, batchsize=512):
        self.epochs = epochs * batchsize 
        for i in range (0, len (X)):
            if len (self.trainX) < self.maxsize: self.trainX.append (None)
            self.trainX [self.index % self.maxsize] = X [i]
            self.index += 1
    def evaluate (self, X, synthesis=False, epochs=None, batchsize=None):
        if epochs is not None and batchsize is not None: self.epochs = epochs * batchsize
        iteration = self.epochs
        X0 = numpy.array (list (self.trainX) + list (X))
        # initial noise must not be zero, otherwise the anomalous test set is trained
        N1 = []
        for i in range (0, len (self.trainX)): N1.append (numpy.zeros (X [0].shape))
        for i in range (0, len (X)): N1.append (numpy.random.random_sample (X [0].shape) * 2 - 1)
        N0, N1 = None, numpy.array (N1)
        # retrain the model as the model could be effected by previous anomalous test set
        self.autoencoder = self.model (height=X0.shape [1], width=X0.shape [2], args=self.args, loss=self.loss)
        while not self.converged (N0, N1) and iteration > 0:
            iteration -= 1
            N0 = N1
            self.autoencoder.fit (X0 - N0, epochs=1)
            X1 = self.autoencoder.predict (X0)
            N1 = self.threshold (X0, X1) 
            if self.args.verbose: print ('Iteration = ' + str (iteration))
        scores = self.scores ([N1 [0 - i] for i in range (0, len (X))])
        syns = [X1 [0 - i] for i in range (0, len (X))]
        return scores, syns
    def train (self, X, epochs=4, batchsize=512):
        return self.fit (X, epochs, batchsize)
    def test (self, X, synthesis=False, epochs=None, batchsize=None):
        return self.evaluate (X, synthesis, epochs, batchsize)
    def converged (self, N0, N1): 
        if N0 is None or N1 is None: return False
        N0 = numpy.reshape (N0, N0.size)
        N1 = numpy.reshape (N1, N1.size)
        return not any (diff > self.hyperepsilon for diff in numpy.absolute (N1 - N0))
    def scores (self, N):
        ret = []
        for i in range (0, len (N)):
            noise = numpy.array (N [i])
            noise = numpy.reshape (noise, noise.size)
            ret.append (numpy.sum (noise ** 2) / noise.size)
        return ret
    def threshold (self, X0, X1):  
        halflambda = self.hyperlambda / 2
        diff = numpy.reshape (X0 - X1, ((len (X0), int (X0.size / len (X0)))))
        if self.hyperlambda == 0: return diff
        noise = numpy.zeros (diff.shape)
        index = numpy.where (diff > halflambda)
        noise [index] = diff [index] - halflambda
        index = numpy.where (numpy.absolute (diff) <= halflambda)
        noise [index] = 0
        index = numpy.where (diff < 0 - halflambda)
        noise [index] = diff [index] + halflambda
        return numpy.reshape (noise, X0.shape)
    @classmethod
    def testModule (cls, args, model=None):
        train = [Figure.getNormalFigure (12, 14) for i in range (0, 1024*4)]
        test1 = [Figure.getNormalFigure (12, 14) for i in range (0, 8)]
        test2 = [Figure.getAbnormalFigure (12, 14) for i in range (0, 8)]
        autoencoder = cls (12, 14, args=args, model=model)
        autoencoder.fit (train)
        scores1, syns1 = autoencoder.evaluate (test1, synthesis=True)
        scores2, syns2 = autoencoder.evaluate (test2, synthesis=True)
        print ('Normal: ' + str (numpy.mean (scores1)))
        print ('Abnormal: ' + str (numpy.mean (scores2)))
        print (syns1 [0])



################################################
###                Deviation                 ###
################################################

class Deviation (list):
    def __init__ (self, obj, args=None):
        filename = None
        # read from file if obj is str
        if isinstance (obj, str):
            if os.path.isfile (obj):
                filename, obj = obj, {}
                if filename [-10: ] == '.deviation': obj = json.loads (open (filename, 'r').read ())
            elif os.path.isdir (obj):
                dirname, obj = obj, {}
                for filename in os.listdir (dirname):
                    filename = os.path.join (dirname, filename)
                    if filename [-10: ] == '.deviation':
                        try:
                            deviation = json.loads (open (filename, 'r').read ())
                            for key in deviation: obj [key] = deviation [key]
                        except: continue # could be an empty file if the user is not presented on the day
        # preprocess if obj is a dict 
        if isinstance (obj, dict):
            # disable features if specified
            for key in list (obj.keys ()):
                if (   (key [-2: ] == '-0' and args.no_level_zero)
                    or ('all' not in args.features and not any (feature in key for feature in args.features))):
                    del (obj [key])
            # flat to one level
            for key in list (obj.keys ()):
                if isinstance (obj [key], list) and isinstance (obj [key][0], list):
                    for level in range (0, len (obj [key])):
                        obj [key + '-' + str (level)] = obj [key][level]
                    del (obj [key])
            # split by timeslices
            arr = []
            for i in range (0, len (obj [list (obj.keys ()) [0]])):
                arr.append ({k: obj [k][i] for k in obj.keys ()})
        # pre-built deviation
        if isinstance (obj, list):
            arr = obj
        # instantiation
        super (Deviation, self).__init__ (arr)
        if filename is not None:
            names = filename.split ('/')
            self.date = names [-2]
            self.user = names [-3]
    def getUser (self):
        return self.user
    def getDate (self):
        return self.date
    def getKeys (self):
        return sorted ([key for key in self [0]])



################################################
###                    User                  ###
################################################

class User (dict):
    def __init__ (self, username, args, init=None):
        if init is not None: super (User, self).__init__ (init)
        else: super (User, self).__init__ ()
        self.username = username
        self.args = args
        self.heatmaps = None
        self.heatdates = None
        self.risk = 'None'
    def append (self, deviation):
        if len (deviation) > 0:
            self.__setitem__ (deviation.date, deviation)
            self.heatmaps = None
            self.heatdates = None
    def getDateRange (self, start=None, end=None):
        dates = sorted (self.keys ())
        iStart, iEnd = 0, len (dates)
        if start and not isinstance (start, str): start, end = start
        while start and iStart < len (dates) and dates [iStart + 1] <= start: iStart += 1
        while end and iEnd > 0 and dates [iEnd - 1] > end: iEnd -= 1
        return dates [iStart: iEnd]
    def sample (self):
        return self [list (self.keys ()) [0]]
    def setRisk (self, risk):
        if isinstance (risk, str):
            self.risk = risk
        else: self.risk = '_'.join ([self.args.model, str (risk)])
    def heatmap (self, dates=None, plot=None):
        # check dates: return None if any date is missed out
        dates = self.getDateRange () if dates is None else dates
        absolute_deviation = True
        if any (d not in self for d in dates): return None
        # make figures
        if self.heatmaps is None or self.heatdates != dates:
            self.heatmaps = []
            self.heatdates = dates
            dateLabels = dates
            featureLabels = self.sample ().getKeys ()
            timeslices = len (self.sample ())
            for timeIndex in range (0, timeslices):
                self.heatmaps.append ([])
                for featureIndex, feature in enumerate (featureLabels):
                    self.heatmaps [timeIndex].append ([])
                    for dateIndex, date in enumerate (dateLabels):
                        if feature not in self [date][timeIndex]: self [date][timeIndex][feature] = 0.0 # default value of deviation = 0.0
                        observation = self [date][timeIndex][feature] / self.args.image_sigma # scale transformation from (-5, 5) to (-1, 1)
                        observation = max (-1.0, min (1.0, observation)) # check boundaries as raw sigma could be 5
                        if observation < 0.0: observation = observation * self.args.image_negweight # assign negative weight
                        if absolute_deviation: observation = (observation + 1) / 2 # scale observation from (-1, 1) to (0, 1)
                        self.heatmaps [timeIndex][featureIndex].append (observation)                    
        # plot figures  
        if plot and self.args.image_dir is not None:
            dateLabels = dates
            featureLabels = self.sample ().getKeys ()
            timeslices = len (self.sample ())
            rows = [self.heatmaps] if plot is True else [self.heatmaps, plot.heatmap (dates=dates)]
            title = '_'.join ([self.args.image_prefix, self.risk, self.username, dateLabels [0], dateLabels [-1]])
            fig, axs = matplotlib.pyplot.subplots (nrows=len (rows), ncols=timeslices, sharex=True, sharey=True)
            for row in range (0, len (rows)):
                figures = rows [row]
                for timeIndex in range (0, timeslices):
                    # color transformation
                    bitmap = []
                    for featureIndex in range (0, len (figures [timeIndex])):
                        bitmap.append ([])
                        for dateIndex in range (0, len (figures [timeIndex][featureIndex])):
                            observation = figures [timeIndex][featureIndex][dateIndex]
                            if absolute_deviation: observation = (observation * 2) - 1 # rescale observation from (0, 1) to (-1, 1) 
                            if observation >= 0.0: observation = [observation, 0.0, 0.0] # red gradient
                            else: observation = [-observation, -observation, -observation] # gray gradient
                            bitmap [featureIndex].append (observation)
                    # plot
                    ax = axs [timeIndex] if len (rows) == 1 else axs [row][timeIndex]
                    im = ax.imshow (numpy.array (bitmap))
                    ax.set_xticks (numpy.arange (len (dateLabels)))
                    ax.set_yticks (numpy.arange (len (featureLabels)))
                    ax.set_xticklabels (dateLabels, fontsize=4)
                    ax.set_yticklabels (featureLabels, fontsize=4)
                    matplotlib.pyplot.setp (ax.get_xticklabels (), rotation=45, ha='right', rotation_mode='anchor')
                    matplotlib.pyplot.setp (ax.get_yticklabels (), rotation=45, ha='right', rotation_mode='anchor')
            fig.suptitle (title)
            fig.tight_layout ()
            matplotlib.pyplot.savefig (os.path.join (self.args.image_dir, title + '.png'), dpi=600)
            matplotlib.pyplot.close (fig)
        return self.heatmaps



################################################
###                User Manager              ###
################################################

class UserManager (dict):
    def __init__ (self, args):
        super (UserManager, self).__init__ ()
        self.common = None
        self.args = args
    def append (self, obj):
        if isinstance (obj, Deviation):
            user = obj.getUser ()
            if user not in self: self [user] = User (user, self.args)
            self [user].append (obj)
            self.common = None
    def sample (self):
        return self [list (self.keys ()) [0]]
    def mean (self):
        if self.common is None:
            timeslices = len (self.sample ().sample ())
            fsum = {}
            # append everything
            for user in self:
                for date in self [user]:
                    if date not in fsum: fsum [date] = [dict () for timeIndex in range (0, timeslices)]
                    for timeIndex in range (0, timeslices):
                        for feature in self [user][date][timeIndex]:
                            if feature not in fsum [date][timeIndex]: fsum [date][timeIndex][feature] = []
                            fsum [date][timeIndex][feature].append (self [user][date][timeIndex][feature])
            # calculate mean
            for date in fsum:
                for timeIndex in range (0, timeslices):
                    for feature in fsum [date][timeIndex]:
                        fsum [date][timeIndex][feature] = numpy.mean (fsum [date][timeIndex][feature])
                fsum [date] = Deviation (fsum [date], self.args)
            self.common = User ('MEAN', self.args, fsum)
        return self.common



################################################
###              Test Figure                 ###
################################################

class Figure (numpy.ndarray):
    def __new__ (cls, h, w=None, args=None, tags=None):
        if w is not None and isinstance (h, int) and isinstance (w, int):
            array = []
            for i in range (0, h): array.append([0.0] * w)
            ret = numpy.asarray (array).view (cls)
        else: ret = numpy.asarray (h).view (cls)
        ret.args = args
        ret.tagl = tags
        return ret
    def tags (self):
        try:
            return list (self.tagl)
        except:
            return self.tagl
    def flat (self):
        return self.reshape (self.size)
    def plot (self, filename, args=None):
        if args: self.args = args
        # color transformation
        width = self.args.image_window * 2
        height = int (self.size / width)
        heatmap = self.reshape (height, width)
        filename = self.args.image_prefix + filename
        bitmap = []
        for i in range (0, height):
            bitmap.append ([])
            for j in range (0, width):
                observation = heatmap [i][j]
                if observation >= 0.0: observation = [observation, 0.0, 0.0] # red gradient
                else: observation = [-observation, -observation, -observation] # gray gradient
                bitmap [i].append (observation)
        # plot
        fig, axs = matplotlib.pyplot.subplots (nrows=1, ncols=1)
        im = axs.imshow (numpy.array (bitmap))
        fig.suptitle (filename)
        fig.tight_layout ()
        matplotlib.pyplot.savefig (os.path.join (self.args.image_dir, filename + '.png'), dpi=600)
        matplotlib.pyplot.close (fig)
    @classmethod
    def getNormalFigure (cls, h, w, args=None, tags=None):
        ret = cls (h, w, args, tags)
        for i in range (0, h, 2):
            for j in range (0, w, 2):
                if random.random () > 0.5: ret [i][j] = random.random () * 2 - 1
        ret [0][0] = 0.88
        ret [0][1] = 0.31415
        return ret
    @classmethod
    def getAbnormalFigure (cls, h, w, args=None, tags=None):
        ret = cls.getNormalFigure (h, w, args, tags)
        for i in range (1, h, 2):
            for j in range (1, w, 2):
                if random.random () > 0.5: ret [i][j] = 1.0
        return ret
    @classmethod
    def Heatmap (cls, heatmap, rows=None, timeslices=None, features=None, dates=None, args=None, tags=None):
        if not rows or not timeslices or not features or not dates:
            rows = len (heatmap)
            timeslices = len (heatmap [0])
            features = len (heatmap [0][0])
            dates = len (heatmap [0][0][0])
        bitmap = [[0.0] * (timeslices * dates) for _ in range (0, (rows * features))]
        for row in range (0, rows):
            for timeIndex in range (0, timeslices):
                for feature in range (0, features):
                    for date in range (0, dates):
                        bitmap [row * features + feature][timeIndex * dates + date] = heatmap [row][timeIndex][feature][date]
        return cls (bitmap, args=args, tags=tags)



################################################
###                   MISC                   ###
################################################

if __name__ == '__main__':
    main ()


