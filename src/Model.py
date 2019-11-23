from __future__ import division
from __future__ import print_function

import sys
import numpy as np
import tensorflow as tf
import os


class DecoderType:
	BestPath = 0
	BeamSearch = 1
	WordBeamSearch = 2

class Model:
    # Model Constants
    batchSize = 50
    imgSize = (800, 64)
    maxTextLen = 100

    def __init__(self, charList, decoderType=DecoderType.BestPath, mustRestore=False):
        self.charList = charList
        self.decoderType = decoderType
        self.mustRestore = mustRestore
        self.snapID = 0

        # CNN
        with tf.name_scope('CNN'):
            with tf.name_scope('Input'):
                self.inputImgs = tf.placeholder(tf.float32, shape=(
                    Model.batchSize, Model.imgSize[0], Model.imgSize[1]))
            cnnOut4d = self.setupCNN(self.inputImgs)

        # RNN
        with tf.name_scope('RNN'):
            rnnOut3d = self.setupRNN(cnnOut4d)

        # # Debuging CTC
        # self.rnnOutput = tf.transpose(rnnOut3d, [1, 0, 2])

        # CTC
        with tf.name_scope('CTC'):
            (self.loss, self.decoder) = self.setupCTC(rnnOut3d)
            self.training_loss_summary = tf.summary.scalar(
                'loss', self.loss)  # Tensorboard: Track loss

        # Optimize NN parameters
        with tf.name_scope('Optimizer'):
            self.batchesTrained = 0
            self.learningRate = tf.placeholder(tf.float32, shape=[])
            self.optimizer = tf.train.RMSPropOptimizer(
                self.learningRate).minimize(self.loss)

        # Initialize TensorFlow
        (self.sess, self.saver) = self.setupTF()

        self.writer = tf.summary.FileWriter(
            './logs', self.sess.graph)  # Tensorboard: Create writer
        self.merge = tf.summary.merge(
            [self.training_loss_summary])  # Tensorboard: Merge

    def setupCNN(self, cnnIn3d):
        """ Create CNN layers and return output of these layers """

        cnnIn4d = tf.expand_dims(input=cnnIn3d, axis=3)

        # First Layer: Conv (5x5) + Pool (2x2) - Output size: 400 x 32 x 64
        with tf.name_scope('Conv_Pool_1'):
            kernel = tf.Variable(
                tf.truncated_normal([5, 5, 1, 64], stddev=0.1))
            conv = tf.nn.conv2d(
                cnnIn4d, kernel, padding='SAME', strides=(1, 1, 1, 1))
            relu = tf.nn.relu(conv)
            pool = tf.nn.max_pool(relu, (1, 2, 2, 1), (1, 2, 2, 1), 'VALID')

        # Second Layer: Conv (5x5) - Output size: 400 x 32 x 128
        with tf.name_scope('Conv_2'):
            kernel = tf.Variable(tf.truncated_normal(
                [5, 5, 64, 128], stddev=0.1))
            conv = tf.nn.conv2d(
                pool, kernel, padding='SAME', strides=(1, 1, 1, 1))
            relu = tf.nn.relu(conv)

        # Third Layer: Conv (3x3) + Pool (2x2) + Simple Batch Norm - Output size: 200 x 16 x 128
        with tf.name_scope('Conv_Pool_BN_3'):
            kernel = tf.Variable(tf.truncated_normal(
                [3, 3, 128, 128], stddev=0.1))
            conv = tf.nn.conv2d(
                relu, kernel, padding='SAME', strides=(1, 1, 1, 1))
            mean, variance = tf.nn.moments(conv, axes=[0])
            batch_norm = tf.nn.batch_normalization(
                conv, mean, variance, offset=None, scale=None, variance_epsilon=0.001)
            relu = tf.nn.relu(batch_norm)
            pool = tf.nn.max_pool(relu, (1, 2, 2, 1), (1, 2, 2, 1), 'VALID')

        # Fourth Layer: Conv (3x3) - Output size: 200 x 16 x 256
        with tf.name_scope('Conv_4'):
            kernel = tf.Variable(tf.truncated_normal(
                [3, 3, 128, 256], stddev=0.1))
            conv = tf.nn.conv2d(
                pool, kernel, padding='SAME', strides=(1, 1, 1, 1))
            relu = tf.nn.relu(conv)

        # Fifth Layer: Conv (3x3) - Output size: 200 x 16 x 256
        with tf.name_scope('Conv_5'):
            kernel = tf.Variable(tf.truncated_normal(
                [3, 3, 256, 256], stddev=0.1))
            conv = tf.nn.conv2d(
                relu, kernel, padding='SAME', strides=(1, 1, 1, 1))
            relu = tf.nn.relu(conv)

        # Sixth Layer: Conv (3x3) + Simple Batch Norm - Output size: 200 x 16 x 512
        with tf.name_scope('Conv_BN_6'):
            kernel = tf.Variable(tf.truncated_normal(
                [3, 3, 256, 512], stddev=0.1))
            conv = tf.nn.conv2d(
                relu, kernel, padding='SAME', strides=(1, 1, 1, 1))
            mean, variance = tf.nn.moments(conv, axes=[0])
            batch_norm = tf.nn.batch_normalization(
                conv, mean, variance, offset=None, scale=None, variance_epsilon=0.001)
            relu = tf.nn.relu(batch_norm)

        # Seventh Layer: Conv (3x3) + Pool (2x2) - Output size: 100 x 8 x 512
        with tf.name_scope('Conv_Pool_7'):
            kernel = tf.Variable(tf.truncated_normal(
                [3, 3, 512, 512], stddev=0.1))
            conv = tf.nn.conv2d(
                relu, kernel, padding='SAME', strides=(1, 1, 1, 1))
            relu = tf.nn.relu(conv)
            pool = tf.nn.max_pool(relu, (1, 2, 2, 1), (1, 2, 2, 1), 'VALID')

        return pool

    def setupRNN(self, rnnIn4d):
        """ Create RNN layers and return output of these layers """
        rnnIn4d = tf.slice(rnnIn4d, [0, 0, 0, 0], [
                           self.batchSize, 100, 1, 512])
        rnnIn3d = tf.squeeze(rnnIn4d)

        # 2 layers of LSTM cell used to build RNN
        numHidden = 512
        cells = [tf.nn.rnn_cell.LSTMCell(
            numHidden, name='basic_lstm_cell') for _ in range(2)]
        stacked = tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)

        # Bi-directional RNN
        # BxTxF -> BxTx2H
        ((forward, backward), _) = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=stacked, cell_bw=stacked, inputs=rnnIn3d, dtype=rnnIn3d.dtype)

        # BxTxH + BxTxH -> BxTx2H -> BxTx1X2H
        concat = tf.expand_dims(tf.concat([forward, backward], 2), 2)

        # Project output to chars (including blank): BxTx1x2H -> BxTx1xC -> BxTxC
        kernel = tf.Variable(tf.truncated_normal(
            [1, 1, numHidden*2, len(self.charList)+1], stddev=0.1))
        return tf.squeeze(tf.nn.atrous_conv2d(value=concat, filters=kernel, rate=1, padding='SAME'), axis=[2])


    def setupCTC(self, ctcIn3d):
        """ Create CTC loss and decoder and return them """
        # BxTxC -> TxBxC
        ctcIn3dTBC = tf.transpose(ctcIn3d, [1, 0, 2])

        # Ground truth text as sparse tensor
        with tf.name_scope('CTC_Loss'):
            self.gtTexts = tf.SparseTensor(tf.placeholder(tf.int64, shape=[
                                           None, 2]), tf.placeholder(tf.int32, [None]), tf.placeholder(tf.int64, [2]))
            # Calculate loss for batch
            self.seqLen = tf.placeholder(tf.int32, [None])
            loss = tf.nn.ctc_loss(labels=self.gtTexts, inputs=ctcIn3dTBC, sequence_length=self.seqLen,
                                  ctc_merge_repeated=True, ignore_longer_outputs_than_inputs=True)
        with tf.name_scope('CTC_Decoder'):
            # Decoder: Best path decoding or Word beam search decoding
            if self.decoderType == DecoderType.BestPath:
                decoder = tf.nn.ctc_greedy_decoder(
                    inputs=ctcIn3dTBC, sequence_length=self.seqLen)
            elif self.decoderType == DecoderType.WordBeamSearch:
                # Import compiled word beam search operation (see https://github.com/githubharald/CTCWordBeamSearch)
                word_beam_search_module = tf.load_op_library(
                    './TFWordBeamSearch.so')

                # Prepare: dictionary, characters in dataset, characters forming words
                chars = codecs.open(FilePaths.fnCharList, 'r', 'utf8').read()
                wordChars = codecs.open(
                    FilePaths.fnWordCharList, 'r', 'utf8').read()
                corpus = codecs.open(FilePaths.fnCorpus, 'r', 'utf8').read()

                # # Decoder using the "NGramsForecastAndSample": restrict number of (possible) next words to at most 20 words: O(W) mode of word beam search
                # decoder = word_beam_search_module.word_beam_search(tf.nn.softmax(ctcIn3dTBC, dim=2), 25, 'NGramsForecastAndSample', 0.0, corpus.encode('utf8'), chars.encode('utf8'), wordChars.encode('utf8'))

                # Decoder using the "Words": only use dictionary, no scoring: O(1) mode of word beam search
                decoder = word_beam_search_module.word_beam_search(tf.nn.softmax(
                    ctcIn3dTBC, dim=2), 25, 'Words', 0.0, corpus.encode('utf8'), chars.encode('utf8'), wordChars.encode('utf8'))

        # Return a CTC operation to compute the loss and CTC operation to decode the RNN output
        return (tf.reduce_mean(loss), decoder)

    def setupTF(self):
        """ Initialize TensorFlow """
        print('Python: ' + sys.version)
        print('Tensorflow: ' + tf.__version__)
        sess = tf.Session()  # Tensorflow session
        saver = tf.train.Saver(max_to_keep=5)  # Saver saves model to file
        modelDir = '../model/'
        latestSnapshot = tf.train.latest_checkpoint(
            modelDir)  # Is there a saved model?
        # If model must be restored (for inference), there must be a snapshot
        if self.mustRestore and not latestSnapshot:
            raise Exception('No saved model found in: ' + modelDir)
        # Load saved model if available
        if latestSnapshot:
            print('Init with stored values from ' + latestSnapshot)
            saver.restore(sess, latestSnapshot)
        else:
            print('Init with new values')
            sess.run(tf.global_variables_initializer())

        return (sess, saver)

	def toSparse(self, texts):
		"put ground truth texts into sparse tensor for ctc_loss"
		indices = []
		values = []
		shape = [len(texts), 0] # last entry must be max(labelList[i])

		# go over all texts
		for (batchElement, text) in enumerate(texts):
			# convert to string of label (i.e. class-ids)
			labelStr = [self.charList.index(c) for c in text]
			# sparse tensor must have size of max. label-string
			if len(labelStr) > shape[1]:
				shape[1] = len(labelStr)
			# put each label into sparse tensor
			for (i, label) in enumerate(labelStr):
				indices.append([batchElement, i])
				values.append(label)

		return (indices, values, shape)


	def decoderOutputToText(self, ctcOutput, batchSize):
		"extract texts from output of CTC decoder"
		
		# contains string of labels for each batch element
		encodedLabelStrs = [[] for i in range(batchSize)]

		# word beam search: label strings terminated by blank
		if self.decoderType == DecoderType.WordBeamSearch:
			blank=len(self.charList)
			for b in range(batchSize):
				for label in ctcOutput[b]:
					if label==blank:
						break
					encodedLabelStrs[b].append(label)

		# TF decoders: label strings are contained in sparse tensor
		else:
			# ctc returns tuple, first element is SparseTensor 
			decoded=ctcOutput[0][0] 

			# go over all indices and save mapping: batch -> values
			idxDict = { b : [] for b in range(batchSize) }
			for (idx, idx2d) in enumerate(decoded.indices):
				label = decoded.values[idx]
				batchElement = idx2d[0] # index according to [b,t]
				encodedLabelStrs[batchElement].append(label)

		# map labels to chars for all batch elements
		return [str().join([self.charList[c] for c in labelStr]) for labelStr in encodedLabelStrs]


	def trainBatch(self, batch):
		"feed a batch into the NN to train it"
		numBatchElements = len(batch.imgs)
		sparse = self.toSparse(batch.gtTexts)
		rate = 0.01 if self.batchesTrained < 10 else (0.001 if self.batchesTrained < 10000 else 0.0001) # decay learning rate
		evalList = [self.optimizer, self.loss]
		feedDict = {self.inputImgs : batch.imgs, self.gtTexts : sparse , self.seqLen : [Model.maxTextLen] * numBatchElements, self.learningRate : rate, self.is_train: True}
		(_, lossVal) = self.sess.run(evalList, feedDict)
		self.batchesTrained += 1
		return lossVal


	def dumpNNOutput(self, rnnOutput):
		"dump the output of the NN to CSV file(s)"
		dumpDir = '../dump/'
		if not os.path.isdir(dumpDir):
			os.mkdir(dumpDir)

		# iterate over all batch elements and create a CSV file for each one
		maxT, maxB, maxC = rnnOutput.shape
		for b in range(maxB):
			csv = ''
			for t in range(maxT):
				for c in range(maxC):
					csv += str(rnnOutput[t, b, c]) + ';'
				csv += '\n'
			fn = dumpDir + 'rnnOutput_'+str(b)+'.csv'
			print('Write dump of NN to file: ' + fn)
			with open(fn, 'w') as f:
				f.write(csv)


	def inferBatch(self, batch, calcProbability=False, probabilityOfGT=False):
		"feed a batch into the NN to recognize the texts"
		
		# decode, optionally save RNN output
		numBatchElements = len(batch.imgs)
		evalRnnOutput = self.dump or calcProbability
		evalList = [self.decoder] + ([self.ctcIn3dTBC] if evalRnnOutput else [])
		feedDict = {self.inputImgs : batch.imgs, self.seqLen : [Model.maxTextLen] * numBatchElements, self.is_train: False}
		evalRes = self.sess.run(evalList, feedDict)
		decoded = evalRes[0]
		texts = self.decoderOutputToText(decoded, numBatchElements)
		
		# feed RNN output and recognized text into CTC loss to compute labeling probability
		probs = None
		if calcProbability:
			sparse = self.toSparse(batch.gtTexts) if probabilityOfGT else self.toSparse(texts)
			ctcInput = evalRes[1]
			evalList = self.lossPerElement
			feedDict = {self.savedCtcInput : ctcInput, self.gtTexts : sparse, self.seqLen : [Model.maxTextLen] * numBatchElements, self.is_train: False}
			lossVals = self.sess.run(evalList, feedDict)
			probs = np.exp(-lossVals)

		# dump the output of the NN to CSV file(s)
		if self.dump:
			self.dumpNNOutput(evalRes[1])

		return (texts, probs)
	

	def save(self):
		"save model to file"
		self.snapID += 1
		self.saver.save(self.sess, '../model/snapshot', global_step=self.snapID)
 
