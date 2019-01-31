package onlineEventdetect;

import java.io.IOException;
import java.io.ObjectInput;
import java.io.ObjectOutput;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import com.aliasi.cluster.LatentDirichletAllocation;
import com.aliasi.corpus.ObjectHandler;
import com.aliasi.stats.Statistics;
import com.aliasi.symbol.SymbolTable;
import com.aliasi.tokenizer.Tokenizer;
import com.aliasi.tokenizer.TokenizerFactory;
import com.aliasi.util.AbstractExternalizable;
import com.aliasi.util.FeatureExtractor;
import com.aliasi.util.Iterators;
import com.aliasi.util.Math;
import com.aliasi.util.ObjectToCounterMap;
import com.aliasi.util.ObjectToDoubleMap;
import com.aliasi.util.Strings;

public class OnlineLatentDirichletAllocation implements Serializable {

	static final long serialVersionUID = 6313662446710242382L;

	private final double mDocTopicPrior;
	private final double[][] mTopicWordProbs;

	/**
	 * Construct a latent Dirichelt allocation (LDA) model using the specified
	 * document-topic prior and topic-word distributions.
	 * 
	 * <p>
	 * The topic-word probability array <code>topicWordProbs</code> represents a
	 * collection of discrete distributions <code>topicwordProbs[topic]</code> for
	 * topics, and thus must satisfy:
	 *
	 * <blockquote>
	 * 
	 * <pre>
	 * topicWordProbs[topic][word] &gt;= 0.0
	 *
	 * <big><big><big>&Sigma;</big></big></big><sub><sub>word &lt; numWords</sub></sub> topicWordProbs[topic][word] = 1.0
	 * </pre>
	 * 
	 * </blockquote>
	 *
	 * <p>
	 * <b>Warning:</b> These requirements are <b>not</b> checked by the constructor.
	 *
	 * <p>
	 * See the class documentation above for an explanation of the parameters and
	 * what can be done with a model.
	 *
	 * @param docTopicPrior
	 *            The document-topic prior.
	 * @param topicWordProbs
	 *            Array of discrete distributions over words, indexed by topic.
	 * @throws IllegalArgumentException
	 *             If the document-topic prior is not finite and positive, or if the
	 *             topic-word probabilities arrays are not all the same length with
	 *             entries between 0.0 and 1.0 inclusive.
	 */
	public OnlineLatentDirichletAllocation(double docTopicPrior, double[][] topicWordProbs) {

		if (docTopicPrior <= 0.0 || Double.isNaN(docTopicPrior) || Double.isInfinite(docTopicPrior)) {
			String msg = "Document-topic prior must be finite and positive." + " Found docTopicPrior=" + docTopicPrior;
			throw new IllegalArgumentException(msg);
		}
		int numTopics = topicWordProbs.length;
		if (numTopics < 1) {
			String msg = "Require non-empty topic-word probabilities.";
			throw new IllegalArgumentException(msg);
		}

		int numWords = topicWordProbs[0].length;
		for (int topic = 1; topic < numTopics; ++topic) {
			if (topicWordProbs[topic].length != numWords) {
				String msg = "All topics must have the same number of words." + " topicWordProbs[0].length="
						+ topicWordProbs[0].length + " topicWordProbs[" + topic + "]=" + topicWordProbs[topic].length;
				throw new IllegalArgumentException(msg);
			}
		}

		for (int topic = 0; topic < numTopics; ++topic) {
			for (int word = 0; word < numWords; ++word) {
				if (topicWordProbs[topic][word] < 0.0 || topicWordProbs[topic][word] > 1.0) {
					String msg = "All probabilities must be between 0.0 and 1.0" + " Found topicWordProbs[" + topic
							+ "][" + word + "]=" + topicWordProbs[topic][word];
					throw new IllegalArgumentException(msg);
				}
			}
		}

		mDocTopicPrior = docTopicPrior;
		mTopicWordProbs = topicWordProbs;
	}

	/**
	 * Returns the number of topics in this LDA model.
	 *
	 * @return The number of topics in this model.
	 */
	public int numTopics() {
		return mTopicWordProbs.length;
	}

	/**
	 * Returns the number of words on which this LDA model is based.
	 *
	 * @return The numbe of words in this model.
	 */
	public int numWords() {
		return mTopicWordProbs[0].length;
	}

	/**
	 * Returns the concentration value of the uniform Dirichlet prior over topic
	 * distributions for documents. This value is effectively a prior count for
	 * topics used for additive smoothing during estimation.
	 *
	 * @return The prior count of topics in documents.
	 */
	public double documentTopicPrior() {
		return mDocTopicPrior;
	}

	/**
	 * Returns the probability of the specified word in the specified topic. The
	 * values returned should be non-negative and finite, and should sum to 1.0 over
	 * all words for a specifed topic.
	 *
	 * @param topic
	 *            Topic identifier.
	 * @param word
	 *            Word identifier.
	 * @return Probability of the specified word in the specified topic.
	 */
	public double wordProbability(int topic, int word) {
		return mTopicWordProbs[topic][word];
	}

	/**
	 * Returns an array representing of probabilities of words in the specified
	 * topic. The probabilities are indexed by word identifier.
	 *
	 * <p>
	 * The returned result is a copy of the underlying data in the model so that
	 * changing it will not change the model.
	 *
	 * @param topic
	 *            Topic identifier.
	 * @return Array of probabilities of words in the specified topic.
	 */
	public double[] wordProbabilities(int topic) {
		double[] xs = new double[mTopicWordProbs[topic].length];
		for (int i = 0; i < xs.length; ++i)
			xs[i] = mTopicWordProbs[topic][i];
		return xs;
	}

	/**
	 * Returns the specified number of Gibbs samples of topics for the specified
	 * tokens using the specified number of burnin epochs, the specified lag between
	 * samples, and the specified randomizer. The array returned is an array of
	 * samples, each sample consisting of a topic assignment to each token in the
	 * specified list of tokens. The tokens must all be in the appropriate range for
	 * this class
	 *
	 * <p>
	 * See the class documentation for more information on how the samples are
	 * computed.
	 *
	 * @param tokens
	 *            The tokens making up the document.
	 * @param numSamples
	 *            Number of Gibbs samples to return.
	 * @param burnin
	 *            The number of samples to take and throw away during the burnin
	 *            period.
	 * @param sampleLag
	 *            The interval between samples after burnin.
	 * @param random
	 *            The random number generator to use for this sampling process.
	 * @return The selection of topic samples generated by this sampler.
	 * @throws IndexOutOfBoundsException
	 *             If there are tokens whose value is less than zero, or whose value
	 *             is greater than the number of tokens in this model.
	 * @throws IllegalArgumentException
	 *             If the number of samples is not positive, the sample lag is not
	 *             positive, or if the burnin period is negative. if the number of
	 *             samples, burnin, and lag are not positive numbers.
	 */
	public short[][] sampleTopics(int[] tokens, int numSamples, int burnin, int sampleLag, Random random) {

		if (burnin < 0) {
			String msg = "Burnin period must be non-negative." + " Found burnin=" + burnin;
			throw new IllegalArgumentException(msg);
		}

		if (numSamples < 1) {
			String msg = "Number of samples must be at least 1." + " Found numSamples=" + numSamples;
			throw new IllegalArgumentException(msg);
		}

		if (sampleLag < 1) {
			String msg = "Sample lag must be at least 1." + " Found sampleLag=" + sampleLag;
			throw new IllegalArgumentException(msg);
		}

		double docTopicPrior = documentTopicPrior();
		int numTokens = tokens.length;

		int numTopics = numTopics();

		int[] topicCount = new int[numTopics];

		short[][] samples = new short[numSamples][numTokens];
		int sample = 0;
		short[] currentSample = samples[0];
		for (int token = 0; token < numTokens; ++token) {
			int randomTopic = random.nextInt(numTopics);
			++topicCount[randomTopic];
			currentSample[token] = (short) randomTopic;
		}

		double[] topicDistro = new double[numTopics];

		int numEpochs = burnin + sampleLag * (numSamples - 1);
		for (int epoch = 0; epoch < numEpochs; ++epoch) {
			for (int token = 0; token < numTokens; ++token) {
				int word = tokens[token];
				int currentTopic = currentSample[token];
				--topicCount[currentTopic];
				if (topicCount[currentTopic] < 0) {
					throw new IllegalArgumentException("bomb");
				}
				for (int topic = 0; topic < numTopics; ++topic) {
					topicDistro[topic] = (topicCount[topic] + docTopicPrior) * wordProbability(topic, word)
							+ (topic == 0 ? 0.0 : topicDistro[topic - 1]);
				}
				int sampledTopic = Statistics.sample(topicDistro, random);
				++topicCount[sampledTopic];
				currentSample[token] = (short) sampledTopic;
			}
			if ((epoch >= burnin) && (((epoch - burnin) % sampleLag) == 0)) {
				short[] pastSample = currentSample;
				++sample;
				currentSample = samples[sample];
				for (int token = 0; token < numTokens; ++token)
					currentSample[token] = pastSample[token];
			}
		}
		return samples;
	}

	/**
	 * Replaced by method {@code bayesTopicEstimate()} because of original
	 * misnaming.
	 *
	 * <p>
	 * <b>Warning:</b> This is actually <b>not</b> a maximum a posterior (MAP)
	 * estimate as suggested by the name.
	 *
	 * @param tokens
	 *            The tokens making up the document.
	 * @param numSamples
	 *            Number of Gibbs samples to return.
	 * @param burnin
	 *            The number of samples to take and throw away during the burnin
	 *            period.
	 * @param sampleLag
	 *            The interval between samples after burnin.
	 * @param random
	 *            The random number generator to use for this sampling process.
	 * @return The selection of topic samples generated by this sampler.
	 * @throws IndexOutOfBoundsException
	 *             If there are tokens whose value is less than zero, or whose value
	 *             is greater than the number of tokens in this model.
	 * @throws IllegalArgumentException
	 *             If the number of samples is not positive, the sample lag is not
	 *             positive, or if the burnin period is negative.
	 */
	double[] mapTopicEstimate(int[] tokens, int numSamples, int burnin, int sampleLag, Random random) {
		return bayesTopicEstimate(tokens, numSamples, burnin, sampleLag, random);
	}

	/**
	 * Return the Bayesian point estimate of the topic distribution for a document
	 * consisting of the specified tokens, using Gibbs sampling with the specified
	 * parameters. The Gibbs topic samples are simply averaged to produce the
	 * Bayesian estimate, which minimizes expected square loss.
	 *
	 * <p>
	 * See the method {@link #sampleTopics(int[],int,int,int,Random)} and the class
	 * documentation for more information on the sampling procedure.
	 *
	 * @param tokens
	 *            The tokens making up the document.
	 * @param numSamples
	 *            Number of Gibbs samples to return.
	 * @param burnin
	 *            The number of samples to take and throw away during the burnin
	 *            period.
	 * @param sampleLag
	 *            The interval between samples after burnin.
	 * @param random
	 *            The random number generator to use for this sampling process.
	 * @return The selection of topic samples generated by this sampler.
	 * @throws IndexOutOfBoundsException
	 *             If there are tokens whose value is less than zero, or whose value
	 *             is greater than the number of tokens in this model.
	 * @throws IllegalArgumentException
	 *             If the number of samples is not positive, the sample lag is not
	 *             positive, or if the burnin period is negative.
	 */
	public double[] bayesTopicEstimate(int[] tokens, int numSamples, int burnin, int sampleLag, Random random) {
		short[][] sampleTopics = sampleTopics(tokens, numSamples, burnin, sampleLag, random);
		int numTopics = numTopics();
		int[] counts = new int[numTopics];
		for (short[] topics : sampleTopics) {
			for (int tok = 0; tok < topics.length; ++tok)
				++counts[topics[tok]];
		}
		double totalCount = 0;
		for (int topic = 0; topic < numTopics; ++topic)
			totalCount += counts[topic];
		double[] result = new double[numTopics];
		for (int topic = 0; topic < numTopics; ++topic)
			result[topic] = counts[topic] / totalCount;
		return result;

	}

	Object writeReplace() {
		return new Serializer(this);
	}

	/**
	 * Run Gibbs sampling for the specified multinomial data, number of topics,
	 * priors, search parameters, randomization and callback sample handler. Gibbs
	 * sampling provides samples from the posterior distribution of topic
	 * assignments given the corpus and prior hyperparameters. A sample is
	 * encapsulated as an instance of class {@link GibbsSample}. This method will
	 * return the final sample and also send intermediate samples to an optional
	 * handler.
	 *
	 * <p>
	 * The class documentation above explains Gibbs sampling for LDA as used in this
	 * method.
	 *
	 * <p>
	 * The primary input is an array of documents, where each document is
	 * represented as an array of integers representing the tokens that appear in
	 * it. These tokens should be numbered contiguously from 0 for space efficiency.
	 * The topic assignments in the Gibbs sample are aligned as parallel arrays to
	 * the array of documents.
	 *
	 * <p>
	 * The next three parameters are the hyperparameters of the model, specifically
	 * the number of topics, the prior count assigned to topics in a document, and
	 * the prior count assigned to words in topics. A rule of thumb for the
	 * document-topic prior is to set it to 5 divided by the number of topics (or
	 * less if there are very few topics; 0.1 is typically the maximum value used).
	 * A good general value for the topic-word prior is 0.01. Both of these priors
	 * will be diffuse and tend to lead to skewed posterior distributions.
	 *
	 * <p>
	 * The following three parameters specify how the sampling is to be done. First,
	 * the sampler is &quot;burned in&quot; for a number of epochs specified by the
	 * burnin parameter. After burn in, samples are taken after fixed numbers of
	 * documents to avoid correlation in the samples; the sampling frequency is
	 * specified by the sample lag. Finally, the number of samples to be taken is
	 * specified. For instance, if the burnin is 1000, the sample lag is 250, and
	 * the number of samples is 5, then samples are taken after 1000, 1250, 1500,
	 * 1750 and 2000 epochs. If a non-null handler object is specified in the method
	 * call, its <code>handle(GibbsSample)</code> method is called with each the
	 * samples produced as above.
	 *
	 * <p>
	 * The final sample in the chain of samples is returned as the result. Note that
	 * this sample will also have been passed to the specified handler as the last
	 * sample for the handler.
	 *
	 * <p>
	 * A random number generator must be supplied as an argument. This may just be a
	 * new instance of {@link java.util.Random} or a custom extension. It is used
	 * for all randomization in this method.
	 *
	 * @param docWords
	 *            Corpus of documents to be processed.
	 * @param numTopics
	 *            Number of latent topics to generate.
	 * @param docTopicPrior
	 *            Prior count of topics in a document.
	 * @param topicWordPrior
	 *            Prior count of words in a topic.
	 * @param burninEpochs
	 *            Number of epochs to run before taking a sample.
	 * @param sampleLag
	 *            Frequency between samples.
	 * @param numSamples
	 *            Number of samples to take before exiting.
	 * @param random
	 *            Random number generator.
	 * @param handler
	 *            Handler to which the samples are sent.
	 * @return The final Gibbs sample.
	 */
	public static GibbsSample gibbsSampler(int[][] docWords, short numTopics, double docTopicPrior,
			double[][] topicWordPriors, int burninEpochs, int sampleLag, int numSamples,

			Random random,

			ObjectHandler<GibbsSample> handler) {

		// validateInputs(docWords,numTopics,docTopicPrior,topicWordPrior,burninEpochs,sampleLag,numSamples);

		int numDocs = docWords.length;
		int numWords = max(docWords) + 1;

		int numTokens = 0;
		for (int doc = 0; doc < numDocs; ++doc)
			numTokens += docWords[doc].length;

		// should inputs be permuted?
		// for (int doc = 0; doc < numDocs; ++doc)
		// Arrays.permute(docWords[doc]);

		short[][] currentSample = new short[numDocs][];
		for (int doc = 0; doc < numDocs; ++doc)
			currentSample[doc] = new short[docWords[doc].length];

		int[][] docTopicCount = new int[numDocs][numTopics];
		int[][] wordTopicCount = new int[numWords][numTopics];
		int[] topicTotalCount = new int[numTopics];

		for (int doc = 0; doc < numDocs; ++doc) {
			for (int tok = 0; tok < docWords[doc].length; ++tok) {
				int word = docWords[doc][tok];
				int topic = random.nextInt(numTopics);
				currentSample[doc][tok] = (short) topic;
				++docTopicCount[doc][topic];
				++wordTopicCount[word][topic];
				++topicTotalCount[topic];
			}
		}

		double[] numWordsTimesTopicWordPriors = new double[numTopics];
		for (int topic = 0; topic < numTopics; topic++) {
			numWordsTimesTopicWordPriors[topic] = 0;
			for (int tok = 0; tok < numWords; tok++) {
				numWordsTimesTopicWordPriors[topic] += topicWordPriors[topic][tok];
			}
		}
		double[] topicDistro = new double[numTopics];
		int numEpochs = burninEpochs + sampleLag * (numSamples - 1);
		for (int epoch = 0; epoch <= numEpochs; ++epoch) {
			double corpusLog2Prob = 0.0;
			int numChangedTopics = 0;
			for (int doc = 0; doc < numDocs; ++doc) {
				int[] docWordsDoc = docWords[doc];
				short[] currentSampleDoc = currentSample[doc];
				int[] docTopicCountDoc = docTopicCount[doc];
				for (int tok = 0; tok < docWordsDoc.length; ++tok) {
					int word = docWordsDoc[tok];
					int[] wordTopicCountWord = wordTopicCount[word];
					int currentTopic = currentSampleDoc[tok];
					if (currentTopic == 0) {
						topicDistro[0] = (docTopicCountDoc[0] - 1.0 + docTopicPrior)
								* (wordTopicCountWord[0] - 1.0 + topicWordPriors[currentTopic][word])
								/ (topicTotalCount[0] - 1.0 + numWordsTimesTopicWordPriors[currentTopic]);
					} else {
						topicDistro[0] = (docTopicCountDoc[0] + docTopicPrior)
								* (wordTopicCountWord[0] + topicWordPriors[currentTopic][word])
								/ (topicTotalCount[0] + numWordsTimesTopicWordPriors[currentTopic]);
						for (int topic = 1; topic < currentTopic; ++topic) {
							topicDistro[topic] = (docTopicCountDoc[topic] + docTopicPrior)
									* (wordTopicCountWord[topic] + topicWordPriors[currentTopic][word])
									/ (topicTotalCount[topic] + numWordsTimesTopicWordPriors[currentTopic])
									+ topicDistro[topic - 1];
						}
						topicDistro[currentTopic] = (docTopicCountDoc[currentTopic] - 1.0 + docTopicPrior)
								* (wordTopicCountWord[currentTopic] - 1.0 + topicWordPriors[currentTopic][word])
								/ (topicTotalCount[currentTopic] - 1.0 + numWordsTimesTopicWordPriors[currentTopic])
								+ topicDistro[currentTopic - 1];
					}
					for (int topic = currentTopic + 1; topic < numTopics; ++topic) {
						topicDistro[topic] = (docTopicCountDoc[topic] + docTopicPrior)
								* (wordTopicCountWord[topic] + topicWordPriors[currentTopic][word])
								/ (topicTotalCount[topic] + numWordsTimesTopicWordPriors[currentTopic])
								+ topicDistro[topic - 1];
					}
					int sampledTopic = Statistics.sample(topicDistro, random);

					// compute probs before updates
					if (sampledTopic != currentTopic) {
						currentSampleDoc[tok] = (short) sampledTopic;
						--docTopicCountDoc[currentTopic];
						--wordTopicCountWord[currentTopic];
						--topicTotalCount[currentTopic];
						++docTopicCountDoc[sampledTopic];
						++wordTopicCountWord[sampledTopic];
						++topicTotalCount[sampledTopic];
					}

					if (sampledTopic != currentTopic)
						++numChangedTopics;
					double topicProbGivenDoc = docTopicCountDoc[sampledTopic] / (double) docWordsDoc.length;
					double wordProbGivenTopic = wordTopicCountWord[sampledTopic]
							/ (double) topicTotalCount[sampledTopic];
					double tokenLog2Prob = Math.log2(topicProbGivenDoc * wordProbGivenTopic);
					corpusLog2Prob += tokenLog2Prob;
				}
			}
			// double crossEntropyRate = -corpusLog2Prob / numTokens;
			if ((epoch >= burninEpochs) && (((epoch - burninEpochs) % sampleLag) == 0)) {
				GibbsSample sample = new GibbsSample(epoch, currentSample, docWords, docTopicPrior, topicWordPriors,
						docTopicCount, wordTopicCount, topicTotalCount, numChangedTopics, numWords, numTokens);
				if (handler != null)
					handler.handle(sample);
				if (epoch == numEpochs)
					return sample;
			}
		}
		throw new IllegalStateException("unreachable in practice because of return if epoch==numEpochs");
	}

	/**
	 * Tokenize an array of text documents represented as character sequences into a
	 * form usable by LDA, using the specified tokenizer factory and symbol table.
	 * The symbol table should be constructed fresh for this application, but may be
	 * used after this method is called for further token to symbol conversions.
	 * Only tokens whose count is equal to or larger the specified minimum count are
	 * included. Only tokens whose count exceeds the minimum are added to the symbol
	 * table, thus producing a compact set of symbol assignments to tokens for
	 * downstream processing.
	 *
	 * <p>
	 * <i>Warning</i>: With some tokenizer factories and or minimum count
	 * thresholds, there may be documents with no tokens in them.
	 *
	 * @param texts
	 *            The text corpus.
	 * @param tokenizerFactory
	 *            A tokenizer factory for tokenizing the texts.
	 * @param symbolTable
	 *            Symbol table used to convert tokens to identifiers.
	 * @param minCount
	 *            Minimum count for a token to be included in a document's
	 *            representation.
	 * @return The tokenized form of a document suitable for input to LDA.
	 */
	public static int[][] tokenizeDocuments(CharSequence[] texts, TokenizerFactory tokenizerFactory,
			SymbolTable symbolTable, int minCount) {
		ObjectToCounterMap<String> tokenCounter = new ObjectToCounterMap<String>();
		for (CharSequence text : texts) {
			char[] cs = Strings.toCharArray(text);
			Tokenizer tokenizer = tokenizerFactory.tokenizer(cs, 0, cs.length);
			for (String token : tokenizer)
				tokenCounter.increment(token);
		}
		tokenCounter.prune(minCount);
		Set<String> tokenSet = tokenCounter.keySet();
		for (String token : tokenSet)
			symbolTable.getOrAddSymbol(token);

		int[][] docTokenId = new int[texts.length][];
		for (int i = 0; i < docTokenId.length; ++i) {
			docTokenId[i] = tokenizeDocument(texts[i], tokenizerFactory, symbolTable);
		}
		return docTokenId;
	}

	/**
	 * Tokenizes the specified text document using the specified tokenizer factory
	 * returning only tokens that exist in the symbol table. This method is useful
	 * within a given LDA model for tokenizing new documents into lists of words.
	 *
	 * @param text
	 *            Character sequence to tokenize.
	 * @param tokenizerFactory
	 *            Tokenizer factory for tokenization.
	 * @param symbolTable
	 *            Symbol table to use for converting tokens to symbols.
	 * @return The array of integer symbols for tokens that exist in the symbol
	 *         table.
	 */
	public static int[] tokenizeDocument(CharSequence text, TokenizerFactory tokenizerFactory,
			SymbolTable symbolTable) {
		char[] cs = Strings.toCharArray(text);
		Tokenizer tokenizer = tokenizerFactory.tokenizer(cs, 0, cs.length);
		List<Integer> idList = new ArrayList<Integer>();
		for (String token : tokenizer) {
			int id = symbolTable.symbolToID(token);
			if (id >= 0)
				idList.add(id);
		}
		int[] tokenIds = new int[idList.size()];
		for (int i = 0; i < tokenIds.length; ++i)
			tokenIds[i] = idList.get(i);

		return tokenIds;
	}

	static int max(int[][] xs) {
		int max = 0;
		for (int i = 0; i < xs.length; ++i) {
			int[] xsI = xs[i];
			for (int j = 0; j < xsI.length; ++j) {
				if (xsI[j] > max)
					max = xsI[j];
			}
		}
		return max;
	}

	static double relativeDifference(double x, double y) {
		return java.lang.Math.abs(x - y) / (java.lang.Math.abs(x) + java.lang.Math.abs(y));
	}

	static void validateInputs(int[][] docWords, short numTopics, double docTopicPrior, double topicWordPrior,
			int burninEpochs, int sampleLag, int numSamples) {

		validateInputs(docWords, numTopics, docTopicPrior, topicWordPrior);

		if (burninEpochs < 0) {
			String msg = "Number of burnin epochs must be non-negative." + " Found burninEpochs=" + burninEpochs;
			throw new IllegalArgumentException(msg);
		}

		if (sampleLag < 1) {
			String msg = "Sample lag must be positive." + " Found sampleLag=" + sampleLag;
			throw new IllegalArgumentException(msg);
		}

		if (numSamples < 1) {
			String msg = "Number of samples must be positive." + " Found numSamples=" + numSamples;
			throw new IllegalArgumentException(msg);
		}
	}

	static void validateInputs(int[][] docWords, short numTopics, double docTopicPrior, double topicWordPrior) {

		for (int doc = 0; doc < docWords.length; ++doc) {
			for (int tok = 0; tok < docWords[doc].length; ++tok) {
				if (docWords[doc][tok] >= 0)
					continue;
				String msg = "All tokens must have IDs greater than 0." + " Found docWords[" + doc + "][" + tok + "]="
						+ docWords[doc][tok];
				throw new IllegalArgumentException(msg);
			}
		}

		if (numTopics < 1) {
			String msg = "Num topics must be positive." + " Found numTopics=" + numTopics;
			throw new IllegalArgumentException(msg);
		}

		if (Double.isInfinite(docTopicPrior) || Double.isNaN(docTopicPrior) || docTopicPrior < 0.0) {
			String msg = "Document-topic prior must be finite and positive." + " Found docTopicPrior=" + docTopicPrior;
			throw new IllegalArgumentException(msg);
		}

		if (Double.isInfinite(topicWordPrior) || Double.isNaN(topicWordPrior) || topicWordPrior < 0.0) {
			String msg = "Topic-word prior must be finite and positive." + " Found topicWordPrior=" + topicWordPrior;
			throw new IllegalArgumentException(msg);
		}
	}

	/**
	 * The <code>LatentDirichletAllocation.GibbsSample</code> class encapsulates all
	 * of the information related to a single Gibbs sample for latent Dirichlet
	 * allocation (LDA). A sample consists of the assignment of a topic identifier
	 * to each token in the corpus. Other methods in this class are derived from
	 * either the topic samples, the data being estimated, and the LDA parameters
	 * such as priors.
	 *
	 * <p>
	 * Instances of this class are created by the sampling method in the containing
	 * class, {@link LatentDirichletAllocation}. For convenience, the sample
	 * includes all of the data used to construct the sample, as well as the
	 * hyperparameters used for sampling.
	 *
	 * <p>
	 * As described in the class documentation for the containing class
	 * {@link LatentDirichletAllocation}, the primary content in a Gibbs sample for
	 * LDA is the assignment of a single topic to each token in the corpus.
	 * Cumulative counts for topics in documents and words in topics as well as
	 * total counts are also available; they do not entail any additional
	 * computation costs as the sampler maintains them as part of the sample.
	 *
	 * <p>
	 * The sample also contains meta information about the state of the sampling
	 * procedure. The epoch at which the sample was produced is provided, as well as
	 * an indication of how many topic assignments changed between this sample and
	 * the previous sample (note that this is the previous sample in the chain, not
	 * necessarily the previous sample handled by the LDA handler; the handler only
	 * gets the samples separated by the specified lag.
	 *
	 * <p>
	 * The sample may be used to generate an LDA model. The resulting model may then
	 * be used for estimation of unseen documents. Typically, models derived from
	 * several samples are used for Bayesian computations, as described in the class
	 * documentation above.
	 *
	 * @author Bob Carpenter
	 * @version 3.3.0
	 * @since LingPipe3.3
	 */
	public static class GibbsSample {
		private final int mEpoch;
		private final short[][] mTopicSample;
		private final int[][] mDocWords;
		private final double mDocTopicPrior;
		private final double[][] mTopicWordPriors;
		private final int[][] mDocTopicCount;
		private final int[][] mWordTopicCount;
		private final int[] mTopicCount;
		private final int mNumChangedTopics;

		private final int mNumWords;
		private final int mNumTokens;

		GibbsSample(int epoch, short[][] topicSample, int[][] docWords, double docTopicPrior,
				double[][] topicWordPriors, int[][] docTopicCount, int[][] wordTopicCount, int[] topicCount,
				int numChangedTopics, int numWords, int numTokens) {

			mEpoch = epoch;
			mTopicSample = topicSample;
			mDocWords = docWords;
			mDocTopicPrior = docTopicPrior;
			mTopicWordPriors = topicWordPriors;
			mDocTopicCount = docTopicCount;
			mWordTopicCount = wordTopicCount;
			mTopicCount = topicCount;
			mNumChangedTopics = numChangedTopics;
			mNumWords = numWords;
			mNumTokens = numTokens;
		}

		/**
		 * Returns the epoch in which this sample was generated.
		 *
		 * @return The epoch for this sample.
		 */
		public int epoch() {
			return mEpoch;
		}

		/**
		 * Returns the number of documents on which the sample was based.
		 *
		 * @return The number of documents for the sample.
		 */
		public int numDocuments() {
			return mDocWords.length;
		}

		/**
		 * Returns the number of distinct words in the documents on which the sample was
		 * based.
		 *
		 * @return The number of words underlying the model.
		 */
		public int numWords() {
			return mNumWords;
		}

		/**
		 * Returns the number of tokens in documents on which the sample was based. Each
		 * token is an instance of a particular word.
		 */
		public int numTokens() {
			return mNumTokens;
		}

		/**
		 * Returns the number of topics for this sample.
		 *
		 * @return The number of topics for this sample.
		 */
		public int numTopics() {
			return mTopicCount.length;
		}

		/**
		 * Returns the topic identifier sampled for the specified token position in the
		 * specified document.
		 *
		 * @param doc
		 *            Identifier for a document.
		 * @param token
		 *            Token position in the specified document.
		 * @return The topic assigned to the specified token in this sample.
		 * @throws IndexOutOfBoundsException
		 *             If the document identifier is not between 0 (inclusive) and the
		 *             number of documents (exclusive), or if the token is not between 0
		 *             (inclusive) and the number of tokens (exclusive) in the specified
		 *             document.
		 */
		public short topicSample(int doc, int token) {
			return mTopicSample[doc][token];
		}

		/**
		 * Returns the word identifier for the specified token position in the specified
		 * document.
		 *
		 * @param doc
		 *            Identifier for a document.
		 * @param token
		 *            Token position in the specified document.
		 * @return The word found at the specified position in the specified document.
		 * @throws IndexOutOfBoundsException
		 *             If the document identifier is not between 0 (inclusive) and the
		 *             number of documents (exclusive), or if the token is not between 0
		 *             (inclusive) and the number of tokens (exclusive) in the specified
		 *             document.
		 */
		public int word(int doc, int token) {
			return mDocWords[doc][token];
		}

		/**
		 * Returns the uniform Dirichlet concentration hyperparameter
		 * <code>&alpha;</code> for document distributions over topics from which this
		 * sample was produced.
		 *
		 * @return The document-topic prior.
		 */
		public double documentTopicPrior() {
			return mDocTopicPrior;
		}

		/**
		 * Returns the uniform Dirichlet concentration hyperparameter
		 * <code>&beta;</code> for topic distributions over words from which this sample
		 * was produced.
		 */
		public double[][] topicWordPriors() {
			return mTopicWordPriors;
		}

		/**
		 * Returns the number of times the specified topic was assigned to the specified
		 * document in this sample.
		 *
		 * @param doc
		 *            Identifier for a document.
		 * @param topic
		 *            Identifier for a topic.
		 * @return The count of the topic in the document in this sample.
		 * @throws IndexOutOfBoundsException
		 *             If the document identifier is not between 0 (inclusive) and the
		 *             number of documents (exclusive) or if the topic identifier is not
		 *             between 0 (inclusive) and the number of topics (exclusive).
		 */
		public int documentTopicCount(int doc, int topic) {
			return mDocTopicCount[doc][topic];
		}

		/**
		 * Returns the length of the specified document in tokens.
		 *
		 * @param doc
		 *            Identifier for a document.
		 * @return The length of the specified document in tokens.
		 * @throws IndexOutOfBoundsException
		 *             If the document identifier is not between 0 (inclusive) and the
		 *             number of documents (exclusive).
		 */
		public int documentLength(int doc) {
			return mDocWords[doc].length;
		}

		/**
		 * Returns the number of times tokens for the specified word were assigned to
		 * the specified topic.
		 *
		 * @param topic
		 *            Identifier for a topic.
		 * @param word
		 *            Identifier for a word.
		 * @return The number of tokens of the specified word assigned to the specified
		 *         topic.
		 * @throws IndexOutOfBoundsException
		 *             If the specified topic is not between 0 (inclusive) and the
		 *             number of topics (exclusive), or if the word is not between 0
		 *             (inclusive) and the number of words (exclusive).
		 */
		public int topicWordCount(int topic, int word) {
			return mWordTopicCount[word][topic];
		}

		/**
		 * Returns the total number of tokens assigned to the specified topic in this
		 * sample.
		 *
		 * @param topic
		 *            Identifier for a topic.
		 * @return The total number of tokens assigned to the specified topic.
		 * @throws IllegalArgumentException
		 *             If the specified topic is not between 0 (inclusive) and the
		 *             number of topics (exclusive).
		 */
		public int topicCount(int topic) {
			return mTopicCount[topic];
		}

		/**
		 * Returns the total number of topic assignments to tokens that changed between
		 * the last sample and this one. Note that this is the last sample in the chain,
		 * not the last sample necessarily passed to a handler, because handlers may not
		 * be configured to handle every * sample.
		 *
		 * @return The number of topics assignments that changed in this sample relative
		 *         to the previous sample.
		 */
		public int numChangedTopics() {
			return mNumChangedTopics;
		}

		/**
		 * Returns the probability estimate for the specified word in the specified
		 * topic in this sample. This value is calculated as a maximum a posteriori
		 * estimate computed as described in the class documentation for
		 * {@link LatentDirichletAllocation} using the topic assignment counts in this
		 * sample and the topic-word prior.
		 *
		 * @param topic
		 *            Identifier for a topic.
		 * @param word
		 *            Identifier for a word.
		 * @return The probability of generating the specified word in the specified
		 *         topic.
		 * @throws IndexOutOfBoundsException
		 *             If the specified topic is not between 0 (inclusive) and the
		 *             number of topics (exclusive), or if the word is not between 0
		 *             (inclusive) and the number of words (exclusive).
		 */
		public double topicWordProb(int topic, int word) {
//			double topicWordPrior = 0;
//			for (int tok = 0; tok < numWords(); tok++) {
//				topicWordPrior += topicWordPriors()[topic][tok];
//			}
			// return (topicWordCount(topic, word) + topicWordPriors()[topic][word])
			// / (topicCount(topic) + topicWordPrior);
			double v1 = topicWordCount(topic, word);
			double v2 = topicCount(topic);
			return v1 / v2;
		}

		/**
		 * Returns the number of times tokens of the specified word appeared in the
		 * corpus.
		 *
		 * @param word
		 *            Identifier of a word.
		 * @return The number of tokens of the word in the corpus.
		 * @throws IndexOutOfBoundsException
		 *             If the word identifier is not between 0 (inclusive) and the
		 *             number of words (exclusive).
		 */
		public int wordCount(int word) {
			int count = 0;
			for (int topic = 0; topic < numTopics(); ++topic)
				count += topicWordCount(topic, word);
			return count;
		}

		/**
		 * Returns the estimate of the probability of the topic being assigned to a word
		 * in the specified document given the topic * assignments in this sample. This
		 * is the maximum a posteriori estimate computed from the topic assignments * as
		 * described in the class documentation for {@link LatentDirichletAllocation}
		 * using the topic assignment counts in this sample and the document-topic
		 * prior.
		 *
		 * @param doc
		 *            Identifier of a document.
		 * @param topic
		 *            Identifier for a topic.
		 * @return An estimate of the probabilty of the topic in the document.
		 * @throws IndexOutOfBoundsException
		 *             If the document identifier is not between 0 (inclusive) and the
		 *             number of documents (exclusive) or if the topic identifier is not
		 *             between 0 (inclusive) and the number of topics (exclusive).
		 */
		public double documentTopicProb(int doc, int topic) {
			return (documentTopicCount(doc, topic) + documentTopicPrior())
					/ (documentLength(doc) + numTopics() * documentTopicPrior());
		}

		/**
		 * Returns an estimate of the log (base 2) likelihood of the corpus given the
		 * point estimates of topic and document multinomials determined from this
		 * sample.
		 *
		 * <p>
		 * This likelihood calculation uses the methods
		 * {@link #documentTopicProb(int,int)} and {@link #topicWordProb(int,int)} for
		 * estimating likelihoods according the following formula:
		 *
		 * <blockquote>
		 * 
		 * <pre>
		 * corpusLog2Probability()
		 * = <big><big><big>&Sigma;</big></big></big><sub><sub>doc,i</sub></sub> log<sub><sub>2</sub></sub> <big><big><big>&Sigma;</big></big></big><sub><sub>topic</sub></sub> p(topic|doc) * p(word[doc][i]|topic)
		 * </pre>
		 * 
		 * </blockquote>
		 *
		 * <p>
		 * Note that this is <i>not</i> the complete corpus likelihood, which requires
		 * integrating over possible topic and document multinomials given the priors.
		 *
		 * @return The log (base 2) likelihood of the training corpus * given the
		 *         document and topic multinomials determined by this sample.
		 */
		public double corpusLog2Probability() {
			double corpusLog2Prob = 0.0;
			int numDocs = numDocuments();
			int numTopics = numTopics();
			for (int doc = 0; doc < numDocs; ++doc) {
				int docLength = documentLength(doc);
				for (int token = 0; token < docLength; ++token) {
					int word = word(doc, token);
					double wordProb = 0.0;
					for (int topic = 0; topic < numTopics; ++topic) {
						double p1 = topicWordProb(topic, word);
						double p2 = documentTopicProb(doc, topic);
						double wordTopicProbGivenDoc = p1 * p2;
						wordProb += wordTopicProbGivenDoc;
					}
					corpusLog2Prob += Math.log2(wordProb);
				}
			}
			return corpusLog2Prob;
		}

		/**
		 * Returns a latent Dirichlet allocation model corresponding to this sample. The
		 * topic-word probabilities are calculated according to
		 * {@link #topicWordProb(int,int)}, and the document-topic prior is as specified
		 * in the call to LDA that produced this sample.
		 *
		 * @return The LDA model for this sample.
		 */
		public OnlineLatentDirichletAllocation lda() {
			int numTopics = numTopics();
			int numWords = numWords();
			double[][] topicWordPrior = topicWordPriors();
			double[][] topicWordProbs = new double[numTopics][numWords];
			for (int topic = 0; topic < numTopics; ++topic) {
				double topicCount = topicCount(topic);
				double denominator = topicCount;
				for (int word = 0; word < numWords; ++word) {
					denominator += topicWordPrior[topic][word];
				}

				for (int word = 0; word < numWords; ++word)
					topicWordProbs[topic][word] = (topicWordCount(topic, word) + topicWordPrior[topic][word])
							/ denominator;
			}
			return new OnlineLatentDirichletAllocation(mDocTopicPrior, topicWordProbs);
		}
	}

	/**
	 * Return a feature extractor for character sequences based on tokenizing with
	 * the specified tokenizer and generating expected values for topics given the
	 * words.
	 *
	 * <p>
	 * The symbol table is used to convert the tokens to word identifiers used for
	 * this LDA model. Symbols not in the symbol table or with values out of range
	 * of the word probabilities are ignored.
	 * 
	 * <p>
	 * The feature names are determined by prefixing the specified string to the
	 * topic number (starting from zero).
	 * 
	 * <p>
	 * In order to maintain some degree of efficiency, the feature extraction
	 * process estimates topic expectations for each feature independently and then
	 * sums them, then normalizes the result so the sum of the values of all
	 * features generated is 1.
	 *
	 * <p>
	 * Given a uniform distribution over topics, the probability of a topic given a
	 * word may be calculated up to a normalizing constant using Bayes's law,
	 *
	 * <blockquote>
	 * 
	 * <pre>
	 * p(topic|word) = p(word|topic) * p(topic) / p(word)
	 *               &prop; p(word|topic) * p(topic)
	 *               = p(word|topic)
	 * </pre>
	 * 
	 * </blockquote>
	 *
	 * Given the finite number of topics, it is easy to normalize the distribution
	 * and compute <code>p(topic|word)</code> from <code>p(word|topic)</code>. The
	 * values in <code>p(topic|word)</code> are precompiled at construction time.
	 *
	 * <p>
	 * The prior document topic count from this LDA model is added to the expected
	 * counts for each topic before normalization.
	 *
	 * <p>
	 * The expectations for topics for each word are then summed, the resulting
	 * feature vector is normalized to sum to 1, and then it is returned.
	 * 
	 * <p>
	 * Thus the value of each feature is proportional to the expected number of
	 * words in that topic plus the document-topic prior, divided by the total
	 * number of words.
	 *
	 * <h3>Serialization</h3>
	 *
	 * The resulting feature extractor may be serialized if its component tokenizer
	 * factory and symbol table are serializable.
	 *
	 * @param tokenizerFactory
	 *            Tokenizer factory to use for token extraction.
	 * @param symbolTable
	 *            Symbol table mapping tokens to word identifiers.
	 * @param featurePrefix
	 *            Prefix of the resulting feature names.
	 * @return A feature extractor over character sequences based on this LDA model.
	 */
	FeatureExtractor<CharSequence> expectedTopicFeatureExtractor(TokenizerFactory tokenizerFactory,
			SymbolTable symbolTable, String featurePrefix) {
		return new ExpectedTopicFeatureExtractor(this, tokenizerFactory, symbolTable, featurePrefix);
	}

	/**
	 * Return a feature extractor for character sequences based on this LDA model
	 * and the specified tokenizer factory and symbol tale, which computes the
	 * unbiased Bayesian least squares estimate for the character sequence's topic
	 * distribution.
	 *
	 * <p>
	 * The method {@link #bayesTopicEstimate(int[],int,int,int,Random)} is used to
	 * compute the topic distribution for the features.
	 *
	 * <p>
	 * The feature names are determined by prefixing the specified string to the
	 * topic number (starting from zero).
	 *
	 * <h3>Serialization</h3>
	 *
	 * The resulting feature extractor may be serialized if its component tokenizer
	 * factory and symbol table are serializable.
	 * 
	 * @param tokenizerFactory
	 *            Tokenizer for character sequences.
	 * @param symbolTable
	 *            Symbol table for mapping tokens to dimensions.
	 * @param burnIn
	 *            Number of initial Gibbs samples to dispose.
	 * @param sampleLag
	 *            The inerval between saved samples after burnin.
	 * @param numSamples
	 *            Number of Gibbs samples to return.
	 * @return Feature extractor with the specified parameters.
	 * @throws IllegalArgumentException
	 *             If the number of samples is not positive, the sample lag is not
	 *             positive, or if the burnin period is negative.
	 */
	FeatureExtractor<CharSequence> bayesTopicFeatureExtractor(TokenizerFactory tokenizerFactory,
			SymbolTable symbolTable, String featurePrefix, int burnIn, int sampleLag, int numSamples) {
		return new BayesTopicFeatureExtractor(this, tokenizerFactory, symbolTable, featurePrefix, burnIn, sampleLag,
				numSamples);
	}

	static String[] genFeatures(String prefix, int numTopics) {
		String[] features = new String[numTopics];
		for (int k = 0; k < numTopics; ++k)
			features[k] = prefix + k;
		return features;
	}

	static class BayesTopicFeatureExtractor implements FeatureExtractor<CharSequence>, Serializable {

		static final long serialVersionUID = 8883227852502200365L;

		private final OnlineLatentDirichletAllocation mLda;
		private final TokenizerFactory mTokenizerFactory;
		private final SymbolTable mSymbolTable;
		private final String[] mFeatures;

		private final int mBurnin;
		private final int mSampleLag;
		private final int mNumSamples;

		public BayesTopicFeatureExtractor(OnlineLatentDirichletAllocation lda, TokenizerFactory tokenizerFactory,
				SymbolTable symbolTable, String featurePrefix, int burnin, int sampleLag, int numSamples) {
			this(lda, tokenizerFactory, symbolTable, genFeatures(featurePrefix, lda.numTopics()), burnin, sampleLag,
					numSamples);
		}

		BayesTopicFeatureExtractor(OnlineLatentDirichletAllocation lda, TokenizerFactory tokenizerFactory,
				SymbolTable symbolTable, String[] features, int burnin, int sampleLag, int numSamples) {
			mLda = lda;
			mTokenizerFactory = tokenizerFactory;
			mSymbolTable = symbolTable;
			mFeatures = features;
			mBurnin = burnin;
			mSampleLag = sampleLag;
			mNumSamples = numSamples;
		}

		public Map<String, Double> features(CharSequence cSeq) {
			int numTopics = mLda.numTopics();
			char[] cs = Strings.toCharArray(cSeq);
			Tokenizer tokenizer = mTokenizerFactory.tokenizer(cs, 0, cs.length);
			List<Integer> tokenIdList = new ArrayList<Integer>();
			for (String token : tokenizer) {
				int symbol = mSymbolTable.symbolToID(token);
				if (symbol < 0 || symbol >= mLda.numWords())
					continue;
				tokenIdList.add(symbol);
			}
			int[] tokens = new int[tokenIdList.size()];
			for (int i = 0; i < tokenIdList.size(); ++i)
				tokens[i] = tokenIdList.get(i).intValue();

			double[] vals = mLda.mapTopicEstimate(tokens, mNumSamples, mBurnin, mSampleLag, new Random());
			ObjectToDoubleMap<String> features = new ObjectToDoubleMap<String>((numTopics * 3) / 2);
			for (int k = 0; k < numTopics; ++k) {
				if (vals[k] > 0.0)
					features.set(mFeatures[k], vals[k]);
			}
			return features;
		}

		Object writeReplace() {
			return new Serializer(this);
		}

		static class Serializer extends AbstractExternalizable {
			static final long serialVersionUID = 6719636683732661958L;
			final BayesTopicFeatureExtractor mFeatureExtractor;

			public Serializer() {
				this(null);
			}

			Serializer(BayesTopicFeatureExtractor featureExtractor) {
				mFeatureExtractor = featureExtractor;
			}

			public void writeExternal(ObjectOutput out) throws IOException {
				out.writeObject(mFeatureExtractor.mLda);
				out.writeObject(mFeatureExtractor.mTokenizerFactory);
				out.writeObject(mFeatureExtractor.mSymbolTable);
				writeUTFs(mFeatureExtractor.mFeatures, out);
				out.writeInt(mFeatureExtractor.mBurnin);
				out.writeInt(mFeatureExtractor.mSampleLag);
				out.writeInt(mFeatureExtractor.mNumSamples);
			}

			public Object read(ObjectInput in) throws IOException, ClassNotFoundException {
				@SuppressWarnings("unchecked")
				OnlineLatentDirichletAllocation lda = (OnlineLatentDirichletAllocation) in.readObject();
				@SuppressWarnings("unchecked")
				TokenizerFactory tokenizerFactory = (TokenizerFactory) in.readObject();
				@SuppressWarnings("unchecked")
				SymbolTable symbolTable = (SymbolTable) in.readObject();
				String[] features = readUTFs(in);
				int burnIn = in.readInt();
				int sampleLag = in.readInt();
				int numSamples = in.readInt();
				return new BayesTopicFeatureExtractor(lda, tokenizerFactory, symbolTable, features, burnIn, sampleLag,
						numSamples);
			}
		}
	}

	static class ExpectedTopicFeatureExtractor implements FeatureExtractor<CharSequence>, Serializable {

		static final long serialVersionUID = -7996546432550775177L;
		private final double[][] mWordTopicProbs;
		private final double mDocTopicPrior;
		private final TokenizerFactory mTokenizerFactory;
		private final SymbolTable mSymbolTable;
		private final String[] mFeatures;

		public ExpectedTopicFeatureExtractor(OnlineLatentDirichletAllocation lda, TokenizerFactory tokenizerFactory,
				SymbolTable symbolTable, String featurePrefix) {
			double[][] wordTopicProbs = new double[lda.numWords()][lda.numTopics()];
			for (int word = 0; word < lda.numWords(); ++word)
				for (int topic = 0; topic < lda.numTopics(); ++topic)
					wordTopicProbs[word][topic] = lda.wordProbability(topic, word);
			for (double[] topicProbs : wordTopicProbs) {
				double sum = com.aliasi.util.Math.sum(topicProbs);
				for (int k = 0; k < topicProbs.length; ++k)
					topicProbs[k] /= sum;
			}
			mWordTopicProbs = wordTopicProbs;
			mDocTopicPrior = lda.documentTopicPrior();
			mTokenizerFactory = tokenizerFactory;
			mSymbolTable = symbolTable;

			mFeatures = genFeatures(featurePrefix, lda.numTopics());
		}

		ExpectedTopicFeatureExtractor(double docTopicPrior, double[][] wordTopicProbs,
				TokenizerFactory tokenizerFactory, SymbolTable symbolTable, String[] features) {
			mWordTopicProbs = wordTopicProbs;
			mDocTopicPrior = docTopicPrior;
			mTokenizerFactory = tokenizerFactory;
			mSymbolTable = symbolTable;
			mFeatures = features;
		}

		public Map<String, Double> features(CharSequence cSeq) {
			int numTopics = mWordTopicProbs[0].length;
			char[] cs = Strings.toCharArray(cSeq);
			Tokenizer tokenizer = mTokenizerFactory.tokenizer(cs, 0, cs.length);
			double[] vals = new double[numTopics];
			Arrays.fill(vals, mDocTopicPrior);
			for (String token : tokenizer) {
				int symbol = mSymbolTable.symbolToID(token);
				if (symbol < 0 || symbol >= mWordTopicProbs.length)
					continue;
				for (int k = 0; k < numTopics; ++k)
					vals[k] += mWordTopicProbs[symbol][k];
			}

			ObjectToDoubleMap<String> featMap = new ObjectToDoubleMap<String>((numTopics * 3) / 2);
			double sum = com.aliasi.util.Math.sum(vals);
			for (int k = 0; k < numTopics; ++k)
				if (vals[k] > 0.0)
					featMap.set(mFeatures[k], vals[k] / sum);
			return featMap;
		}

		Object writeReplace() {
			return new Serializer(this);
		}

		static class Serializer extends AbstractExternalizable {
			static final long serialVersionUID = -1472744781627035426L;
			final ExpectedTopicFeatureExtractor mFeatures;

			public Serializer() {
				this(null);
			}

			public Serializer(ExpectedTopicFeatureExtractor features) {
				mFeatures = features;
			}

			public void writeExternal(ObjectOutput out) throws IOException {
				out.writeDouble(mFeatures.mDocTopicPrior);
				out.writeInt(mFeatures.mWordTopicProbs.length);
				for (int w = 0; w < mFeatures.mWordTopicProbs.length; ++w)
					writeDoubles(mFeatures.mWordTopicProbs[w], out);
				out.writeObject(mFeatures.mTokenizerFactory);
				out.writeObject(mFeatures.mSymbolTable);
				writeUTFs(mFeatures.mFeatures, out);
			}

			public Object read(ObjectInput in) throws IOException, ClassNotFoundException {
				double docTopicPrior = in.readDouble();
				int numWords = in.readInt();
				double[][] wordTopicProbs = new double[numWords][];
				for (int w = 0; w < numWords; ++w)
					wordTopicProbs[w] = readDoubles(in);
				@SuppressWarnings("unchecked")
				TokenizerFactory tokenizerFactory = (TokenizerFactory) in.readObject();
				@SuppressWarnings("unchecked")
				SymbolTable symbolTable = (SymbolTable) in.readObject();
				String[] features = readUTFs(in);
				return new ExpectedTopicFeatureExtractor(docTopicPrior, wordTopicProbs, tokenizerFactory, symbolTable,
						features);
			}
		}
	}

	static class Serializer extends AbstractExternalizable {
		static final long serialVersionUID = 4725870665020270825L;
		final OnlineLatentDirichletAllocation mLda;

		public Serializer() {
			this(null);
		}

		public Serializer(OnlineLatentDirichletAllocation lda) {
			mLda = lda;
		}

		public Object read(ObjectInput in) throws IOException {
			double docTopicPrior = in.readDouble();
			int numTopics = in.readInt();
			double[][] topicWordProbs = new double[numTopics][];
			for (int i = 0; i < topicWordProbs.length; ++i)
				topicWordProbs[i] = readDoubles(in);
			return new LatentDirichletAllocation(docTopicPrior, topicWordProbs);
		}

		public void writeExternal(ObjectOutput out) throws IOException {
			out.writeDouble(mLda.mDocTopicPrior);
			out.writeInt(mLda.mTopicWordProbs.length);
			for (double[] topicWordProbs : mLda.mTopicWordProbs)
				writeDoubles(topicWordProbs, out);
		}
	}

}
