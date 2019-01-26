package util;
import java.io.IOException;

import org.apache.log4j.Logger;

import com.aliasi.corpus.ObjectHandler;
import com.aliasi.util.Strings;

import onlineEventdetect.OnlineLatentDirichletAllocation;

public class LdaReportingHandler implements ObjectHandler<OnlineLatentDirichletAllocation.GibbsSample> {
	private Logger logger = Logger.getLogger(LdaReportingHandler.class);
	private final long mStartTime;

	public LdaReportingHandler() {
		mStartTime = System.currentTimeMillis();
	}

	public void handle(OnlineLatentDirichletAllocation.GibbsSample sample) {
		if ((sample.epoch() % 100) == 0) {
			logger.debug(String.format("Epoch=%3d   elapsed time=%s", sample.epoch(),
					Strings.msToString(System.currentTimeMillis() - mStartTime)));
			double corpusLog2Prob = sample.corpusLog2Probability();
			logger.debug(String.format("\tlog2 p(corpus|phi,theta)=%f\ttoken cross-entropy rate=%f", corpusLog2Prob,
					(-corpusLog2Prob / sample.numTokens())));
		}
	}

	public void fullReport(OnlineLatentDirichletAllocation.GibbsSample sample, int maxWordsPerTopic, int maxTopicsPerDoc,
			boolean reportTokens, String outputFile, String type) throws IOException {
	}

	static double binomialZ(double wordCountInDoc, double wordsInDoc, double wordCountinCorpus, double wordsInCorpus) {
		double pCorpus = wordCountinCorpus / wordsInCorpus;
		double var = wordsInCorpus * pCorpus * (1 - pCorpus);
		double dev = Math.sqrt(var);
		double expected = wordsInDoc * pCorpus;
		double z = (wordCountInDoc - expected) / dev;
		return z;
	}

}