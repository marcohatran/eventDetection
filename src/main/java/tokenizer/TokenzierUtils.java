package tokenizer;

import com.aliasi.tokenizer.TokenizerFactory;

public class TokenzierUtils {
	public static TokenizerFactory getAnsjTokenzierFactory() {
		TokenizerFactory tokenizerFactory = AnsjSelfDicTokenizerFactory
				.getIstance();
		return tokenizerFactory;
	}

	public static TokenizerFactory getStopTokenizerFactory() {
		TokenizerFactory stopFactory = new StopWordTokenierFactory(getAnsjTokenzierFactory());
		return stopFactory;
	}

	public static TokenizerFactory getNGramTokenizerFactory(int count) {
		NGramTokenizerBasedOtherTokenizerFactory factory = new NGramTokenizerBasedOtherTokenizerFactory(
				getStopTokenizerFactory(), count, count);
		return factory;
	}
	
	public static TokenizerFactory getStopNatureStopWordFactory() {
		TokenizerFactory stopFactory = new StopWordTokenierFactory(getAnsjTokenzierFactory());
		stopFactory = new StopNatureTokenizerFactory(stopFactory);
		return stopFactory;
	}
}
