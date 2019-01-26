package tokenizer;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

import org.ansj.domain.Result;
import org.ansj.domain.Term;
import org.ansj.library.DicLibrary;
import org.ansj.splitWord.analysis.ToAnalysis;
import org.apache.commons.lang3.exception.ExceptionUtils;
import org.nlpcn.commons.lang.tire.domain.Forest;

import com.aliasi.tokenizer.Tokenizer;
import com.aliasi.tokenizer.TokenizerFactory;

import util.NLPContants;
import util.RegexUtils;

public class AnsjSelfDicTokenizerFactory implements Serializable, TokenizerFactory {
	private static final long serialVersionUID = 572943028477125945L;
	private static String md5path = null;
	private static ToAnalysis toAnalysis = null;

	public void insertWord(String word) throws Exception {
		synchronized (DicLibrary.class) {
			DicLibrary.insert(md5path, word);
		}
	}

	public static void setDicLibrary(String path) throws Exception {
	}

	private AnsjSelfDicTokenizerFactory() {
		String path = NLPContants.BASE_DIR + "dictionary.dic";
		toAnalysis = new ToAnalysis();
		try {
			md5path = RegexUtils.md5Encode(path);
			DicLibrary.putIfAbsent(md5path, path);
			Forest forest = DicLibrary.get(md5path);
			if (forest != null) {
				toAnalysis.setForests(forest);
			}
		} catch (Exception e) {
			System.err.println(ExceptionUtils.getStackTrace(e));
		}
	}

	private static volatile AnsjSelfDicTokenizerFactory instance;

	@Override
	public Tokenizer tokenizer(char[] ch, int start, int length) {
		return new AnsjTokenizer(ch, start, length);
	}

	public static AnsjSelfDicTokenizerFactory getIstance() {
		if (instance == null) {
			synchronized (AnsjSelfDicTokenizerFactory.class) {
				if (instance == null) {
					instance = new AnsjSelfDicTokenizerFactory();
				}
			}
		}
		return instance;
	}

	class AnsjTokenizer extends Tokenizer {
		private List<Term> parse = new ArrayList<Term>();
		private int currentPos = -1;
		StringBuffer sb = new StringBuffer();

		public AnsjTokenizer(char[] ch, int start, int length) {
			String text = String.valueOf(ch);
			Result result = null;
			try {
				synchronized (DicLibrary.class) {
					result = toAnalysis.parseStr(text);
				}

			} catch (Exception e) {
				e.printStackTrace();
			}
			parse = result.getTerms();
		}

		@Override
		public String nextToken() {
			if (parse == null || currentPos >= parse.size() - 1)
				return null;
			else {
				currentPos++;
				Term term = parse.get(currentPos);
				String result = String.format("%s/%s/%s", term.getName(), term.getNatureStr(), term.getOffe());
				return result;
			}
		}

	}
}
