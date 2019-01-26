package onlineEventdetect;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import com.aliasi.symbol.SymbolTable;

import onlineEventdetect.OnlineLatentDirichletAllocation.GibbsSample;
import onlineEventdetect.bean.Document;
import onlineEventdetect.bean.TimeSlice;
import tokenizer.TokenzierUtils;
import util.NLPContants;

/**
 * @ClassName: TimeSliceGenerator
 * @Description:文本时间片生成器
 * @author: 龚帅宾
 * @date: 2019年1月24日 上午10:43:03
 *
 */
public class TimeSliceGenerator {
	private static int currentNum = 0;
	private static TimeSlice timeSlice;

	/**
	 * 根据新闻文档生成时间片
	 * 
	 * @param newRawList
	 * @return
	 */
	public static TimeSlice generateTimeSlice(String newsRaw, SymbolTable mTokenIndexMap) {
		String[] splits = newsRaw.split(NLPContants.TAB);
		String id = splits[0];
		String title = splits[1];
		String content = splits[2];
		if (0 == currentNum) {
			timeSlice = new TimeSlice(id);
		}
		if (currentNum < NLPContants.SLICE_MAX) {
			Document doc = new Document(id, title + "," + content, timeSlice.getTokenIndexInSlice(),
					TokenzierUtils.getStopTokenizerFactory());
			timeSlice.getDocumentList().add(doc);
			currentNum++;
		} else {
			currentNum = 0;
			return timeSlice;
		}

		return null;
	}

	public static int[][] transformDocuments(TimeSlice timeSlice) {
		int[][] docWords = new int[NLPContants.SLICE_MAX][];
		List<Document> list = timeSlice.getDocumentList();
		for (int i = 0; i < NLPContants.SLICE_MAX; i++) {
			Document document = list.get(i);
			int[] tmp = document.getDocmentContentIdx();
			docWords[i] = tmp;
		}
		return docWords;
	}

	/**
	 * 生产Φ,主题下面各个词的分布
	 * 
	 * @throws Exception
	 */
	public static List<Map<Integer, Double>> generateTopicWordProbInTimeSlice(GibbsSample sample, TimeSlice timeSlice) {
		int topicNum = sample.numTopics();
		SymbolTable sliceSymble = timeSlice.getTokenIndexInSlice();
		List<Map<Integer, Double>> allTopicWordProbs = new ArrayList<Map<Integer, Double>>();
		int sliceWordCount = sliceSymble.numSymbols();
		for (int topic = 0; topic < topicNum; topic++) {
			Map<Integer, Double> topicWordProbs = new HashMap<Integer, Double>();
			for (int wordIndex = 0; wordIndex < sliceWordCount; wordIndex++) {
				double prob = sample.topicWordProb(topic, wordIndex);// 获得topic下词的概率
				String word = sliceSymble.idToSymbol(wordIndex);// 根据时间片的临时索引获得在时间片中词
				int globalWordIndex = NLPContants.GLOBAL_WORD_INDEX.symbolToID(word);// 获得全局索引
				if (-1 == globalWordIndex) {
					System.err.println(String.format("globalword can not find %s", word));
					prob = 0;
				}
				topicWordProbs.put(globalWordIndex, prob);
			}
			allTopicWordProbs.add(topicWordProbs);
		}
		return allTopicWordProbs;
	}

}
