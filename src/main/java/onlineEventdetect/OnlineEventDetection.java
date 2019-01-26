package onlineEventdetect;

import java.io.IOException;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Random;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingDeque;

import com.aliasi.symbol.SymbolTable;
import com.aliasi.util.ObjectToDoubleMap;

import onlineEventdetect.bean.Document;
import onlineEventdetect.bean.TimeSlice;
import util.LdaReportingHandler;
import util.NLPContants;
import util.NewsProducers;

public class OnlineEventDetection extends Thread {
	private int topicNum;
	private LinkedList<TimeSlice> timeSliceQueue;
	private LinkedList<double[]> distanceMatrix;
	private SymbolTable mTokenIndexMap;// total token index
	private BlockingQueue<String> cacheQueue;// 新闻内存缓冲区

	public OnlineEventDetection(int i, BlockingQueue<String> queue) {
		super();
		topicNum = i;
		timeSliceQueue = new LinkedList<TimeSlice>();
		distanceMatrix = new LinkedList<double[]>();
		this.cacheQueue = queue;
	}

	public int[][] transform(TimeSlice slice) {
		int[][] docWords = new int[NLPContants.SLICE_MAX][];
		List<Document> list = slice.getDocumentList();
		for (int i = 0; i < NLPContants.SLICE_MAX; i++) {
			Document document = list.get(i);
			int[] tmp = document.getDocmentContentIdx();
			docWords[i] = tmp;
		}
		return docWords;
	}

	@Override
	public void run() {
		LdaReportingHandler handler = new LdaReportingHandler();
		while (true) { 
			String getNewsString = null;
			try {
				getNewsString = cacheQueue.take();
				if (null == getNewsString) {
					continue;
				}
				TimeSlice generateSlice = TimeSliceGenerator.generateTimeSlice(getNewsString, mTokenIndexMap);
				if (null == generateSlice) {
					continue;
				}
				SymbolTable tokenSliceIndex = generateSlice.getTokenIndexInSlice();// 获得该时间片的词的索引
				if (timeSliceQueue.size() == 0) {
					// 首次运行
					int[][] docWords = TimeSliceGenerator.transformDocuments(generateSlice);
					int wordNum = tokenSliceIndex.numSymbols();
					double[][] topicWordPriors = new double[topicNum][wordNum];
					for (int topic = 0; topic < topicNum; topic++) {
						for (int tok = 0; tok < wordNum; tok++) {
							topicWordPriors[topic][tok] = 0.1;
						}
					}
					OnlineLatentDirichletAllocation.GibbsSample gibbsSample = OnlineLatentDirichletAllocation
							.gibbsSampler(docWords, (short)topicNum, 0.1, topicWordPriors, 0, 1, 1000, new Random(1l), handler);
					TimeSliceGenerator.generateTopicWordProbInTimeSlice(gibbsSample, generateSlice);
					timeSliceQueue.offer(generateSlice);

				} else if (timeSliceQueue.size() < NLPContants.QUEUE_MAX) {

					int[][] docWords = TimeSliceGenerator.transformDocuments(generateSlice);
					int wordNum = tokenSliceIndex.numSymbols();
					double[][] topicWordPriors = new double[topicNum][wordNum];
					for (int topic = 0; topic < topicNum; topic++) {
						for (int tok = 0; tok < wordNum; tok++) {
							topicWordPriors[topic][tok] = 0.1;
						}
					}

					// LDA的Gibbs采样
					OnlineLatentDirichletAllocation.GibbsSample gibbsSample = OnlineLatentDirichletAllocation
							.gibbsSampler(docWords, (short)topicNum, 0.1, topicWordPriors, 0, 1, 1000, new Random(1l), handler);

					List<Map<Integer, Double>> allTopicWordProbs = TimeSliceGenerator
							.generateTopicWordProbInTimeSlice(gibbsSample, generateSlice);
					generateSlice.setTopicWordProb(allTopicWordProbs);
					// 计算该时间片与上一时间片的对称kl散度距离
					double[] distance = calDistanceBetweenTopics(timeSliceQueue.getLast(), generateSlice);
					distanceMatrix.offer(distance);
					timeSliceQueue.offer(generateSlice);
					int hotTopicId = getTopicIndexByPercByLocal(distance);
					if (hotTopicId != -1) {
						System.out.println(hotTopicId);
					}

				} else if (timeSliceQueue.size() > NLPContants.QUEUE_MAX) {

					// 根据线性时间片的衰减权值计算主题词的先验参数
					double[][] topicWordPriors = calTopicWordPriors(timeSliceQueue, generateSlice);

					// 江Document文档对象转化成LDA可训练的输入模式，词的索引为该时间片的局部索引
					int[][] docWords = TimeSliceGenerator.transformDocuments(generateSlice);

					// LDA的Gibbs采样
					OnlineLatentDirichletAllocation.GibbsSample gibbsSample = OnlineLatentDirichletAllocation
							.gibbsSampler(docWords, (short)topicNum, 0.1, topicWordPriors, 0, 1, 1000, new Random(1l), handler);

					// 计算改时间片在全局映射下的各个主题的词分布
					List<Map<Integer, Double>> allTopicWordProbs = TimeSliceGenerator
							.generateTopicWordProbInTimeSlice(gibbsSample, generateSlice);
					generateSlice.setTopicWordProb(allTopicWordProbs);

					// 计算该时间片与上一时间片的对称kl散度距离
					double[] distance = calDistanceBetweenTopics(timeSliceQueue.getLast(), generateSlice);
					distanceMatrix.offer(distance);
					timeSliceQueue.offer(generateSlice);
					distanceMatrix.poll();
					timeSliceQueue.poll();

					int hotTopicId = getTopicIndexByPercByLocal(distance);
					if (hotTopicId != -1) {
						System.out.println(hotTopicId);
					}
				}
			} catch (InterruptedException e) {
				e.printStackTrace();
				Thread.currentThread().interrupt();
			}
		}
	}

	public int getTopicIndexByPercByLocal(double[] distance) {
		double sum = com.aliasi.util.Math.sum(distance);
		double percT = sum * NLPContants.CL;
		for (int i = 0; i < distance.length; i++) {
			if (distance[i] > percT) {
				return i;
			}
		}
		return -1;
	}

	public double[] calDistanceBetweenTopics(TimeSlice beforeTimeSlice, TimeSlice currentTimeSlice) {
		List<Map<Integer, Double>> curTwprobs = currentTimeSlice.getTopicWordProb();
		List<Map<Integer, Double>> beforeTwprobs = beforeTimeSlice.getTopicWordProb();
		double[] topicDis = new double[topicNum];
		for (int topic = 0; topic < topicNum; topic++) {
			double jsdis = jsDivergence(beforeTwprobs.get(topic), curTwprobs.get(topic));
			topicDis[topic] = jsdis;
		}
		return topicDis;
	}

	public double jsDivergence(Map<Integer, Double> p, Map<Integer, Double> q) {
		Map<Integer, Double> m = new HashMap<Integer, Double>();
		for (Integer pi : p.keySet()) {
			if (q.get(pi) == null) {
				m.put(pi, p.get(pi) / 2);
			} else {
				m.put(pi, (p.get(pi) + q.get(pi)) / 2.0);
			}
		}
		for (Integer qi : q.keySet()) {
			if (p.get(qi) == null) {
				m.put(qi, q.get(qi) / 2);
			} else {
				m.put(qi, (p.get(qi) + q.get(qi)) / 2.0);
			}
		}
		return (klDivergence(p, m) + klDivergence(q, m)) / 2.0;
	}

	private double klDivergence(Map<Integer, Double> p, Map<Integer, Double> m) {
		double divergence = 0.0;
		for (Integer i : m.keySet()) {
			if (p.get(i) != null && p.get(i) != 0) {
				double vp = p.get(i);
				double vq = m.get(i);
				divergence += vp * com.aliasi.util.Math.log2(vp / vq);
			}
		}
		return divergence;
	}

	public double[][] calTopicWordPriors(LinkedList<TimeSlice> historyQueue, TimeSlice currentTimeSlice) {
		SymbolTable tokenIndexInSlice = currentTimeSlice.getTokenIndexInSlice();
		double[][] topicWordPriors = new double[topicNum][tokenIndexInSlice.numSymbols()];
		for (int topic = 0; topic < topicNum; topic++) {
			ObjectToDoubleMap<Integer> otd = new ObjectToDoubleMap<>();
			double startweight = 0.01;
			int sliceNum = 0;
			for (TimeSlice slice : historyQueue) {
				List<Map<Integer, Double>> list = slice.getTopicWordProb();
				Map<Integer, Double> map = list.get(topic);
				for (Entry<Integer, Double> entry : map.entrySet()) {
					otd.increment(entry.getKey(), entry.getValue() * (sliceNum * 0.02 + startweight));
				}
			}
			for (Integer index : otd.keySet()) {
				String word = NLPContants.GLOBAL_WORD_INDEX.idToSymbol(index);
				int indexInSlice = tokenIndexInSlice.symbolToID(word);
				if (indexInSlice == -1) {
					continue;
				}
				topicWordPriors[topic][indexInSlice] = otd.getValue(index);
			}
		}
		return topicWordPriors;
	}

	public List<String> receiver() {
		return null;
	}

	public static void main(String[] args) throws IOException {
		BlockingQueue<String> queue = new LinkedBlockingDeque<>(200);
		NewsProducers produces = new NewsProducers(queue);
		OnlineEventDetection onlineEd = new OnlineEventDetection(50, queue);
		produces.start();
		onlineEd.start();
	}

}
