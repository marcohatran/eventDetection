package util;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.apache.commons.lang3.exception.ExceptionUtils;

/**
 * 各种初始化
 * @author 龚帅宾
 *
 */
public class DictionaryResource {
	
//	private Logger logger = Logger.getLogger(DictionaryResource.class);
	/**
	 * 词向量
	 */
	private List<double[]> WORD_VECTOR_LIST = new ArrayList<double[]>();
	
	/**
	 * 词的索引
	 */
	private Map<String, Integer> WORD_INDEX_MAP = new HashMap<String, Integer>();
	
	/**
	 * 词的列表
	 */
	private List<String> WORDLIST = new ArrayList<String>();
	private int W2VLENGTH = 200;
	/**
	 * 词IDF值
	 */
	private Map<String, Double> IDFMAP = new HashMap<String, Double>();

	private DictionaryResource() {
		long s = System.currentTimeMillis();
		System.out.println("init w2v, please waiting ...");
//		logger.info("init w2v, please waiting ...");
//		initW2V();
//		logger.info("init idf, please waiting ...");
		System.out.println("init idf, please waiting ...");
		initIDF();
//		logger.info(String.format("init cost:%ds", (System.currentTimeMillis() - s)/1000));
		System.out.println(String.format("init cost:%ds", (System.currentTimeMillis() - s)/1000));
	}

	
	public List<double[]> getWORD_VECTOR_LIST() {
		return WORD_VECTOR_LIST;
	}


	public Map<String, Integer> getWORD_INDEX_MAP() {
		return WORD_INDEX_MAP;
	}


	public List<String> getWORDLIST() {
		return WORDLIST;
	}

	public Map<String, Double> getIDFMAP() {
		return IDFMAP;
	}


	private void initW2V() {
		BufferedReader br = null;
		InputStream in = null;
		String line;
		try {
			in = new FileInputStream(new File(LoadConf.getIstance().getProperty("dic") + "w2v"));
			br = new BufferedReader(new InputStreamReader(in));
			line = br.readLine();
			String[] splits = null;
			int i = 0;
			while (line != null) {
				splits = line.split("\\s+");
				if (splits.length != (W2VLENGTH + 1)) {
					continue;
				}
				String w = splits[0];
				WORD_INDEX_MAP.put(w, i);
				WORDLIST.add(w);
				double[] v = new double[W2VLENGTH];
				for (int l = 1; l < splits.length; l++) {
					v[l - 1] = Double.parseDouble(splits[l]);
				}
				WORD_VECTOR_LIST.add(v);
				line = br.readLine();
				i++;
			}
		} catch (FileNotFoundException e1) {
			System.err.println(ExceptionUtils.getRootCauseMessage(e1));
		}catch (IOException e) {
			System.err.println(ExceptionUtils.getRootCauseMessage(e));
		} finally {
			if (br != null) {
				try {
					br.close();
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
			if (in != null) {
				try {
					in.close();
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
		}
	}

	private void initIDF() {
		BufferedReader br = null;
		InputStream in = null;
		String line;
		try {
			in = new FileInputStream(new File(LoadConf.getIstance().getProperty("dic") + "IDF.dic"));
			br = new BufferedReader(new InputStreamReader(in));
			line = br.readLine();
			String[] splits = null;
			while (line != null) {
				splits = line.split("\t");
				if (splits.length != 2) {
					continue;
				}
				String word = splits[0];
				double idf = Double.parseDouble(splits[1]);
				if(IDFMAP.get(word) != null) {
					continue;
				}
				IDFMAP.put(word, idf);
				line = br.readLine();
			}
			br.close();
			in.close();
		} catch (FileNotFoundException e1) {
			System.err.println(ExceptionUtils.getRootCauseMessage(e1));
		} catch (IOException e) {
			System.err.println(ExceptionUtils.getRootCauseMessage(e));
		}finally {
			if (br != null) {
				try {
					br.close();
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
			if (in != null) {
				try {
					in.close();
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
		}
	}
	
	public static DictionaryResource getInstance(){
		return SingletonHolder.instance;
	}
	
	private static class SingletonHolder{        
        private static DictionaryResource instance = new DictionaryResource();        
    }
}
