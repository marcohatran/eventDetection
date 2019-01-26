package util;

import com.aliasi.symbol.MapSymbolTable;
import com.aliasi.symbol.SymbolTable;

public class NLPContants {
	public final static String ROOTPATH  = "ROOT_DIC_PATH";
	public final static String BASE_DIR = LoadConf.getIstance().getProperty(NLPContants.ROOTPATH);
	
	
	public final static Integer SLICE_MAX = 1000;
	public static final String TAB = "\t";
	
	/*全局词索引*/
	public static SymbolTable GLOBAL_WORD_INDEX = new MapSymbolTable();
	
	public final static Integer QUEUE_MAX = 10;
	
	public final static Double EP = 0.0000000000000000000000001;
	public static final double CL = 0.9;
}
