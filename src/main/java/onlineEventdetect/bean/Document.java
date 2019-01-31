package onlineEventdetect.bean;

import java.util.ArrayList;
import java.util.List;
import com.aliasi.symbol.SymbolTable;
import com.aliasi.tokenizer.TokenizerFactory;

import util.NLPContants;

public class Document {
	private String docuemntId;
	private int[] docmentContentIdx;

	public String getDocuemntId() {
		return docuemntId;
	}

	
	
	public int[] getDocmentContentIdx() {
		return docmentContentIdx;
	}

	public Document(String docuemntId, String content, SymbolTable sliceSymbolTable,TokenizerFactory factory) {
		super();
		List<Integer> tmpSymbolList = new ArrayList<Integer>();
		this.docuemntId = docuemntId;
		for (String token : factory.tokenizer(content.toCharArray(), 0, content.length())) {
			String[] splits = token.split("/");
			if(splits[0].length() < 2) {
				continue;
			}
			int symbol = sliceSymbolTable.getOrAddSymbol(splits[0]);
			NLPContants.GLOBAL_WORD_INDEX.getOrAddSymbol(splits[0]);
			tmpSymbolList.add(symbol);
		}
		docmentContentIdx = new int[tmpSymbolList.size()];
		for (int i = 0; i < tmpSymbolList.size(); i++) {
			docmentContentIdx[i] = tmpSymbolList.get(i);
		}
	}

}
