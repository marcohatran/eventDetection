package onlineEventdetect.bean;

import java.util.List;
import java.util.Map;

import com.aliasi.symbol.MapSymbolTable;
import com.aliasi.symbol.SymbolTable;

public class TimeSlice {
	private String mSliceId;
	private List<Document> documentList;
	private SymbolTable tokenIndexInSlice;

	public SymbolTable getTokenIndexInSlice() {
		return tokenIndexInSlice;
	}

	public void setTokenIndexInSlice(SymbolTable tokenIndexInSlice) {
		this.tokenIndexInSlice = tokenIndexInSlice;
	}

	private List<Map<Integer,Double>> allTopicWordProbs;
	
	

	public String getmSliceId() {
		return mSliceId;
	}

	public void setmSliceId(String mSliceId) {
		this.mSliceId = mSliceId;
	}

	public List<Map<Integer, Double>> getTopicWordProb() {
		return allTopicWordProbs;
	}

	public void setTopicWordProb(List<Map<Integer, Double>> allTopicWordProbs) {
		this.allTopicWordProbs = allTopicWordProbs;
	}

	public List<Document> getDocumentList() {
		return documentList;
	}

	public void setDocumentList(List<Document> documentList) {
		this.documentList = documentList;
	}

	public TimeSlice(String sliceId) {
		mSliceId = sliceId;
		tokenIndexInSlice = new MapSymbolTable();
	}

}
