package dataprepare;

public class Data {

	public String userStr;
	public String productStr;
	public String reviewText;
	public int goldRating;
	public int predictedRating;
	public String[] words;
	
	// the propertyStr is only used for generating the data for jmars
	public String propertyStr;
	
	public Data(
			String userStr, 
			String productStr, 
			String reviewText,
			int goldRating,
			String propertyStr) {
		this.userStr = userStr;
		this.productStr = productStr;
		this.reviewText = reviewText;
		this.goldRating = goldRating;
		this.predictedRating = -1;
		this.propertyStr = propertyStr;
	}
	
	public Data(
			String userStr, 
			String productStr, 
			String reviewText,
			int goldRating) {
		this.userStr = userStr;
		this.productStr = productStr;
		this.reviewText = reviewText;
		this.goldRating = goldRating;
		this.predictedRating = -1;
	}
	
	public Data(String userStr, 
			String productStr, 
			String[] xWords,
			int goldRating) {
		this.userStr = userStr;
		this.productStr = productStr;
		this.goldRating = goldRating;
		this.predictedRating = -1;
		this.words = new String[xWords.length];
		System.arraycopy(xWords, 0, this.words, 0, xWords.length);
	}
	
	public String toString()
	{
		return userStr + "\t\t" + productStr + "\t\t" + goldRating + "\t\t" + reviewText;
	}
}
