package evaluationMetric;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import dataprepare.IO;

public class Metric {

	public static void calcMetric(List<Integer> goldLabelList, List<Integer> predictLabelList)
	{
		if(goldLabelList.size() != predictLabelList.size())
		{
			System.err.println("gold length and predict length don't match!");
		}
		
		HashMap<Integer, ClassCount> classMap = new HashMap<Integer, ClassCount>();
		
		for(int i = 0; i < goldLabelList.size(); i++)
		{
			int goldLabel = goldLabelList.get(i);
			int predictLabel = predictLabelList.get(i);
			
			if(!classMap.containsKey(goldLabel))
			{
				classMap.put(goldLabel, new ClassCount());
			}
			if(!classMap.containsKey(predictLabel))
			{
				classMap.put(predictLabel, new ClassCount());
			}
			
			classMap.get(predictLabel).predict++;
			classMap.get(goldLabel).gold++;
			if(goldLabel == predictLabel)
			{
				classMap.get(predictLabel).match++;
			}
		}
		
		for(Integer label: classMap.keySet())
		{
			classMap.get(label).update();
			System.out.println("label: " + label + "\t" + classMap.get(label));
		}
		
		//accuracy, macro-F
		int totalMatch = 0;
		int totalGold = 0;
		double macroF= 0.0;
		
		for(Integer label: classMap.keySet())
		{
			totalMatch += classMap.get(label).match;
			totalGold += classMap.get(label).gold;
			
			macroF += classMap.get(label).f;
		}
		
		double accuracy = 1.0 * totalMatch / totalGold;
		System.out.println("Accuracy: " + accuracy);
		System.out.println("macro-F: "+ macroF/classMap.size());
		
		// mae = 1/N * (|gold_i - predict_i|)
		// rmse = sqrt(1/N * (gold_i - predict_i) * (gold_i - predict_i))
		// mse = 1/N * (gold_i - predict_i) * (gold_i - predict_i)
		int maeDistance = 0;
		int rmseDistance = 0;
		
		for(int i = 0; i < goldLabelList.size(); i++)
		{
			int goldLabel = goldLabelList.get(i);
			int predictLabel = predictLabelList.get(i);
			
			maeDistance = maeDistance + Math.abs(goldLabel - predictLabel);
			rmseDistance = rmseDistance + (goldLabel - predictLabel) * (goldLabel - predictLabel);
		}
		
		double mae = 1.0 * maeDistance / goldLabelList.size();
		double rmse = 1.0 * Math.sqrt(1.0 * rmseDistance / goldLabelList.size());
		double mse = 1.0 * rmseDistance / goldLabelList.size();
		
		System.out.println("MAE: " + mae);
		System.out.println("RMSE: " + rmse);
		System.out.println("MSE: " + mse);
	}
	
	public static void calcMetric(String goldFile, String predictFile)
	{
		ArrayList<String> goldList = IO.readFile(goldFile, "utf8");
		ArrayList<String> preList = IO.readFile(predictFile, "utf8");
		
		List<Integer> goldLabelList = new ArrayList<Integer>();
		List<Integer> predictLabelList = new ArrayList<Integer>();
		
		for(int i = 0; i < goldList.size(); i++)
		{
			String goldLine = goldList.get(i);
			String predictLine = preList.get(i);
			
			int goldLabel = Integer.parseInt(goldLine.split(" ")[0]);
			int predictLabel = (int)Double.parseDouble(predictLine);

			goldLabelList.add(goldLabel);
			predictLabelList.add(predictLabel);
		}
		
		calcMetric(goldLabelList, predictLabelList);
	}
}
class ClassCount{
	
	int gold = 0;
	int predict = 0;
	int match = 0;
	
	double p;
	double r;
	double f;
	
	public void update()
	{
		if(this.predict == 0)
		{
			p = 0.0;
		}
		else
		{
			p = 1.0 * this.match / this.predict;
		}
		
		r = 1.0 * this.match / this.gold;
		
		if(p < 0.000001 || r < 0.000001)
		{
			f = 0.0;
		}
		else
		{
			f = 2 * (p * r)/(p + r);
		}
	}
	
	public String toString()
	{
		String prf = "P: " + p + "\tR: " + r + "\tF: " + f;
		return prf;
	}
}