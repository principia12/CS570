package model.nn_composition;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import duyuNN.*;
import duyuNN.combinedLayer.*;

public class DocSimplifiedLSTM implements NNInterface{

	// sentence(i) -> lstm(i)
	// tanh(i-1) -> lstm(i)
	// lstm(i) -> tanh(i)
	
	// tanh(i) -> lstm(i+1)
	// output is tanhList.get(tanhList.size() - 1)
	
	public List<SentenceConvMultiFilter> sentenceConvList;
	public List<SimplifiedLSTMLayer> simplifiedLSTMList;
	public List<TanhLayer> tanhList;
	
	public int linkId;
	public int hiddenLength;
	
	public DocSimplifiedLSTM(
			LookupLinearTanh seedLLT1,
			LookupLinearTanh seedLLT2,
			LookupLinearTanh seedLLT3,
			int[][] wordIdMatrix,
			SimplifiedLSTMLayer seedSimplifiedLSTM) throws Exception
	{
		hiddenLength = seedLLT1.outputLength;
		sentenceConvList = new ArrayList<SentenceConvMultiFilter>();
		
		int maxWindowSize = Math.max(seedLLT1.lookup.inputLength, 
				Math.max(seedLLT2.lookup.inputLength, seedLLT3.lookup.inputLength));
		
		for(int i = 0; i < wordIdMatrix.length; i++)
		{
			if(wordIdMatrix[i].length >= maxWindowSize)
			{
				sentenceConvList.add(
					new SentenceConvMultiFilter(wordIdMatrix[i], seedLLT1, seedLLT2, seedLLT3));
			}
			else
			{
				continue;
			}
		}
		
		if(sentenceConvList.size() == 0)
		{
			return;
		}
		
		// new hidden layers
		simplifiedLSTMList = new ArrayList<SimplifiedLSTMLayer>();
		tanhList = new ArrayList<TanhLayer>();
		for(int i = 0; i < sentenceConvList.size(); i++)
		{
			simplifiedLSTMList.add((SimplifiedLSTMLayer) seedSimplifiedLSTM.cloneWithTiedParams());
			tanhList.add(new TanhLayer(hiddenLength));
		}
		
		// link. important
		for(int i = 0; i < sentenceConvList.size(); i++)
		{
			sentenceConvList.get(i).link(simplifiedLSTMList.get(i), 0);
			if(i > 0)
			{
				tanhList.get(i - 1).link(simplifiedLSTMList.get(i), 1);
			}
			else
			{
			}
			
			simplifiedLSTMList.get(i).link(tanhList.get(i));
		}
	}

	@Override
	public void randomize(Random r, double min, double max) {
		// TODO Auto-generated method stub
	}

	@Override
	public void forward() {
		// be careful about the order. it is important.
		for(int i = 0; i < sentenceConvList.size(); i++)
		{
			sentenceConvList.get(i).forward();
			simplifiedLSTMList.get(i).forward();
			tanhList.get(i).forward();
		}
	}

	@Override
	public void backward() {
		// the order is important. Be careful
		for(int i = sentenceConvList.size() - 1; i >= 0 ; i--)
		{
			tanhList.get(i).backward();
			simplifiedLSTMList.get(i).backward();
			sentenceConvList.get(i).backward();
		}
	}

	@Override
	public void update(double learningRate) {
		for(SentenceConvMultiFilter layer: sentenceConvList)
		{
			layer.update(learningRate);
		}
		
		// update linear
		for(SimplifiedLSTMLayer layer: simplifiedLSTMList)
		{
			layer.update(learningRate);
		}
	}

	@Override
	public void updateAdaGrad(double learningRate, int batchsize) {
		
	}

	@Override
	public void clearGrad() {
		
		for(int i = 0; i < sentenceConvList.size(); i++)
		{
			sentenceConvList.get(i).clearGrad();
			simplifiedLSTMList.get(i).clearGrad();
			tanhList.get(i).clearGrad();
		}
		
		sentenceConvList.clear();
		simplifiedLSTMList.clear();
		tanhList.clear();
	}

	@Override
	public void link(NNInterface nextLayer, int id) throws Exception {
		Object nextInputG = nextLayer.getInputG(id);
		Object nextInput = nextLayer.getInput(id);
		
		double[] nextI = (double[])nextInput;
		double[] nextIG = (double[])nextInputG; 
		
		if(nextI.length != tanhList.get(tanhList.size() - 1).output.length 
				|| nextIG.length != tanhList.get(tanhList.size() - 1).outputG.length)
		{
			throw new Exception("The Lengths of linked layers do not match.");
		}
		tanhList.get(tanhList.size() - 1).output = nextI;
		tanhList.get(tanhList.size() - 1).outputG = nextIG;
	}

	@Override
	public void link(NNInterface nextLayer) throws Exception {
		// TODO Auto-generated method stub
		link(nextLayer, linkId);
	}

	@Override
	public Object getInput(int id) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public Object getOutput(int id) {
		// TODO Auto-generated method stub
		return tanhList.get(tanhList.size() - 1).output;
	}

	@Override
	public Object getInputG(int id) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public Object getOutputG(int id) {
		// TODO Auto-generated method stub
		return tanhList.get(tanhList.size() - 1).outputG;
	}

	@Override
	public Object cloneWithTiedParams() {
		// TODO Auto-generated method stub
		return null;
	}
}