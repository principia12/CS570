package model.nn_composition;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import duyuNN.AverageLayer;
import duyuNN.MultiConnectLayer;
import duyuNN.NNInterface;
import duyuNN.combinedLayer.LookupLinearTanh;

public class SentenceConvOneFilter implements NNInterface{
	List<LookupLinearTanh> LLTlist;
	MultiConnectLayer connect;
	public AverageLayer average;
	
	int linkId;
	int outputLength;
	
	public SentenceConvOneFilter()
	{
	}
	
	public SentenceConvOneFilter(
			int[] wordIds,
			LookupLinearTanh seedLLT
		) throws Exception 
	{
		int windowSizeLookup = seedLLT.lookup.inputLength;
		LLTlist = new ArrayList<LookupLinearTanh>();
		
		for(int i = 0; i < wordIds.length - windowSizeLookup + 1; i++)
		{
			LookupLinearTanh tmpLLT = seedLLT.cloneWithTiedParams();
			for(int j = 0; j < windowSizeLookup; j++)
			{
				tmpLLT.lookup.input[j] = wordIds[i + j];
			}
			LLTlist.add(tmpLLT);
		}
		
		int[] connectInputLengths = new int[LLTlist.size()];
		Arrays.fill(connectInputLengths, LLTlist.get(0).outputLength);
		
		connect = new MultiConnectLayer(connectInputLengths);
		for(int k = 0; k < LLTlist.size(); k++)
		{
			LLTlist.get(k).link(connect, k);
		}
		
		average = new AverageLayer(connect.outputLength, LLTlist.get(0).outputLength);
		connect.link(average);
		
		linkId = 0;
		outputLength = average.outputLength;
	}
	
	@Override
	public void randomize(Random r, double min, double max) {
		
	}

	@Override
	public void forward() {
		for(LookupLinearTanh layer: LLTlist)
		{
			layer.forward();
		}
		
		connect.forward();
		average.forward();
	}

	@Override
	public void backward() {
		average.backward();
		connect.backward();
		for(LookupLinearTanh layer: LLTlist)
		{
			layer.backward();
		}
	}

	@Override
	public void update(double learningRate) {
		for(LookupLinearTanh layer: LLTlist)
		{
			layer.update(learningRate);
		}
	}
	
	@Override
	public void updateAdaGrad(double learningRate, int batchsize) {
		// TODO Auto-generated method stub
	}

	@Override
	public void clearGrad() {
		// TODO Auto-generated method stub
		for(LookupLinearTanh layer: LLTlist)
		{
			layer.clearGrad();
		}
		
		connect.clearGrad();
		average.clearGrad();
		
		LLTlist.clear();
	}

	@Override
	public void link(NNInterface nextLayer, int id) throws Exception {
		Object nextInputG = nextLayer.getInputG(id);
		Object nextInput = nextLayer.getInput(id);
		
		double[] nextI = (double[])nextInput;
		double[] nextIG = (double[])nextInputG; 
		
		if(nextI.length != average.output.length || nextIG.length != average.outputG.length)
		{
			throw new Exception("The Lengths of linked layers do not match.");
		}
		average.output = nextI;
		average.outputG = nextIG;
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
		return average.output;
	}

	@Override
	public Object getInputG(int id) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public Object getOutputG(int id) {
		// TODO Auto-generated method stub
		return average.outputG;
	}

	@Override
	public Object cloneWithTiedParams() {
		return null;
	}
}
