package duyuNN;

public class MathOp {

	public static void xDotApb(double[] x,
			double[][] A,
			double[] b,
			double[] results)
	{
		int row = x.length;
		int col = results.length;
		
		for(int i = 0; i < col; i++)
		{
			results[i] = b[i];
		}
		
		for(int i = 0; i < row; i++)
		{
			for(int j = 0; j < col; j++)
			{
				results[j] += A[i][j] * x[i];
			}
		}
	}
	
	public static void Axpy(double[][] A, 
			double[] x,
			double[] results)
	{
		int row = results.length;
		int col = x.length;
		
		for(int i = 0; i < row; i++)
		{
			for(int j = 0; j < col; j++)
			{
				results[i] += A[i][j] * x[j];
			}
		}
	}
	
	
	public static void Axpb(double[][] A, 
			double[] x,
			double[] b,
			double[] results)
	{
		int row = results.length;
		int col = x.length;
		
		for(int i = 0; i < row; i++)
		{
			results[i] = b[i];
		}
		
		for(int i = 0; i < row; i++)
		{
			for(int j = 0; j < col; j++)
			{
				results[i] += A[i][j] * x[j];
			}
		}
	}
	
	public static void xdotA(double[] x, double[][] A, double[] results)
    {
		int row = x.length;
		int col = results.length;
		
        for (int j = 0; j < col; ++j)
        {
            results[j] = 0;
        }

        for (int i = 0; i < row; ++i)
        {
            for (int j = 0; j < col; ++j)
            {
            	results[j] += x[i] * A[i][j];
            }
        }
    }
	
	public static void A_add_xTmulty(double[] x, double[] y, double[][] A)
    {
        int row = x.length;
        int col = y.length;

        for (int i = 0; i < row; ++i)
        {
            for (int j = 0; j < col; ++j)
            {
                A[i][j] += x[i] * y[j];
            }
        }
    }
	
	public static double dotProduct(double[] x, double[] y)
	{
		double dotP = 0;
		for(int i = 0; i < x.length; i++)
		{
			dotP += x[i] * y[i];
		}
		return dotP;
	}
}
