package dataprepare;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.Map.Entry;

public class IO {

	public static DecimalFormat decimalFormat = new DecimalFormat("#0.000000");   
	
	public static ArrayList<String> readFile(String inPath, String encoding)
	{
		ArrayList<String> list = new ArrayList<String>();
		try{
			BufferedReader reader = new BufferedReader(new InputStreamReader(
					new FileInputStream(inPath) , encoding));
			String line = null;
			while((line = reader.readLine()) != null)
			{
				list.add(line);
			}
			reader.close();
		}
		catch(IOException e){
			e.printStackTrace();
		}
		return list;
	}
	
	public static String readFileStr(String inPath, String encoding)
	{
		String content = "";
		try{
			BufferedReader reader = new BufferedReader(new InputStreamReader(
					new FileInputStream(inPath) , encoding));
			String line = null;
			while((line = reader.readLine()) != null)
			{
				content = content + line + "\n";
			}
			reader.close();
		}
		catch(IOException e){
			e.printStackTrace();
		}
		return content;
	}
	
	public static HashSet<String> readFileSet(String inPath, String encoding)
	{
		ArrayList<String> list = readFile(inPath, encoding);
		HashSet<String> set = new HashSet<String>();
		set.addAll(list);
		return set;
	}
	
	/**
	 * The input format is => word + "\t" + index(frequency)
	 * @param inPath
	 * @param encoding
	 * @return
	 */
	public static LinkedHashMap<String, Integer> readMap(String inPath, String encoding)
	{
		LinkedHashMap<String, Integer> map = new LinkedHashMap<String, Integer>();
		ArrayList<String> tmpList = readFile(inPath, encoding);
		for(String tmpLine: tmpList)
		{
			String[] set = tmpLine.split("\t");
			map.put(set[0], Integer.parseInt(set[1]));
		}
		return map;
	}
	/**
	 * 
	 * @param outPath
	 * @param map
	 * @param encoding
	 * @param isDescend 1 if order in ascend; 0 if order in descend
	 */
	public static void writeMap(
			HashMap<String, Integer> map,
			String outPath, 
			String encoding, 
			int isAscend)
	{
		ArrayList<String> outlist = new ArrayList<String>();
		
		ArrayList<Entry> tmpList = new ArrayList<Entry>();
		tmpList.addAll(map.entrySet());
		if(isAscend == 1)
		{
			Collections.sort(tmpList, new EntryAscendComparator());
		}
		else if(isAscend == 0)
		{
			Collections.sort(tmpList, new EntryDescendComparator());
		}
		for(Entry entry: tmpList)
		{
			outlist.add(entry.getKey() + "\t" + entry.getValue());
		}
		
		IO.writeFile(outPath, outlist, encoding);
	}
	
	public static void writeSet(String outPath, HashSet<String> set, String encoding)
	{
		ArrayList<String> list = new ArrayList<String>();
		list.addAll(set);
		writeFile(outPath, list, encoding);
	}
	
	public static void writeIntMap(String outPath, HashMap<Integer, Integer> map, String encoding, int isAscend)
	{
		ArrayList<String> outlist = new ArrayList<String>();
		
		ArrayList<Entry> tmpList = new ArrayList<Entry>();
		tmpList.addAll(map.entrySet());
		if(isAscend == 1)
		{
			Collections.sort(tmpList, new EntryAscendComparator());
		}
		else if(isAscend == 0)
		{
			Collections.sort(tmpList, new EntryDescendComparator());
		}
		for(Entry entry: tmpList)
		{
			outlist.add(entry.getKey() + "\t" + entry.getValue());
		}
		
		IO.writeFile(outPath, outlist, encoding);
	}
	
	public static void writeFile(String outPath, Collection<String> list, String encoding)
	{
		try{
			PrintWriter writer = new PrintWriter(new BufferedWriter(new OutputStreamWriter(
			          new FileOutputStream(outPath), encoding)));
			for(String line: list)
			{
				writer.write(line + "\n");
			}
			writer.close();
		}
		catch(IOException e){
			e.printStackTrace();
		}
	}
	
	public static void writeIntFile(String outPath, Collection<Integer> list, String encoding)
	{
		try{
			PrintWriter writer = new PrintWriter(new BufferedWriter(new OutputStreamWriter(
			          new FileOutputStream(outPath), encoding)));
			for(Integer line: list)
			{
				writer.write(line + "\n");
			}
			writer.close();
		}
		catch(IOException e){
			e.printStackTrace();
		}
	}
	
	/**
	 * 获取dir文件夹下所有文件列表
	 * @param dir
	 */
	public static ArrayList<String> getFileNames(String dir){
		ArrayList<String> fileNames = new ArrayList<String>();
		File dirFile = new File(dir);
		File[] _files = dirFile.listFiles();
		for(File file: _files){
			if(file.isFile()){
				try {
					fileNames.add(file.getCanonicalPath());
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
		}
		return fileNames;
	}
	
	public static ArrayList<String> getDirNames(String dir){
		ArrayList<String> fileNames = new ArrayList<String>();
		File dirFile = new File(dir);
		File[] _files = dirFile.listFiles();
		for(File file: _files){
			if(file.isDirectory()){
				try {
					fileNames.add(file.getCanonicalPath());
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
		}
		return fileNames;
	}
}
class EntryAscendComparator implements Comparator{

	@Override
	public int compare(Object arg0, Object arg1) {
		// TODO Auto-generated method stub
		Entry<String, Integer> e0 = (Entry<String, Integer>) arg0;
		Entry<String, Integer> e1 = (Entry<String, Integer>) arg1;
		
		return e0.getValue().compareTo(e1.getValue());
	}
}
class EntryDescendComparator implements Comparator{

	@Override
	public int compare(Object arg0, Object arg1) {
		// TODO Auto-generated method stub
		Entry<String, Integer> e0 = (Entry<String, Integer>) arg0;
		Entry<String, Integer> e1 = (Entry<String, Integer>) arg1;
		
		return e1.getValue().compareTo(e0.getValue());
	}

}
