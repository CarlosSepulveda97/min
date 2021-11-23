package cl.lai.datamining.evaluacion2;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Discretize;

public class App_loader_data_set {
	
	public static Instances load_normal_data_set_discretized(String file, int numeroDivision)throws Exception{
		BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(new File(file))));
   	 
    	Map<Integer,String> correspondencia = new HashMap<Integer,String>();
    	
    	String aux = reader.readLine();
    	int totalRegistros = 380;
    	
    	String arrays[] = aux.split(";");
    	
    	Instances dptos = null;
    	ArrayList<Attribute> attributes = new ArrayList<Attribute>();
    	for(int i=3; i<arrays.length-1;i++){
    		attributes.add(new Attribute(arrays[i]));
    	}
    	//considerar el ultimo atributo como target 
    	ArrayList<String> clasesPreviamenteDefinida = new ArrayList<String>(); 
    	while((aux=reader.readLine())!=null){
    		arrays = aux.split(";");
    		if(!clasesPreviamenteDefinida.contains(arrays[8]) )
    			clasesPreviamenteDefinida.add(arrays[8]);
    	}
    	Attribute classAttribute = new Attribute("Empresa",clasesPreviamenteDefinida);
    	attributes.add(classAttribute);
    	
    	Instances isTrainingSet = new Instances("traning", attributes, totalRegistros);
    	isTrainingSet.setClassIndex(classAttribute.index());
    	
    	reader = new BufferedReader(new InputStreamReader(new FileInputStream(new File(file))));
    	reader.readLine();
    	int filas = 1;
    	//Lectura de cada instancia
      	while((aux=reader.readLine())!=null){
     		arrays = aux.split(";");
     		correspondencia.put(filas,arrays[0]);
     		
     		DenseInstance inst = new DenseInstance(isTrainingSet.numAttributes());
     		for(int at=0,   i=3; i<arrays.length-1;i++,at++){
     			double valor = Double.parseDouble(arrays[i]);
     			inst.setValue(attributes.get(at), valor);
        	}
     		inst.setValue(classAttribute, arrays[arrays.length-1]);
     		 
     		isTrainingSet.add(inst);
     		filas++;
    	} 
      	
      	Discretize discretizeNumeric = new Discretize();
      	discretizeNumeric.setInputFormat(isTrainingSet); 
    	discretizeNumeric.setOptions(new String[] {
    			"-B",  String.valueOf(numeroDivision),  // numero de division
    			"-R",  "first-last"}); //rango de numero (desde atributo 1 hasta atributo 2)
    	isTrainingSet = Filter.useFilter(isTrainingSet, discretizeNumeric);        
     	  
      	return isTrainingSet;
	}
	
	public static Instances load_data_set_keresone(String file)throws Exception{
		BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(new File(file))));
   	 
    	Map<Integer,String> correspondencia = new HashMap<Integer,String>();
    	
    	String aux = reader.readLine();
    	int totalRegistros = 380;
    	
    	String arrays[] = aux.split(";");
    	
    	Instances dptos = null;
    	ArrayList<Attribute> attributes = new ArrayList<Attribute>();
    	for(int i=3; i<arrays.length-1;i++){
    		attributes.add(new Attribute(arrays[i]));
    	}
    	 
    	
    	Instances isTrainingSet = new Instances("traning", attributes, totalRegistros);
    	isTrainingSet.setClassIndex(attributes.size()-1);
    	 
    	int filas = 1;
    	//Lectura de cada instancia
      	while((aux=reader.readLine())!=null){
     		arrays = aux.split(";");
     		correspondencia.put(filas,arrays[0]);
     		
     		DenseInstance inst = new DenseInstance(isTrainingSet.numAttributes());
     		for(int at=0,   i=3; i<arrays.length-1;i++,at++){
     			double valor = Double.parseDouble(arrays[i]);
     			inst.setValue(attributes.get(at), valor);
        	}
     		 
     		isTrainingSet.add(inst);
     		filas++;
    	} 
      	return isTrainingSet;
	}
	
	public static Instances load_normal_data_set(String file)throws Exception{
		BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(new File(file))));
   	 
    	Map<Integer,String> correspondencia = new HashMap<Integer,String>();
    	
    	String aux = reader.readLine();
    	int totalRegistros = 380;
    	
    	String arrays[] = aux.split(";");
    	
    	Instances dptos = null;
    	ArrayList<Attribute> attributes = new ArrayList<Attribute>();
    	for(int i=3; i<arrays.length-1;i++){
    		attributes.add(new Attribute(arrays[i]));
    	}
    	//considerar el ultimo atributo como target 
    	ArrayList<String> clasesPreviamenteDefinida = new ArrayList<String>(); 
    	while((aux=reader.readLine())!=null){
    		arrays = aux.split(";");
    		if(!clasesPreviamenteDefinida.contains(arrays[8]) )
    			clasesPreviamenteDefinida.add(arrays[8]);
    	}
    	Attribute classAttribute = new Attribute("Empresa",clasesPreviamenteDefinida);
    	attributes.add(classAttribute);
    	
    	Instances isTrainingSet = new Instances("traning", attributes, totalRegistros);
    	isTrainingSet.setClassIndex(classAttribute.index());
    	
    	reader = new BufferedReader(new InputStreamReader(new FileInputStream(new File(file))));
    	reader.readLine();
    	int filas = 1;
    	//Lectura de cada instancia
      	while((aux=reader.readLine())!=null){
     		arrays = aux.split(";");
     		correspondencia.put(filas,arrays[0]);
     		
     		DenseInstance inst = new DenseInstance(isTrainingSet.numAttributes());
     		for(int at=0,   i=3; i<arrays.length-1;i++,at++){
     			double valor = Double.parseDouble(arrays[i]);
     			inst.setValue(attributes.get(at), valor);
        	}
     		inst.setValue(classAttribute, arrays[arrays.length-1]);
     		 
     		isTrainingSet.add(inst);
     		filas++;
    	} 
      	return isTrainingSet;
	}
	
	public static Instances load_clustering_data_set(String file)throws Exception{
		BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(new File(file))));
   	 
    	Map<Integer,String> correspondencia = new HashMap<Integer,String>();
    	
    	String aux = reader.readLine();
    	int totalRegistros = 380;
    	
    	String arrays[] = aux.split(";");
    	
    	Instances dptos = null;
    	ArrayList<Attribute> attributes = new ArrayList<Attribute>();
    	for(int i=3; i<arrays.length-1;i++){
    		attributes.add(new Attribute(arrays[i]));
    	}
    	 
    	Instances isTrainingSet = new Instances("traning", attributes, totalRegistros);
     	
     	int filas = 1;
    	//Lectura de cada instancia
      	while((aux=reader.readLine())!=null){
     		arrays = aux.split(";");
     		correspondencia.put(filas,arrays[0]);
     		
     		DenseInstance inst = new DenseInstance(isTrainingSet.numAttributes());
     		for(int at=0,   i=3; i<arrays.length-1;i++,at++){
     			double valor = Double.parseDouble(arrays[i]);
     			inst.setValue(attributes.get(at), valor);
        	}
     		 
     		isTrainingSet.add(inst);
     		filas++;
    	} 
      	return isTrainingSet;
	}
}
