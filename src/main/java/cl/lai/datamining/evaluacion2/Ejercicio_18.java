package cl.lai.datamining.evaluacion2;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.Map;

import weka.associations.Apriori;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.lazy.IBk;
import weka.classifiers.trees.J48;
import weka.clusterers.SimpleKMeans;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

public class Ejercicio_18 extends App_loader_data_set{
	
	public static Instances load_normal_data_set(String file)throws Exception{
		//Lectura de archivo general
		BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(new File(file))));
    	//Lectura de archivo para obtener los valores nominales
    	 
    	Map<Integer,String> correspondencia = new HashMap<Integer,String>();
    	
    	String aux = reader.readLine();
    	int totalRegistros = 6329;
    	
    	String arrays[] = aux.split(";");
    	
    	ArrayList<Attribute> attributes = new ArrayList<Attribute>();
    	//omitir el primer, segundo atributo y el último atributo, y considerar los 4 atributos restantes: 
    	for(int i=3; i<arrays.length-1;i++){
    		attributes.add(new Attribute(arrays[i]));
    	}
    	//considerar el ultimo atributo como target 
    	ArrayList<String> clasesPreviamenteDefinida = new ArrayList<String>(); 
    	clasesPreviamenteDefinida.add("SHELL"); 
    	clasesPreviamenteDefinida.add("COPEC"); 
    	clasesPreviamenteDefinida.add("PETROBRAS");
    	clasesPreviamenteDefinida.add("JVL COMBUSTIBLES");
    	clasesPreviamenteDefinida.add("GYT COMBUSTIBLES");
    	clasesPreviamenteDefinida.add("ABASTIBLE");
    	clasesPreviamenteDefinida.add("ECOGREEN LTDA.");
    	clasesPreviamenteDefinida.add("TERPEL");
    	clasesPreviamenteDefinida.add("Petroval");
    	clasesPreviamenteDefinida.add("AUTOGASCO");
    	clasesPreviamenteDefinida.add("HOLA!");
    	clasesPreviamenteDefinida.add("Combustibles Endless.com");
    	clasesPreviamenteDefinida.add("BULL ENERGY");
    	clasesPreviamenteDefinida.add("ECCO");
    	clasesPreviamenteDefinida.add("JLC");
    	clasesPreviamenteDefinida.add("DELPA");
    	clasesPreviamenteDefinida.add("Sin Bandera");
    	clasesPreviamenteDefinida.add("Combustible Alhue");
    	clasesPreviamenteDefinida.add("SESA");
    	clasesPreviamenteDefinida.add("Del Sol Combustibles");
    	clasesPreviamenteDefinida.add("VIVA COMBUSTIBLES");
    	clasesPreviamenteDefinida.add("SINHEL");
    	clasesPreviamenteDefinida.add("ENEX");
    	clasesPreviamenteDefinida.add("CAVE");
    	clasesPreviamenteDefinida.add("COMBUSTIBLES AMADE");
    	
    	
    	Attribute classAttribute = new Attribute("marca",clasesPreviamenteDefinida);
    	attributes.add(classAttribute);
    	
    	Instances isTrainingSet = new Instances("traning", attributes, totalRegistros);
    	isTrainingSet.setClassIndex(classAttribute.index());
     	int filas = 1;
    	//Lectura de cada instancia
      	while((aux=reader.readLine())!=null){
     		arrays = aux.split(";");
     		correspondencia.put(filas,arrays[0]);
     		
     		DenseInstance inst = new DenseInstance(attributes.size());
     		for(int at=0,   i=3; i<arrays.length-1;i++,at++){
     			double valor = Double.parseDouble(arrays[i]);
      			inst.setValue(attributes.get(at), valor);
        	}
     		System.out.println("arrays[arrays.length-1]: " + arrays[arrays.length-1]);
     		inst.setValue(classAttribute, arrays[arrays.length-1]);
     		 
     		isTrainingSet.add(inst);
     		filas++;
    		 
    	} 
      	
      	DenseInstance inst = new DenseInstance(6);
		
		Attribute a1 = attributes.get(0);
		Attribute a2 = attributes.get(1);
		Attribute a3 = attributes.get(2);
		Attribute a4 = attributes.get(3);
		Attribute a5 = attributes.get(4);
		inst.setValue(a1,762);
		inst.setValue(a2,800);
		inst.setValue(a3,733);
		inst.setValue(a4,534);
		inst.setValue(a5,605);
		
		// asociar la instancia a un dataset
		Instances dataset = new Instances("probar", attributes, 1);
		dataset.setClassIndex(classAttribute.index());
		inst.setDataset(dataset);

		//RNN
		MultilayerPerceptron rnn = new MultilayerPerceptron();
		String[] options = new String[] { "-N", "500" };// hacer que se itera 100 veces
		rnn.setOptions(options);
		rnn.buildClassifier(isTrainingSet);
		
		

		int indiceClasificado = (int) rnn.classifyInstance(inst);

		System.out.println("RNN: " + clasesPreviamenteDefinida.get(indiceClasificado));
		System.out.println("----------------------------------------------------");
		
		//KNN
		IBk knn = new IBk();
      	String[] options2 = new String[]{"-K","7"};//hacer que 5 sea el knn
        knn.setOptions(options2);
        knn.buildClassifier(isTrainingSet);
      	
    	int indiceClasificado2 = (int) knn.classifyInstance(inst);
        
    	System.out.println("KNN: " + clasesPreviamenteDefinida.get(indiceClasificado2));
    	System.out.println("----------------------------------------------------");
    	
    	//Naive
    	NaiveBayes tree = new NaiveBayes();
      	String[] options3 = new String[1];
        options3[0] = "-D"; 
        tree.setOptions(options3);
        tree.buildClassifier(isTrainingSet);
         
        
        double probabilidadPorGrupo[] = tree.distributionForInstance(inst);
		int indiceClasificado4 = (int) tree.classifyInstance(inst);
		for(int i=0; i<probabilidadPorGrupo.length;i++){
			//System.out.println(String.format("Probabilidad del SECTOR [%s] : %s", clasesPreviamenteDefinida.get(i), String.valueOf(probabilidadPorGrupo[i])));
		}
		System.out.println("NAIVE: "+clasesPreviamenteDefinida.get(indiceClasificado4));
		System.out.println("----------------------------------------------------");
		
		J48 tree2 = new J48();
        String[] options4 = new String[1];
        options4[0] = "-U"; 
        tree2.setOptions(options4);
        tree2.buildClassifier(isTrainingSet);
		 
        int indiceClasificado5 = (int) tree.classifyInstance(inst);
		System.out.println("ARBOL: "+clasesPreviamenteDefinida.get(indiceClasificado5));
		System.out.println("----------------------------------------------------");
    	
      	return isTrainingSet;
	}
	
	public static void main( String[] args ) throws Exception{
		
		//String current = new java.io.File( "." ).getCanonicalPath()+"/src/resources/";
		//String file = current + "arriendo_dpto_categoria_numerica_clasificacion.csv";
		String current = new java.io.File( "." ).getCanonicalPath()+"/src/resources/";
		String file = current + "bencineras_rm.csv";
		Instances isTrainingSet = load_normal_data_set(file);
		
      	//TODO implementar lo que el documento señala
      	
	  	System.out.println("Atributos seleccionados");
	  	Enumeration<Attribute> enu = isTrainingSet.enumerateAttributes();
		while (enu.hasMoreElements()) {
			Attribute attr = enu.nextElement();
			System.out.println(attr);
		}
		
    }
}
