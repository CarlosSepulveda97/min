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
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.clusterers.SimpleKMeans;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

public class Ejercicio_12 extends App_loader_data_set{
	
	public static Instances load_normal_data_set(String file)throws Exception{
		//Lectura de archivo general
		BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(new File(file))));
    	//Lectura de archivo para obtener los valores nominales
    	 
    	Map<Integer,String> correspondencia = new HashMap<Integer,String>();
    	
    	String aux = reader.readLine();
    	int totalRegistros = 6329;
    	
    	String arrays[] = aux.split(";");
    	
    	Instances dptos = null;
    	ArrayList<Attribute> attributes = new ArrayList<Attribute>();
    	//omitir el primer, segundo atributo y el último atributo, y considerar los 4 atributos restantes: 
    	for(int i=3; i<arrays.length-1;i++){
    		attributes.add(new Attribute(arrays[i]));
    	}
    	
    	//attributes.add(new Attribute(arrays[5]));
    	//attributes.add(new Attribute(arrays[7]));
    	
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
     		for(int at=0,   i=5; i<arrays.length-1;i++,at++){
     			double valor = Double.parseDouble(arrays[i]);
      			inst.setValue(attributes.get(at), valor);
        	}
     		//System.out.println("arrays[arrays.length-1]: " + arrays[arrays.length-1]);
     		inst.setValue(classAttribute, arrays[arrays.length-1]);
     		 
     		isTrainingSet.add(inst);
     		filas++;
    		 
    	} 
      	
    	NaiveBayes tree = new NaiveBayes();
      	String[] options = new String[1];
        options[0] = "-D"; 
        tree.setOptions(options);
        tree.buildClassifier(isTrainingSet);
		 
		DenseInstance inst = new DenseInstance(6);
		
		Attribute a1 = attributes.get(1);
		Attribute a2 = attributes.get(2);
		Attribute a3 = attributes.get(3);
		Attribute a4 = attributes.get(4);
		Attribute a5 = attributes.get(5);
		inst.setValue(a1,770);
		inst.setValue(a2,798);
		inst.setValue(a3,780);
		inst.setValue(a4,550);
		inst.setValue(a5,618);
		 
		 
		//asociar la instancia a un dataset
		Instances dataset = new Instances("probar", attributes, 1);
		dataset.setClassIndex(classAttribute.index());
		inst.setDataset(dataset); 
		
		double probabilidadPorGrupo[] = tree.distributionForInstance(inst);
		
		int indiceClasificado = (int) tree.classifyInstance(inst);
		
		System.out.println("marca clasificada: "+clasesPreviamenteDefinida.get(indiceClasificado));
      	
		Evaluation evalTest = new Evaluation(isTrainingSet);
		evalTest.evaluateModel(tree, isTrainingSet);
		
		  for (int i = 0; i < clasesPreviamenteDefinida.size(); ++i) {
		    	System.out.printf("Results for class %d (%s):\n", i, clasesPreviamenteDefinida.get(i));
		    	System.out.printf("  F-Measure: %6.4f\n", evalTest.fMeasure(i));
		    	System.out.println();
	       	}
		
      	
      	return isTrainingSet;
}

public static void main( String[] args ) throws Exception{

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
