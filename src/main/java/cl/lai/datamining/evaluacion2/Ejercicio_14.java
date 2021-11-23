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
import weka.classifiers.lazy.LWL;
import weka.classifiers.trees.J48;
import weka.clusterers.SimpleKMeans;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.gui.treevisualizer.PlaceNode2;
import weka.gui.treevisualizer.TreeVisualizer;

public class Ejercicio_14 extends App_loader_data_set{
	
	public static void main(String args[])throws Exception{
		String current = new java.io.File( "." ).getCanonicalPath()+"/src/resources/";
		String file = current + "bencineras_rm.csv";
		Instances datosEntrenamiento = load_data_set_keresone(file);
		
		//TODO Implementar
		
		//System.out.println(datosEntrenamiento);
		
		int totalRegistros = 1000;

		String aux = ("2B93;B95;B97;DIESEL;keresone;marca;"); 
		Map<Integer,String> correspondencia = new HashMap<Integer,String>();
		
    	String arrays[] = aux.split(";");
    	ArrayList<Attribute> attributes = new ArrayList<Attribute>(5);
    	Attribute classAttribute = new Attribute("precio");
    	//omitir el primer, segundo atributo y el último atributo, y considerar los 4 atributos restantes: 
    	for(int i=0; i<arrays.length-1;i++){
    		attributes.add(new Attribute(arrays[i]));
    	}
    	//considerar el ultimo atributo como target 
    	
    	Instances isTrainingSet = new Instances("traning", attributes, totalRegistros);
    	isTrainingSet.setClass(classAttribute);
    	isTrainingSet.setClassIndex(4);
    	
    	isTrainingSet.addAll(datosEntrenamiento);
    	
    	LWL l = new LWL();
		l.setKNN(4);
		l.buildClassifier(isTrainingSet);
		
		
		//double []valores_probar = new double[]{0,261,168,303,666};
		//DenseInstance inst = new DenseInstance(1,valores_probar);
		DenseInstance inst = new DenseInstance(5);
		
		Attribute b39 = datosEntrenamiento.attribute(0);
		Attribute b95 = datosEntrenamiento.attribute(1);
		Attribute b97 = datosEntrenamiento.attribute(2);
		Attribute DIESEL = datosEntrenamiento.attribute(3);
		Attribute keresone = datosEntrenamiento.attribute(4);
		//Attribute marca = datosEntrenamiento.attribute(5);
		
		inst.setValue(b39, 779);
		inst.setValue(b95, 810);
		inst.setValue(b97, 843);
		inst.setValue(DIESEL, 578);
		//inst.setValue(keresone, 618);
		//inst.setValue(marca,  );
		
		//asociar la instancia a un dataset
		Instances dataset = new Instances("probar", attributes, 1);
		dataset.setClassIndex(0);
		inst.setDataset(dataset); 
		
		
		double precio = l.classifyInstance(inst);
		 
		System.out.println("Precio kerosenne: "+ precio);
 	}
}
