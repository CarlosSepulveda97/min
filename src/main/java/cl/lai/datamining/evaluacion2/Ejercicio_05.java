package cl.lai.datamining.evaluacion2;

import weka.clusterers.SimpleKMeans;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instances;

public class Ejercicio_05 extends App_loader_data_set{
	
	public static void main(String args[])throws Exception{
		String current = new java.io.File( "." ).getCanonicalPath()+"/src/resources/";
		String file = current + "bencineras_rm.csv";
		Instances datosEntrenamiento = load_clustering_data_set(file);
		
		//TODO Implementar		
		SimpleKMeans kMeans = new SimpleKMeans();
        kMeans.setNumClusters(3);
        kMeans.buildClusterer(datosEntrenamiento); 
        
        DenseInstance inst = new DenseInstance(5);
        
        Attribute b93 = datosEntrenamiento.attribute(0);
        Attribute b95 = datosEntrenamiento.attribute(1);
        Attribute b97 = datosEntrenamiento.attribute(2);
        Attribute DIESEL = datosEntrenamiento.attribute(3);
        Attribute keresone = datosEntrenamiento.attribute(4);
        
		inst.setValue(b93, 530);
		inst.setValue(b95, 650);
		inst.setValue(b97, 660);
		inst.setValue(DIESEL, 700);
		inst.setValue(keresone, 613);
        
        datosEntrenamiento.add(inst);
        String grupo = "";        
        for (int i = 0; i < datosEntrenamiento.numInstances(); i++) {
        	if (kMeans.clusterInstance(datosEntrenamiento.instance(i)) == 0) {
        		grupo = "barato";
        	}else if(kMeans.clusterInstance(datosEntrenamiento.instance(i)) == 1) {
        		grupo = "intermedio";
        	}else {
        		grupo = "caro";
        	}
        	
        	if(i==datosEntrenamiento.numInstances()-1) {
        		System.out.println("-----------------------------------------------");
        		System.out.println(datosEntrenamiento.get(i)+ " esta en cluster " + kMeans.clusterInstance(datosEntrenamiento.instance(i)) + " - " + grupo);  
        		System.out.println("-----------------------------------------------");   
        	}
    	 }

 	}
}
