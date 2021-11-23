package cl.lai.datamining.evaluacion2;

import weka.associations.Apriori;
import weka.core.Instances;

public class Ejercicio_02 extends App_loader_data_set{
	
	public static void main(String args[])throws Exception{
		String current = new java.io.File( "." ).getCanonicalPath()+"/src/resources/";
		String file = current + "bencineras_rm.csv";
		//el 2 es el rango de división
		Instances datosEntrenamiento = load_normal_data_set_discretized(file, 3);
		//TODO Implementar 
		
		Apriori aprioriObj = new Apriori();
		
			try {
				String c="0.95";
				String m="0.8";
				String n="2";
				
				String []options =  {"-C",c,"-N",n,"-M",m};
	    		aprioriObj.setOptions(options);
	    		aprioriObj.buildAssociations(datosEntrenamiento);
			}catch(Exception e) {
				
			}
			System.out.println(aprioriObj);

 	}
}
