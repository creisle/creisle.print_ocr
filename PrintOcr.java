/*****************************************************************************************************************
*
*  SENG 474: Datamining Final Project
*
*  Title: optical chacater recognition of printed text
*  Authors: Caralyn Reisle and Sunette Mynhardt
*  Date: 2014 March 25 (T)
*
* To compile: javac -cp weka.jar clusteringwithbitset.java
* To Run: java -cp weka.jar:. clusteringwithbitset <test image> <optional parameters>
* 
*  Purpose: this program uses a database of letter images to build a classifier for the times
*  	new roman font using either a neural network or svm depending on the users choice
*  	this classifier is build using weka from inside this program (i.e. you need the weka-jar file)
*  	takes an input test image and does a simple clustering of connected black pixels to separate the
*  	image into components. we then compute the attributes for this component and use the classifier to
*  	determine the most likely letter
*  	once we have the letters we determing spacing based on the positions of the components in the original
*  	image and added spaces and line breaks where required
*  	the resulting text taken from the image is sent to an output text file in the same directory
*
*  references:
* 	1. Neves, E. et al. (1997). IEEE. A Multi-Font Character Recognition Based on its Fundamental Features by Artificial Neural Networks
* 	2. Shrivastava, V. & Sharma, N. (2012). SIPIJ 3(5). Artificial Neural Network Based Optical Character Recognition.
* 	3. Hall, M. et al. (2009); The WEKA Data Mining Software: An Update; SIGKDD Explorations, Volume 11, Issue 1.
* 	4. Tautu E.D. & Leon, F. (2012). Bul. Inst. Polit. Optical Character Recognition using support vector machines.
* 	5. Sadri, J. et al. (2003). MVIP. Application of Support Vector Machines for Recognition of Handwritten Arabic/Persian Digits.
* 	6. Rashnodi, O. (2011). International journal of computer applications 29(12). Persian Handwritten Digit Recognition using Support Vector Machines.
*
******************************************************************************************************************/

import java.awt.Image;
import java.awt.image.BufferedImage;
import java.io.*;
import java.awt.Graphics;
import javax.imageio.ImageIO;
import java.awt.Color;
import java.util.*;
import weka.core.*;
import weka.classifiers.*;
import weka.classifiers.functions.SMO;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.functions.MultilayerPerceptron;

public class PrintOcr{
	
	public static boolean output = false; //used in outputting extra information for debugging purposes
	
	public static final int black = 0;
	public static final int red = 16711680;
	public static final int white = 16777215;
	public static final int blue = 255;
	public static final String[] alpha_tnr = {"a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "fr", "fi", "fo", "rt", "tu", "vw", "KL", "WN", "JU", "?", ".", "(", ")", "]", "[", "-", ";", "<", ">", ","}; 
	public static final String[] attributes = {"V10", "V30", "V50", "V80", "V90", "H10", "H30", "H50", "H80", "H90", "hsymm", "vsymm", "cc", "hw", "c", "q1", "q2", "q3", "q4", "Ih50", "Ih30", "Ih80", "Iv30", "Iv50", "Iv80", "class"};
	
	public static String weka = "";
	
	public static void main(String[] args){
		boolean notest = false;
		boolean train = false;
		boolean evaluateModel = false;
		boolean annmodeltype = true;
		String font = "TimesNewRoman";
		
		System.out.println("***************************************");
		System.out.println("*           Start of program          *");
		System.out.println("***************************************\n");
		/************** command line arguments ****************/
		
		String filename = "";
		
		if(args.length>0){
			filename = args[0];
		}
		
		for(int i=1; i<args.length; i++){
			if(args[i].equals("train")){
				train = true;
			}else if(args[i].equals("output")){
				output = true;
			}else if(args[i].equals("notest")){
				notest = true;
			}else if(args[i].equals("eval")){
				evaluateModel = true;
			}else if(args[i].equals("svm")){
				annmodeltype = false;
			}
		}
		
		if(train){
			generate_arff(font);
		}
		
		/* ======================= BUILD THE MODEL ========================*/
		Classifier model = null;
		Instances data = null;
		
		if(!annmodeltype){
			model = (Classifier)new SMO(); //SMO()
		}else{
			model = (Classifier)new MultilayerPerceptron();
		}
		//read in our data file (either built just now or already there)
		System.out.println("reading the input arff font library data file ... ");
		try{
			//load the data from the arff file
			DataSource src = new DataSource("font.arff");
			data = src.getDataSet();
			if (data.classIndex() == -1){
				data.setClassIndex(data.numAttributes() - 1);
			}
		}catch(Exception e){
			System.err.println("Error: the arff file did not load correctly. please ensure to run in taining mode if the arff file does not exist in the smae directory as this program");
			System.exit(1);
		}
		
		System.out.printf("Building the classifier using the WEKA %s classifier function .... \n", annmodeltype? "MultilayerPerceptron" : "SMO");
		//set the options and build the classifier
		try{
			if(!annmodeltype){
				String[] opt = {"-M"};
				model.setOptions(opt);
			}
			
			model.buildClassifier(data); //trows java.lan.Exception is classifier isn't built correctly
		}catch(Exception e){
			System.err.println("Error: the model failed to build correctly");
			System.exit(1);
		}
		
		if(evaluateModel){
			System.out.println("Evaluating our classifier ... ");
			//now we will evaluate the model
			try{
				Evaluation eTest = new Evaluation(data);
				eTest.evaluateModel(model, data);
				//Random rand = new Random(1); // using seed = 1
				//int folds = 10;
				//eTest.crossValidateModel(model, testset, folds, rand);
				String summary = eTest.toSummaryString();
				System.out.println(summary);
			}catch(Exception e){
				System.err.println("Error: exception evaluating the test model");
				System.exit(1);
			}
		}
		
		//loading in an image
		if(notest){ System.exit(0); } //program ends here is we don't wish to use the classifier at all
		
		String outputfilename = "out.txt";
		while(true){
			if(process_test_file(filename, outputfilename, model, data)<0){
				System.out.printf("Error in processing the file %s\n", filename);
			}else{
				System.out.printf("File %s was sucessfully processed.\n To run another file enter <inputfile.jpg> <outputfile name> followed by the optional parameter: <output> if chosen for debugging\nEnter q to quit\n\n", filename);
				Scanner in = new Scanner(System.in);
				filename = in.next();
				if(filename.equals("q")||filename.equals("Q")){
					System.exit(0);
				}
				outputfilename = in.next();
			}
		}
		
		
		
	}
	
	/*****************************************************************
	 *
	 * function: process_test_file()
	 * purpose: for each input test jpg file isolates, orders and classifys components. adds spaces and new line
	 * 	characters where apprpriate and outputs the result to a text file
	 * input:
	 * 	filename, outputfilename
	 * 	model: the classifier model generated using weka
	 * 	data: the training dataset (we will need to add new instances to this dataset)
	 * output:
	 * 	returns -1 if an error occurred. otherwise returns 0
	 *
	 *****************************************************************/
	public static int process_test_file(String filename, String outputfilename, Classifier model, Instances data){
		BufferedImage img_color = null;
		System.out.printf("Loading the input image %s... \n", filename);
		try{
			img_color = ImageIO.read(new File(filename));
		}catch (IOException e){
			System.err.println("Error: test image not read correctly\n");
			return -1;
		}
		
		//get the heights and widths of the image
		int img_width = img_color.getWidth();
		int img_height = img_color.getHeight();
		
		System.out.println("Converting Image Data to Binary ... ");
		BitSet img_data = convertImageToBinary(img_color);		
		
		System.out.println("Isolating Putative Text Components ... ");
		ArrayList<Region> components = cluster_connected_pixels(img_data, img_width, img_height);
		
		if(output){ System.out.println("Outlining components on result image ... "); outline_components(img_color, components, red); }
		
		System.out.println("Order components by line ... ");
		ArrayList<ArrayList<Region>> lineslist = order_by_line(components, img_data, img_width, img_height);
		
		if(output){ for(ArrayList<Region> line: lineslist){ outline_components(img_color, line, blue); } }
		
		System.out.println("Scanning for spaces ... ");
		double[] spaces = define_spaces(lineslist, img_width, img_height); //array storing the [average space between letters, average space between words]
		
		String text = "";
		
		int count = 0;
		
		System.out.println("Classifying test Components ... \n");
		for(ArrayList<Region> line: lineslist){
			for(int i=0; i<line.size(); i++){
				Region r = line.get(i);
				//System.out.printf("Component "+(++count)+": ");
				int[][] rmat = getRegionMatrix(img_data, img_width, img_height, r);
				LetterVector vector = compute_attribute_vector(rmat);
				
				String ch = getLetter(model, data, vector);
				System.out.printf("%s ", ch);
				if(spaces!=null&&i>0){
					int space_from_prev = r.getXmin() - line.get(i-1).getXmax();
					if(space_from_prev>0){ //ignore overlapping characters
						if(Math.abs(spaces[0]-space_from_prev)>Math.abs(spaces[1]-space_from_prev)){
							//this value is closer to the space between words average value. add a space to the output
							text += ' ';
						}
					}
					
				}
				
				text += ch;
			}
			text += '\n';
		}
		System.out.printf("\n\nThe text is recognized as \n\n%s\n\n", text);
		
		//output the resulting image
		if(output){
			System.out.println("Outputting the result image ... ");
			try{
				File outputfile = new File("result.jpg");
				ImageIO.write(img_color, "jpg", outputfile);
			}catch (IOException e){
				System.out.println("Error: problem outputting component outline image file");
				return -1;
			}
		}
		
		//output the resulting text file
		System.out.println("Outputting the result text ... ");
		try{
			File outputfile = new File(outputfilename);
			BufferedWriter output = new BufferedWriter(new FileWriter(outputfile));
			output.write(text);
			output.close();
		}catch (IOException e){
			System.out.println("Error: problem outputting the results text file");
			return -1;
		}
		return 0;
	}
	/*****************************************************************
	 *
	 * function: outline_components()
	 * purpose: for a list of components, it outlines their "box" on the original image. useful for debugging.
	 * 	allows one to determine if the classification is incorrect or if the letters are not being isolated properly
	 * input:
	 * 	img: the img we are drawing are component "boxes" on
	 * 	setlist: list of components (Regions)
	 * 	color: the color we want to outline them in
	 * output:
	 * 	just updates the buffered image. no output
	 *
	 *****************************************************************/
	public static void outline_components(BufferedImage img, ArrayList<Region> setlist, int color){
		
		//int color = 0;
		for(int i=0; i<setlist.size(); i++){
			//draw horizontal lines
			Region temp = setlist.get(i);
			
			for(int x=temp.getXmin(); x<temp.getXmax(); x++){
				img.setRGB(x, temp.getYmin(), color);
				img.setRGB(x, temp.getYmax(), color);
			}
			for(int y=temp.getYmin(); y<temp.getYmax(); y++){
				img.setRGB(temp.getXmin(), y, color);
				img.setRGB(temp.getXmax(), y, color);
			}
		}
	}
	
	/*****************************************************************
	 *
	 * function: generate_arff()
	 * purpose: generates the arff file for our font training data. this is only called if the train=true tag is
	 * 	used in running the program. 
	 * input:
	 * 	font: the font we wish to train (currently always the default, TimesNewRoman) 
	 * output:
	 * 	to the current directory, outputs the font.arff file
	 *
	 *****************************************************************/
	public static void generate_arff(String font){
		weka += "% training data\n";
		weka += "@RELATION \"Times New Roman\"\n\n";
		for(int i=0; i<attributes.length-1; i++){
			weka += "@ATTRIBUTE "+attributes[i]+" NUMERIC\n";
		}
		weka += "@ATTRIBUTE "+attributes[attributes.length-1]+" {";
		for(int i=0; i<alpha_tnr.length-1; i++){
			weka += "\""+alpha_tnr[i]+"\",";
		}
		weka += "\""+alpha_tnr[alpha_tnr.length-1]+"\"}\n\n@DATA\n";
		ArrayList<LetterVector> training_set = train_font(font);
		try{
			File outputfile = new File("font.arff");
			BufferedWriter output = new BufferedWriter(new FileWriter(outputfile));
			output.write(weka);
			output.close();
		}catch (IOException e){
			System.out.println("problem outputting the arff file");
		}
	}
	
	/*****************************************************************
	 *
	 * function: count_closed_components()
	 * purpose: inverts the binary matrix (flips 0's and 1's) and then uses the cluster connected pixels method to
	 * 	determine the number of elements in a region. then we ignore any components that have max or min x or y values on
	 * 	the component border. the components that are left are isolated whitespace, i.e. closed components
	 * 	this is one of the attributes we use in our attribute vectors and helps us distinguish i's and l's
	 * input:
	 * 	points: binary data for the component
	 * output:
	 * 	returns the number of closed components found
	 *
	 *****************************************************************/ 
	public static ArrayList<Region> count_closed_components(int[][] points){
		//go through the array vertically, then horizontally
		int set_num = 0; 
		int width = points[0].length;
		int height = points.length;
		ArrayList<Region> setlist = new ArrayList<Region>(); //list of all our sets
		//System.out.print("cluster_connected_pixels - go through all the rows\n");
		int[][] pixels = new int[height][width];
		//create int array for easier manipulation. later can change this
		for(int i=0; i<height; i++){
			for(int j=0; j<width; j++){
				if(points[i][j]==0){
					pixels[i][j] = 1;
				}else{
					pixels[i][j] = 0;
				}
			}
		}
		
		//checked up to here
		
		for(int i=0; i<height; i++){ //number of rows, y position
			if(pixels[i][0]>0){
				pixels[i][0] = ++set_num;
				Region temp = new Region();
				temp.addColor(set_num);
				setlist.add(temp);
			}
			for(int j=1; j<width; j++){ //number of columns, x position
				if(pixels[i][j-1]>0&&pixels[i][j]>0){
					pixels[i][j] = pixels[i][j-1];
				}else if(pixels[i][j]>0){
					pixels[i][j] = ++set_num;
					Region temp = new Region();
					temp.addColor(set_num);
					setlist.add(temp);
				}
			}
		}
		//pass over the array a second time to join adjacent sets
		for(int i=0; i<width; i++){ //number of columns. note this only works if all the rows are the same length
			for(int j=1; j<height; j++){ //number of rows. this goes through each column by row position
				//col number = i, row number = j
				int left = pixels[j-1][i];
				int curr = pixels[j][i];
				if(left>0&&curr>0){
					//System.out.printf("\n\nneed to merge the set %d with the set %d\n", left, curr);
					Region temp1 = null;
					Region temp2 = null;
					//System.out.printf("left = %d and current = %d\n", left, curr);
					for(int k=0; k<setlist.size(); k++){
						if(setlist.get(k).isEquivalent(left)){
							temp1 = setlist.get(k);
						}
						if(setlist.get(k).isEquivalent(curr)){
							temp2 = setlist.get(k);
						}
					}
					if(temp1!=null&&temp2!=null){
						if(!temp1.equals(temp2)){
							//System.out.printf("merging\n");
							//System.out.println(temp1.toString()+"\n"+temp2.toString());
							temp1.mergeSets(temp2);
							setlist.remove(temp1);
							setlist.remove(temp2);
							setlist.add(temp1);
						}else{
							//System.out.println("these sets are already merged\n");
							
						}
					}
				}
			}
		}
		//System.out.print("cluster_connected_pixels - go through all the pixels again\n");
		for(int i=0; i<pixels.length; i++){ //number of rows, y position
			for(int j=0; j<pixels[i].length; j++){ //number of columns, x position
				Region temp = null;
				for(int k=0; k<setlist.size(); k++){
					if(setlist.get(k).isEquivalent(pixels[i][j])){
						temp = setlist.get(k);
					}
				}
				if(temp!=null){
					pixels[i][j] = temp.getColorID();
					temp.updateX(j);
					temp.updateY(i);
				}	
			}
		}
		
		boolean flag = true;
		while(flag){
			flag = false;
			for(Region r: setlist){
				//ignore close components that are on the borders
				if(r.getXmin()==0||r.getXmax()==width-1||r.getYmin()==0||r.getYmax()==height-1){
					flag = true;
					setlist.remove(r);
					break;
				}
				//now ignore single uncolored pixels
				if(r.getXmin()==r.getXmax()&&r.getYmin()==r.getYmax()){
					flag = true;
					setlist.remove(r);
					break;
				}
				
			}
		}
		return setlist;
	}
	
	/*****************************************************************
	 *
	 * function: count_components()
	 * purpose: uses the cluster connected pixels method to determine the number of elements in a region (important if we have merged)
	 * 	this is one of the attributes we use in our attribute vectors and helps us distinguish i's and l's
	 * input:
	 * 	points: binary data for the component
	 * output:
	 * 	returns the number of components found
	 *
	 *****************************************************************/
	public static double count_components(int[][] points){
		//go through the array vertically, then horizontally
		int set_num = 0; 
		int width = points[0].length;
		int height = points.length;
		ArrayList<Region> setlist = new ArrayList<Region>(); //list of all our sets
		//System.out.print("cluster_connected_pixels - go through all the rows\n");
		int[][] pixels = new int[height][width];
		//create int array for easier manipulation. later can change this
		for(int i=0; i<height; i++){
			for(int j=0; j<width; j++){
				if(points[i][j]==0){
					pixels[i][j] = 0;
				}else{
					pixels[i][j] = 1;
				}
			}
		}
		
		//checked up to here
		
		for(int i=0; i<height; i++){ //number of rows, y position
			if(pixels[i][0]>0){
				pixels[i][0] = ++set_num;
				Region temp = new Region();
				temp.addColor(set_num);
				setlist.add(temp);
			}
			for(int j=1; j<width; j++){ //number of columns, x position
				if(pixels[i][j-1]>0&&pixels[i][j]>0){
					pixels[i][j] = pixels[i][j-1];
				}else if(pixels[i][j]>0){
					pixels[i][j] = ++set_num;
					Region temp = new Region();
					temp.addColor(set_num);
					setlist.add(temp);
				}
			}
		}
		//pass over the array a second time to join adjacent sets
		for(int i=0; i<width; i++){ //number of columns. note this only works if all the rows are the same length
			for(int j=1; j<height; j++){ //number of rows. this goes through each column by row position
				//col number = i, row number = j
				int left = pixels[j-1][i];
				int curr = pixels[j][i];
				if(left>0&&curr>0){
					//System.out.printf("\n\nneed to merge the set %d with the set %d\n", left, curr);
					Region temp1 = null;
					Region temp2 = null;
					//System.out.printf("left = %d and current = %d\n", left, curr);
					for(int k=0; k<setlist.size(); k++){
						if(setlist.get(k).isEquivalent(left)){
							temp1 = setlist.get(k);
						}
						if(setlist.get(k).isEquivalent(curr)){
							temp2 = setlist.get(k);
						}
					}
					if(temp1!=null&&temp2!=null){
						if(!temp1.equals(temp2)){
							//System.out.printf("merging\n");
							//System.out.println(temp1.toString()+"\n"+temp2.toString());
							temp1.mergeSets(temp2);
							setlist.remove(temp1);
							setlist.remove(temp2);
							setlist.add(temp1);
						}else{
							//System.out.println("these sets are already merged\n");
							
						}
					}
				}
			}
		}
		//System.out.print("cluster_connected_pixels - go through all the pixels again\n");
		for(int i=0; i<pixels.length; i++){ //number of rows, y position
			for(int j=0; j<pixels[i].length; j++){ //number of columns, x position
				Region temp = null;
				for(int k=0; k<setlist.size(); k++){
					if(setlist.get(k).isEquivalent(pixels[i][j])){
						temp = setlist.get(k);
					}
				}
				if(temp!=null){
					pixels[i][j] = temp.getColorID();
					temp.updateX(j);
					temp.updateY(i);
				}	
			}
		}
		return (double)setlist.size();
	}
		
	/*****************************************************************
	 *
	 * function: cluster_connected_pixels()
	 * purpose: initial step in processing the binary image data. takes in a bitset and clusters adjacent balck pixels into components.
	 * 	stores there max and min x and y values into a list of regions for generating sub arrays later. makes 3 passes of the data to do
	 * 	this, along with many many passes over the putative component sets (this is what takes so long)
	 * input:
	 * 	points: binary data for our test image
	 * 	width: width of the test image
	 * 	height: height of the test image
	 * output:
	 * 	returns the list of regions representing our putative components
	 *
	 *****************************************************************/
	public static ArrayList<Region> cluster_connected_pixels(BitSet points, int width, int height){
		//go through the array vertically, then horizontally
		int set_num = 0; 
		int[][] pixels = new int[height][width];
		
		ArrayList<Region> setlist = new ArrayList<Region>(); //list of all our sets
		//System.out.print("cluster_connected_pixels - go through all the rows\n");
		
		//create int array for easier manipulation. later can change this
		for(int i=0; i<height; i++){
			for(int j=0; j<width; j++){
				pixels[i][j] = points.get(i*width+j) ? 1: 0;
			}
		}
		
		//checked up to here
		
		for(int i=0; i<height; i++){ //number of rows, y position
			if(pixels[i][0]>0){
				pixels[i][0] = ++set_num;
				Region temp = new Region();
				temp.addColor(set_num);
				setlist.add(temp);
			}
			for(int j=1; j<width; j++){ //number of columns, x position
				if(pixels[i][j-1]>0&&pixels[i][j]>0){
					pixels[i][j] = pixels[i][j-1];
				}else if(pixels[i][j]>0){
					pixels[i][j] = ++set_num;
					Region temp = new Region();
					temp.addColor(set_num);
					setlist.add(temp);
				}
			}
		}
		//System.out.println("number of sets after going through the rows is \n"+setlist.size());
		//print_array_fraction(pixels, 1);
		//System.out.print("cluster_connected_pixels - go through all the columns\n");
		//pass over the array a second time to join adjacent sets
		for(int i=0; i<width; i++){ //number of columns. note this only works if all the rows are the same length
			for(int j=1; j<height; j++){ //number of rows. this goes through each column by row position
				//col number = i, row number = j
				int left = pixels[j-1][i];
				int curr = pixels[j][i];
				if(left>0&&curr>0){
					//System.out.printf("\n\nneed to merge the set %d with the set %d\n", left, curr);
					Region temp1 = null;
					Region temp2 = null;
					
					//check how many points of connection there are
					
					
					//System.out.printf("left = %d and current = %d\n", left, curr);
					for(int k=0; k<setlist.size(); k++){
						if(setlist.get(k).isEquivalent(left)){
							temp1 = setlist.get(k);
						}
						if(setlist.get(k).isEquivalent(curr)){
							temp2 = setlist.get(k);
						}
					}
					if(temp1!=null&&temp2!=null){
						if(!temp1.equals(temp2)){
							//System.out.printf("merging\n");
							//System.out.println(temp1.toString()+"\n"+temp2.toString());
							temp1.mergeSets(temp2);
							setlist.remove(temp1);
							setlist.remove(temp2);
							setlist.add(temp1);
						}else{
							//System.out.println("these sets are already merged\n");
							
						}
					}
					
					
				}
			}
		}
		//System.out.print("cluster_connected_pixels - go through all the pixels again\n");
		for(int i=0; i<pixels.length; i++){ //number of rows, y position
			for(int j=0; j<pixels[i].length; j++){ //number of columns, x position
				Region temp = null;
				for(int k=0; k<setlist.size(); k++){
					if(setlist.get(k).isEquivalent(pixels[i][j])){
						temp = setlist.get(k);
					}
				}
				if(temp!=null){
					pixels[i][j] = temp.getColorID();
					temp.updateX(j);
					temp.updateY(i);
				}	
			}
		}
		//System.out.printf("the number of setlist is %d\n", setlist.size());
		/*
		for(Region r: setlist){
			for(int i=r.xmin; i<=r.xmax; i++){
				for(int j=r.ymin; j<r.ymax; j++){
					if(r.isEquivalent(pixels[i][j])){
						
					}
				}
			}
		}*/
		
		return setlist;
	}
	
	/*****************************************************************
	 * !!!!!!!!CURRENTLY NOT IN USE!!!!!!!!!!!
	 * function: find_avg_element_size()
	 * purpose: computes the average size of all our extracted components
	 * input:
	 * 	elements: the list of the components we extracted from the test image
	 * output:
	 * 	returns the int value representing the average size
	 *
	 *****************************************************************/
	public static int find_avg_element_size(ArrayList<Region> elements){
		int area = 0;
		for(int i=0; i<elements.size(); i++){
			Region temp = elements.get(i);
			int w = temp.getXmax() - temp.getXmin();
			int h = temp.getYmax() - temp.getYmin();
			area += w*h;
		}
		return area/elements.size();
	}
	
	/*****************************************************************
	 * 
	 * function: getRegionMatrix()
	 * purpose: takes the stored bitset and outputs a binary value int array of the region
	 * input:
	 * 	r: the component we want the submatrix of the original image for
	 * 	w: width of the input image
	 * 	h: height of the input image
	 * 	pixels: BitSet containing the image data
	 * output:
	 * 	returns the 2D array of values from the original image representing the space assigned to the component
	 *
	 *****************************************************************/
	public static int[][] getRegionMatrix(BitSet pixels, int w, int h, Region r){
		//copy area to an int array of bits
		//change to a bitset?
		int hs = r.getYmax()-r.getYmin()+1;
		int ws = r.getXmax()-r.getXmin()+1;
		int[][] subset = new int[hs][ws];
		for(int i=0; i<hs; i++){
			for(int j=0; j<ws; j++){
				//index in the new array is i, j
				//corresponds to (i+ymin, j+xmin) in the old array
				//this is equivalent to index = (i+ymin)*width + (j+xmin)
				int index = (i+r.getYmin())*w + (j+r.getXmin());
				if(pixels.get(index)){
					subset[i][j] = 1;
				}
				if(output){ System.out.printf(subset[i][j]+" "); }
			}
			if(output){ System.out.println(); }
		}
		if(output){ System.out.println(); }
		return subset;
	}
	
	/*****************************************************************
	 * 
	 * function: order_by_line()
	 * purpose: put all the components into their natural ordering
	 * input:
	 * 	components: list of all the extracted components from the original test image
	 * 	w: width of the input image
	 * 	h: height of the input image
	 * 	pixels: BitSet containing the image data
	 * output:
	 * 	returns the list of lines where lines are the lists of components 
	 *
	 *****************************************************************/
	public static ArrayList<ArrayList<Region>> order_by_line(ArrayList<Region> components, BitSet pixels, int w, int h){
				
		ArrayList<ArrayList<Region>> lineslist = new ArrayList<ArrayList<Region>>();
		
//inefficient! could use improvement 
		// group elements by overlapping y regions into different lists representing the lines of text in our document
		while(components.size()>0){
			Region r = components.remove(0); //check which line this component belongs to
			boolean found = false;
			for(ArrayList<Region> line: lineslist){
				for(Region m: line){
					if(r.checkRangeOverlap(m)){
						line.add(r);
						found = true;
						break;
					}
				}
				if(found){
					break;
				}
			}
			if(!found){ //not found, need to add a new line
				ArrayList<Region> line = new ArrayList<Region> ();
				line.add(r);
				lineslist.add(line);
			}
		}
		
		
		
		//order each line by x value and merge overlapping components
		for(ArrayList<Region> line: lineslist){
			Collections.sort(line, new RegionXComparator());
			for(int i=1; i<line.size(); i++){
//changed this from more than 50, to any overlap
				if(line.get(i).getDomainOverlapAsPercent(line.get(i-1))>0.25){
					line.get(i-1).mergeSets(line.get(i));
					line.remove(i);
					i=1;
				}else if(line.get(i).checkDomainOverlap(line.get(i-1))){ //remove this 25% from the first letter so as to avoid noise
					line.get(i-1).setXmax(line.get(i).getXmin()-1);
				}
			}
		}
		
		
		//order the lines vertically and put into the new list of lists elements that we will return
		ArrayList<ArrayList<Region>> elements = new ArrayList<ArrayList<Region>>();
		
		while(lineslist.size()>0){
			ArrayList<Region> line = lineslist.get(0); //get the first element
			for(int i=1; i<lineslist.size(); i++){
				if(line.get(0).getCY()>lineslist.get(i).get(0).getCY()){
					line = lineslist.get(i);
				}
			}
			lineslist.remove(line);
			elements.add(line);
		}
		
		return elements;
	}
	
	
	/*****************************************************************
	 * 
	 * function: define_spaces()
	 * purpose: to identify putative values for spaces between letters, and then to separate these into two sets,
	 * 	while minimizing the sse of both sets. allowing us to split the putative space values into two categories:
	 * 	spaces between letters and spaces between words.
	 * input:
	 * 	lineslist: ordered list of lines, oredered line: list of components
	 * 	w: width of the input image
	 * 	h: height of the input image
	 * output:
	 * 	returns null if there are no spaces expected. else returns a double arrray with [average letter space, average word space]
	 *
	 *****************************************************************/
	public static double[] define_spaces(ArrayList<ArrayList<Region>> lineslist, int w, int h){
		ArrayList<Integer> set = new ArrayList<Integer>();
		for(ArrayList<Region> line: lineslist){
			for(int i=1; i<line.size(); i++){
				int temp = line.get(i).getXmin()-line.get(i-1).getXmax();
				if(temp>0){
					set.add(temp);
				}
			}
		}
		Collections.sort(set);
		
		ArrayList<Integer> set2 = new ArrayList<Integer>();
		double unsplit_sse = list_sse(set);
		double split_max = unsplit_sse;
		double prev_max = unsplit_sse;
		boolean has_words = false;
		
		while(Double.compare(prev_max, split_max)>=0&&set.size()>0){
			prev_max = split_max;
			set2.add(set.remove(0));
			double sse1 = list_sse(set);
			double sse2 = list_sse(set2);
			
			if(Double.compare(sse1, sse2)>=0){
				split_max = sse1;
			}else{
				split_max = sse2;
			}
			
			if(Double.compare(unsplit_sse, sse1+sse2)>0){
				has_words = true;
			}
			if(output){ System.out.printf("the sse of the single set is %f and the sse of the two sets is %f and %f\n", unsplit_sse, sse1, sse2);}
		}
		
		//revert the last change since the previous sets were better which is why the loop quit
		set.add(set2.remove(set2.size()-1));
		
		if(has_words){
			double[] result = {list_avg(set2), list_avg(set)};
			return result;
		}else{
			return null;
		}
		
	}
	
	/*****************************************************************
	 * 
	 * function: weight_quadrants()
	 * purpose: for a binary array of pixels, compute the proportion of black pixels that belong to each
	 * 	respective quadrant {q1: top-left, q2: top-right, q3: bottom-right, q4: bottom-left}
	 * input:
	 * 	pixels: binary array for the component
	 * output:
	 * 	returns an array of doubles [q1, q2, q3, q4]
	 *
	 *****************************************************************/
	public static double[] weight_quadrants(int[][] pixels){
		double top_left = 0;
		double bottom_left = 0;
		double top_right = 0;
		double bottom_right = 0;
		for(int i=0; i<pixels.length/2; i++){ //top
			for(int j=0; j<pixels[i].length/2; j++){ //left
				if(pixels[i][j]>0){
					top_left++;
				}
			}
			
			for(int j=pixels[i].length/2; j<pixels[i].length; j++){ //right
				if(pixels[i][j]>0){
					top_right++;
				}
			}
		}
		for(int i=pixels.length/2; i<pixels.length; i++){ //bottom
			for(int j=0; j<pixels[i].length/2; j++){ //left
				if(pixels[i][j]>0){
					bottom_left++;
				}
			}
			
			for(int j=pixels[i].length/2; j<pixels[i].length; j++){ //right
				if(pixels[i][j]>0){
					bottom_right++;
				}
			}
		}
		double sum = top_left+bottom_left+top_right+bottom_right;
		double[] quadrants = {top_left/sum, top_right/sum, bottom_left/sum, bottom_right/sum};
		return quadrants;
	}
	
	/*****************************************************************
	 * 
	 * function: list_avg()
	 * purpose: computes the average of a list of integer values
	 * input:
	 * 	ArrayList of integers
	 * output:
	 * 	returns the average
	 *
	 *****************************************************************/
	public static double list_avg(ArrayList<Integer> set){
		if(set.isEmpty()){
			return -1;
		}
		//calculate the average of the set
		double average = 0;
		for(Integer i: set){
			average += (double)i;
		}
		average = average/set.size();
		return average;
	}
	
	/*****************************************************************
	 * 
	 * function: list_sse()
	 * purpose: computes the sum squared error for a list of integers
	 * input:
	 * 	ArrayList of integers
	 * output:
	 * 	returns the sum squared error for these integers
	 *
	 *****************************************************************/
	public static double list_sse(ArrayList<Integer> set){
		if(set.isEmpty()){
			return 0;
		}
		//calculate the average of the set
		double average = list_avg(set);
		if(average<0){
			return 0;
		}
		
		//now calculate the see
		double sse = 0;
		for(Integer i: set){
			sse += Math.pow((average-i), 2);
		}
		return sse;
	}
	
	/*****************************************************************
	 * 
	 * function: compute_attribute_vector()
	 * purpose: given an input component matrix we assume this is a value letter and then compute attribute values to
	 * 	be stored in a vector that we will later use in classifying this component or in training the model
	 * input:
	 * 	pixels: binary matrix of the letter
	 * output:
	 * 	returns the LetterVector that stores the vector of attribute values for this component
	 *
	 *****************************************************************/
	public static LetterVector compute_attribute_vector(int[][] pixels){
		
		int w = pixels[0].length;
		int h = pixels.length;
		LetterVector v = new LetterVector();
		v.add(pixel_col_sum(pixels, w*1/10)/h); // V10 sum
		v.add(pixel_col_sum(pixels, w*3/10)/h); // V30 sum
		v.add(pixel_col_sum(pixels, w*5/10)/h); // V50 sum
		v.add(pixel_col_sum(pixels, w*8/10)/h); // V80 sum
		v.add(pixel_col_sum(pixels, w*9/10)/h); // V90 sum
		
		v.add(pixel_row_sum(pixels, h*1/10)/w); // H10 sum
		v.add(pixel_row_sum(pixels, h*3/10)/w); // H30 sum
		v.add(pixel_row_sum(pixels, h*5/10)/w); // H50 sum
		v.add(pixel_row_sum(pixels, h*8/10)/w); // H80 sum
		v.add(pixel_row_sum(pixels, h*9/10)/w); // H90 sum
		
		v.add(reflect_over_vertical(pixels)); // horizontal symmetry
		v.add(reflect_over_horizontal(pixels)); // vertical symmetry
		
		v.add(count_closed_components(pixels).size());//number of closed components
		v.add((double)h/w, 5); // height to width ratio
		v.add(count_components(pixels), 0.1); //number of components (if not a merged char will be zero)
		double[] quadrants = weight_quadrants(pixels);
		v.add(quadrants[0], 2); //q1
		v.add(quadrants[1], 2); //q2
		v.add(quadrants[2], 2); //q3
		v.add(quadrants[3], 2); //q4
		v.add(pixel_row_intersects(pixels, h*3/10)); //Ih50
		v.add(pixel_row_intersects(pixels, h*5/10)); //Ih50
		v.add(pixel_row_intersects(pixels, h*8/10)); //Ih50
		v.add(pixel_col_intersects(pixels, w*3/10)); //Iv50
		v.add(pixel_col_intersects(pixels, w*5/10)); //Iv50
		v.add(pixel_col_intersects(pixels, w*8/10)); //Iv50
		
		/* outputs the values of the vector */
		if(output){
			System.out.print("vector values = [");
			for(int i=0; i<v.size(); i++){
				System.out.printf("%.2f ", v.valueAt(i));
			}
			System.out.println("]");
		}
		return v;
	}
	
	/*****************************************************************
	 * 
	 * function: getLetter()
	 * purpose: to classify a test instance using our weka classifcation model
	 * input:
	 * 	model: the trained model we generated using weka and font training data
	 * 	data: the set of training data, need to add the new instances to this set of instances
	 * 	curr_vector: the LetterVecotr that contains the attribute vector for the letter we wish to classify
	 * output:
	 * 	returns the model likely letter according to our classification model
	 *
	 *****************************************************************/
	public static String getLetter(Classifier model, Instances data, LetterVector curr_vector){
		Instance temp = new Instance(data.numAttributes());
		data.add(temp);
		for(int i=0; i<data.numAttributes()-1; i++){
			temp.setValue(data.attribute(i), curr_vector.valueAt(i));
		}
		
		try{
			double[] fDistribution = model.distributionForInstance(temp);
			int max_index = 0;
			for(int i=1; i<fDistribution.length; i++){
				if(Double.compare(fDistribution[i], fDistribution[max_index])>0){
					max_index = i;
				}
			}
			//System.out.printf("the max_index is %d and the character this corresponds to is %s\n", max_index, alpha_tnr[max_index]);
			return alpha_tnr[max_index];
		}catch(Exception e){
			System.err.println("Error occured while classifying the test instance");
			return null;
		}
	}
	
	/*****************************************************************
	 * 
	 * function: pixel_row_sum()
	 * input:
	 * 	pixels (binary array of pixels),
	 * 	row_index: the row we want the pixel sum for
	 * output:
	 * 	the sum of black pixels/the width of the row
	 *
	 *****************************************************************/
	public static double pixel_row_sum(int[][] pixels, int row_index){
		//find the number of bits set true between row_index*width (inclusive) to (row_index+1)*width (exclusive)
		double sum = 0;
		for(int i=0; i<pixels[row_index].length; i++){
			sum += pixels[row_index][i];
		}
		return sum;
	}
	
	/*****************************************************************
	 * 
	 * function: pixel_row_intersects()
	 * input:
	 * 	pixels (binary array of pixels),
	 * 	row_index: the row we want the pixel intersects for
	 * output:
	 * 	the number of times the pixels change from black to white (or vice versa) in that row
	 *
	 *****************************************************************/
	public static double pixel_row_intersects(int[][] pixels, int row_index){
		//find the number of bits set true between row_index*width (inclusive) to (row_index+1)*width (exclusive)
		int intersects = 0;
		for(int i=1; i<pixels[row_index].length; i++){
			if(pixels[row_index][i]!=pixels[row_index][i-1]){
				intersects++;
			}
		}
		return intersects;
	}
	
	/*****************************************************************
	 * 
	 * function: pixel_col_sum()
	 * input:
	 * 	pixels (binary array of pixels),
	 * 	col_index: the col we want the pixel sum for
	 * output:
	 * 	the sum of black pixels/the height of the col
	 *
	 *****************************************************************/
	public static double pixel_col_sum(int[][] pixels, int col_index){
		//sum every col_index'th element
		double sum = 0;
		for(int i=0; i<pixels.length; i++){
			sum += pixels[i][col_index];
		}
		return sum;
	}
	
	/*****************************************************************
	 * 
	 * function: pixel_col_intersects()
	 * input:
	 * 	pixels (binary array of pixels),
	 * 	col_index: the col we want the pixel intersects for
	 * output:
	 * 	the number of times the pixels change from black to white (or vice versa) in that col
	 *
	 *****************************************************************/
	public static double pixel_col_intersects(int[][] pixels, int col_index){
		//sum every col_index'th element
		double intersects = 0;
		for(int i=1; i<pixels.length; i++){
			if(pixels[i-1][col_index]!=pixels[i][col_index]){
				intersects++;
			}
		}
		return intersects;
	}
	
	/*****************************************************************
	 * 
	 * function: reflect_over_vertical()
	 * input:
	 * 	pixels (binary array of pixels),
	 * output:
	 * 	the value representing the horisontal symmetry proportion
	 *
	 *****************************************************************/
	public static double reflect_over_vertical(int[][] pixels){
		int h = pixels.length;
		int w = pixels[0].length;
		double score = 0;
		for(int i=0; i<h; i++){
			for(int j=0; j<w/2; j++){
				if(pixels[i][j]==pixels[i][w-j-1]){
					score++;
				}
			}
		}
		return score/(h*w);
	}
	
	/*****************************************************************
	 * 
	 * function: reflect_over_horizontal()
	 * input:
	 * 	pixels (binary array of pixels),
	 * output:
	 * 	the value representing the vertical symmetry proportion
	 *
	 *****************************************************************/
	public static double reflect_over_horizontal(int[][] pixels){
		int h = pixels.length;
		int w = pixels[0].length;
		double score = 0;
		for(int i=0; i<w; i++){
			for(int j=0; j<h/2; j++){
				if(pixels[j][i]==pixels[h-j-1][i]){
					score++;
				}
			}
		}
		return score/(h*w);
	}
	
	/*****************************************************************
	 * 
	 * function: train_font()
	 * purpose: generate a list of attribute vectors for each letter in the font library
	 * input:
	 * 	String font: the font we wish to train (currently always the default, TimesNewRoman)
	 * output:
	 * 	the list of vectors with their associated known character
	 *
	 *****************************************************************/
	public static ArrayList<LetterVector> train_font(String font){
		ArrayList<LetterVector> alpha_vectors = new ArrayList<LetterVector>();
		String[] alpha = font.equals("TimesNewRoman")? alpha_tnr: alpha_tnr;
		System.out.printf("Training for font = %s on the following characters:\n", font);
		for(int i=0; i<alpha.length; i++){
			train_letter(alpha[i], font, alpha_vectors);			
		}
		System.out.println("\nFinished Training!\n");
		return alpha_vectors;
	}

	/*****************************************************************
	 * 
	 * function: train_letter()
	 * purpose: finds all files associated with a given letter, if there are files it computes the attribute vectors for these images
	 * 	then these vectors are stored in our alpha_vectors library so we can later use them in training our classifier model
	 * input:
	 * 	String font: the font we wish to train (currently always the default, TimesNewRoman)
	 * 	String curr: the letter/String we are looking for
	 * 	alpha_vectors: the list where we will build our library of trained computed vectors
	 * output:
	 * 	none
	 *
	 *****************************************************************/
	public static void train_letter(String curr, String font, ArrayList<LetterVector> alpha_vectors){
		System.out.printf("%s ", curr);
		String foldername = font+"/"; //directory name
		if(curr.equals(".")){
			foldername += "symbol/period";
		}else if(curr.matches("[A-Z]")){
			foldername += "alpha/"+curr.toLowerCase()+"/uppercase";
		}else if(curr.matches("[a-z]")){
			foldername += "alpha/"+curr.toLowerCase()+"/lowercase";
		}else if(curr.matches("[A-Z][A-Z]")){ //double letter combo
			foldername += "alpha/"+curr.toLowerCase()+"/uppercase";
		}else if(curr.matches("[a-z][a-z]")){
			foldername += "alpha/"+curr.toLowerCase()+"/lowercase";
		}else if(curr.matches("[0-9]")){
			foldername += "number/"+curr;
		}else{
			foldername += "symbol/"+curr;
		}
		try{
			File folder = new File(foldername);
			if(folder==null){
				System.out.println("tried opening the direcotry");
				return;
			}
			File[] listOfFiles = folder.listFiles();
			if(listOfFiles==null){
				System.out.printf("\ndid not find files in the directory %s \n", foldername);
				return;
			}
			
			for (File file : listOfFiles) {
			    if (file.isFile()) {
				if(!file.getName().matches(".*.jpg")){ //ignore anything that isn't a jpg file
					continue;
				}
				BufferedImage img_color = ImageIO.read(file);
				//get the heights and widths of the image
				int img_width = img_color.getWidth();
				int img_height = img_color.getHeight();
				BitSet img_data = convertImageToBinary(img_color);
				//output_binary_image(img_data, img_width, img_height, "t");
				//outputBitSet(img_data, img_width, img_height);
				
				ArrayList<Region> components = cluster_connected_pixels(img_data, img_width, img_height);
				Region r = null;
				while(components.size()>0){
					if(r==null){
						r = components.remove(0);
					}else{
						r.mergeSets(components.remove(0));
					}
				}
				//if more than one component merge them since we know this should be just one component
				//outline_components(img_color, components);
				int[][] curr_region = getRegionMatrix(img_data, img_width, img_height, r);
				LetterVector v = compute_attribute_vector(curr_region);
				v.setClassifier(curr);
				alpha_vectors.add(v);
				weka += v.toString()+", \""+curr+"\"\n";
			    }
			}
			
			
		}catch (IOException e){
			System.err.printf("error opening files in directory  %s\n", curr);
		}
		
	}
	
	/*****************************************************************
	 * 
	 * function: outputBitSet()
	 * purpose: purely for debugging purposes. prints a bitset of a component, this we let us see how
	 * 	components are being classified on the command line and what bits are being clustered together
	 * input:
	 * 	data: the bitset for the component
	 * 	w: width of the component
	 * 	h: height of the component
	 * output:
	 * 	none
	 *
	 *****************************************************************/
	public static void outputBitSet(BitSet data, int w, int h){
		for(int i=0; i<h; i++){
			for(int j=0; j<w; j++){
				if(data.get(i*w+j)){
					System.out.printf("%d ", 1);
				}else{
					System.out.printf("%d ", 0);
				}
			}
			System.out.println();
		}
	}
	
	
	/*****************************************************************
	 * 
	 * function: convertImageToBinary()
	 * purpose: initial step. takes in a color image, converts it to greyscale, then computes an average
	 * 	greyscale value and uses this as a threshold for determining a binary matrix. returns the bitset representing the binary matrix
	 * input:
	 * 	img: buffered image that we want to conver
	 * output:
	 * 	BitSet: the binary pixels values of the input image
	 *
	 *****************************************************************/
	public static BitSet convertImageToBinary(BufferedImage img){
		//make the image grayscale
		int width = img.getWidth();
		int height = img.getHeight();
		BufferedImage img_gr = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY); //create a new grascale buffered image
		Graphics g = img_gr.getGraphics();
		g.drawImage(img, 0, 0, null);
		g.dispose();
		
		//now find the average pixel value
		int[][] pixels = new int[height][width];
		int average = 0;
		int min = -1;
		int max = 0;
		for(int i=0; i<height; i++){
			for(int j=0; j<width; j++){
				Color c = new Color(img_gr.getRGB(j, i));
				//rgb for each color should be same in grayscale
				pixels[i][j] = c.getRed();
				average += pixels[i][j];
				if(min<0){
					min = pixels[i][j];
				}else{
					if(pixels[i][j]<min){
						min = pixels[i][j];
					}
				}
				if(pixels[i][j]>max){
					max = pixels[i][j];
				}
			}
		}
		int threshold = average/(width*height);
		
		//choose the average as the threshold. 1 means a point exists, 0 means this point is blank
		BitSet points = new BitSet(width*height);
		for(int i=0; i<height; i++){
			for(int j=0; j<width; j++){
				if(pixels[i][j]<threshold){ //lower numbers are black points
					points.set(i*width+j);
				}
			}
		}
		return points;
	}
}

class LetterVector{
	private static int attr_num = 19;
	private ArrayList<Double> vector;
	private ArrayList<Double> weights;
	private int index_count = 0;
	private String classifier;
	
	
	public LetterVector(){
		this.vector = new ArrayList<Double>();
		this.weights = new ArrayList<Double>();
		this.classifier = null;
	}
	
	public LetterVector(String c){
		this.vector = new ArrayList<Double>();
		this.weights = new ArrayList<Double>();
		this.classifier = c;
	}
	
	public void add(double value, double weight){
		this.vector.add(new Double(value));
		this.weights.add(new Double(weight));
	}
	
	
	public void add(double value){
		this.vector.add(new Double(value));
		this.weights.add(new Double(1));
	}
	
	public double valueAt(int index){
		if(index>this.vector.size()){
			return -1;
		}else{
			return this.vector.get(index);
		}
	}
	
	public void setClassifier(String c){
		this.classifier = c;
	}
	
	public int size(){ return this.vector.size(); }
	
	public String getClassifier(){ return this.classifier; }
	
	//make sure you always add attr in the same order and amount
	public double computeDistanceTo(LetterVector in){
		int len = this.vector.size();
		if(in.vector.size()<this.vector.size()){
			len = in.vector.size();
		}
		double distance = 0;
		for(int i=0; i<len; i++){
			distance += in.weights.get(i)*Math.abs(this.vector.get(i)-in.vector.get(i));
		}
		return distance;
	}
	
	public String toString(){
		String result = "";
		for(int i=0; i<this.vector.size(); i++){
			if(i==this.vector.size()-1){
				result += this.vector.get(i);
			}else{
				result += this.vector.get(i)+", ";
			}
			
		}
		return result;
	}
}

class RegionXComparator implements Comparator<Region> {
    @Override
    public int compare(Region a, Region b) {
	return a.getXmin() - b.getXmin();
    }
}

class Region{
	private TreeSet<Integer> equivalent_colors;
	private int color_id;
	private int xmin;
	private int ymin;
	private int xmax;
	private int ymax;
	private char ch;
	private ArrayList<int[]> noise;
	
	public int getCX(){ return (this.xmax-this.xmin)/2 + this.xmin;}
	public int getCY(){ return (this.ymax-this.ymin)/2 + this.ymin;}
	
	public Region(){
		this.equivalent_colors = new TreeSet<Integer>();
		this.xmin = -1;
		this.xmax = -1;
		this.ymin = -1;
		this.ymax = -1;
		this.color_id = 0;
		this.ch = ' ';
		this.noise = new ArrayList<int[]>();
	}
	
	private TreeSet<Integer> getColors(){
		return equivalent_colors;
	}
	
	public int getColorID(){ return color_id; }
	public int getXmin(){ return xmin; }
	public int getYmin(){ return ymin; }
	public int getXmax(){ return xmax; }
	public int getYmax(){ return ymax; }
	
	public boolean isEquivalent(int c){
		if(equivalent_colors.contains(c)){
			return true;
		}else{
			return false;
		}
	}
	
	public void addColor(int c){
		equivalent_colors.add(new Integer(c));
		color_id = equivalent_colors.first().intValue();
	}
	
	public void mergeSets(Region c){
		equivalent_colors.addAll(c.getColors());
		color_id = equivalent_colors.first().intValue();
		this.updateX(c.getXmin());
		this.updateX(c.getXmax());
		this.updateY(c.getYmax());
		this.updateY(c.getYmin());
	}
	
	public void markNoise(int xmin, int xmax, int ymin, int ymax){
		int[] n = {xmin, ymin, xmax, ymax};
		noise.add(n);
	}
	
	public ArrayList<int[]> getNoiseList(){
		return this.noise;
	}
	
	public void setXmax(int x){
		this.xmax = x;
	}
	
	public void updateX(int x){
		if(xmax<0){
			xmax = x;
		}
		if(xmin<0){
			xmin = x;
		}
		if(x>xmax){
			xmax = x;
		}
		if(x<xmin){
			xmin = x;
		}
	}
	
	public void updateY(int y){
		if(ymax<0){
			ymax = y;
		}
		if(ymin<0){
			ymin = y;
		}
		if(y>ymax){
			ymax = y;
		}
		if(y<ymin){
			ymin = y;
		}
	}
	
	public String toString(){
		String result = "";
		result += "color_id = "+color_id;
		result += " min ("+this.xmin+", "+this.ymin+") ";
		result += "max ("+this.xmax+", "+this.ymax+") ";
		result += "equivalent sets = ";
		for(Integer n: equivalent_colors){
			result += n+" ";
		}
		return result;
	}
	
	public void setChar(char ch){
		this.ch = ch;
	}
	
	public char getChar(){ return this.ch; }
	
	public boolean checkRangeOverlap(Region r){
		if(r.getYmin()<this.getYmin()){ //...r...this...
			if(r.getYmax()>=this.getYmin()){
				return true;
			}
		}else{ //...this...r...
			if(this.getYmax()>=r.getYmin()){
				return true;
			}
		}
		return false;
	}
	
	//return false if the overlap is less than half the smaller domain
	public boolean checkDomainOverlap(Region r){		
		if(r.getXmin()<this.xmin){ //...r...this...
			if(r.getXmax()>=this.xmin){
				return true;
			}
		}else{ //...this...r...
			if(this.xmax>=r.getXmin()){
				return true;
			}
		}
		return false;
	}
	
	public double getDomainOverlapAsPercent(Region r){
		double smaller_domain = 0;
		if(r.getXmax()-r.getXmin()<this.xmax - this.xmin){
			smaller_domain = r.getXmax()-r.getXmin()+1;
		}else{
			smaller_domain = this.xmax - this.xmin + 1;
		}
		if(r.getXmin()<this.getXmin()){ //...r...this...
			if(r.getXmax()>=this.xmin){
				return (r.getXmax() - this.xmin + 1)/smaller_domain;
			}
		}else{ //...this...r...
			if(this.xmax>=r.getXmin()){
				return (this.xmax - r.getXmin() + 1)/smaller_domain;
			}
		}
		return 0;
	}
	
}