Optical Character Recognition of printed text
=================

**Authors: Caralyn Reisle and Sunette Mynhardt**

Date: 2014 March 27



Project Goal: take jpg file input, extract text, output to plain text file (Final Project for a datamining course)

Notes: currently only works for clean (no noise) straight images. Later versions will attempt more image cleaning in the pre-processing steps as well as a more robust scanning algorithm for isolating putative text components. See presentation.pdf for more details and references.

##How it works?
- Pre-Processing:
    - Convert Image to Binary
    - Clustering to isolate text components
- Classification: Training
    - Compute Attribute vectors for each component binary array
    - Output the training image vectors to an arff file
- Classification: Model
    - build the classification model using the weka (http://www.cs.waikato.ac.nz/ml/weka/) java api and the output arff file
- Other processing:
    - for test images, finding lines of text. Separating text components based on overlapping range values
    - finding spaces between words. Makes the assumption that there are two types of spaces in the image (words vs characters). Computes a list of values for spaces and separates it into two groups attempting to minimize the maximum sum squared error (sse). Compares values when outputing text to the averages of these two groups


##Library Structure for training images:

alpha characters: \<Font>/\<alpha>/\<character>/\<uppercase or lowercase>

symbols: \<Font>/\<symbol>/\<character>

numbers: \<Font>/\<number>/\<digit>

each directory must contain at leat two test images (better results with more images) that contain only 1 instance of that character

to train/add a new character, create a directory for that character in the approriate location and then add the same character to the global variable alpha_tnr. note that at this point in time this program is built to train on TimesNewRoman fonts only

##Runnning/Compiling on command line

**compile**: javac -cp weka.jar PrintOcr.java

note: you will need to have the weka.jar file in the same directory or specify a different classpath

**run**: java -cp weka.jar:. PrintOcr \<test image> \<optional parameters>

optional parameters:

1. **train**: generates the arff file. note you will need the training image library for this
2. **output**: additional ouput for debugging purposes. outputs to command line as well as producing a copy of the original image that shows the performance of the clustering algorithm
3. **svm**: if included weka will generate the classifier model using SMO model, else will use the MultilayerPerceptron model
4. **eval**: outputs an evaluation summary of the model tested on the training data
