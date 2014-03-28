Optical Character Recognition of printed text
=================

Authors: Caralyn Reisle and Sunette Mynhardt

Final Project for a datamining course

Project Goal: take jpg file input, extract text, output to plain text file

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
