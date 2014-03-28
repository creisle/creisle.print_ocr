creisle.print_ocr
=================

Authors: Caralyn Reisle and Sunette Mynhardt

Final Project for a datamining course

Project goal was to take a jpg as input and produce a plain text file with the printed text extracted from the image. Computes character attributes and uses Weka to build ann and svm models for character clasification.


Library Structure for training images:

<Font>/<alpha>/<character>/<uppercase or lowercase>
<Font>/<symbol>/<character>
<Font>/<number>/<digit>

each directory must contain at leat two test images (better results with more images) that contain only 1 instance of that character

to train/add a new character, create a directory for that character in the approriate location and then add the same character to the global variable alpha_tnr. note that at this point in time this program is built to train on TimesNewRoman fonts only
