# FairDeFace
a framework to evaluate face obfuscation methods and their fairness


FairDeFace consists of three sections:
1- Face Obfuscation
2- Quality Analysis: Face Detection
3- Privacy Analysis: Verification and Identification
4- Utility Analysis


1- Face Obfuscation:
inputs: datasets
output: obfuscated datasets
You can access the datasets through this link: [https://drive.google.com/drive/folders/1vUkg2rBfg7-Ff0QWwyGf3aFhUbaLWNLJ?usp=sharing](https://drive.google.com/drive/folders/1z9vHJRBny8isg7wmA5w4KufM6VeBRJj0?usp=drive_link)


Inside the datasets directory are multiple zip files containing the original and obfuscated datasets. Every dataset needed should be unzipped inside the datasets directory. The datasets directory should be in the main project directory. 

Original: Inside the Original directory, there are 4 demographics based directories:BFW, DemogPairs, RFW, and CelebA
You can choose one or more datasets and obfuscate them using your chosen method. 

The results for 7 obfuscation methods are available. 

To generate the obfuscated faces, install the needed libraries and run this command in the terminal:

python obfuscation.py --method_file_path < the path of the obfuscation method code > --method <choose a name for your method such as Pixelation> --datasets_path < the source datasets path which is defaulted to datasets> --datasets <the lists of source datasets that are separated with comma defaulted to BFW, DemogPairs, RFW, CelebA>

Example: python obfuscation.py --method_file_path  Pixelation.py --method Pixelation --datasets BFW,DemogPairs

The above example uses Pixelation.py to generate a folder named Pixelation in the datasets folder having obfuscated datasets
note that you need to download your chosen obfuscation model, install the libraries, and make sure they properly work.
You can use Anaconda and our yml files, but they might not be compatible with your system.

2- Detection
command example: python Detection_deepface.py --method Pixelation --datasets_path datasets --detection_methods mtcnn, opencv

--datasets_path and --detection_methods are defaulted to datasets and mtcnn

Detection Results:
command example: python Detection_results.py --detection_file Detection_Pixelation_mtcnn_000151.pkl  --pass_rates no  --fig_style all
choose yes for pass_rates to show the passing rate results

3- Verification
command example: python Verification_DeepFace.py --method Pixelation --same_photo yes --same_identity no --detector mtcnn --models ArcFace --step 500 --datasets BFW,DemogPairs

same_photo is used for the two scenarios in verification and same_identity is used for TPRs and TFRs for face recognition. Here, becase Pixelation is an obfuscation metho, only same_photo values are considered. If the method is Original, only same_identity is considered. The defaults values for the face detector, verification model, and datasets are as in the example.

Verification Results:
command example: python Verification_results.py --file_path Verification_Pixelation_same-photo_000041.pkl   --fig_style all

4-Identification
First, the representations are collected
command example: python Identification_DeepFace.py --method Pixelation  --detector mtcnn --FR_models ArcFace --step 500 --datasets BFW,DemogPairs

Identification SVM
command example: python Identification_SVM.py --obfuscation_file Identification_Pixelation_000385.pkl  --original_file Identification_Original_000862.pkl  --train_test_ratio .7 --scenarios 1,2  
For scenarios, choose 1 when training each demographic separately, or 2 when training all demographics together. The default is 1,2.


Identification Results:
command example: python Verification_results.py --file_path Identification_Scores_Pixelation_000016.pkl  --fig_style all

5- Attributes
First, the attributes are collected:
command example: python Attribute_DeepFace.py --method Pixelation --datasets_path datasets 

Then, they are compared to the original and the results are shown:
command example: python Attribute_results.py --obfuscation_file Attribute_DP2_000181.pkl --original_file Attribute_Original_000382.pkl --fig_style all
