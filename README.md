# FairDeFace
A Framework to Evaluate Face Obfuscation Methods and Their Fairness

---

## Overview
FairDeFace is a framework designed to evaluate face obfuscation methods with a focus on fairness. It consists of five main components:

1. **Face Obfuscation**
2. **Quality Analysis: Face Detection**
3. **Privacy Analysis: Verification and Identification**
4. **Attribute Analysis**

---

## 1. Face Obfuscation

### Inputs and Outputs
- **Input**: Datasets
- **Output**: Obfuscated datasets

### Dataset Preparation
Datasets can be accessed through this link: [Dataset Link](https://drive.google.com/drive/folders/1z9vHJRBny8isg7wmA5w4KufM6VeBRJj0?usp=drive_link).

1. Download the datasets and unzip them into the `datasets` directory in the main project folder.
2. The `Original` folder contains four demographic-based datasets: BFW, DemogPairs, RFW, and CelebA.
3. Obfuscate one or more datasets using your chosen method.

Results for seven obfuscation methods are provided as examples.

### Generating Obfuscated Faces
To generate obfuscated datasets, run the following command:

```bash
python obfuscation.py --method_file_path <path_to_obfuscation_method_code> \
                      --method <method_name> \
                      --datasets_path <path_to_datasets> \
                      --datasets <dataset_list>
```

#### Example:
```bash
python obfuscation.py --method_file_path Pixelation.py \
                      --method Pixelation \
                      --datasets_path datasets \
                      --datasets BFW,DemogPairs
```
This creates a folder named `Pixelation` in the `datasets` directory containing the obfuscated datasets.

**Note**:
- Ensure you have downloaded the required obfuscation models and installed all necessary libraries.
- Consider using Anaconda for dependency management with the provided `.yml` files, although compatibility may vary.

---

## 2. Face Detection

### Detect Faces
Run the following command to analyze face detection:

```bash
python Detection_deepface.py --method <obfuscation_method> \
                             --datasets_path <datasets_path> \
                             --detection_methods <methods_list>
```

#### Example:
```bash
python Detection_deepface.py --method Pixelation \
                             --datasets_path datasets \
                             --detection_methods mtcnn,opencv
```

### View Detection Results
To analyze detection results:

```bash
python Detection_results.py --detection_file <detection_results_file> \
                            --pass_rates <yes|no> \
                            --fig_style <style>
```

#### Example:
```bash
python Detection_results.py --detection_file Detection_Pixelation_mtcnn_000151.pkl \
                            --pass_rates no \
                            --fig_style all
```

---

## 3. Privacy Analysis: Verification and Identification

### Verification
Perform verification analysis with the following command:

```bash
python Verification_DeepFace.py --method <obfuscation_method> \
                                --same_photo <yes|no> \
                                --same_identity <yes|no> \
                                --detector <detector> \
                                --models <models> \
                                --step <step> \
                                --datasets <dataset_list>
```

#### Example:
```bash
python Verification_DeepFace.py --method Pixelation \
                                --same_photo yes \
                                --same_identity no \
                                --detector mtcnn \
                                --models ArcFace \
                                --step 500 \
                                --datasets BFW,DemogPairs
```
**Note**: Use `same_photo` for obfuscation methods like Pixelation. For original datasets, use `same_identity`.

### View Verification Results
```bash
python Verification_results.py --file_path <verification_results_file> \
                               --fig_style <style>
```

#### Example:
```bash
python Verification_results.py --file_path Verification_Pixelation_same-photo_000041.pkl \
                               --fig_style all
```

### Identification
#### Collect Representations
```bash
python Identification_DeepFace.py --method <obfuscation_method> \
                                  --detector <detector> \
                                  --FR_models <models> \
                                  --step <step> \
                                  --datasets <dataset_list>
```

#### Train and Test SVM
```bash
python Identification_SVM.py --obfuscation_file <obfuscation_file> \
                             --original_file <original_file> \
                             --train_test_ratio <ratio> \
                             --scenarios <scenario_list>
```
- **Scenario 1**: Train each demographic separately.
- **Scenario 2**: Train all demographics together.

#### Example:
```bash
python Identification_SVM.py --obfuscation_file Identification_Pixelation_000385.pkl \
                             --original_file Identification_Original_000862.pkl \
                             --train_test_ratio 0.7 \
                             --scenarios 1,2
```

### View Identification Results
```bash
python Verification_results.py --file_path <identification_scores_file> \
                               --fig_style <style>
```

#### Example:
```bash
python Verification_results.py --file_path Identification_Scores_Pixelation_000016.pkl \
                               --fig_style all
```

---



## 4. Attribute Analysis

### Collect Attributes
```bash
python Attribute_DeepFace.py --method <obfuscation_method> \
                             --datasets_path <datasets_path>
```

### Compare Attributes
```bash
python Attribute_results.py --obfuscation_file <obfuscation_file> \
                            --original_file <original_file> \
                            --fig_style <style>
```

#### Example:
```bash
python Attribute_results.py --obfuscation_file Attribute_DP2_000181.pkl \
                            --original_file Attribute_Original_000382.pkl \
                            --fig_style all
```

---

## Contribution
Feel free to contribute to FairDeFace by submitting pull requests or reporting issues.

---

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Acknowledgements
Special thanks to all contributors and researchers in the field of privacy-preserving AI.

