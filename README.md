# Generalized Seizure Detection Algorithm

### Goal
Low-income regions of the world cannot afford professional-grade EEG equipment to detect seizures. 
This project seeks to determine whether a machine-learning algorithm can effectively predict seizures using only 3 channels of EEG data.

### Data Processing

<img src=https://user-images.githubusercontent.com/38896123/157976056-786e429f-b6b6-429b-94e0-864455d44b84.jpg width=400>

#### Steps
1. Collect the EEG data from three channels of the CHB-MIT database (channels from frontal, temporal, and parietal lobes)
2. Filter raw EEG data through 8 bandpass filters
3. Separate the EEG readings into 2-second epochs
4. Attain a power value for each of the epochs (3 channels of readings, 8 filters, 24 power values per timestep)
5. Split full dataset into training and testing set. Training set is 80% of total compared to testing set which is 20% of the total.
6.  Train each model over 100 epochs  on each subset of the data (3 subsets, 3 models, 9 total model variants)

![variables](https://user-images.githubusercontent.com/38896123/157975895-2cb0edcb-c04b-45f7-9ad3-f3de26ae1cdd.jpg)

7.  Evaluate each model using test data (Acquire Confusion Matrix)
8. Convert confusion matrix into precision, recall, specificity, and sensitivity
9. Convert specificity to false positives per hour

### Model

Models were developed in three levels of increasing complexity.

#### Model Architecture

<img src = https://user-images.githubusercontent.com/38896123/157976688-02b98ebb-cd0a-4525-8c80-de884ea3f111.jpg width=400>

#### Performance

![performance](https://user-images.githubusercontent.com/38896123/157976423-7c6a7af9-76cd-4a5b-b42a-2a20f64b765c.jpg)

### Results & Conclusion

Our best model, RNN1, performed comparatively to various existing algorithms. 
Compared to neural network-based algorithms Reveal, CNet, and Sensa, we achieve significantly higher sensitivity, with each of them yielding 76%, 35.4%, and 48.2% respectively. 
However, Reveal and Sensa yielded fewer false positives per hour. We postulate that this is due to the Reveal and Sensa algorithms being programmed with physiologically relevant rules, prompting the tradeoff of specificity over false positives. 

However, RNN1 was significantly outperformed by the algorithm presented in Guttag-Shoeb, which utilized a Support Vector Machine on the CHB-MIT database. 
A notable distinction is that the Guttag-Shoeb algorithm utilized 23 channels of EEG, while our models were using 3 channels.

We conclude that the usage of lesser channel EEG data for generalized seizure detection is possible and can perform well. 
However, further real-world testing is required to determine our model’s realistic performance.
