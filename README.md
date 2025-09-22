A demo implementation of the SDD task experiment described in the submission, entitled **Open-World Pedestrian Trajectory Prediction**.



**Installation & Requirements**

The current version of the codes has been tested with Python 3.8.20 on both Windows and Linux operating systems with the following requirements:

- 1x RTX 3080
- cuda==12.7
- numpy==1.21.4
- scipy==1.5.0
- torch==2.0.0
- tqdm==4.48.0
- pyyaml==5.3.1
- matplotlib==3.2.2
- pandas==1.0.5
- opencv-python==4.4.0.42
- segmentation_models_pytorch==0.1.0

Please install the necessary packages


To download the pre-trained segmentation model weights, refer to https://drive.google.com/file/d/1u4hTk_BZGq1929IxMPLCrDzoG3wsZnsa/view?usp=sharing. Place the **SDD_segmentation.pth** file in the **segmentation_models** folder.
We declare that link originates from a published work cited in this paper. This link contains no information relevant to the present work. This codebase remains entirely anonymous.




**How To Use**

For ease of implementation, the detection and accommodation processes are evaluated separately. 

First, run the following command to complete the accommodation process and test predictive performance. 

`python train_GMPDR_accommodation.py`

Subsequently, run the following command to test the detection process, which verifies whether detection succeeds when new movement patterns are introduced.

``python inference_GMPDR_detection.py``



**License**
The codes are proprietary. Please do not distribute.

