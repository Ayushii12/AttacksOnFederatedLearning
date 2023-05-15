# AttacksOnFederatedLearning 

Federated Learning (FL) has become one of the most extensively utilized distributed training approaches since it allows users to access large datasets without really sharing them. Only the updated model parameters are exchanged with the central server after the model has been trained locally on the devices holding the data. <br>

 <h3> Objective </h3> 
Exploiting the distributed nature of the FL technique to manipulate the behavior of the model. <br>
The study explores targeted clean-label attack in which adversaries inject poisoned images into compromised clients’ dataset <br>
to alter the behaviour of the model on a specific target image at test time. <br>

<h3> Dataset </h3>
The standard CIFAR10 dataset is used in this study to conduct various experiments and manipulate the image classifier. 

<h3> Experiments </h3>

***Success Rate Calculation*** <br>
The success rate is determined by randomly selecting 30 images from a specific class and measuring the model's ability to misclassify them. For instance, if the model misclassifies 20 out of the 30 selected images, the attack success rate is calculated as 20/30. <br>

***Experiment 1*** <br>
In this experiment, we investigate the impact of varying the number of poisonous images while keeping the number of malicious clients constant. The range of the number of poisonous images tested is [0, 10, 20, 30, 40, 50, 60, 70, 80]. <br>

***Experiment 2***<br>
This experiment aims to examine the attack success rate when the number of malicious clients is varied while maintaining a constant number of poisonous images. The range of the number of malicious clients tested is [0, 1, 2, 3, 4, 5, 6]. <br>

***Experiment 3***<br>
Here, we explore the relationship between the number of poisonous images injected into the dataset and the overall model accuracy. This experiment focuses on observing the changes in the model's accuracy as the number of poisonous images increases. <br>

<h3> Results </h3>
[1] The attack’s impact grows in direct proportion to the number of injected poisonous images and malicious client (i.e. controlled by adversaries) participating in the FL process. <br>
[2] The behavior of a FL model can be altered maliciously towards a specific target image without significantly affecting the model’s overall accuracy. <br>


