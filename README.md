# MedicalDiagnosis-KNN-NB
Comparing the accuracy of KNN and Naive Bayes for detecting hepatitis and diabetes


**Simple Diagnoses using KNN and Decision Trees**

**Nicholas Dahdah (260915130), Jonayed Islam (260930902), Jasper Yun (260651891)**

**Abstract**

Machine learning models are increasingly applied to classification tasks in which they can achieve greater accuracy than expert human equivalents. However, complex models are computationally expensive; simpler models may offer similar performance at a reduced cost. We implemented K-nearest neighbors (KNN) and decision tree (DT) models which were trained on benchmark hepatitis and diabetes datasets. In the training process, we examined the cross-correlation between features and outcomes to determine the best features to use. We used 10-fold cross-validation to compare the training accuracy against the validation accuracy. On the hepatitis dataset, we achieved testing accuracies of 76.7% and 83.8%, for the KNN and DT, respectively. On the diabetes dataset, we achieved testing accuracies of 64.4% and 68.6%, for the KNN, and DT, respectively. While this performance does not surpass human expert performance, the results of our classifiers can be used to aid doctors in the automated rapid classification of patients.

**Introduction **

Simple machine learning models can be used for classification with surprisingly good accuracy. We implemented K-nearest neighbors (KNN) and decision tree models on two benchmark datasets, hepatitis [1] and diabetes [2].

Various previous papers have used the hepatitis dataset to compare different ML models for classification accuracy. For instance, [3] implemented a classifier using a nonlinear-weighted Bayes discriminant function; their best results on the Hepatitis dataset achieved a 95.4% training accuracy with 79.4% test accuracy when using 6.5 features, on average, of the 19 total features in the dataset. In work by the same authors using a genetic algorithm combined with the KNN, their classifier achieved 86.0% training and 69.6% test accuracies using 8.1 features.

The diabetes dataset has been used in research by [2] in which an ensemble system of ML models was applied to predict the presence of DR. The ensemble combines the outputs of multiple models to produce a more accurate prediction. Using this method, the ensemble system achieved 94% sensitivity, 90% specificity, and 90% accuracy.

**Datasets**

The two datasets used in this project are the Hepatitis Data Set and the Diabetic Retinopathy (DR) Debrecen Data Set, both pulled from the UC Irvine (UCI) Machine Learning Repository (MLR). The Hepatitis Data Set contains information from 155 subjects with hepatitis regarding the fatality of the disease, along with information about each subject and their symptoms. Unfortunately, this dataset contains missing values, labeled with a “?,” which were removed before training any model, so as to not skew classification. The DR Debrecen Dataset contains information extracted from 1151 samples of the Messidor image set pertaining to the characteristics of retinal abnormalities and whether the sample contains signs of DR. This dataset does not have any missing values.

To better understand the data, our team computed the correlation between the feature we aimed to predict (fatality of hepatitis and signs of diabetic retinopathy) and each of the other features in the dataset and compiled it into Table 1. This identified the features that were most related to our feature of interest. For the hepatitis data, the four most correlated features were ascites, albumin, histology, and protime. For the diabetes data, the four most correlated features were the first four MA detection features. This correlation information also allowed us to drop certain features when training our KNN model and DT, as relatively little information about the feature we aim to predict is gained from features that do not correlate strongly with it. This will be addressed more at the end of the Results section.

It is worth noting that when training the KNN model, the hepatitis and diabetes datasets were normalized since KNN is sensitive to scaling. We did this to equally weigh each feature before training. Normalizing the dataset for KNN improves accuracy. We did not normalize the datasets for the DT.

**Table 1. Correlation of features to diagnosis of hepatitis fatality (left) and DR (right)**


<table>
  <tr>
   <td>

<p id="gdcalert1" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image1.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert2">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


<img src="images/image1.png" width="" alt="alt_text" title="image_tooltip">


<p id="gdcalert2" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image2.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert3">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


<img src="images/image2.png" width="" alt="alt_text" title="image_tooltip">

   </td>
   <td>

<p id="gdcalert3" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image3.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert4">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


<img src="images/image3.png" width="" alt="alt_text" title="image_tooltip">


<p id="gdcalert4" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image4.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert5">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


<img src="images/image4.png" width="" alt="alt_text" title="image_tooltip">

   </td>
  </tr>
</table>


According to the donation policy of the UCI MLR, all datasets require explicit permission to be made public and all personally identifiable information must be removed. While this policy protects subjects from violations of privacy, which is an important ethical consideration, it does not ensure that the datasets themselves are representative of present-day reality. Verification by independent third parties, as well as the revisitation of older datasets to ensure the data is still in-line with current medical standards, would substantiate the credibility of these datasets.

**Results**

After tuning the hyperparameters of the KNN and DT models, K and maximum depth, respectively, via cross-validation, we evaluated our models using a test set of data unique from the training and validation data. The accuracy of each model on each dataset is presented in Table 2. The DT model outperforms the KNN model by a margin of 7.1% and 4.2% on the hepatitis dataset and diabetes datasets, respectively. Tables 3 and 4 show the confusion matrix of both the KNN and DT models run on test sets of the hepatitis and diabetes datasets. 

**Table 2. KNN and DT test accuracy**


<table>
  <tr>
   <td>
   </td>
   <td><strong>KNN</strong>
   </td>
   <td><strong>DT</strong>
   </td>
  </tr>
  <tr>
   <td>Hepatitis
   </td>
   <td>76.7% (K = 7, Euclidean)
   </td>
   <td>83.8% (Max depth = 5, Entropy)
   </td>
  </tr>
  <tr>
   <td>Diabetes
   </td>
   <td>64.4% (K = 22, Euclidean)
   </td>
   <td>68.6% (Max depth = 8, Gini)
   </td>
  </tr>
</table>


**Table 3. KNN confusion matrix from test set**


<table>
  <tr>
   <td>

<table>
  <tr>
   <td>
   </td>
   <td colspan="3" ><strong>PREDICTED</strong>
   </td>
  </tr>
  <tr>
   <td rowspan="3" ><strong>ACTUAL</strong>
   </td>
   <td>Total: 30
   </td>
   <td>Live
   </td>
   <td>Die
   </td>
  </tr>
  <tr>
   <td>Live
   </td>
   <td>22
   </td>
   <td>6
   </td>
  </tr>
  <tr>
   <td>Die
   </td>
   <td>1
   </td>
   <td>1
   </td>
  </tr>
</table>


   </td>
   <td>

<table>
  <tr>
   <td>
   </td>
   <td colspan="3" ><strong>PREDICTED</strong>
   </td>
  </tr>
  <tr>
   <td rowspan="3" ><strong>ACTUAL</strong>
   </td>
   <td>Total: 275
   </td>
   <td>DR+
   </td>
   <td>DR-
   </td>
  </tr>
  <tr>
   <td>DR+
   </td>
   <td>82
   </td>
   <td>41
   </td>
  </tr>
  <tr>
   <td>DR-
   </td>
   <td>57
   </td>
   <td>95
   </td>
  </tr>
</table>


   </td>
  </tr>
</table>


**Table 4. DT confusion matrix from test set**


<table>
  <tr>
   <td>

<table>
  <tr>
   <td>
   </td>
   <td colspan="3" ><strong>PREDICTED</strong>
   </td>
  </tr>
  <tr>
   <td rowspan="3" ><strong>ACTUAL</strong>
   </td>
   <td>Total: 30
   </td>
   <td>Live
   </td>
   <td>Die
   </td>
  </tr>
  <tr>
   <td>Live
   </td>
   <td>23
   </td>
   <td>1
   </td>
  </tr>
  <tr>
   <td>Die
   </td>
   <td>4
   </td>
   <td>2
   </td>
  </tr>
</table>


   </td>
   <td>

<table>
  <tr>
   <td>
   </td>
   <td colspan="3" ><strong>PREDICTED</strong>
   </td>
  </tr>
  <tr>
   <td rowspan="3" ><strong>ACTUAL</strong>
   </td>
   <td>Total: 275
   </td>
   <td>DR+
   </td>
   <td>DR-
   </td>
  </tr>
  <tr>
   <td>DR+
   </td>
   <td>118
   </td>
   <td>0
   </td>
  </tr>
  <tr>
   <td>DR-
   </td>
   <td>56
   </td>
   <td>101
   </td>
  </tr>
</table>


   </td>
  </tr>
</table>


_KNN_

To determine the effect of K on the KNN model’s testing accuracy, we ran a 10-fold cross-validation on the hepatitis and diabetes datasets while varying the value of K. For the hepatitis and diabetes datasets, K was varied from 1 to 25 and 1 to 100, respectively; the maximum value of K was limited by the size of the available dataset. Figure 1 shows the validation accuracy obtained from this process as the value of K was varied. There is no specific trend when varying K, but we note that the optimal values for K are 7 and 22 for the hepatitis and diabetes datasets, respectively.



<p id="gdcalert5" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image5.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert6">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image5.png "image_tooltip")


<p id="gdcalert6" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image6.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert7">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image6.png "image_tooltip")


**Fig. 1. Validation accuracy with varying K for the hepatitis (left) and diabetes (right) datasets**

We also examined the effect of varying the model’s distance function. A 10-fold cross-validation with varying distances yielded the validation accuracies shown in Table 5.

**Table 5. KNN validation accuracy versus distance function**


<table>
  <tr>
   <td><strong>Distance Function</strong>
   </td>
   <td><strong>Hepatitis</strong>
   </td>
   <td><strong>Diabetes</strong>
   </td>
  </tr>
  <tr>
   <td>Euclidean
   </td>
   <td>88.0% (K = 7)
   </td>
   <td>66.7% (K = 22)
   </td>
  </tr>
  <tr>
   <td>Manhattan
   </td>
   <td>88.0% (K = 7)
   </td>
   <td>65.1% (K = 22)
   </td>
  </tr>
</table>


Figures 2 and 3 show the decision boundaries (of the most correlated features) obtained from training the KNN model on the hepatitis and diabetes datasets. These decision boundaries form complex regions, which may indicate overfitting of the model. However, reducing the numerous features to a 2-dimensional decision boundary introduces a large loss of information, but including more than 2 features is difficult to illustrate in a meaningful way. 



<p id="gdcalert7" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image7.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert8">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image7.png "image_tooltip")
        

<p id="gdcalert8" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image8.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert9">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image8.png "image_tooltip")


**Fig. 2.  KNN decision boundary of the hepatitis dataset**



<p id="gdcalert9" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image9.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert10">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image9.png "image_tooltip")
        

<p id="gdcalert10" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image10.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert11">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image10.png "image_tooltip")


**Fig. 3.  KNN decision boundary of the diabetes dataset**

Using the results of the experimentation, we determined the features that when removed, led to an increase in accuracy, and we removed these features before training our models. As a result, the validation accuracy on the hepatitis and diabetes datasets became 86.67% and 67.33%, respectively, using 12 of 19 and 12 of 20 features. This reduces the complexity and run-time of the machine learning models while improving the overall validation accuracy. However, we note that specific combinations of removed features may yield even better validation accuracies, as evidenced in Table 7.

**Table 7. Validation accuracy versus removed features for the hepatitis (left) and diabetes (right) datasets.**



<p id="gdcalert11" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image11.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert12">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image11.png "image_tooltip")


<p id="gdcalert12" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image12.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert13">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image12.png "image_tooltip")


_Decision Tree_

To determine the effect of tree depth on the accuracy of the DT model, we ran a 10-fold cross-validation on the hepatitis and diabetes datasets, varying max depth from 1 to 20 to find its optimal value. The validation accuracies achieved are shown in Fig. 4. The optimal values for maximum tree depth are 5 and 8 for the hepatitis and diabetes datasets, respectively. It is worth noting that we saw no change in accuracy for depth greater than 12.



<p id="gdcalert13" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image13.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert14">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image13.png "image_tooltip")


<p id="gdcalert14" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image14.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert15">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image14.png "image_tooltip")


**Fig. 4. Validation accuracy with varying max depth for the hepatitis (left) and diabetes (right) datasets**

For both datasets, we found that varying the cost function did not produce significantly different results. A 10-fold cross-validation with varying cost functions yields the validation accuracies shown in Table 6. 

**Table 6. DT validation accuracy versus cost function**


<table>
  <tr>
   <td><strong>Cost Function</strong>
   </td>
   <td><strong>Hepatitis</strong>
   </td>
   <td><strong>Diabetes</strong>
   </td>
  </tr>
  <tr>
   <td>Misclassification
   </td>
   <td>82.5% (Max depth = 5)
   </td>
   <td>64.2% (Max depth = 8)
   </td>
  </tr>
  <tr>
   <td>Entropy
   </td>
   <td>80.0% (Max depth = 5)
   </td>
   <td>64.5% (Max depth = 8)
   </td>
  </tr>
  <tr>
   <td>Gini Index
   </td>
   <td>83.8% (Max depth = 5)
   </td>
   <td>63.6% (Max depth = 8)
   </td>
  </tr>
</table>


Figures 5 and 6 show the decision boundaries (of the most correlated features) obtained from training the DT model on the hepatitis and diabetes datasets. Unlike the KNN decision boundaries, the DT decision boundaries are much clearer, indicating distinct boundaries for specific diagnoses. Low albumin levels and medium ascites levels were indicative of death from hepatitis.

The decision boundary plots of the diabetes data show two groups. Those on the top left with higher MA3 or MA2 values and lower MA1 and MA4 values were more likely to have DR. 



<p id="gdcalert15" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image15.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert16">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image15.png "image_tooltip")


<p id="gdcalert16" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image16.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert17">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image16.png "image_tooltip")


**Fig. 5.  DT decision boundary of the hepatitis dataset**



<p id="gdcalert17" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image17.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert18">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image17.png "image_tooltip")


<p id="gdcalert18" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image18.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert19">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image18.png "image_tooltip")


**Fig. 6.  DT decision boundary of the diabetes dataset**

**Discussion and Conclusion**

We can extract certain trends from the results presented above. For additional insight, we also noted the effect of dropping certain features that were uncorrelated to the feature we aimed to predict.

Firstly, we note there is a strong link between the accuracy of both the KNN and DT models and the correlation of the features. Since the hepatitis dataset has features with higher correlations to the feature we aim to predict, we obtain test accuracies that are relatively higher than those of the diabetes dataset, which has features that are not as highly correlated with the output. 

For the KNN model, we examined the effect of different distance functions and values of K on validation accuracy. For the DT model, we examined the effect of different cost functions and tree depths on validation accuracy. There are only minor differences in accuracy when using different distance and cost functions. We conclude that in the scope of these datasets, it is not the most important hyperparameter to tune for our models. However, the value of K for KNN and the tree depth for DT are more important. For KNN models, smaller values of K cause the model to overfit to the training data, while very large values of K cause the model to underfit. As we varied K in our experiment, we were able to find an optimal value that maximized validation accuracy. For DT models, when the depth is too shallow the model tends to underfit the data, whereas overly deep trees tend to overfit. As we varied the tree depth in our experiment, we were able to find an optimal value that maximized validation accuracy. 

To improve the accuracy of these models, it would be interesting to work with larger datasets to provide more information to our models. Examining the dropping features to a greater extent would also prove useful, as the results we achieved were already promising. We may want to investigate variations of our models, such as a weighted KNN model where the classification task weighs each of the K-nearest neighbors by the distance from the test point or a DT model with pruning. These more advanced techniques are able to capture more information from the data, and as such, would likely yield higher test accuracies.

Overall, the KNN and DT models were both successful in diagnosing both the fatality of hepatitis and signs of diabetic retinopathy. Given that both models have better prediction accuracy than random selection, these models can help medical professionals make better decisions. However, since their accuracy is not sufficiently higher for diagnosing patients with certainty, these models cannot be solely relied on. Rather, an approach of using collaborative intelligence is advisable; doctors can use the models to get an initial idea before prescribing further tests. These models can thus help rapidly flag more serious cases of hepatitis or diabetes. 

**Statement of Contributions**

Nicholas handled the cross-validation and hyperparameter optimization of the models. Jonayed handled the training and experimentation of the decision trees. Jasper handled the training and experimentation of the KNN model.

**References** 


    [1] D. Dua and C. Graff. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science. 


    [2] B. Antal and A. Hajdu. “An ensemble-based system for automatic screening of diabetic retinopathy,” arXiv:1410.8576 [cs], Oct. 2014.


    [3] M. L. Raymer, T. E. Doom, L. A. Kuhn and W. F. Punch. “Knowledge discovery in medical and biological datasets using a hybrid Bayes classifier/evolutionary algorithm,” in _IEEE Transactions on Systems, Man, and Cybernetics, Part B (Cybernetics), _vol. 33, no. 5, pp. 802-813, Oct. 2003, doi: 10.1109/TSMCB.2003.816922.
