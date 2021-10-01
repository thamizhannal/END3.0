## Assignment2

1. Rewrite the whole excel sheet showing backpropagation. Explain each major step, and write it on Github. 
   1. **Use exactly the same values for all variables as used in the class**
   2. Take a screenshot, and show that screenshot in the readme file
   3. Excel file must be there for us to cross-check the image shown on readme (no image = no score)
   4. Explain each major step
   5. Show what happens to the error graph when you change the learning rate from [0.1, 0.2, 0.5, 0.8, 1.0, 2.0] 



### Example Neural Network

![alt text](https://raw.githubusercontent.com/thamizhannal/END3.0/main/Session%202%20-%20Backprop%2C%20embeddings%20and%20Language%20Models/images/NeuralNetwork.png)

### Forward Pass:

![alt text](https://raw.githubusercontent.com/thamizhannal/END3.0/main/Session%202%20-%20Backprop%2C%20embeddings%20and%20Language%20Models/images/forwardpass.png)

### Back Propagation:

![alt text](https://raw.githubusercontent.com/thamizhannal/END3.0/main/Session%202%20-%20Backprop%2C%20embeddings%20and%20Language%20Models/images/BackPropGradComp.png)

### Error Values for various Learning rates
This is error values against various learning rates, as epoch progresses error start reducing.

![alt text](https://raw.githubusercontent.com/thamizhannal/END3.0/main/Session%202%20-%20Backprop%2C%20embeddings%20and%20Language%20Models/images/ErrorTable.png)

### Various Learning Rate: Error Vs Iteration graph:
Error graph Vs various learning rate shows that when error rate was 2.0 network converges in 40 epochs.
![alt text](https://raw.githubusercontent.com/thamizhannal/END3.0/main/Session%202%20-%20Backprop%2C%20embeddings%20and%20Language%20Models/images/LearningRateGraph.png)
