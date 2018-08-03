**** SIMULATED ANNEALING ALGORITHM ****



Dependencies:-


1. 
The CSV file should be in the same directory and the named should be properly changed in the main program.

2. The program expects at least python 2.7.x

Program Flow:-

1. Access the CSV File.
2. Create Boolean Array with size equal to the total number of attributes.
3. In each iteration of simulated annealing, we call the gradient descent which iterates 50k times per call, thus training the model.
4. Call our simulated annealing algorithm which starts from a random feature subset. 
5. If the accuracy of the successor is greater than that of the predecessor, we treat the successor as the new predecessor.
6. In the later case, we call our probability function to choose a neighbour based on the given parameters.
7. Then simulated annealing returns the optimal attribute subset at our disposal.

Credits :-



Credit:-

1. (Dataset Courtesy)
https://vincentarelbundock.github.io/Rdatasets/datasets.html
https://archive.ics.uci.edu/ml/datasets.html