**** HILL CLIMBING ALGORITHM ****



Dependencies:-



1. The CSV file should be in the same directory and the named should be properly changed in the main program.

2. The program expects at least python 2.7.x

Program Flow:-

1. Access the CSV File.
2. Create Boolean Array with size equal to the total number of attributes.
3. In each iteration of hill climbing, we call the gradient descent which iterates 50k times per call, thus training the model.
4. Call our hill climbing algorithm which starts from a random feature subset and is a non decreasing function. 
5. The accuracy keeps increasing or stays the same until the first decrease happens. 
6. Then hill climbing returns the locally optimal attribute subset at our disposal.



Credits :-
1. (Dataset Courtesy)
https://vincentarelbundock.github.io/Rdatasets/datasets.html
https://archive.ics.uci.edu/ml/datasets.html