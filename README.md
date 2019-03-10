# Unsupervised-Learning
This program uses Kohenen and K-means learning to assist more complex unsupervised learning networks, helping to determine the node whose weight vector is nearest to an input pattern.
Change the LEARNING_ALGORITHM constant to 1 or 2 depending on the type of learning you choose to use.
Terminating conditions for both algorithms are 15 epochs.
For the K-means learning algorithm, I found that I get consistent results after around 15 epochs, which is why I set the number of epochs to 15.
For the Kohonen learning algorithm, results were consistent after 3 epochs.
To get the sum squared error for each input, uncomment the print statement in the EuclideanDistance function.
