#We need numpy for the mathematical computation
import numpy
#we need matplotlib for plotting the graph
import matplotlib.pyplot as plotter

#Here we have defined the polynomial for generating sample values
def p(x):
    return numpy.exp(-numpy.power(x,4))*(2+numpy.sin(5*x)+numpy.sin(-2*numpy.square(x)))

#The below function assigs the value of sigma accordingly
def sigma_setter(iteration_value):
    if iteration_value == 0:
        return .05  #Low value of sigma
    if iteration_value == 1:
        return 1    #Medium value of sigma
    if iteration_value == 2:
        return 5    #High value of sigma
    if iteration_value == 3:
        return 50   #Very high value of sigma

def main():
    #N is the total number of iterations
    iterations = 1500
    #We store the Samples generated in X
    x = numpy.zeros(iterations,dtype=numpy.float)
    #The initial value has been set to 0
    x[0] = -1
    for j in range(0,5):
        sigma = sigma_setter(j)
        for i in range(0, iterations-1):
            #We get the squeare of the sigma
            sigma_square = numpy.square(sigma)
            #We draw samples from normal (Gaussian) Distribution with sd as the square of sigma
            Candidate = numpy.random.normal(x[i],sigma_square,None)
            #We again draw samples but the with modified mean
            next_candidate = numpy.random.normal(Candidate,sigma_square,None)
            #And now we generate the alpha
            alpha=min(1, (p(Candidate)*next_candidate)/(p(x[i])*Candidate))
            #We randomly generate a number between 0 and 1
            u_generator=numpy.random.uniform(0,1)
            #Now we compare the generated value and determine the next sample
            if  u_generator<alpha:
                x[i+1] = Candidate
            else:
                x[i+1] = x[i]

        #First we plot Quantity vs Samples of each quantity
        plotter.title('sigma = '+str(sigma))
        #X-axis is names as samples generated
        plotter.xlabel('Samples Generated')
        #Y-axis is the quantity
        plotter.ylabel('Quantitiy')
        #We create the histogram with color cyan
        plotter.hist(x, bins=30, color='cyan')
        #We plot Quantitiy vs The samples
        plotter.show()
        #Now we plot Sample value generated each iteration
        plotter.title('sigma = '+str(sigma))
        #X-axis is names as the number of iteration
        plotter.xlabel('iteration')
        #Y-axis is names as sample value
        plotter.ylabel('Sample value')
        # x is the list which has all the samples
        plotter.plot(x)
        #We plot the iteration vs sample values per iteration
        plotter.show()

if __name__ == '__main__':
    main()
