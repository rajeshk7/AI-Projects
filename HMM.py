import numpy
import matplotlib.pyplot as ploter
import matplotlib.pyplot as plotac

# Introducing the obstacles
def initialize_Matrix():
        #We read the obstacles stored in the file Obs.txt
        obscount = 0
        with open("Obs.txt", "r") as f:
                for line in f:
                        numbers = line.split()
                        counter = 0
                        obscount = obscount + 1
                        arr = [0,0]
                        for number in numbers:
                                arr[counter] = int(number)
                                counter = counter + 1
                        Matrix[arr[0]][arr[1]] = 0
        emptycount = 64 - obscount
        print("The given grid is :")
        print(Matrix)
        print("1 : Empty and 0 : Obstacle")
        return emptycount

# We find all the empty squares in the Matrix
def init_empty_sq():
    for x in range(0,4):
        for y in range(0,16):
            if(Matrix[x][y]==1):
                #We append all those to the empty list
                empty_sq.append((x,y))

#We here populate the Transition matrix
def trans_model(emp):
    for i in range(0,emp):
        (x ,y)=empty_sq[i]
        Neigh=Neighbour(x,y)
        Ns=len(Neigh)
        for j in range(0,Ns):
            index=empty_sq.index(Neigh[j])
            #initially its 1/Ns for all, that is all of them are equiprobable
            Trans_mat[i][index]=1/Ns

# Calculate sensor model at time t
# (x,y) is true location of the robot at time t
def sensor_model(x,y,epsilon,emp):
    #First we receive the reading from the sensor with some error
    R=sensor_reading(x,y)
    for i in range(0,emp):
        (a,b)=empty_sq[i]
        A=sensor_reading(a,b)
        #We then calculate the error of moving to neighbour
        dit=error(A,R)
        t=numpy.power(1-epsilon,4-dit)*numpy.power(epsilon,dit)
        Obs_mat[i][i]=t;

# All the possible neighbour of a current location (x,y)
def Neighbour(x,y):
    a_list=[]
    if x-1>=0 and Matrix[x-1][y]==1:
        a_list.append((x-1,y))
    if x+1<X and Matrix[x+1][y]==1:
        a_list.append((x+1,y))
    if y+1<Y and Matrix[x][y+1]==1:
        a_list.append((x,y+1))
    if y-1>=0 and Matrix[x][y-1]==1:
        a_list.append((x,y-1))
    return a_list

# Getting the sensor reading given a location (x,y)
def sensor_reading(x,y):
    reading=list("1111")
    if x-1<0 or Matrix[x-1][y]==0:
        reading[0]="0"
    if x+1>=X or Matrix[x+1][y]==0:
        reading[1]="0"
    if y+1>=Y or Matrix[x][y+1]==0:
        reading[2]="0"
    if y-1<0 or Matrix[x][y-1]==0:
        reading[3]="0"
    return reading

# Error is discrepancy between true value of a square and sensor reading
def error(A,R):
    count=0;
    for i in range(0,4):
        if A[i]!= R[i]:
            count=count+1
    return count

#viterbi algorithm to find most likely path
def viterbi_algo(m):
    g=numpy.multiply(numpy.transpose(Trans_mat),numpy.transpose(m))
    t=numpy.amax(g,axis=1)
    m=numpy.matmul(Obs_mat,t)
    max_ind=numpy.argmax(m)
    return m,max_ind

#We here derive the path accuracy
def path_acc(e_path,a_path):
    count=0;
    for i in range(0,len(e_path)):
         if e_path[i]==a_path[i]:
             count=count+1;
    return count/len(e_path)

#The below function first generates random initial location and then moves
#It gets the sensor values with 5 values of sigma values
#We plot all the data we get with respect to each sigma value
def Driver_HMM(emp):
    error_val=[0.0,0.02,0.05,0.1,0.2]
    for _error in range(0,5):
        epsilon=error_val[_error]
        avg_arr_err=[]
        avg_arr_acc=[]
        for iter in range(0,400):
            f=numpy.full([emp,1],1/emp) #prior
            m=numpy.full([emp,1],1/emp)
            arr_err=[]
            arr_acc=[]
            e_path=[]
            a_path=[]
            while(True):
                ind=numpy.random.randint(0,emp, size=None)
                (a,b)=empty_sq[ind]
                N=Neighbour(a,b)
                if(len(N)!=0):
                    break;
            start=1
            for i in range(0,emp):
                if start!=1:
                    N=Neighbour(a,b)
                    Nind=numpy.random.randint(0,len(N), size=None)
                    (a,b)=N[Nind]
                sensor_model(a,b,epsilon,emp)           # calculate O matrix
                f=numpy.matmul(Obs_mat,numpy.matmul(numpy.transpose(Trans_mat),f))     #find posterior
                ind=numpy.argmax(f)
                x,y=empty_sq[ind]
                arr_err.append(abs(a-x)+abs(b-y));
                m,z=viterbi_algo(m)
                e_path.append(empty_sq[z])
                a_path.append((a,b))
                acc=path_acc(e_path,a_path)
                arr_acc.append(acc)
                start=0;
            avg_arr_err.append(arr_err)
            avg_arr_acc.append(arr_acc)
        cal_avg_err=[]
        cal_avg_acc=[]
        for i in range(0,emp):
            cal_avg_err.append(0)
            cal_avg_acc.append(0)
        for i in range(0,400):
            for j in range(0,emp):
                cal_avg_err[j]=cal_avg_err[j]+avg_arr_err[i][j]
                cal_avg_acc[j]=cal_avg_acc[j]+avg_arr_acc[i][j]
        for i in range(0,emp):
            cal_avg_err[i]=cal_avg_err[i]/400
            cal_avg_acc[i]=cal_avg_acc[i]/400
        
        '''
        #The below part prints the error
        #That is Number of Observations vs The localization Error
        ploter.title('Error')
        ploter.xlabel('Number of Observations')
        ploter.ylabel('Localization error')
        ploter.plot(cal_avg_err)
        ploter.legend(["0.00","0.02","0.05","0.10","0.20"])
        
        '''
        #The below part prints the Accuracy
        #That is Number of Observations vs The path accuracy
        plotac.title('Accuracy')        
        plotac.xlabel('Number of Observations')
        plotac.ylabel('Path Accuracy')
        plotac.plot(cal_avg_acc)       
        plotac.legend(["0.00","0.02","0.05","0.10","0.20"]) 
        
        
    #ploter.show()
    plotac.show()

if __name__ == '__main__':
    
    X=4                        						# Number of rows is 4
    Y=16											# Number of columns is 16                   
    empty_sq=[]                 					# Empty locations in the Matrix, container
    Matrix=numpy.ones([4,16])						# Matrix with all empty cells
    emp=initialize_Matrix()							# We introduce obstacles to the matrix
    Trans_mat=numpy.zeros([emp,emp],) 				# Transition matrix
    Obs_mat=numpy.zeros([emp,emp])    				# Emission matrix
    #We find all the empty squares according to the obstacles
    init_empty_sq()
    #Here we populate the transition matrix
    trans_model(emp)
    #Now we call the HMM function 
    #It runs 400 times for each of the values of epsilon
    Driver_HMM(emp)
