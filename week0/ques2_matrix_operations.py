import numpy as np

size1=eval(input("enter size of array1 in the format (no. of row,no.of column):"))
entries1=list(map(int,input("enter entries of array1 separated by space: ").split()))
size2=eval(input("enter number size of array2 in the format (no. of row,no.of column):"))
entries2=list(map(int,input("enter entries of array2 separated by space: ").split()))

array1= np.array(entries1).reshape(size1)
array2= np.array(entries2).reshape(size2)

operation=input("Specify the operation to be performed with the two 2d arrays {dot/matrix multiplication}:")

if operation.lower() == "dot":
    if size1==size2:
        i=0;j=0
        dot_arr= np.zeros(size1)
        for i in range(size1[0]) :
            for j in range(size1[1]) :             
             dot_arr[i,j]= array1[i,j]*array2[i,j]
        print(f"Dot product of array1 & array2 is:\n{dot_arr}")
    else: print("Dot product for matrices of unequal sizes is not possible.")

elif operation.lower() =='matrix multiplication':
    if size1[1] !=size2[0] :
        print('Matrix multiplication is not possible if no. of column in array1 is not equal to no. of rows in array2')
    else :
        i=0;j=0;k=0
        matmul=np.zeros((size1[0],size2[1]))
        for i in range(size1[0]):
            for j in range(size2[1]):
                for k in range(size1[1]==size2[0]):
                 matmul[i, j] = np.sum(array1[i, k] * array2[k, j])
        print(f"matrix multiplication of array1 & array2 is:\n {matmul}")

else :
    print('Invalid operation. Choose operation from dot/matrix multiplication')