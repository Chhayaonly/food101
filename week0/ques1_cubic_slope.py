def slope_of_cubic() :
    t=eval(input("Enter the coefficients of the cubic polynomial in the format (cubic,quadratic,linear,constant):"))
    x= int(input('Enter value of x at which slope of polynomial is to be calculated:'))
    Slope= int(t[0])*3*x*x + int(t[1])*2*x + int(t[2])
    print(f"Slope of given cubic polynomial at x={x} is {Slope}")

slope_of_cubic()
