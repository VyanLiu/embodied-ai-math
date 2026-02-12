import numpy as np

#1.8 definition Suppose S is an order set, E belongs to S, and E is bounded above.
#Suppose there exists an alpha belonging to S with the following properties:
#   (i). alpha is an upper bound of E;
#   (ii). If Gamma is smaller than alpha, then gamma is not an upper bound of E;
#alpha is called as the supreme of E, denoted by alpha = sup E

def is_supreme(e_set, s_set):
    m = len(s_set)
    n = len(e_set)
    f_is_ordered = True
    f_is_supreme = False
    f_is_belonging = False
    supreme = 0
    for i in range(m-1):
        if s_set[i] > s_set[i + 1]:
            f_is_ordered = False
            break
    if f_is_ordered:
        for i in range(n):
            for j in range(m):
                if e_set[i] == s_set[j]:
                    f_is_belonging = True
        if f_is_belonging & n < 10000:
            for i in range(m):
                f_is_upper_bounded = True
                for j in range(n):
                    if e_set[j] > s_set[i]:
                        f_is_upper_bounded = False
                        break
                if f_is_upper_bounded:
                    f_is_supreme = True
                    supreme = s_set[i]
                    break
        else:
            f_is_supreme = False
    else:
        f_is_supreme = False
    return f_is_supreme, supreme

#Schwarz inequality:
#1.35 Theorem If a1,...,an and b1,...,bn are complex numbers, then |Sigma a * b*|^2 <= Sigma|a|^2 * Sigma|b|^2

def check_schwarz_complex(a, b):
    f_is_valid = False
    dot_product_ab = np.vdot(a, b) # vdot means complex_number * conjugate of complex_number
    dot_product_aa = np.vdot(a, a)
    dot_product_bb = np.vdot(b, b)

    left = np.abs(dot_product_ab) ** 2
    right = dot_product_aa* dot_product_bb

    print("Schwarz Left Result:", left)
    print("Schwarz Right Result:", right)

    if left <= right + 1e-12:
        f_is_valid = True
    else:
        f_is_valid = False
    return f_is_valid


if __name__ == "__main__":
    A = [1, 3, 5, 6, 12, 14, 11, 16, 17, 18, 21, 0, 32, 12, 15]
    B = [1, 3, 5, 7, 9, 11,13,15,17,19,21,23,25,27,29,31,33,35,37]
    C = [1, 100, 200, 300, 5, 7, 11]
    D = [11, 15, 18, 19, 23, 25, 27]
    E = [11, 13, 15, 17, 19, 21, 23, 27, 29]
    f_is_sup, sup = is_supreme(E, B)
    print("C has a supreme:", f_is_sup)
    if f_is_sup:
        print("supreme:\t", sup)

    A = np.array([1+3j, 3+5j, 5+7j, 7+9j, 9+11j])
    B = np.array([2+4j, 4+6j, 6+8j, 8+10j, 10+12j])
    f_schwarz_valid = check_schwarz_complex(A, B)
    print("Schwarz Valid?", f_schwarz_valid)


