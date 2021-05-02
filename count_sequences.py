from fractions import Fraction

# Generate binomial coefficients (n choose k) for n, k < 100
memo = {}
def binom(n, k):
    if (n, k) in memo:
        return memo[(n, k)]
    if k == 0:
        return 1
    ans = binom(n, k-1) * (n - k + 1) // k
    memo[(n, k)] = ans
    return ans
binoms = [[binom(i, j) for j in range(100)] for i in range(100)]

def G_partial_prime(sums, p, d, x, cap, num_at_cap):
    '''
    Consider sequences B in (Z/pZ)^d containing only elements which are 
    between x and p-1, such that no element occurs more than cap times, such 
    that after being concatenated with a fixed sequence A, the resulting 
    sequence is zero-free. Returns the sum, over all such B, of the reciprocal 
    of the number of elements (in A + B) which occur cap times. The entire 
    sequence A is not passed; rather, sums and num_at_cap contain all the 
    necessary information about A.

    Parameters
    ----------
    sums : int
        The set of all subset sums of A in Z/pZ. Represented as an integer,
        where the bit at the s-th position indicates whether s is a sum of 
        a subset of A.
    p : int
        The modulus, must be prime.
    d : int
        The length of desired sequences.
    x : int
        The lower bound for elements of the sequences.
    cap : int
        The cap on the number of times each sequence element may occur.
    num_at_cap : int
        The number of elements of A which are equal to cap.

    Returns
    -------
    int
        The sum over all such sequences B of the reciprocal of the number
        of elements in A + B which occur cap times.
    '''
    
    univ = (1 << p) - 1
    
    # If x is p, then the sequence must be complete; returns (1 / num_at_cap)
    # if d is 0, and 0 otherwise,
    if x == p: 
        return Fraction(int(d == 0), num_at_cap)
    
    ans = 0
        
    # Casework on the number of occurrences of x – let that be i. We must
    # have i <= cap.
    for i in range(min(d, cap) + 1):
        
        # Use recursion to find the answer for each such i. We update sums,
        # and then we need to find sequences of length d-i using elements
        # from x+1 to p-1. If i = cap, then we increment num_at_cap.
        # Also, we need to multiply by (d choose i) to account for the 
        # number of ways to pick which indices in the sequence are equal to x.
        ans += (G_partial_prime(sums, p, d - i, x + 1, cap, 
                                num_at_cap + (i == cap)) 
                * binoms[d][i])
        
        # If -x is already a sum, then we cannot add another copy of x, so we
        # break.
        if sums & (1 << (p - x)): 
            break
        
        # Updates sums to also include all new sums created by adding another
        # copy of x. Uses bitwise arithmetic to speed up computation.
        a, b = sums << x, sums >> (p - x)
        sums |= a
        sums |= b
        sums &= univ
        
    return ans

def G_prime(p, d):
    '''
    For prime p, finds the total number of sequences in (Z/pZ)^d with no 
    subset sum equal to zero.

    Parameters
    ----------
    p : int
        The modulus, must be prime.
    d : int
        The length of the sequences.

    Returns
    -------
    int
        The total number of sequences in (Z/pZ)^d with no subset sum equal to 
        zero.
    '''
    
    ans = 0
    for c in range(1, d + 1):
        num = (G_partial_prime((1 << (c+1)) - 1, p, d - c, 2, c, 1) * 
               binoms[d][c])
        # print("partial result:", n, d, c, num)
        ans += num
    return ans * (p-1)


d = 9
for p in [103, 107, 109, 113, 127, 131]:
    g = G_prime(p, d)
    print(f'|G({p}, {d})| = {g}')
    
    
