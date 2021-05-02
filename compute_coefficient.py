# Code to count 0-1 matrices of rank 4 with distinct rows and columns and no 
# zero columns

import numpy as np
from numpy.linalg import det, inv
# from random import random
import itertools as it
from fractions import Fraction
from time import time

T = time()

# generate list of factorials
facs = [1]
while len(facs) < 20:
    facs.append(len(facs) * facs[-1])
    
# generate list of binomial coefficients
memo = {}
def binom(n, k):
    if (n, k) in memo:
        return memo[(n, k)]
    if k == 0:
        return 1
    ans = binom(n, k-1) * (n - k + 1) // k
    memo[(n, k)] = ans
    return ans
binoms = [[binom(i, j) for j in range(20)] for i in range(20)]

# converts m*n-bit binary number to a 0-1 m by n matrix
def bin_to_matrix(b, m, n):
    M = np.zeros((m,n), dtype=int)
    for i in range(m):
        for j in range(n):
            M[i][j] = int((b & (1 << (i*n+j))) > 0)
    return M

# converts n-bit binary number to 0-1 list of length n
def bin_to_list(b, n):
    return [int((b & (1 << i)) > 0) for i in range(n)]

# We start by assuming there are no zero rows; we will add these back later.

# We will first generate all matrices up to both row and column permutation,
# and correct for this later. Also, since the conditions are symmetric in 
# rows and columns, we will assume for now that n_r >= n_c (# rows < # cols).

# Part 1: Assuming that the columns have been permuted so that the first four 
# columns are rank 4, generate all possibilities for the first 4 columns, up 
# to row permutation. For each possibility X, the rows will be permuted so
# that the top 4x4 submatrix is full-rank.

q = list(range(1, 2**16, 2)) 
# We represent each possible row as a binary number from 1 to 15, and each 
# set of rows as a 16-bit binary number whose last bit is 0. q contains all
# possible sets which haven't been used yet.
conv = np.array([1,2,4,8])

seed_matrices = [[] for i in range(16)] # matrices X will be stored here


for b in range(2**16):
    M = bin_to_matrix(b, 4, 4) # iterate over possibilities for the top 4 by 4
                               # submatrix
    if det(M) != 0:
        s = M@conv
        if not (s[0] < s[1] < s[2] < s[3]):
            continue # Require top 4 by 4 submatrix to have its rows in order
            
        A = np.rint(inv(M) * 6).astype(int) 
        assert (M@A == np.eye(4, dtype=int) * 6).all()
        # Set A = 6 * M^-1 (this will always have integer entries)
        
        M_bits = np.bitwise_or.reduce(1 << s) # set representation of the rows
                                              # of M
        new_q = []
        for c in q:
            if M_bits & c == M_bits: # if c contains all rows of M
                l = list(s)
                c ^= M_bits
                for i in range(1, 16):
                    if c & (1 << i):
                        l.append(i) # unroll c into a list, starting with the
                                    # 4 rows in M
                l = [bin_to_list(i, 4) for i in l] # convert list of rows to
                                                   # matrix X
                seed_matrices[len(l)].append((np.array(l), A)) # store X and A
            else:
                new_q.append(c)
        q = new_q
        
print('seeds generated')


# Part 2: For each matrix X, generate all matrices Y whose first 4 columns
# are X, again up to column permutation, such that n_r >= n_c. This will 
# generate all possible matrices, up to row permutation, column permutation,
# and transposition, and with no zero rows.
        
all_matrices = [[[] for j in range(16)] for i in range(16)]

cols_4 = [[int((k & (1 << i)) > 0) for i in range(4)] for k in range(16)]
cols_4 = np.array(cols_4) # array of all possible nonzero columns of 4 bits

for nr in range(4, 16):
    
    for X, A in seed_matrices[nr]: # (Recall that A is 6 times the inverse of
                                   # the first 4 rows of X)
        used_cols = conv @ X[:4]
        poss_cols = [] # list of all possible new columns that can be added to
                       # X while maintaining rank 4
                        
        if nr > 4: 
            for k in range(1, 16): # loop over first 4 elements of column;
                                   # since the first 4 rows of X are rank 4,
                                   # these elements determine the rest of the
                                   # column.
                
                if k in used_cols:
                    continue
                col_start = cols_4[k]
                col = X @ A @ col_start 
                # the rest of the column is X * M^-1 * v, where v is the first
                # 4 elements, and M is the first 4 rows of X. col is 6 times 
                # the actual column
                assert (col[:4] == col_start * 6).all()
                
                works = True
                for x in col[4:]:
                    if x != 0 and x != 6:
                        works = False
                        break
                if not works:
                    continue
                # checking that the column is 6 times a 0-1 vector; if not, it
                # is not a valid column to add
                
                poss_cols.append((col // 6)[..., None])
            
        for nc in range(4, min(nr, 4 + len(poss_cols)) + 1):
            for cols in it.combinations(poss_cols, nc - 4):
                Y = np.concatenate((X,) + cols, axis=1)
                all_matrices[nr][nc].append(Y)
        # generating one matrix for each subset of possible columns (such that
        # n_r >= n_c is still satisfied)
                
print('generated all_matrices')
                
                
# print(np.array([[len(x) for x in y] for y in all_matrices])[4:,4:])


# Part 3: generate all column permutations of matrices, so that our set of
# matrices is just up to row permutation (and transposition)


all_matrices_2 = [[set() for j in range(16)] for i in range(16)]
num_eq_classes = np.zeros((16,16),dtype=int)

def to_sorted_tuple(Y):
    '''Sorts rows of matrix Y (given as list of tuples) in lexicographic 
    order'''
    return tuple(sorted(tuple(r) for r in Y)) 

for nr in range(4, 16):
    for nc in range(4, nr+1):
        for Y in all_matrices[nr][nc]:
            Yt = to_sorted_tuple(Y)
            if Yt in all_matrices_2[nr][nc]:
                continue # if Y is already in the list, then all its 
                         # permutations will also already be in the list
            for perm in it.permutations(range(nc), nc):
                Y2t = to_sorted_tuple(Y[:, perm])
                all_matrices_2[nr][nc].add(Y2t)
            num_eq_classes[nr][nc] += 1
            
print('generated all_matrices_2')


# Step 4: Adjust for transposition and zero rows
                
total_counts = [[0 for j in range(16)] for i in range(17)] 
# contains the number of nr by nc matrices satisfying the condition, up to
# row permutation
                
# First, deal with transposition: there are n_r! ways to permute the rows,
# so we need to multiply by n_r! and divide by n_c!
for nr in range(4, 16):
    for nc in range(4, nr+1):
        total_counts[nr][nc] = len(all_matrices_2[nr][nc])
        total_counts[nc][nr] = total_counts[nr][nc] * facs[nr] // facs[nc]
        
# Now, deal with zero rows. We can add a zero row to any matrix, so just
# add in the count of n_r-1 by n_c matrices.
for nr in range(16, 4, -1):
    for nc in range(4, 16):
        total_counts[nr][nc] += total_counts[nr-1][nc]
        

# Step 5: generate the coefficients using the formula derived.
        
coeffs = [Fraction(0)] * 17

for nr in range(4, 17):
    for nc in range(4, 16):
        # |S| = nr, ell = nc
        for i in range(1, nr+1):
            coeffs[i] += Fraction(((-1)**(nc+nr-i) *
                                   binoms[nr][i] *
                                   total_counts[nr][nc]), 
                                  facs[nc])
            
def get_coeff(d):
    return sum(i**d * coeffs[i] for i in range(17))
        
print('\ncoefficients:')

for i in range(1, 17):
    print(f'{coeffs[i]} * {i}^n')
    
print(f'\nTotal time: {time() - T} s')
