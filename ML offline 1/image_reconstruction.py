import cv2
import numpy as np
import matplotlib.pyplot as plt

def lower_rank_approximation(A,k):
    # SVD decomposition
    U,S,V_T = np.linalg.svd(A)  
    V_k = V_T[:k,:]
    U_k = U[:,:k]
    Sk = S[:k]
    A_k = np.matmul(np.matmul(U_k,np.diag(Sk)),V_k)
    return A_k

img = cv2.imread('./image.jpg')
#Converting to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#gray = cv2.resize(gray, (512, 792))
#print(gray.dtype)
#print(gray)

# Original image show
#cv2.imshow('Original',gray)

# Image shape and minimum of them finding
n,m = gray.shape
min_dim = min(n,m)


total_row, total_column ,current_row, current_column, k  = 2, 5 ,0 , 0, 1
step_increase = min_dim**(1/(total_row*total_column))

fig, th_pic_in_axis = plt.subplots(total_row, total_column, figsize=(18,25))
plt.subplots_adjust(top = 0.9, bottom=0.1, hspace=0.5, wspace=0.4)


while k <= min_dim:

    A_K = lower_rank_approximation(gray, k)
    A_K = A_K.astype(np.uint8) # from float to uint8 as gray's type is also uint8

    # Image reconstruction show in plot figure
    th_pic_in_axis[current_row, current_column].set_title('n_components = {}'.format(k))
    th_pic_in_axis[current_row, current_column].imshow(A_K, cmap='gray')
    
    current_column = (current_column + 1) % total_column
    if current_column == 0:
        current_row = current_row + 1

    k *= round(step_increase)

    
plt.show()


