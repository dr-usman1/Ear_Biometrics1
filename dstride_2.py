# The given code defines a function `tile_size` that calculates and prints the tile sizes and possible stride options for two layers in a convolutional neural network (CNN). This function seems to be used for determining possible configurations of convolutional and pooling layers based on input parameters.

# Here's what the code does step by step:

# 1. The function takes several input parameters:
#    - IFM_L1: Dimension of the input feature map (IFM) for layer 1.
#    - IFM_L2: Dimension of the input feature map (IFM) for layer 2.
#    - kernel_L1: Kernel size for layer 1.
#    - kernel_L2: Kernel size for layer 2.
#    - stride_kernel_L1: Kernel stride for layer 1.
#    - stride_kernel_L2: Kernel stride for layer 2.
#    - output_pixel: Desired number of pixels at the output.
#    - pool_window_L1: Pooling window size after layer 1.
#    - pool_window_L2: Pooling window size after layer 2.
#    - stride_pool_L1: Pooling stride for layer 1.
#    - stride_pool_L2: Pooling stride for layer 2.

# 2. The function calculates intermediate values to determine the dimensions of the tiles for layers 1 and 2.

# 3. The following calculations are performed:
#    - inp2: Size of the intermediate feature map after pooling for layer 2.
#    - inc2: Size of the intermediate feature map after convolution for layer 2.
#    - inp1: Size of the intermediate feature map after pooling for layer 1.
#    - inc1: Size of the intermediate feature map after convolution for layer 1.

# 4. The function initializes two empty lists, `stride_layer_1` and `stride_layer_2`, to store possible stride values for layer 1 and layer 2 respectively.

# 5. Two for loops iterate over possible stride values from 1 to `inc1` (for layer 1) and from 1 to `inc2` (for layer 2).
#    - Inside each loop, a calculation is performed to determine the "movement" (how much the convolutional kernel moves across the feature map given the stride and kernel size). If this movement results in an integer value, it is considered a valid stride option.
#    - If the movement is an integer (i.e., `movement.is_integer()` is true), the stride value is appended to the corresponding stride list (`stride_layer_1` or `stride_layer_2`).

# 6. After both loops complete, the function prints the following information:
#    - Tile size for layer 1 (`inc1`).
#    - Tile size for layer 2 (`inc2`).
#    - Possible stride options for layer 1 (`stride_layer_1`).
#    - Possible stride options for layer 2 (`stride_layer_2`).

# In summary, this function is designed to calculate and print possible tile sizes and stride options for two layers in a CNN based on the provided input parameters. It can help in configuring the convolutional and pooling layers of a CNN architecture to achieve a desired output size.
    
def tile_size(IFM_L1, IFM_L2, kernel_L1, kernel_L2, stride_kernel_L1, stride_kernel_L2, output_pixel, pool_window_L1, pool_window_L2, stride_pool_L1, stride_pool_L2):
    inp2 = ((output_pixel - 1)*stride_pool_L2) + pool_window_L2
    inc2 = ((inp2 - 1)*stride_kernel_L2) + kernel_L2
    inp1 = ((inc2 - 1)*stride_pool_L1) + pool_window_L1
    inc1 = ((inp1 - 1)*stride_kernel_L1) + kernel_L1
    stride_layer_1 = []
    
    for i in range(1,inc1+1):
        movement = ((IFM_L1-inc1)/i) + 1
        # print(movement1)
        if (movement.is_integer()):
             stride_layer_1.append(i)


    stride_layer_2 = []
    for i in range(1,inc2+1):
        movement = ((IFM_L2-inc2)/i) + 1
        # print(movement1)
        if (movement.is_integer()):
                stride_layer_2.append(i)
    return(print('Tile_1 size:', inc1, '\nTile_2 size:', inc2, '\nTile_1 stride options:', stride_layer_1, '\nTile_2 stride options:', stride_layer_2))
# def tile_size(IFM1, IFM2, k1, k2, s1=1, s2=1, out=1, kp1=2, kp2=2, sp1=2, sp2=2):
#     inp2 = ((out - 1) * sp2) + kp2
#     inc2 = ((inp2 - 1) * s2) + k2
#     inp1 = ((inc2 - 1) * sp1) + kp1
#     inc1 = ((inp1 - 1) * s1) + k1
#     movement = []
#     for i in range(1, inc1):
#         movement1 = ((IFM1 - inc1) / i) + 1
#         if movement1.is_integer():
#             movement.append(i)
#     movement2 = []
#     for i in range(1, inc2):
#         movement1 = ((IFM2 - inc2) / i) + 1
#         if movement1.is_integer():
#             movement2.append(i)
    
#     largest_movement1 = max(movement)
#     optimal_stride_tile1 = largest_movement1 // 2 if (largest_movement1 // 2) in movement else largest_movement1
    
#     largest_movement2 = max(movement2)
#     optimal_stride_tile2 = largest_movement2 // 2 if (largest_movement2 // 2) in movement2 else largest_movement2
    
#     print('Tile_1 size:', inc1)
#     print('Tile_2 size:', inc2)
#     print('Tile_1 stride options:', movement)
#     print('Tile_2 stride options:', movement2)
#     print('Optimal stride option for Tile 1:', optimal_stride_tile1)
#     print('Optimal stride option for Tile 2:', optimal_stride_tile2)

# Example usage
tile_size(IFM_L1=118, IFM_L2=31, kernel_L1=11, kernel_L2=5, stride_kernel_L1= 4, stride_kernel_L2=1, output_pixel=1, pool_window_L1=3, pool_window_L2=3, stride_pool_L1=2, stride_pool_L2=2)

