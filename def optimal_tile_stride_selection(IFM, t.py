def calculate_tile_moves(IFM_H, IFM_W, kernel_size, tile_size):
    possible_strides = []
    tile_moves = []

    for stride in range(1, max(IFM_H, IFM_W) + 1):
        if IFM_H - kernel_size + stride >= 0 and IFM_W - kernel_size + stride >= 0:
            possible_strides.append(stride)
            h_tile_moves = (IFM_H - kernel_size) // stride + 1
            w_tile_moves = (IFM_W - kernel_size) // stride + 1
            tile_moves.append(h_tile_moves * w_tile_moves)

    return possible_strides, tile_moves

# User-defined parameters
IFM_H = 224   # Height of input feature map
IFM_W = 224   # Width of input feature map
kernel_size = 11   # Size of the convolutional kernel (k*k)
tile_size = 67     # Size of the tile

possible_strides, tile_moves = calculate_tile_moves(IFM_H, IFM_W, kernel_size, tile_size)

print("Possible Stride_tile values:", possible_strides)
print("Corresponding number of tile moves for each stride:", tile_moves)
