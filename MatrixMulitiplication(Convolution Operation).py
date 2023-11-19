#define matrixA
MatrixA = tf.Variable([[5, 10, 15], [10, 20, 30], [100, 150, 200]])

#define matrix B
MatrixB = tf.Variable([[1, 0, -1], [2, 0, -2], [1, 0, -1]])

mult_out = tf.math.multiply(MatrixA, MatrixB)

conv_out = tf.math.reduce_sum(mult_out)

Matrix_A = np.array([[5,10,15],[10,20,30],[100,150,200]])
Matrix_B = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])

def convolution2D(input_matrix, kernel):
    input_height, input_width = input_matrix.shape
    kernel_height, kernel_width = kernel.shape

    output_height = input_height - kernel_height + 1
    output_width = input_width - kernel_width + 1

    output_matrix = np.zeros((output_height, output_width))

    for i in range(output_height):
        for j in range(output_width):
            roi = input_matrix[i:i + kernel_height, j:j + kernel_width]
          
            elementwise_product = roi * kernel
          
            output_matrix[i, j] = np.sum(elementwise_product)

    return output_matrix

convolution2D(Matrix_A,Matrix_B)

