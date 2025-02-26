import torch




class FuncPool:
    
    def sigmoid(self, x):
        return torch.sigmoid(x)

    def relu(self, x):
        return torch.relu(x)

    def tanh(self, x):
        return torch.tanh(x)
    
    # def exp(self, x):
    #     return torch.exp(x)
    
    # def log(self, x):
    #     # convert 0 and minus x values to 1
    #     # not in inplace operations
    #     for elem in x:
    #         if elem < 0:
    #             elem = 1
    #     return torch.log(x)
    
    # def dot(self, x, y):
    #     return torch.unsqueeze(torch.dot(x, y),dim=-1)

    # def mean(self, x):
    #     mean = torch.mean(x, dim=-1).unsqueeze(dim=-1)
    #     return mean
    
    def fft(self, x):
        out = torch.fft.fft(x)
        # output should be separated into real and imaginary parts
        # convert to imaginary part
        return out.real
    

    def matmul(self, x, y):
        return torch.mul(x, y)
        
    def add(self, x, y):
        res = torch.add(x, y)
        if torch.any(res >= 100):
            return x
        return res    
    def sub(self, x, y):
        return torch.sub(x, y)
    
    def hadamard(self, x, y):
        res = torch.mul(x, y)
        if torch.any(res >= 100):
            return x
        return res    
    # def div(self, x, y):
    #     # make exception for y elem is 0
    #     # check if certain elem is 0
    #     if torch.any(y == 0):
    #         return x
        
    #     return torch.div(x, y)
    
    def sin(self, x):
       
        return torch.sin(x)
    
    def cos(self, x):
        return torch.cos(x)
    
    
    
class Trigonometric:
    def sin(self, x):
        return torch.sin(x)
    
    def cos(self, x):
        return torch.cos(x)
    
    def tan(self, x):
        res = torch.tan(x)
        
        if torch.all(res<=1) and torch.all(res>=-1):
            return res
        return x
    
    def cot(self, x):
        res = 1/torch.tan(x)
        if torch.all(res<=1) and torch.all(res>=-1):
            return res
        return x
    
    def sec(self, x):
        res = 1/torch.cos(x)
        if torch.all(res<=1) and torch.all(res>=-1):
            return res
        return x
    
    def csc(self, x):
        res = 1/torch.sin(x)
        if torch.all(res<=1) and torch.all(res>=-1):
            return res
        return x
    

class Hyperbolic:
    
    
    # clamp values to avoid inf through multiple layers
    def sinh(self, x):
        res = torch.sinh(x)
        if torch.all(res<=1) and torch.all(res>=-1):
            return res
        return x
    
    def cosh(self, x):
        res = torch.cosh(x)
        if torch.all(res<=1) and torch.all(res>=-1):
            return res
        return x
    
    def tanh(self, x):
        res = torch.tanh(x)
        if torch.all(res<=1) and torch.all(res>=-1):
            return res
        return x
    
    # def coth(self, x):
    #     res = 1/torch.tanh(x)
    #     if torch.all(res<=1) and torch.all(res>=-1):
    #         return res
    #     return x
    
    def sech(self, x):
        res = 1/torch.cosh(x)
        if torch.all(res<=1) and torch.all(res>=-1):
            return res
        return x
    
    def csch(self, x):
        res = 1/torch.sinh(x)
        if torch.all(res<=1) and torch.all(res>=-1):
            return res
        return x
    
class InverseTrigonometric:
    def asin(self, x):
        res = torch.asin(x)
        if torch.all(res<=1) and torch.all(res>=-1):
            return res
        return x
    
    def acos(self, x):
        res = torch.acos(x)
        if torch.all(res<=1) and torch.all(res>=-1):
            return res
        return x
    
    def atan(self, x):
        res = torch.atan(x)
        if torch.all(res<=1) and torch.all(res>=-1):
            return res
        return x
    
    def acot(self, x):
        return torch.atan(1/x)
    
    def asec(self, x):
        res = torch.acos(1/x)
        if torch.all(res<=1) and torch.all(res>=-1):
            return res
        return x
    
    def acsc(self, x):
        res = torch.asin(1/x)
        if torch.all(res<=1) and torch.all(res>=-1):
            return res
        return x
    
class InverseHyperbolic:
    # with inf values, return x
    def asinh(self, x):
        res = torch.asinh(x)
        if torch.all(res<=1) and torch.all(res>=-1):
            return res
        return x
    def acosh(self, x):
        res = torch.acosh(x)
        if torch.all(res<=1) and torch.all(res>=-1):
            return res
        return x
    
    def atanh(self, x):
        res = torch.atanh(x)
        if torch.all(res<=1) and torch.all(res>=-1):
            return res
        return x
    
    def acoth(self, x):
        res = torch.atanh(1/x)
        if torch.all(res<=1) and torch.all(res>=-1):
            return res
        return x
    
    def asech(self, x):
        res = torch.acosh(1/x)
        if torch.all(res<=1) and torch.all(res>=-1):
            return res
        return x
    
    def acsch(self, x):
        res = torch.asinh(1/x)
        if torch.all(res<=1) and torch.all(res>=-1):
            return res
        return x
    
class Power:
    # this will make value to be almost inf. do not use this class. 
    # def pow(self, x, y):
    #     if torch.any(x == float('inf')):
    #         return x
    #     return torch.pow(x, y)
    
    def sqrt(self, x):
        if torch.any(x == float('inf')):
            return x
        return torch.sqrt(x)
    
    def rsqrt(self, x):
        if torch.any(x == float('inf')):
            return x
        return torch.rsqrt(x)
    
    # def square(self, x):
    #     if torch.any(x == float('inf')):
    #         return x
    #     return torch.square(x)
    
    # def cube(self, x):
    #     if torch.any(x == float('inf')):
    #         return x
    #     return torch.pow(x, 3)
    
    # def reciprocal(self, x):
    #     if torch.any(x == 0):
    #         return x
    #     if torch.any(x == float('inf')):
    #         return x
    #     return torch.reciprocal(x)
        
class Fourier:
    # utilize real part only. 
    def fft(self, x):
        out = torch.fft.fft(x)
        if torch.any(out.real >= 100):
            return x
        return out.real
    
    def ifft(self, x):
        out = torch.fft.ifft(x)
        if torch.any(out.real >= 100):
            return x
        return out.real
    
    def rfft(self, x):
        out = torch.fft.rfft(x)
        if torch.any(out.real >= 100):
            return x
        return out.real
    
    # def irfft(self, x):
    #     out = torch.fft.irfft(x)
    #     return out.real

    
# class Exponential:
#     def exp(self, x):
#         if torch.any(x == float('inf')):
#             return x
#         return torch.exp(x)
    
#     def log(self, x):
#         if torch.any(x == float('inf')):
#             return x
        
#         if torch.any(x <= 1):
#             return x
        
#         return torch.log(x)

#     def log10(self, x):
#         if torch.any(x == float('inf')):
#             return x
            
#         if torch.any(x <= 1):
#             return x
        
#         return torch.log10(x)
    
#     def log2(self, x):
#         if torch.any(x == float('inf')):
#             return x
            
#         if torch.any(x <= 1):
#             return x
        
#         return torch.log2(x)
    
#     def expm1(self, x):
#         if torch.any(x == float('inf')):
#             return x
            
#         if torch.any(x <= 1):
#             return x
        
#         return torch.expm1(x)
    
#     def log1p(self, x):
#         if torch.any(x == float('inf')):
#             return x
            
#         if torch.any(x <= 1):
#             return x
        
#         return torch.log1p(x)

class Mathmatical:
    def abs(self, x):
        return torch.abs(x)
    
    def ceil(self, x):
        return torch.ceil(x)
    
    def floor(self, x):
        return torch.floor(x)
    
    def round(self, x):
        return torch.round(x)
    
    def trunc(self, x):
        return torch.trunc(x)
    
    def frac(self, x):
        return torch.frac(x)
    
    def sign(self, x):
        return torch.sign(x)
    
    def neg(self, x):
        return torch.neg(x)
    
    # def reciprocal(self, x):
    #     if torch.any(x == 0):
    #         return x
    #     return torch.reciprocal(x)
    
    # def square(self, x):
    #     return torch.square(x)
    
    # def sqrt(self, x):
    #     if torch.any(x <= 0):
    #         return x
    #     return torch.sqrt(x)
    
    # def rsqrt(self, x):
    #     if torch.any(x <= 0):
    #         return x
    #     return torch.rsqrt(x)
    
   
    

    def remainder(self, x, y):
        if torch.any(y == float(0.0)):
            return x
        return torch.remainder(x, y)

    
    def clamp(self, x, min, max):
        min = -1.0
        max = 1.0
        
        return torch.clamp(x, min, max)
    
    def maximum(self, x, y):
        return torch.maximum(x, y)
    
    def minimum(self, x, y):
        return torch.minimum(x, y)
    
    def clip(self, x):
        min = -1.0
        max = 1.0
        return torch.clip(x, min, max)
    
    # def sort(self, x):
    #     dim = -1 
    #     descending=False
    #     return torch.sort(x, dim, descending).values
    
    
    # def argsort(self, x):
    #     dim = -1 
    #     descending=False
    #     return torch.argsort(x, dim, descending)
    
    # def topk(self, x, k, dim, largest=True, sorted=True):
    #     return torch.topk(x, k, dim, largest, sorted)
    
    # def kthvalue(self, x, k, dim, keepdim=False):
    #     return torch.kthvalue(x, k, dim, keepdim)
    
    # def unique(self, x, sorted=True, return_inverse=False, return_counts=False, dim=0):
    #     return torch.unique(x, sorted, return_inverse, return_counts, dim)
    
class Comparison:
    # convert bool to float, 0.0 if false, 1.0 if true
    def equal(self, x, y):
        res = torch.eq(x, y)
        return res.float()

    def not_equal(self, x, y):
        res = torch.ne(x, y)
        return res.float()
    
    def greater(self, x, y):
        res = torch.gt(x, y)
        return res.float()
    
    def greater_equal(self, x, y):
        res = torch.ge(x, y)
        return res.float()
    
    def less(self, x, y):
        res = torch.lt(x, y)
        return res.float()
    
    def less_equal(self, x, y):

        res = torch.le(x, y)
        return res.float()
    
    def logical_and(self, x, y):

        res = torch.logical_and(x, y)
        return res.float()
    
    def logical_or(self, x, y):

        res = torch.logical_or(x, y)
        return res.float()
    
    def logical_not(self, x):

        res = torch.logical_not(x)
        return res.float()
    
    def logical_xor(self, x, y):

        res = torch.logical_xor(x, y)
        return res.float()
    
    def maximum(self, x, y):

        res = torch.maximum(x, y)
        return res.float()
    
    def minimum(self, x, y):

        res = torch.minimum(x, y)
        return res.float()
    

class Bitwise:
    def bitwise_and(self, x, y):
        return torch.bitwise_and(x, y)
    
    def bitwise_or(self, x, y):
        return torch.bitwise_or(x, y)
    
    def bitwise_xor(self, x, y):
        return torch.bitwise_xor(x, y)
    
    def bitwise_not(self, x):
        return torch.bitwise_not(x)
    
    


class Neural:

    # def linear(self, x, weight, bias=None):
    #     return torch.nn.functional.linear(x, weight, bias)
    
    def relu(self, x):
        return torch.nn.functional.relu(x)
    
    def prelu(self, x, weight):
        return torch.nn.functional.prelu(x, weight)
    
    def elu(self, x, alpha=1.0):
        return torch.nn.functional.elu(x, alpha)
    
    def leaky_relu(self, x, negative_slope=0.01):
        return torch.nn.functional.leaky_relu(x, negative_slope)
    
    def rrelu(self, x, lower=1.0/8.0, upper=1.0/3.0, training=False, inplace=False):
        return torch.nn.functional.rrelu(x, lower, upper, training, inplace)
    
    def selu(self, x):
        return torch.nn.functional.selu(x)
    
    def celu(self, x, alpha=1.0):
        return torch.nn.functional.celu(x, alpha)
    
    def gelu(self, x):
        return torch.nn.functional.gelu(x)
    
    def hardshrink(self, x, lambd=0.5):
        return torch.nn.functional.hardshrink(x, lambd)
    
    def hardtanh(self, x, min_val=-1.0, max_val=1.0):
        return torch.nn.functional.hardtanh(x, min_val, max_val)
    
    def softplus(self, x, beta=1, threshold=20):
        return torch.nn.functional.softplus(x, beta, threshold)
    
    def softshrink(self, x, lambd=0.5):
        return torch.nn.functional.softshrink(x, lambd)
    
    def softsign(self, x):
        return torch.nn.functional.softsign(x)
    
    def tanhshrink(self, x):
        return torch.nn.functional.tanhshrink(x)
    
    def threshold(self, x, threshold, value, inplace=False):
        return torch.nn.functional.threshold(x, threshold, value, inplace)
    
    def sigmoid(self, x):
        return torch.nn.functional.sigmoid(x)
    
    def hardtanh(self, x, min_val=-1.0, max_val=1.0):
        return torch.nn.functional.hardtanh(x, min_val, max_val)
    
    def hardswish(self, x ):
        return torch.nn.functional.hardswish(x)
    
class NeuralConv:
    def conv1d(self, x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        return torch.nn.functional.conv1d(x, weight, bias, stride, padding, dilation, groups)
    
    def conv2d(self, x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        return torch.nn.functional.conv2d(x, weight, bias, stride, padding, dilation, groups)
    
    def conv3d(self, x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        return torch.nn.functional.conv3d(x, weight, bias, stride, padding, dilation, groups)
    
    def conv_transpose1d(self, x, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1):
        return torch.nn.functional.conv_transpose1d(x, weight, bias, stride, padding, output_padding, groups, dilation)
    
    def conv_transpose2d(self, x, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1):
        return torch.nn.functional.conv_transpose2d(x, weight, bias, stride, padding, output_padding, groups, dilation)
    
    def conv_transpose3d(self, x, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1):
        return torch.nn.functional.conv_transpose3d(x, weight, bias, stride, padding, output_padding, groups, dilation)
    
    def unfold(self, x, kernel_size, dilation=1, padding=0, stride=1):
        return torch.nn.functional.unfold(x, kernel_size, dilation, padding, stride)
    
    def fold(self, x, output_size, kernel_size, dilation=1, padding=0, stride=1):
        return torch.nn.functional.fold(x, output_size, kernel_size, dilation, padding, stride)
    
    def max_pool1d(self, x, kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False):
        return torch.nn.functional.max_pool1d(x, kernel_size, stride, padding, dilation, return_indices, ceil_mode)
    
    def max_pool2d(self, x, kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False):
        return torch.nn.functional.max_pool2d(x, kernel_size, stride, padding, dilation, return_indices, ceil_mode)
    
    def max_pool3d(self, x, kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False):
        return torch.nn.functional.max_pool3d(x, kernel_size, stride, padding, dilation, return_indices, ceil_mode)
    
    def avg_pool1d(self, x, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None):
        return torch.nn.functional.avg_pool1d(x, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override)
    
    def avg_pool2d(self, x, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None):
        return torch.nn.functional.avg_pool2d(x, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override)
    
    def avg_pool3d(self, x, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None):
        return torch.nn.functional.avg_pool3d(x, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override)
    
    def adaptive_max_pool1d(self, x, output_size, return_indices=False):
        return torch.nn.functional.adaptive_max_pool1d(x, output_size, return_indices)
    
    def adaptive_max_pool2d(self, x, output_size, return_indices=False):
        return torch.nn.functional.adaptive_max_pool2d(x, output_size, return_indices)
    
    def adaptive_max_pool3d(self, x, output_size, return_indices=False):
        return torch.nn.functional.adaptive_max_pool3d(x, output_size, return_indices) 
    
    def adaptive_avg_pool1d(self, x, output_size):
        return torch.nn.functional.adaptive_avg_pool1d(x, output_size)
    
    def adaptive_avg_pool2d(self, x, output_size):
        return torch.nn.functional.adaptive_avg_pool2d(x, output_size)
    
    def adaptive_avg_pool3d(self, x, output_size):
        return torch.nn.functional.adaptive_avg_pool3d(x, output_size)
    
    def dropout(self, x, p=0.5, training=True, inplace=False):
        return torch.nn.functional.dropout(x, p, training, inplace)
    
    def feature_alpha_dropout(self, x, p=0.5, training=True):
        return torch.nn.functional.feature_alpha_dropout(x, p, training)
    
    def feature_dropout(self, x, p=0.5, training=True):
        return torch.nn.functional.feature_dropout(x, p, training)
    
    def alpha_dropout(self, x, p=0.5, training=True):
        return torch.nn.functional.alpha_dropout(x, p, training)
    
    def dropout2d(self, x, p=0.5, training=True, inplace=False):
        return torch.nn.functional.dropout2d(x, p, training, inplace)
    
    def dropout3d(self, x, p=0.5, training=True, inplace=False):
        return torch.nn.functional.dropout3d(x, p, training, inplace)

class NeuralTimeSeries:

    def lstm(self, x, hx, weight, bias, batch_sizes, dropout, training, bidirectional, batch_first):
        return torch.nn.functional.lstm(x, hx, weight, bias, batch_sizes, dropout, training, bidirectional, batch_first)
    
    def gru(self, x, hx, weight, bias, batch_sizes, dropout, training, bidirectional, batch_first):
        return torch.nn.functional.gru(x, hx, weight, bias, batch_sizes, dropout, training, bidirectional, batch_first)
    
    def rnn(self, x, hx, weight, bias, batch_sizes, dropout, training, bidirectional, batch_first):
        return torch.nn.functional.rnn(x, hx, weight, bias, batch_sizes, dropout, training, bidirectional, batch_first)
    
    def pad_packed_sequence(self, x, batch_first, padding_value, total_length):
        return torch.nn.functional.pad_packed_sequence(x, batch_first, padding_value, total_length)
    
    def pack_padded_sequence(self, x, batch_first, enforce_sorted):
        return torch.nn.functional.pack_padded_sequence(x, batch_first, enforce_sorted)
    
    def interpolate(self, x, size, scale_factor, mode, align_corners):
        return torch.nn.functional.interpolate(x, size, scale_factor, mode, align_corners)
    
    def grid_sample(self, x, grid, mode, padding_mode, align_corners):
        return torch.nn.functional.grid_sample(x, grid, mode, padding_mode, align_corners)
    
    def affine_grid(self, theta, size):
        return torch.nn.functional.affine_grid(theta, size)
    
    def unfold(self, x, kernel_size, dilation, padding, stride):
        return torch.nn.functional.unfold(x, kernel_size, dilation, padding, stride)
    
    def fold(self, x, output_size, kernel_size, dilation, padding, stride):
        return torch.nn.functional.fold(x, output_size, kernel_size, dilation, padding, stride)
    
    def lp_pool1d(self, x, norm_type, kernel_size, stride, ceil_mode):
        return torch.nn.functional.lp_pool1d(x, norm_type, kernel_size, stride, ceil_mode)
    
    def lp_pool2d(self, x, norm_type, kernel_size, stride, ceil_mode):
        return torch.nn.functional.lp_pool2d(x, norm_type, kernel_size, stride, ceil_mode)
    

class LinearAlgebra:
    def cholesky(self, x, upper=False):
        return torch.linalg.cholesky(x, upper)
    
    def cholesky_inverse(self, x, upper=False):
        return torch.linalg.cholesky_inverse(x, upper)
    
    def cholesky_solve(self, x1, x2, upper=False):
        return torch.linalg.cholesky_solve(x1, x2, upper)
    
    def qr(self, x, some=True):
        return torch.qr(x, some)
    
    def svd(self, x, some=True, compute_uv=True):
        return torch.svd(x, some, compute_uv)
    
    def pinverse(self, x, rcond=1e-15):
        return torch.pinverse(x, rcond)
    
    def matrix_rank(self, x, tol=None, symmetric=False):
        return torch.matrix_rank(x, tol, symmetric)
    
    def lstsq(self, x, y, rcond=1e-15):
        return torch.lstsq(x, y, rcond)
    
    def tensorinv(self, x, ind=2):
        return torch.tensorinv(x, ind)
    
    def tensorsolve(self, x, y, dims=None):
        return torch.tensorsolve(x, y, dims)
    
    def eig(self, x, eigenvectors=False):
        return torch.eig(x, eigenvectors)
    
    def eigh(self, x, UPLO='L'):
        return torch.eigh(x, UPLO)
    
    def eigvalsh(self, x, UPLO='L'):
        return torch.eigvalsh(x, UPLO)
    
    def symeig(self, x, eigenvectors=False, upper=True):
        return torch.symeig(x, eigenvectors, upper)
    
    def svdvals(self, x):
        return torch.svdvals(x)
    
    def multi_dot(self, x):
        return torch.multi_dot(x)
    
    def norm(self, x, p='fro', dim=None, keepdim=False, out=None):
        return torch.norm(x, p, dim, keepdim, out)
    
    def tensor_diag(self, x, diagonal=0):
        return torch.tensor_diag(x, diagonal)
    
    def tensor_diag_embed(self, x, offset=0, dim1=0, dim2=1):
        return torch.tensor_diag_embed(x, offset, dim1, dim2)

class VectorArithmetic:
    def dot(self, x, y):
        return torch.dot(x, y)
    
    def vdot(self, x, y):
        return torch.vdot(x, y)
    
    def inner(self, x, y):
        return torch.inner(x, y)
    
    def outer(self, x, y):
        return torch.outer(x, y)
    
    def cross(self, x, y):
        return torch.cross(x, y)
    
    def linalg_matrix_power(self, x, n):
        return torch.linalg.matrix_power(x, n)
    
    def linalg_norm(self, x, ord=None, axis=None, keepdims=False):
        return torch.linalg.norm(x, ord, axis, keepdims)
    
    def linalg_vector_norm(self, x, ord=None, dim=None, keepdim=False):
        return torch.linalg.vector_norm(x, ord, dim, keepdim)
    
    def linalg_matrix_rank(self, x, tol=None, hermitian=False):
        return torch.linalg.matrix_rank(x, tol, hermitian)
    
    def linalg_tensorinv(self, x, ind=2):
        return torch.linalg.tensorinv(x, ind)
    
    def linalg_tensorsolve(self, x, y, dims=None):
        return torch.linalg.tensorsolve(x, y, dims)
    
    def linalg_eig(self, x, UPLO='L'):
        return torch.linalg.eig(x, UPLO)
    
    def linalg_eigh(self, x, UPLO='L'):
        return torch.linalg.eigh(x, UPLO)
    
    def linalg_eigvalsh(self, x, UPLO='L'):
        return torch.linalg.eigvalsh(x, UPLO)
    
    def linalg_svd(self, x, full_matrices=True, compute_uv=True):
        return torch.linalg.svd(x, full_matrices, compute_uv)
    
    def linalg_svdvals(self, x):
        return torch.linalg.svdvals(x)
    
    def linalg_multi_dot(self, x):
        return torch.linalg.multi_dot(x)
    
    def linalg_matrix_norm(self, x, ord=None, axis=None, keepdims=False):
        return torch.linalg.matrix_norm(x, ord, axis, keepdims)
    
        
     
     
     
FUNC_CLASSES = {
    # 'trig': Trigonometric,
    # 'hyperbolic': Hyperbolic,
    # 'inv_trig': InverseTrigonometric,
    # 'inv_hyperbolic': InverseHyperbolic,
    # # # # # 'power': Power,
    # 'fourier': Fourier,
    # # # # # 'exp': Exponential,
    # 'math': Mathmatical,
    # 'comparison': Comparison,
    # 'bitwise': Bitwise,
    # ------------------------------------------------------
    # 'neural': Neural,
    # 'neural_conv': NeuralConv,
    # 'neural_time_series': NeuralTimeSeries,
    # 'linear_algebra': LinearAlgebra,
    # 'vector_arithmetic': VectorArithmetic
        }

    

