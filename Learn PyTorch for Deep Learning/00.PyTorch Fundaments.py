

import torch


scalar = torch.tensor(7)
print(scalar)
print("demention fo scalar (torch)",scalar.ndim)
# Get the Python number within a tensor (only works with one-element tensors)
print(scalar.item())


vector =torch.tensor([7,17])
print(vector)
print("Dimention of vector",vector.ndim)
print(vector.shape)

# Tensor
TENSOR = torch.tensor([[[1, 2, 3],
                        [3, 6, 9],
                        [2, 4, 5]]])

print(TENSOR.shape)

#random tensors  
print(torch.rand(size=(3,4)))
print(torch.rand(size=(224,224,3)))
print(torch.zeros(size=(3,4)))
print(torch.ones(size=(3,4)))

#specific tensor data
# Default datatype for tensors is float32
float_32_tensor = torch.tensor([3.0, 6.0, 9.0],
                               dtype=None, # defaults to None, which is torch.float32 or whatever datatype is passed
                               device=None, # defaults to None, which uses the default tensor type
                               requires_grad=False) # if True, operations performed on the tensor are recorded 

print(float_32_tensor)
print(float_32_tensor.shape, float_32_tensor.dtype, float_32_tensor.device)

#matrix multiplication
BasicTensor=torch.tensor([1,2,3])
print(BasicTensor*BasicTensor,torch.matmul(BasicTensor,BasicTensor))

# Shapes need to be in the right way  
tensor_A = torch.tensor([[1, 2],
                         [3, 4],
                         [5, 6]], dtype=torch.float32)

tensor_B = torch.tensor([[7, 10],
                         [8, 11], 
                         [9, 12]], dtype=torch.float32)

print(tensor_A.shape,tensor_B.shape,tensor_B.T.shape)
print(torch.mm(tensor_A, tensor_B.T))

x= torch.arange(0,100,10)
print(f"Minimum: {x.min()}, Maximuim: {x.max()}, Mean: {x.type(torch.float32).mean()}, Sum: {x.sum()}")
print(f"Index where min val occurs {x.argmin()}, Index where max val occurs{x.argmax()}")

x=torch.arange(0.,100.,10.)
print(x.dtype)
x_16=x.type(torch.float16)
print(x_16.dtype)

#tensor manipulation with rshape,stack,permute...

x=torch.arange(1.,8.)
print(x,x.shape)
x_reshaped=x.reshape(1,7)
print(x_reshaped, x_reshaped.shape)