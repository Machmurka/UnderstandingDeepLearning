

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
x_view=x.view(1,7)
x_view[:, 0]=5
print(f"Changing the view changes the original tensor too {x_view},{x}")

#stacking tensors 
x_stack=torch.stack([x,x,x,x],dim=0)
x_stack1=torch.stack([x,x,x,x],dim=1)

print(f"x stacked on dimention 0 :{x_stack}'\n X stacked on dimention 1'{x_stack1} stuff stacked vertically")

#usign squeeze
print(f"Previous tensor ^ it's shape {x_reshaped}{x_reshaped.shape}")
x_squeezed= x_reshaped.squeeze()
print(f"tensor squeezed ^ it's shape {x_squeezed}{x_squeezed.shape}")
x_unsqueezed = x_squeezed.unsqueeze(dim=0)
print(f"tensor unsqueezed ^ it's shape {x_unsqueezed}{x_unsqueezed.shape}")

x_original=torch.rand(size=(5,4,3))
x_permutate=x_original.permute(2,0,1) #shift ax 0->1 1->2 2->0
print(f"Original shape: \n {x_original} \n New shape \n {x_permutate}")

x=torch.arange(1,10).reshape(1,3,3)
#print(f"First square bracket \n {x[0]} \n Second square bracket \n {x[0][0]} \n Third square bracket \n {x[0][0][0]} ")
print(x)
print(f"get all values from 0th dimension and 0 index of 1st dimenstion {x[:, 0]}")
print(f"get all values of 0th & 1st dimension but only index 1 of 2nd dimension {x[:, :, 1]}") 
print(f"{x[:, 1, 1]}")
print(f"Same as x[0][0]: {x[0,0,:]}")

#numpy to torch
import numpy as np
array=np.arange(1.0,9.0)
tensor_np=torch.from_numpy(array)
