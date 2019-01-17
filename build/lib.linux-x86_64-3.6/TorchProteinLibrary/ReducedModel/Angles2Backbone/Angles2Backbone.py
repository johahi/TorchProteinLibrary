import torch
from torch.autograd import Function
from torch.autograd import Variable
from torch.nn.modules.module import Module
import _ReducedModel
import math

class Angles2BackboneFunction(Function):
	"""
	Backbone angles -> backbone coordinates function
	"""
	@staticmethod
	def forward(ctx, input, angles_length):
		ctx.angles_max_length = torch.max(angles_length)
		ctx.atoms_max_length = 3*ctx.angles_max_length
		if len(input.size())==3:
			batch_size = input.size(0)
			output_coords_gpu = torch.zeros(batch_size, 3*(ctx.atoms_max_length), dtype=torch.float, device='cuda')
			ctx.A = torch.zeros(batch_size, 16*ctx.atoms_max_length, dtype=torch.float, device='cuda')
		else:
			raise Exception('Angles2BackboneFunction: ', 'Incorrect input size:', input.size()) 

		_ReducedModel.Angles2Backbone_forward( input, output_coords_gpu, angles_length, ctx.A)
													
		if math.isnan(output_coords_gpu.sum()):
			raise(Exception('Angles2BackboneFunction: forward Nan'))

		ctx.save_for_backward(input, angles_length)
		return output_coords_gpu
			
	@staticmethod
	def backward(ctx, gradOutput):
		gradOutput = gradOutput.contiguous()
		input_angles, angles_length = ctx.saved_tensors
		if len(input_angles.size()) == 3:
			batch_size = input_angles.size(0)
			gradInput_gpu = torch.zeros(batch_size, 2, ctx.angles_max_length, dtype=torch.float, device='cuda')
			dr_dangle = torch.zeros(batch_size, 2, 3*ctx.atoms_max_length*ctx.angles_max_length, dtype=torch.float, device='cuda')
		else:
			raise(Exception('Angles2BackboneFunction: backward size', input_angles.size()))		
			
		_ReducedModel.Angles2Backbone_backward(gradInput_gpu, gradOutput.data, input_angles, angles_length, ctx.A, dr_dangle)
		
		if math.isnan(torch.sum(gradInput_gpu)):
			raise(Exception('Angles2BackboneFunction: backward Nan'))		
		
		return Variable(gradInput_gpu), None, None

class Angles2Backbone(Module):
	def __init__(self):
		super(Angles2Backbone, self).__init__()
				
	def forward(self, input, angles_length):
		# stringListTensor = Variable(convertStringList(sequences))
				
		# self.num_atoms = []
		# for seq in sequences:
		# 	self.num_atoms.append(cppPDB2Coords.getSeqNumAtoms(seq, self.add_term))
		# num_atoms = Variable(torch.IntTensor(self.num_atoms))

		return Angles2BackboneFunction.apply(input, angles_length)