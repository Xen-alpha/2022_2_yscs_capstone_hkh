import torch
import random
import pytorchfi
from pytorchfi.core import FaultInjection
from bitstring import BitArray

class add_input_layer(torch.nn.Module):
    '''
    You can use this class when you want fault injection to input tensor itself.    
    '''
    def __init__(self, model, *args):
        super().__init__(*args)
        self.input_layer = torch.nn.Identity()
        self.model = model

    def forward(self, x):
        input = self.input_layer(x)
        output = self.model(input)
        return output


class single_bit_flip_model(FaultInjection):
    def __init__(self, model, batch_size, flip_bit_pos=None, save_log_list=False, **kwargs):
        super().__init__(model, batch_size, **kwargs)
        self.flip_bit_pos = flip_bit_pos
        self.save_log_list = save_log_list

        self.log_original_value = []
        self.log_original_value_bin = []
        self.log_error_value = []
        self.log_error_value_bin = []
        self.log_bit_pos = []

    def reset_log(self):
        '''
        You MUST call this function after single inference if save_log_list=True
        '''
        self.log_original_value = []
        self.log_original_value_bin = []
        self.log_error_value = []
        self.log_error_value_bin = []
        self.log_bit_pos = []

    def _single_bit_flip(self, orig_value, bit_pos):
        # set data type
        save_type = orig_value.dtype
        orig_value = orig_value.cpu().item()
        length = None
        if save_type == torch.float32:
            length = 32
        elif save_type == torch.float64:
            length = 64
        else:
            raise AssertionError(f'Unsupported Data Type: {save_type}')

        # single bit flip
        orig_arr = BitArray(float = orig_value, length = length)
        error = list(map(int, orig_arr.bin))
        error[bit_pos] = (error[bit_pos] + 1) % 2
        error = ''.join(map(str, error))
        error = BitArray(bin=error)
        new_value = error.float

        if self.save_log_list:
            self.log_original_value.append(orig_value)
            self.log_original_value_bin.append(orig_arr.bin)
            self.log_error_value.append(new_value)
            self.log_error_value_bin.append(error.bin)
            self.log_bit_pos.append(bit_pos)

        return torch.tensor(new_value, dtype=save_type)

    def weight_single_bit_flip_function(self, weight, position):
    
        bits = weight[position].dtype
        if bits == torch.float32:
            bits = 32
        elif bits == torch.float64:
            bits = 64
        else:
            raise AssertionError(f'Unsupported data type {bits}')

        rand_bit = random.randint(0, bits - 1) if self.flip_bit_pos is None else self.flip_bit_pos

        return self._single_bit_flip(weight[position], rand_bit)

    # structure from pytorchfi/neuron_error_models/single_bit_flip_func/single_bit_flip_signed_across_batch
    def neuron_single_bit_flip_function(self, module, input_val, output):
        corrupt_conv_set = self.corrupt_layer
        
        bits = output.dtype
        if bits == torch.float32:
            bits = 32
        elif bits == torch.float64:
            bits = 64
        else:
            raise AssertionError(f'Unsupported data type {bits}')
            
        if type(corrupt_conv_set) is list:
            inj_list = list(
                filter(
                    lambda x: corrupt_conv_set[x] == self.current_layer,
                    range(len(corrupt_conv_set)),
                )
            )
            for i in inj_list:
                self.assert_injection_bounds(index=i)
                prev_value = output[self.corrupt_batch[i]][self.corrupt_dim[0][i]][
                    self.corrupt_dim[1][i]
                ][self.corrupt_dim[2][i]]

                rand_bit = random.randint(0, bits - 1) if self.flip_bit_pos is None else self.flip_bit_pos

                new_value = self._single_bit_flip(prev_value, rand_bit)

                output[self.corrupt_batch[i]][self.corrupt_dim[0][i]][
                    self.corrupt_dim[1][i]
                ][self.corrupt_dim[2][i]] = new_value

        else:
            if self.current_layer == corrupt_conv_set:
                prev_value = output[self.corrupt_batch][self.corrupt_dim[0]][
                    self.corrupt_dim[1]
                ][self.corrupt_dim[2]]

                rand_bit = random.randint(0, bits - 1)

                new_value = self._single_bit_flip(prev_value, rand_bit)

                output[self.corrupt_batch][self.corrupt_dim[0]][self.corrupt_dim[1]][
                    self.corrupt_dim[2]
                ] = new_value     

        self.update_layer()
        if self.current_layer >= len(self.output_size):
            self.reset_current_layer()