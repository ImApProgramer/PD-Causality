import torch
import random
import numpy as np

#要启用增强的话必须改动
class MirrorReflection:
    """
    Do horizontal flipping for each frame of the sequence.
    
    Args:
      format (str): Skeleton format. By default it expects it to be h36m (as motion encoders are mostly trained on that)
    """

    def __init__(self, format='h36m', data_dim=2):
        if format == 'h36m':
            self.left = [14, 15, 16, 1, 2, 3]
            self.right = [11, 12, 13, 4, 5, 6]
        elif format == 'ntu25':
            self.left = [4, 5, 6, 7, 12, 13, 14, 15, 21, 22]
            self.right = [8, 9, 10, 11, 16, 17, 18, 19, 23, 24]
        else:
            raise NotImplementedError("Skeleton format is not supported.")
        self.data_dim = data_dim
    
    def __call__(self, sample):
        sequence, label, labels_str = sample['encoder_inputs'], sample['label'], sample['labels_str']       #此时要注意对于GCN来说：sample['encoder_inputs']样子是(3, 81, 25, 1)
        if isinstance(sequence, np.ndarray):
            sequence = torch.from_numpy(sequence)



        if sequence.ndim == 4:      #GCN专属逻辑
            # ✅ GCN格式 (C, T, V, M)
            mirrored_sequence = sequence.clone()
            mirrored_sequence[0] *= -1  # 这个东西的值是[T,V,1]，是X轴分量，这里将它镜像反转

            # Swap left ↔ right joints
            mirrored_sequence[:, :, self.left, :] = sequence[:, :, self.right, :]
            mirrored_sequence[:, :, self.right, :] = sequence[:, :, self.left, :]

        else:
            if self.data_dim == 3:
                merge_last_dim = 0
                if sequence.ndim == 2:
                    sequence = sequence.view(-1, 17, 3)  # Reshape sequence back to N x 17 x 3
                    merge_last_dim = 1
            mirrored_sequence = sequence.clone()
            mirrored_sequence[:, :, 0] *= -1
            mirrored_sequence[:, self.left + self.right, :] = mirrored_sequence[:, self.right + self.left, :]

            if self.data_dim == 3 and merge_last_dim: # Reshape sequence back to N x 51
                    N = np.shape(mirrored_sequence)[0]
                    mirrored_sequence = mirrored_sequence.reshape(N, -1)

        return {
            'encoder_inputs': mirrored_sequence,
            'label': label,
            'labels_str': labels_str
        }


class RandomRotation:
    """
    Rotate randomly all the joints in all the frames.

    Args:
       min_rotate (int): Minimum degree of rotation angle.
       max_rotate (int): Maximum degree of rotation angle.
    """


    def __init__(self, min_rotate, max_rotate, data_dim=2):
        self.min_rotate, self.max_rotate = min_rotate, max_rotate
        self.data_dim = data_dim
    
    def _create_3d_rotation_matrix(self, axis, rotation_angle):
        theta = rotation_angle * (torch.pi / 180)
        if axis == 0:  # x-axis
            rotation_matrix = torch.tensor([[1, 0, 0],
                            [0, torch.cos(theta), torch.sin(theta)],
                            [0, -torch.sin(theta), torch.cos(theta)]])
        elif axis == 1:  # y-axis
            rotation_matrix = torch.tensor([[torch.cos(theta), 0, -torch.sin(theta)],
                            [0, 1, 0],
                            [torch.sin(theta), 0, torch.cos(theta)]])
        elif axis == 2:  # z-axis
            rotation_matrix = torch.tensor([[torch.cos(theta), torch.sin(theta), 0],
                            [-torch.sin(theta), torch.cos(theta), 0],
                            [0, 0, 1]])
        return rotation_matrix

    def _create_rotation_matrix(self):
        rotation_angle = torch.FloatTensor(1).uniform_(self.min_rotate, self.max_rotate)
        theta = rotation_angle * (torch.pi / 180)

        rotation_matrix = torch.tensor([[torch.cos(theta), -torch.sin(theta)],
                                        [torch.sin(theta), torch.cos(theta)]])
        return rotation_matrix
    
    def __call__(self, sample):
        sequence, label, labels_str = sample['encoder_inputs'], sample['label'], sample['labels_str']
        if isinstance(sequence, np.ndarray):
            sequence = torch.from_numpy(sequence)

        if sequence.ndim == 4:
            C, T, V, M = sequence.shape
            rotated_sequence = sequence.clone()

            total_axis = [0, 1, 2]
            main_axis = random.randint(0, 2)  # 随机选一个主轴
            for axis in total_axis:
                if axis == main_axis:
                    rotation_angle = torch.FloatTensor(1).uniform_(self.min_rotate, self.max_rotate)
                else:
                    rotation_angle = torch.FloatTensor(1).uniform_(self.min_rotate / 10, self.max_rotate / 10)
                rotation_matrix = self._create_3d_rotation_matrix(axis, rotation_angle).to(sequence.device)

                # [C,T,V,M] → [T,V,M,C] → 乘旋转矩阵 → 再变回去
                rotated_sequence = rotated_sequence.permute(1, 2, 3, 0)  # [T, V, M, C]
                rotated_sequence = torch.matmul(rotated_sequence, rotation_matrix)  # (T, V, M, 3)
                rotated_sequence = rotated_sequence.permute(3, 0, 1, 2)  # [C, T, V, M]




        else:
            if self.data_dim == 2:
                rotation_matrix = self._create_rotation_matrix()

                sequence_has_confidence_score = sequence.shape[-1] == 3
                if sequence_has_confidence_score:
                    rotated_sequence = sequence[..., :2] @ rotation_matrix
                    rotated_sequence = torch.cat((rotated_sequence, sequence[..., 2].unsqueeze(-1)), dim=-1)
                else:
                    rotated_sequence = sequence @ rotation_matrix
            else:
                merge_last_dim = 0
                if sequence.ndim == 2:
                    sequence = sequence.view(-1, 17, 3)  # Reshape sequence back to N x 17 x 3
                    merge_last_dim = 1
                rotated_sequence = sequence.clone()
                total_axis = [0, 1, 2]
                main_axis = random.randint(0, 2)
                for axis in total_axis:
                    if axis == main_axis:
                        rotation_angle = torch.FloatTensor(1).uniform_(self.min_rotate, self.max_rotate)
                        rotation_matrix = self._create_3d_rotation_matrix(axis, rotation_angle)
                    else:
                        rotation_angle = torch.FloatTensor(1).uniform_(self.min_rotate/10, self.max_rotate/10)
                        rotation_matrix = self._create_3d_rotation_matrix(axis, rotation_angle)
                    rotated_sequence = rotated_sequence @ rotation_matrix
                if merge_last_dim: # Reshape sequence back to N x 51
                    N = np.shape(rotated_sequence)[0]
                    rotated_sequence = rotated_sequence.reshape(N, -1)
            
        return {
            'encoder_inputs': rotated_sequence,
            'label': label,
            'labels_str': labels_str
        }
    

class RandomNoise:
    """
    Adds noise randomly to each join separately from normal distribution.
    """
    def __init__(self, mean=0, std=0.01, data_dim=2):
        self.mean = mean
        self.std = std
        self.data_dim = data_dim

    def __call__(self, sample):
        sequence, label, labels_str = sample['encoder_inputs'], sample['label'], sample['labels_str']
        if isinstance(sequence, np.ndarray):
            sequence = torch.from_numpy(sequence)
        noise = torch.normal(self.mean, self.std, size=sequence.shape)
        noise_sequence = sequence + noise

        return {
            'encoder_inputs': noise_sequence,
            'label': label,
            'labels_str': labels_str
        }
        
        
class axis_mask:
    def __init__(self, data_dim=3):
        self.data_dim = data_dim
    
    def Zero_out_axis(self, sequence):
        axis_next = random.randint(0, self.data_dim-1) 
        temp = sequence.clone()
        T, J, C = sequence.shape
        x_new = torch.zeros(T, J, device=temp.device)
        temp[:, :, axis_next] = x_new
        return temp

    def Zero_out_axis_GCN(self, sequence):

        axis_next = random.randint(0, self.data_dim - 1)

        temp = sequence.clone()
        temp[axis_next, :, :, :] = 0    #对于(x,y,z)中的一个维度对应的分量，全部清零

        return temp

    def __call__(self, sample):
        
        sequence, label, labels_str = sample['encoder_inputs'], sample['label'], sample['labels_str']
        if isinstance(sequence, np.ndarray):
                sequence = torch.from_numpy(sequence)

        if sequence.ndim==4:
            masked_sequence = self.Zero_out_axis_GCN(sequence)
            return {
                'encoder_inputs': masked_sequence,
                'label': label,
                'labels_str': labels_str
            }
        else:
            if self.data_dim > 2 :
                if self.data_dim == 3:
                    merge_last_dim = 0
                    if sequence.ndim == 2:
                        sequence = sequence.view(-1, 17, 3)  # Reshape sequence back to N x 17 x 3
                        merge_last_dim = 1
                masked_sequence = self.Zero_out_axis(sequence)

                if self.data_dim == 3 and merge_last_dim: # Reshape sequence back to N x 51
                        N = np.shape(masked_sequence)[0]
                        masked_sequence = masked_sequence.reshape(N, -1)

                return {
                        'encoder_inputs': masked_sequence,
                        'label': label,
                        'labels_str': labels_str
                    }
            else:
                return {
                        'encoder_inputs': sequence,
                        'label': label,
                        'labels_str': labels_str
                    }
