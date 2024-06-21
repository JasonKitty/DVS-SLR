import os
import torch
import numpy as np

from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from PIL import Image
from torchvision import transforms
import os
import numpy as np
import sys
sys.path.append(r'/home/spikingjelly')

from spikingjelly.datasets import integrate_events_segment_to_frame

class DVS346Sign(Dataset):
    def __init__(self, root_dir, users, labels, lights, positions, dt=50):
        self.root_dir = root_dir
        self.users = users
        self.labels = labels
        self.lights = lights
        self.positions = positions
        self.dt = dt
        self.data_list, self.label_list = self.get_data_label_file()

    # need to overload
    def __len__(self):
        return int(len(self.users)*len(self.labels)*len(self.lights)*len(self.positions))

    # need to overload
    def __getitem__(self, idx):
        event_file, frame_file = self.data_list[idx]
        label = self.label_list[idx]
        events = np.load(event_file)
        frames = np.load(frame_file, allow_pickle=True)
        sampled_frames = self.temporal_sample(frames)
        spike_frames, rgb_frames = self.generate_bimodal_synchronous_frame_stream(sampled_frames, events)
        spike_frames, rgb_frames = self.pad_frames(spike_frames, rgb_frames, int(6/(self.dt / 1e3)))
        spike_frames = spike_frames.astype(np.float64)
        rgb_frames = rgb_frames.astype(np.float64)
        return (spike_frames, rgb_frames), label

    def get_data_label_file(self):
        data_list = []
        label_list = []
        for user in self.users:
            for i, label in enumerate(self.labels):
                for light in self.lights:
                    for pos in self.positions:
                        event_name = '_'.join([user, label, light, pos, 'event'])
                        event_file = os.path.join(self.root_dir, user, label, event_name, event_name+'.npz')
                        frame_name = '_'.join([user, label, light, pos, 'frame'])
                        frame_file = os.path.join(self.root_dir, user, label, frame_name, frame_name+'.npz')
                        data_list.append((event_file, frame_file))
                        label_list.append(i)
        return data_list, label_list
    
    def temporal_sample(self, frames):
        sampled_frames = {'t':[], 'imgs':[]}
        t_frame = np.array(list(map(lambda x:int(x), frames['t'])))
        imgs = frames['imgs']
        if self.dt == 50:
            sampled_frames['t'] = t_frame
            sampled_frames['imgs'] = imgs
            return sampled_frames
        
        else:
            sampled_frames['t'] = np.arange(t_frame[0], t_frame[-1], self.dt * 1e6)
            sampled_imgs = []
            j = 0
            for i in range(len(sampled_frames['t'])):
                while j < len(t_frame):
                    if sampled_frames['t'][i] - t_frame[j] > 50 * 1e6:
                        j += 1
                    else:
                        sampled_frames['imgs'].append(imgs[j])
                        break
            sampled_frames['imgs'] = np.stack(sampled_frames['imgs'], axis=0)
            return sampled_frames

        
    
    def pad_frames(self, spike_frames, rgb_frames, length=120):
        if len(spike_frames)>length:
            return spike_frames[:length], rgb_frames[:length]
        elif len(spike_frames)==length:
            return spike_frames, rgb_frames
        else:
            last_spike_frame = spike_frames[-1]
            last_rgb_frame = rgb_frames[-1]
            num_pad = length - len(spike_frames)
            
            pad_spike_frames = np.repeat(last_spike_frame[np.newaxis,:, :, :], num_pad, axis=0)
            pad_rgb_frames = np.repeat(last_rgb_frame[np.newaxis,:, :, :], num_pad, axis=0)
            
            padded_spike_frames = np.concatenate((spike_frames, pad_spike_frames), axis=0)
            padded_rgb_frames = np.concatenate((rgb_frames, pad_rgb_frames), axis=0)
            return padded_spike_frames, padded_rgb_frames
        
    def generate_bimodal_synchronous_frame_stream(self, frames, events, H=260, W=346):
        t_event = events['t']
        
        t_frame, imgs = frames['t'], frames['imgs']
        t_frame, imgs = self.align_frame(t_frame, imgs, t_event)

        spike_frames = []
        for t_frame_ind in range(len(t_frame)-1):
            t_l, t_r = t_frame[t_frame_ind], t_frame[t_frame_ind+1]
            t_l_ind, t_r_ind = np.searchsorted(t_event, t_l), np.searchsorted(t_event, t_r)
            spike_frame = integrate_events_segment_to_frame(events['x'], events['y'], events['p'], H, W, t_l_ind, t_r_ind)
            spike_frames.append(np.expand_dims(spike_frame, 0))
        spike_frames = np.concatenate(spike_frames)
        imgs = np.transpose(imgs, (0, 3, 1, 2))

        return spike_frames, imgs[1:]
    
    def align_frame(self, t_frame, imgs, t_event):
        while t_frame[0] < t_event[0]:
            t_frame = t_frame[1:]
            imgs = imgs[1:]
        while t_frame[-1] > t_event[-1]:
            t_frame = t_frame[:-1]
            imgs = imgs[:-1]
        return t_frame, imgs
    
    
# class DVS346Sign(Dataset):
#     def __init__(self, root_dir, users, labels, lights, positions, transform=None):
#         self.root_dir = root_dir
#         self.users = users
#         self.labels = labels
#         self.lights = lights
#         self.positions = positions
#         self.transform = transform
#         self.data_list, self.label_list = self.get_data_label_file()

#     # need to overload
#     def __len__(self):
#         return int(len(self.users)*len(self.labels)*len(self.lights)*len(self.positions))

#     # need to overload
#     def __getitem__(self, idx):
#         event_file, frame_file = self.data_list[idx]
#         label = self.label_list[idx]
#         events = np.load(event_file)
#         frames = np.load(frame_file, allow_pickle=True)
#         spike_frames, rgb_frames = self.generate_bimodal_synchronous_frame_stream(frames, events)
#         spike_frames, rgb_frames = self.pad_frames(spike_frames, rgb_frames, 120)
#         spike_frames = spike_frames.astype(np.float64)
#         rgb_frames = rgb_frames.astype(np.float64)
#         return (spike_frames, rgb_frames), label

#     def get_data_label_file(self):
#         data_list = []
#         label_list = []
#         for user in self.users:
#             for i, label in enumerate(self.labels):
#                 for light in self.lights:
#                     for pos in self.positions:
#                         event_name = '_'.join([user, label, light, pos, 'event'])
#                         event_file = os.path.join(self.root_dir, user, label, event_name, event_name+'.npz')
#                         frame_name = '_'.join([user, label, light, pos, 'frame'])
#                         frame_file = os.path.join(self.root_dir, user, label, frame_name, frame_name+'.npz')
#                         data_list.append((event_file, frame_file))
#                         label_list.append(i)
#         return data_list, label_list
    
#     def pad_frames(self, spike_frames, rgb_frames, length=120):
#         if len(spike_frames)>length:
#             return spike_frames[:length], rgb_frames[:length]
#         elif len(spike_frames)==length:
#             return spike_frames, rgb_frames
#         else:
#             last_spike_frame = spike_frames[-1]
#             last_rgb_frame = rgb_frames[-1]
#             num_pad = length - len(spike_frames)
            
#             pad_spike_frames = np.repeat(last_spike_frame[np.newaxis,:, :, :], num_pad, axis=0)
#             pad_rgb_frames = np.repeat(last_rgb_frame[np.newaxis,:, :, :], num_pad, axis=0)
            
#             padded_spike_frames = np.concatenate((spike_frames, pad_spike_frames), axis=0)
#             padded_rgb_frames = np.concatenate((rgb_frames, pad_rgb_frames), axis=0)
#             return padded_spike_frames, padded_rgb_frames
        
#     def generate_bimodal_synchronous_frame_stream(self, frames, events, H=260, W=346):
#         t_event = events['t']
#         t_frame = np.array(list(map(lambda x:int(x), frames['t'])))
#         imgs = frames['imgs']

#         t_frame, imgs = self.align_frame(t_frame, imgs, t_event)

#         spike_frames = []
#         for t_frame_ind in range(len(t_frame)-1):
#             t_l, t_r = t_frame[t_frame_ind], t_frame[t_frame_ind+1]
#             t_l_ind, t_r_ind = np.searchsorted(t_event, t_l), np.searchsorted(t_event, t_r)
#             spike_frame = integrate_events_segment_to_frame(events['x'], events['y'], events['p'], H, W, t_l_ind, t_r_ind)
#             spike_frames.append(np.expand_dims(spike_frame, 0))
#         spike_frames = np.concatenate(spike_frames)
#         imgs = np.transpose(imgs, (0, 3, 1, 2))

#         return spike_frames, imgs[1:]
    
#     def align_frame(self, t_frame, imgs, t_event):
#         while t_frame[0] < t_event[0]:
#             t_frame = t_frame[1:]
#             imgs = imgs[1:]
#         while t_frame[-1] > t_event[-1]:
#             t_frame = t_frame[:-1]
#             imgs = imgs[:-1]
#         return t_frame, imgs



# class DVS346Sign(Dataset):
#     def __init__(self, root_dir, users, labels, lights, positions, transform=None):
#         self.root_dir = root_dir
#         self.users = users
#         self.labels = labels
#         self.lights = lights
#         self.positions = positions
#         self.transform = transform
#         self.data_list, self.label_list = self.get_data_label_file()

#     # need to overload
#     def __len__(self):
#         return len(self.data_list)

#     # need to overload
#     def __getitem__(self, idx):
#         event_file, frame_file = self.data_list[idx]
#         label = self.label_list[idx]
#         spike_frames = np.load(event_file, allow_pickle=True, mmap_mode='r')['arr_0'].astype(np.float32)
#         rgb_frames = np.load(frame_file, allow_pickle=True, mmap_mode='r')['arr_0'].astype(np.float32)
#         return (spike_frames, rgb_frames), label

#     def get_data_label_file(self):
#         data_list = []
#         label_list = []
#         for user in self.users:
#             for i, label in enumerate(self.labels):
#                 for light in self.lights:
#                     for pos in self.positions:
#                         event_name = '_'.join([user, label, light, pos, 'event'])
#                         event_file = os.path.join(self.root_dir, user, label, event_name, event_name+'.npz')
#                         frame_name = '_'.join([user, label, light, pos, 'frame'])
#                         frame_file = os.path.join(self.root_dir, user, label, frame_name, frame_name+'.npz')
#                         data_list.append((event_file, frame_file))
#                         label_list.append(i)
#         return data_list, label_list

# class DVS346Sign(Dataset):
#     def __init__(self, root_dir, users, labels, lights, positions, transform=None):
#         self.root_dir = root_dir
#         self.users = users
#         self.labels = labels
#         self.lights = lights
#         self.positions = positions
#         self.transform = transform
#         self.data_list, self.label_list = self.get_data_label_file()

#     # need to overload
#     def __len__(self):
#         return len(self.data_list)

#     # need to overload
#     def __getitem__(self, idx):
#         event_file, frame_file = self.data_list[idx]
#         label = self.label_list[idx]
#         spike_frames = np.load(event_file)[:,:,2:258,45:45+256]
# #         print(spike_frames.shape)
# #         rgb_frames = np.load(frame_file)[:,:,2:258,45:45+256]
#         rgb_frames = 1
#         return (spike_frames, rgb_frames), label

#     def get_data_label_file(self):
#         data_list = []
#         label_list = []
#         for user in self.users:
#             for i, label in enumerate(self.labels):
#                 for light in self.lights:
#                     for pos in self.positions:
#                         event_name = '_'.join([user, label, light, pos, 'event'])
#                         event_file = os.path.join(self.root_dir, user, label, event_name+'.npy')
#                         frame_name = '_'.join([user, label, light, pos, 'frame'])
#                         frame_file = os.path.join(self.root_dir, user, label, frame_name+'.npy')
#                         data_list.append((event_file, frame_file))
#                         label_list.append(i)
#         return data_list, label_list

# class DVS346Sign(Dataset):
#     def __init__(self, root_dir, users, labels, lights, positions, transform=None):
#         self.root_dir = root_dir
#         self.users = users
#         self.labels = labels
#         self.lights = lights
#         self.positions = positions
#         self.transform = transform
#         self.data_list, self.label_list = self.get_data_label_file()

#     # need to overload
#     def __len__(self):
#         return len(self.data_list)

#     # need to overload
#     def __getitem__(self, idx):
#         file = self.data_list[idx]
#         label = self.label_list[idx]
#         frames = np.load(file)[:, :, 2:258, 45:45+256]
#         return frames, label

#     def get_data_label_file(self):
#         data_list = []
#         label_list = []
#         for user in self.users:
#             for i, label in enumerate(self.labels):
#                 for light in self.lights:
#                     for pos in self.positions:
#                         name = '_'.join([user, label, light, pos])
#                         file = os.path.join(self.root_dir, user, label, name+'.npy')
#                         data_list.append(file)
#                         label_list.append(i)
#         return data_list, label_list

# class DVS346Sign(Dataset):
#     def __init__(self, root_dir, users, labels, lights, positions, transform=None):
#         self.root_dir = root_dir
#         self.users = users
#         self.labels = labels
#         self.lights = lights
#         self.positions = positions
#         self.transform = transform
#         self.data_list, self.label_list = self.get_data_label_file()

#     # need to overload
#     def __len__(self):
#         return len(self.data_list)

#     # need to overload
#     def __getitem__(self, idx):
#         file = self.data_list[idx]
#         label = self.label_list[idx]
#         frames = np.load(file)
#         return frames, label

#     def get_data_label_file(self):
#         data_list = []
#         label_list = []
#         for user in self.users:
#             for i, label in enumerate(self.labels):
#                 for light in self.lights:
#                     for pos in self.positions:
#                         name = '_'.join([user, label, light, pos])
#                         file = os.path.join(self.root_dir, user, label, name+'.npy')
#                         data_list.append(file)
#                         label_list.append(i)
#         return data_list, label_list