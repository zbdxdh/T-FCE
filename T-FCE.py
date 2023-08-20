from scipy.io import wavfile
import matplotlib.pyplot as plt
import pywt
from scipy.fftpack import fft
from torch.utils.data import DataLoader
import os
from torch.utils.data import Dataset, Subset
import numpy as np
import torch
from torch import nn
import math
from torch import optim
from torch.nn.modules.loss import CrossEntropyLoss
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import random

save_model_path = 'Model'
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

#Parameters for reproduction
def init_seeds(seed=0):
    torch.manual_seed(seed)  # sets the seed for generating random numbers.
    torch.cuda.manual_seed(
        seed)  # Sets the seed for generating random numbers for the current GPU. It’s safe to call this function if CUDA is not available; in that case, it is silently ignored.
    # torch.cuda.manual_seed_all(seed) # Sets the seed for generating random numbers on all GPUs. It’s safe to call this function if CUDA is not available; in that case, it is silently ignored.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
random_seed = 2
init_seeds(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)


def has_file_allowed_extension(filename: str, extensions) -> bool:
    """Checks if a file is an allowed extension.
    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions if isinstance(extensions, str) else tuple(extensions))


def is_wav_file(filename: str) -> bool:
    """Checks if a file is an allowed image.py extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image.py extension
    """
    return has_file_allowed_extension(filename, ("wav",))


def find_classes(directory: str):
    """Finds the class folders in a dataset.

    See :class:`DatasetFolder` for details.
    """
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    if not classes:
        raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    print(classes)
    print(class_to_idx)
    return classes, class_to_idx


def make_dataset(
        directory: str,
        class_to_idx=None,
        extensions=None,
        is_valid_file=None,
):
    """Generates a list of samples of a form (path_to_sample, class).

    See :class:`DatasetFolder` for details.

    Note: The class_to_idx parameter is here optional and will use the logic of the ``find_classes`` function
    by default.
    """
    directory = os.path.expanduser(directory)

    if class_to_idx is None:
        _, class_to_idx = find_classes(directory)
    elif not class_to_idx:
        raise ValueError("'class_to_index' must have at least one entry to collect any samples.")

    both_none = extensions is None and is_valid_file is None
    both_something = extensions is not None and is_valid_file is not None
    if both_none or both_something:
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")

    if extensions is not None:
        def is_valid_file(x: str) -> bool:
            return has_file_allowed_extension(x, extensions)  # type: ignore[arg-type]

    instances = []
    available_classes = set()
    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(directory, target_class)
        if not os.path.isdir(target_dir):
            continue
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    item = path, class_index
                    instances.append(item)

                    if target_class not in available_classes:
                        available_classes.add(target_class)

    empty_classes = set(class_to_idx.keys()) - available_classes
    if empty_classes:
        msg = f"Found no valid file for the classes {', '.join(sorted(empty_classes))}. "
        if extensions is not None:
            msg += f"Supported extensions are: {extensions if isinstance(extensions, str) else ', '.join(extensions)}"
        raise FileNotFoundError(msg)

    return instances


class DatasetFolder(Dataset):
    """A generic data loader.

    This default directory structure can be customized by overriding the
    :meth:`find_classes` method.

    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (tuple[string]): A list of allowed extensions.
            both extensions and is_valid_file should not be passed.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        is_valid_file (callable, optional): A function that takes path of a file
            and check if the file is a valid file (used to check of corrupt files)
            both extensions and is_valid_file should not be passed.

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image.py in the dataset
    """

    def __init__(
            self,
            root: str,
            transform=None,
            target_transform=None,
            is_valid_file=None,
    ) -> None:
        super().__init__()
        self.root = root
        classes, class_to_idx = self.find_classes(self.root)
        samples = self.make_dataset(self.root, class_to_idx, "wav", is_valid_file)

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]
        self.train = False

    @staticmethod
    def make_dataset(
            directory: str,
            class_to_idx,
            extensions=None,
            is_valid_file=None,
    ):
        """Generates a list of samples of a form (path_to_sample, class).

        This can be overridden to e.g. read files from a compressed zip file instead of from the disk.

        Args:
            directory (str): root dataset directory, corresponding to ``self.root``.
            class_to_idx (Dict[str, int]): Dictionary mapping class name to class index.
            extensions (optional): A list of allowed extensions.
                Either extensions or is_valid_file should be passed. Defaults to None.
            is_valid_file (optional): A function that takes path of a file
                and checks if the file is a valid file
                (used to check of corrupt files) both extensions and
                is_valid_file should not be passed. Defaults to None.

        Raises:
            ValueError: In case ``class_to_idx`` is empty.
            ValueError: In case ``extensions`` and ``is_valid_file`` are None or both are not None.
            FileNotFoundError: In case no valid file was found for any class.

        Returns:
            List[Tuple[str, int]]: samples of a form (path_to_sample, class)
        """
        if class_to_idx is None:
            # prevent potential bug since make_dataset() would use the class_to_idx logic of the
            # find_classes() function, instead of using that of the find_classes() method, which
            # is potentially overridden and thus could have a different logic.
            raise ValueError("The class_to_idx parameter cannot be None.")
        return make_dataset(directory, class_to_idx, extensions=extensions, is_valid_file=is_valid_file)

    def find_classes(self, directory: str):
        """Find the class folders in a dataset structured as follows::

            directory/
            ├── class_x
            │   ├── xxx.ext
            │   ├── xxy.ext
            │   └── ...
            │       └── xxz.ext
            └── class_y
                ├── 123.ext
                ├── nsdf3.ext
                └── ...
                └── asd932_.ext

        This method can be overridden to only consider
        a subset of classes, or to adapt to a different dataset directory structure.

        Args:
            directory(str): Root directory path, corresponding to ``self.root``

        Raises:
            FileNotFoundError: If ``dir`` has no class folders.

        Returns:
            (Tuple[List[str], Dict[str, int]]): List of all classes and dictionary mapping each class to an index.
        """
        return find_classes(directory)

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        # print(path)
        samplerate, sample = wavfile.read(path)
        # print(len(sample))
        interval = 768000
        sample = np.nan_to_num(sample, nan=1, posinf=10000)
        # random sampling for training
        if not self.train:
            out = np.zeros((interval, sample.shape[1])) if len(sample.shape) > 1 else np.zeros((interval,))
            if len(sample) <= interval:
                out[:len(sample)] = sample
            else:
                start = np.random.randint(0, len(sample) - interval)
                out = sample[start:start + interval]
        else:
            out = sample

        if len(out.shape) != 2:
            out = np.expand_dims(out, 1)
            out = np.repeat(out, 2, 1)
            # print("out shape", out.shape)
        if len(out.shape) > 1:
            out[:, 1] = out[:, 0]
        return out, target

    def __len__(self) -> int:
        return len(self.samples)


def get_dataloader(ds, bs, split):
    '''
        Get dataloaders of the training and validation set.

        Parameter:
            train_ds: Dataset
                Training set
            valid_ds: Dataset
                Validation set
            bs: Int
                Batch size

        Return:
            (train_dl, valid_dl): Tuple of DataLoader
                Dataloaders of training and validation set.
    '''

    # Stratified Sampling for train and val
    train_idx, validation_idx = train_test_split(np.arange(len(ds)),
                                                 test_size=0.1,
                                                 random_state=999,
                                                 shuffle=True,
                                                 stratify=ds.targets)

    # Subset dataset for train and val
    train_ds = Subset(ds, train_idx)
    test_ds = Subset(ds, validation_idx)

    train_ds.train = True
    test_ds.train = False
    # print(len(ds))
    # print(range(len(ds) -int(len(ds)*split)))
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True),
        # test batch size=1 for different length
        DataLoader(test_ds, batch_size=1),
    )

def apply_fft(x, fs, num_samples):
    f = np.linspace(0.0, (fs / 2.0), num_samples // 2)
    freq_values = fft(x)
    freq_values = 2.0 / num_samples * np.abs(freq_values[0:num_samples // 2])
    return f, freq_values


# add channel attention
class Conv_1D_2L_CA(nn.Module):
    def __init__(self, n_in, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.n_in = n_in
        self.bn1 = nn.BatchNorm1d(2)
        self.layer1 = nn.Sequential(
            nn.Conv1d(2, 32, (9,), stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.MaxPool1d(2, stride=2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv1d(32, 64, (5,), stride=2, padding=0),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.AvgPool1d(2, stride=2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv1d(64, 64, (5,), stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.AvgPool1d(2, stride=2)
        )

        # self.linear1 = nn.Linear(self.n_in * 128 // 4, 4)
        self.linear1 = nn.Linear(64 * 4, 4)
        self.linear2 = nn.Linear(16, 4)
        dim = 64 * 4
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x1, x2):
        x1 = x1.transpose(1, 2)
        B, C, N = x1.shape
        # apply log to x1
        x1 = self.bn1(x1.float())
        x2 = (x2 - x2.mean(1, keepdims=True)) / (x2.std(1, keepdims=True) + 1e-5)
        x2 = x2.float()

        # print(x1)
        # print(x2)
        # print(x1.dtype)

        # print("input1 shape", x1.shape)
        # print("input2 shape", x2.shape)
        x1 = self.layer1(x1)
        x1 = self.layer2(x1)
        x = self.layer3(x1).transpose(-1, -2)
        B, N, C = x.shape
        x = x.reshape(B, N // 4, C * 4)

        # channel attention
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C) + x
        x = self.proj(x)
        x = self.proj_drop(x)

        x = x.mean(1)
        # print("attention output shape", x.shape)

        return self.linear1(x) + self.linear2(x2.squeeze())
        # return self.linear2(x2.squeeze())


def validate(model, dl, loss_func):
    total_loss = 0.0
    total_size = 0
    predictions = []
    y_true = []
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # device = torch.device("cpu")
    for i, (xb, yb) in enumerate(dl):
        features = np.repeat(np.nan, 2 * m * num_features).reshape(2, m * num_features)
        freq = [[[], []] for _ in range(len(xb))]
        amp = [[[], []] for _ in range(len(xb))]
        xb = xb.numpy()
        for k in range(len(xb)):
            # left
            wp = pywt.WaveletPacket(xb[k][:, 0], wavelet=wavelet_function,
                                    maxlevel=num_levels)  # Wavelet packet transformation
            packet_names = [node.path for node in wp.get_level(num_levels, "natural")]
            for j in range(num_features):
                new_wp = pywt.WaveletPacket(data=None, wavelet=wavelet_function, maxlevel=num_levels)
                new_wp[packet_names[j]] = wp[packet_names[j]].data
                reconstructed_signal = new_wp.reconstruct(
                    update=False)  # Signal reconstruction from wavelet packet coefficients
                f, c = apply_fft(reconstructed_signal, fs, len(reconstructed_signal))
                z = abs(c)

                # Find  m  highest amplitudes of the spectrum and their corresponding frequencies:
                maximal_idx = np.argpartition(z, -m)[-m:]
                high_amp = z[maximal_idx]
                high_freq = f[maximal_idx]
                feature = high_amp * high_freq
                # print("high amp", high_amp)
                # print("high freq", high_freq)
                amp[k][0].append(high_amp[0])
                freq[k][0].append(high_freq[0])
                l = 0
                for f in feature:
                    features[0, j * m + l] = f
                    l = l + 1
            # right
            wp = pywt.WaveletPacket(xb[k][:, 1], wavelet=wavelet_function,
                                    maxlevel=num_levels)  # Wavelet packet transformation
            packet_names = [node.path for node in wp.get_level(num_levels, "natural")]
            for j in range(num_features):
                new_wp = pywt.WaveletPacket(data=None, wavelet=wavelet_function, maxlevel=num_levels)
                new_wp[packet_names[j]] = wp[packet_names[j]].data
                reconstructed_signal = new_wp.reconstruct(
                    update=False)  # Signal reconstruction from wavelet packet coefficients
                f, c = apply_fft(reconstructed_signal, fs, len(reconstructed_signal))
                z = abs(c)

                # Find  m  highest amplitudes of the spectrum and their corresponding frequencies:
                maximal_idx = np.argpartition(z, -m)[-m:]
                high_amp = z[maximal_idx]
                high_freq = f[maximal_idx]
                feature = high_amp * high_freq
                # print("high amp", high_amp)
                # print("high freq", high_freq)
                amp[k][1].append(high_amp[0])
                freq[k][1].append(high_freq[0])
                l = 0
                for f in feature:
                    features[1, j * m + l] = f
                    l = l + 1
        freq = np.array(freq)
        amp = np.array(amp)
        freq = torch.tensor(freq).reshape(len(xb), -1)
        amp = torch.tensor(amp).reshape(len(xb), -1)
        # print(yb.dtype)
        # print(yb)
        # print(freq.shape)
        xf = freq * amp

        xb, yb, xf = torch.tensor(xb).to(device), torch.tensor(yb).to(device), torch.tensor(xf).to(device)

        loss, batch_size, pred = loss_batch(model, loss_func, xb, xf, yb)
        total_loss += loss * batch_size
        total_size += batch_size
        predictions.append(pred)
        y_true.append(yb.cpu().numpy())
    mean_loss = total_loss / total_size
    predictions = np.concatenate(predictions, axis=0)
    y_true = np.concatenate(y_true, axis=0)
    accuracy = np.mean((predictions == y_true))
    return mean_loss, accuracy, (y_true, predictions)



def loss_batch(model, loss_func, xb, xf, yb, opt=None):
    '''
        Parameter:
            model: Module
                Your neural network model
            loss_func: Loss
                Loss function, e.g. CrossEntropyLoss()
            xb: Tensor
                One batch of input x
            yb: Tensor
                One batch of true label y
            opt: Optimizer
                Optimizer, e.g. SGD()

        Return:
            loss.item(): Python number
                Loss of the current batch
            len(xb): Int
                Number of examples of the current batch
            pred: ndarray
                Predictions (class with highest probability) of the minibatch
                input xb
    '''
    out = model(xb, xf)
    # print("out shape", out.shape)

    # print(out)
    # print(yb)
    loss = loss_func(out, yb)
    pred = torch.argmax(out, dim=1).cpu().numpy()

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb), pred


# Source of this code: https://github.com/nachiket273/One_Cycle_Policy
class OneCycle(object):
    """
    In paper (https://arxiv.org/pdf/1803.09820.pdf), author suggests to do one cycle during
    whole run with 2 steps of equal length. During first step, increase the learning rate
    from lower learning rate to higher learning rate. And in second step, decrease it from
    higher to lower learning rate. This is Cyclic learning rate policy. Author suggests one
    addition to this. - During last few hundred/thousand iterations of cycle reduce the
    learning rate to 1/100th or 1/1000th of the lower learning rate.
    Also, Author suggests that reducing momentum when learning rate is increasing. So, we make
    one cycle of momentum also with learning rate - Decrease momentum when learning rate is
    increasing and increase momentum when learning rate is decreasing.
    Args:
        nb              Total number of iterations including all epochs
        max_lr          The optimum learning rate. This learning rate will be used as highest
                        learning rate. The learning rate will fluctuate between max_lr to
                        max_lr/div and then (max_lr/div)/div.
        momentum_vals   The maximum and minimum momentum values between which momentum will
                        fluctuate during cycle.
                        Default values are (0.95, 0.85)
        prcnt           The percentage of cycle length for which we annihilate learning rate
                        way below the lower learnig rate.
                        The default value is 10
        div             The division factor used to get lower boundary of learning rate. This
                        will be used with max_lr value to decide lower learning rate boundary.
                        This value is also used to decide how much we annihilate the learning
                        rate below lower learning rate.
                        The default value is 10.
    """

    def __init__(self, nb, max_lr, momentum_vals=(0.95, 0.85), prcnt=10, div=10):
        self.nb = nb
        self.div = div
        self.step_len = math.ceil(self.nb * (1 - prcnt / 100) / 2)
        self.high_lr = max_lr
        self.low_mom = momentum_vals[1]
        self.high_mom = momentum_vals[0]
        self.prcnt = prcnt
        self.iteration = 0
        self.lrs = []
        self.moms = []

    def calc(self):
        self.iteration += 1
        lr = self.calc_lr()
        mom = self.calc_mom()
        return (lr, mom)

    def calc_lr(self):
        if self.iteration == self.nb:
            self.iteration = 0
            self.lrs.append(self.high_lr / self.div)
            return self.high_lr / self.div
        if self.iteration > 2 * self.step_len:
            ratio = (self.iteration - 2 * self.step_len) / (self.nb - 2 * self.step_len)
            lr = self.high_lr * (1 - 0.99 * ratio) / self.div
        elif self.iteration > self.step_len:
            ratio = 1 - (self.iteration - self.step_len) / self.step_len
            lr = self.high_lr * (1 + ratio * (self.div - 1)) / self.div
        else:
            ratio = self.iteration / self.step_len
            lr = self.high_lr * (1 + ratio * (self.div - 1)) / self.div
        self.lrs.append(lr)
        return lr

    def calc_mom(self):
        if self.iteration == self.nb:
            self.iteration = 0
            self.moms.append(self.high_mom)
            return self.high_mom
        if self.iteration > 2 * self.step_len:
            mom = self.high_mom
        elif self.iteration > self.step_len:
            ratio = (self.iteration - self.step_len) / self.step_len
            mom = self.low_mom + ratio * (self.high_mom - self.low_mom)
        else:
            ratio = self.iteration / self.step_len
            mom = self.high_mom - ratio * (self.high_mom - self.low_mom)
        self.moms.append(mom)
        return mom


def update_lr(optimizer, lr):
    for g in optimizer.param_groups:
        g['lr'] = lr


def update_mom(optimizer, mom):
    for g in optimizer.param_groups:
        g['momentum'] = mom


ds = DatasetFolder("dataset_back")
path_save = "./data_save/"
train_dl, valid_dl = get_dataloader(ds, 2, 0.8)

#####INPUTS / Parameters #############
fs = 44100
wavelet_function = "db4"
num_levels = 3  # k parameter
m = 1  # m parameter
num_features = 2 ** num_levels
#### HYPERPARAMETERS ####
bs = 64
lr = 0.0001
wd = 1e-5
betas = (0.99, 0.999)

model = Conv_1D_2L_CA(m * num_features)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# device = torch.device("cpu")

model.to(device)
opt = optim.Adam(model.parameters(), lr=lr, betas=betas, weight_decay=wd)
loss_func = CrossEntropyLoss()

## Train
epochs = 150

print(
    'EPOCH', '\t',
    'Train Loss', '\t',
    'Val Loss', '\t',
    'Train Acc', '\t',
    'Val Acc', '\t')
# Initialize dic to store metrics for each epoch.
metrics_dic = {}
metrics_dic['train_loss'] = []
metrics_dic['train_accuracy'] = []
metrics_dic['val_loss'] = []
metrics_dic['val_accuracy'] = []


one_cycle = OneCycle(bs, lr)

train_acc = []
val_acc = []

train_losses = []
val_losses = []
for epoch in range(epochs):
    # Train
    model.train()
    train_loss = 0.0
    train_accuracy = 0.0
    num_examples = 0
    for i, (xb, yb) in enumerate(train_dl):
        features = np.repeat(np.nan, 2 * m * num_features).reshape(2, m * num_features)
        freq = [[[], []] for _ in range(len(xb))]
        amp = [[[], []] for _ in range(len(xb))]
        xb = xb.numpy()
        for k in range(len(xb)):
            # left
            wp = pywt.WaveletPacket(xb[k][:, 0], wavelet=wavelet_function,
                                    maxlevel=num_levels)  # Wavelet packet transformation
            packet_names = [node.path for node in wp.get_level(num_levels, "natural")]
            for j in range(num_features):
                new_wp = pywt.WaveletPacket(data=None, wavelet=wavelet_function, maxlevel=num_levels)
                new_wp[packet_names[j]] = wp[packet_names[j]].data
                reconstructed_signal = new_wp.reconstruct(
                    update=False)  # Signal reconstruction from wavelet packet coefficients
                f, c = apply_fft(reconstructed_signal, fs, len(reconstructed_signal))
                z = abs(c)

                # Find  m  highest amplitudes of the spectrum and their corresponding frequencies:
                maximal_idx = np.argpartition(z, -m)[-m:]
                high_amp = z[maximal_idx]
                high_freq = f[maximal_idx]
                feature = high_amp * high_freq
                # print("high amp", high_amp)
                # print("high freq", high_freq)
                amp[k][0].append(high_amp[0])
                freq[k][0].append(high_freq[0])
                # print(amp)
                l = 0
                for f in feature:
                    features[0, j * m + l] = f
                    l = l + 1
            # right
            wp = pywt.WaveletPacket(xb[k][:, 1], wavelet=wavelet_function,
                                    maxlevel=num_levels)  # Wavelet packet transformation
            packet_names = [node.path for node in wp.get_level(num_levels, "natural")]
            for j in range(num_features):
                new_wp = pywt.WaveletPacket(data=None, wavelet=wavelet_function, maxlevel=num_levels)
                new_wp[packet_names[j]] = wp[packet_names[j]].data
                reconstructed_signal = new_wp.reconstruct(
                    update=False)  # Signal reconstruction from wavelet packet coefficients
                f, c = apply_fft(reconstructed_signal, fs, len(reconstructed_signal))
                z = abs(c)

                # Find  m  highest amplitudes of the spectrum and their corresponding frequencies:
                maximal_idx = np.argpartition(z, -m)[-m:]
                high_amp = z[maximal_idx]
                high_freq = f[maximal_idx]
                feature = high_amp * high_freq
                # print("high amp", high_amp)
                # print("high freq", high_freq)
                amp[k][1].append(high_amp[0])
                freq[k][1].append(high_freq[0])
                l = 0
                for f in feature:
                    features[1, j * m + l] = f
                    l = l + 1
        freq = np.array(freq)
        amp = np.array(amp)
        freq = torch.tensor(freq).reshape(len(xb), -1)
        amp = torch.tensor(amp).reshape(len(xb), -1)
        # print(yb.dtype)
        # print(yb)
        # print(freq.shape)
        xf = freq * amp

        N, T, V = xb.shape

        xb, yb, xf = torch.tensor(xb).to(device), torch.tensor(yb).to(device), torch.tensor(xf).to(device)

        print(i)

        loss, batch_size, pred = loss_batch(model, loss_func, xb, xf, yb, opt)
        print(loss)
        # lr, mom = one_cycle.calc()
        # update_lr(opt, lr)
        # update_mom(opt, mom)

    # Validate
    model.eval()
    with torch.no_grad():
        val_loss, val_accuracy, result_valid = validate(model, valid_dl, loss_func)

        train_loss, train_accuracy, result_train = validate(model, train_dl, loss_func)
    print("saving result")
    np.save(path_save + "result_valid_" + str(epoch), np.array(result_valid))
    np.save(path_save + "result_train_" + str(epoch), np.array(result_train))

    print(val_accuracy)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_acc.append(train_accuracy)
    val_acc.append(val_accuracy)
    metrics_dic['val_loss'].append(val_loss)
    metrics_dic['val_accuracy'].append(val_accuracy)
    metrics_dic['train_loss'].append(train_loss)
    metrics_dic['train_accuracy'].append(train_accuracy)

    print(
        f'{epoch} \t',
        f'{train_loss:.05f}', '\t',
        f'{val_loss:.05f}', '\t',
        f'{train_accuracy:.05f}', '\t'
                                  f'{val_accuracy:.05f}', '\t')

metrics = pd.DataFrame.from_dict(metrics_dic)

torch.save(model.state_dict(), f'{save_model_path}/model.pth')


