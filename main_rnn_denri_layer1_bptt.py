import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR,MultiStepLR
from torch.utils import data
from dataset_processing import data_generator, Epilepsiae_iEEG, Reading_FB_training
from sklearn.metrics import roc_auc_score, precision_score, recall_score, roc_curve
from sklearn import metrics
torch.manual_seed(42)
batch_size = 200
from SNN_layers.spike_dense import *
from SNN_layers.spike_neuron import *
from SNN_layers.spike_rnn import *
thr_func = ActFun_adp.apply
from SNN_layers.liquid_neuron import *
import argparse
import time
import os
import torch.nn.functional as F
import seaborn as sns
from spikingjelly.activation_based import neuron
from spikingjelly import visualizing
import matplotlib

parser = argparse.ArgumentParser(description='Sequential Decision Making..')
parser.add_argument('--dataset', type=str,
                    default='TUH_RAW',
                    help='path to load the model')
parser.add_argument('--load', type=str,
                    default="",
                    help='path to load the model')
parser.add_argument('--debias', action='store_true', help='FedDyn debias algorithm')
parser.add_argument('--save', type=str, default='./models/',
                    help='path to load the model')
parser.add_argument('--branches', type=int, default= '4',
                    help="number of branches")
parser.add_argument('--batch', type=int, default= '256',
                    help="number of branches")
parser.add_argument('--n', type=int, default= '100',
                    help="number of neurons")
parser.add_argument('--ICA', type=bool, default= True,
                    help="Process by ICA")
parser.add_argument('--alpha', type=float, default=.1, help='Alpha')
parser.add_argument('--beta', type=float, default=0.5, help='Beta')
parser.add_argument('--rho', type=float, default=0.0, help='Rho')
parser.add_argument('--lmbda', type=float, default=2.0, help='Lambda')
args = parser.parse_args()

ICA = args.ICA
torch.manual_seed(42)
batch_size = args.batch

# # # TUH TRAINING or Testing #FOR 3 layers

train_loader, test_loader, seq_length, input_channels, n_classes = data_generator(args.dataset, batch_size, ICA)
output_file_fol = "./Results/" + str(batch_size) + 'B-' + str(3) + 'L-' + str(
    args.branches) + 'BR-' + str(args.n) + 'n-' + str(ICA) + '' + str(args.dataset) + "/"


os.makedirs(output_file_fol, exist_ok=True)

output_dim = 2

is_bias=True

def get_stats_named_params( model ):

    named_params = {}
    for name, param in model.named_parameters():
        sm, lm, dm = param.detach().clone(), 0.0*param.detach().clone(), 0.0*param.detach().clone()
        named_params[name] = (param, sm, lm, dm)
    return named_params
def post_optimizer_updates( named_params, args, epoch ):

    alpha = args.alpha
    beta = args.beta
    rho = args.rho

    for name in named_params:
        param, sm, lm, dm = named_params[name]
        param = param.to(device)
        sm = sm.to(device)
        lm = lm.to(device)
        dm = dm.to(device)
        if args.debias:
            beta = (1. / (1. + epoch))
            sm.data.mul_( (1.0-beta) )
            sm.data.add_( beta * param )

            rho = (1. / (1. + epoch))
            dm.data.mul_( (1.-rho) )
            dm.data.add_( rho * lm )
        else:
            lm.data.add_( -alpha * (param - sm) )
            sm.data.mul_( (1.0-beta) )
            sm.data.add_( beta * param - (beta/alpha) * lm )

def get_regularizer_named_params( named_params, args, _lambda=1.0 ):

    alpha = args.alpha
    rho = args.rho
    regularization = torch.zeros( [], device=device)
    for name in named_params:
        param, sm, lm, dm = named_params[name]
        regularization += (rho-1.) * torch.sum( param.to(device) * lm.to(device) )
        if args.debias:
            regularization += (1.-rho) * torch.sum( param.to(device) * dm.to(device) )
        else:
            r_p = _lambda * 0.5 * alpha * torch.sum( torch.square(param.to(device) - sm.to(device)))
            regularization += r_p
            # print(name,r_p)
    return regularization

def reset_named_params(named_params, args):

    if args.debias: return

    for name in named_params:
        param, sm, lm, dm = named_params[name]
        param.data.copy_(sm.data)

class RNN_test(nn.Module): #DH-SRNN model

    def __init__(self,):
        super(RNN_test, self).__init__()

        self.n = 50

        self.rnn_1 = spike_rnn_test_denri_wotanh_R(input_channels,self.n, tau_ninitializer = 'uniform',low_n = 0,high_n = 4,vth=1,dt =1,branch =4,device=device,bias=is_bias)
        self.rnn_2 = spike_rnn_test_denri_wotanh_R(self.n,self.n*2, tau_ninitializer = 'uniform',low_n = 0,high_n = 4,vth=1,dt =1,branch =4,device=device,bias=is_bias)
        self.liquid_1 = SNN_rec_cell(self.n*2,self.n*2,is_rec=True)
        self.dense_1 = readout_integrator_test(self.n*2,output_dim,dt=1,device=device,bias=is_bias)

        torch.nn.init.xavier_normal_(self.dense_1.dense.weight)

        if is_bias:
            torch.nn.init.constant_(self.dense_1.dense.bias,0)

    def forward(self,input):

        # input.to(device)
        b,seq_length,input_dim = input.shape

        self.dense_1.set_neuron_state(b) #initalizating variables
        self.rnn_1.set_neuron_state(b)   #initializing variables
        self.rnn_2.set_neuron_state(b)

        layer_i = 0
        output = torch.zeros(b, output_dim).to(device)

        Acc_spk = []
        fr = 0

        Accumulated_spikes = []
        Acc_mem_2 = []
        Acc_spikes_2 = []
        Acc_0 = []
        total = []
        total0 = []

        Accumulated_spikes_layer1, Accumulated_spikes_layer2, Accumulated_zeros_layer1, Accumulated_zeros_layer2 = \
        [[] for _ in range(4)]

        Acc_mem, Acc_spikes, Acc_spikes2 = [[] for _ in range(3)]
        Acc_spikes_d1, Acc_spikes_d2, Acc_spikes_d3, Acc_spikes_d4 = [[] for _ in range(4)]
        Acc_spikes_d5, Acc_spikes_d6, Acc_spikes_d7, Acc_spikes_d8 = [[] for _ in range(4)]

        Acc_beta1_1, Acc_beta1_2, Acc_beta1_3, Acc_beta1_4 = [[] for _ in range(4)]
        Acc_beta2_1, Acc_beta2_2, Acc_beta2_3, Acc_beta2_4 = [[] for _ in range(4)]

        Acc_alpha1 = []

        Acc_Thres, Acc_TauMLiq, Acc_TauAdpLiq = [[] for _ in range(3)]
        Mem_ltc_list, Spk_ltc_list = [[] for _ in range(2)]

        for i in range(seq_length):

            input_x = input[:,i,:].reshape(b,input_dim)

            mem_layer1, spike_layer1, d1_1, d1_2, d1_3, d1_4, beta1, alpha1 = self.rnn_1.forward(input_x)
            mem_layer2, spike_layer2, d2_1, d2_2, d2_3, d2_4, beta2, alpha2 = self.rnn_2.forward(spike_layer1)

            if i == 0:
                h = model.liquid_1.init_hidden(b)
            else:
                h = tuple(v.detach() for v in h)

            mem_3, spk_3, b_1, TauMLiq, TauAdpLiq = self.liquid_1.forward(spike_layer2, h[layer_i * 3],
                                              h[1 + layer_i * 3], h[2 +layer_i *3])

            Mem_ltc_list.append(mem_3)
            Spk_ltc_list.append (spk_3)

            mem_dense = self.dense_1.forward(spk_3)

            if i>0:
                output += mem_dense

            # Count spikes (1s)
            spike_count_layer1 = spike_layer1.detach().cpu().sum().item()
            spike_count_layer2 = spike_layer2.detach().cpu().sum().item()

            # Calculate number of zeros (non-spikes)
            total_elements_layer1 = spike_layer1.numel()
            total_elements_layer2 = spike_layer2.numel()
            zero_count_layer1 = total_elements_layer1 - spike_count_layer1
            zero_count_layer2 = total_elements_layer2 - spike_count_layer2

            # Accumulate spikes and zeros
            Accumulated_spikes_layer1.append(spike_count_layer1)
            Accumulated_spikes_layer2.append(spike_count_layer2)
            Accumulated_zeros_layer1.append(zero_count_layer1)
            Accumulated_zeros_layer2.append(zero_count_layer2)

            # Accumulate firing rate over time
            fr += (spike_layer1.detach().cpu().numpy().mean() + spike_layer2.detach().cpu().numpy().mean()) / 2.

            Acc_mem.append(mem_layer1)
            Acc_spikes.append(spike_layer1)
            Acc_spikes2.append(spike_layer2)

            Acc_spikes_d1.append(d1_1)
            Acc_spikes_d2.append(d1_2)
            Acc_spikes_d3.append(d1_3)
            Acc_spikes_d4.append(d1_4)

            Acc_spikes_d5.append(d2_1)
            Acc_spikes_d6.append(d2_2)
            Acc_spikes_d7.append(d2_3)
            Acc_spikes_d8.append(d2_4)

            Acc_beta1_1.append(beta1[:, 0])  # 256, branches(4)
            Acc_beta1_2.append(beta1[:, 1])  # 256, branches(4)
            Acc_beta1_3.append(beta1[:, 2])  # 256, branches(4)
            Acc_beta1_4.append(beta1[:, 3])  # 256, branches(4)

            Acc_beta2_1.append(beta2[:, 0])  # 256, branches(4)
            Acc_beta2_2.append(beta2[:, 1])  # 256, branches(4)
            Acc_beta2_3.append(beta2[:, 2])  # 256, branches(4)
            Acc_beta2_4.append(beta2[:, 3])  # 256, branches(4)

            Acc_alpha1.append(alpha1)  # 256, branches(4)

            Acc_Thres.append(b_1)
            Acc_TauMLiq.append(TauMLiq)
            Acc_TauAdpLiq.append(TauAdpLiq)

        def plotting_beta(beta1,beta2,beta3,beta4,output_file_f):

            plt.figure(figsize=(12, 6))

            y1 = [tensor.cpu().detach().numpy() for tensor in beta1]
            y1 = np.array(y1)
            y1 = y1[:, :]

            y2 = [tensor.cpu().detach().numpy() for tensor in beta2]
            y2 = np.array(y2)
            y2 = y2[:, :]

            y3 = [tensor.cpu().detach().numpy()  for tensor in beta3]
            y3 = np.array(y3)
            y3 = y3[:, :]

            y4 = [tensor.cpu().detach().numpy()  for tensor in beta4]
            y4 = np.array(y4)
            y4 = y4[:,: ]

            y1 = y1.flatten()
            y2 = y2.flatten()
            y3 = y3.flatten()
            y4 = y4.flatten()

            #Freedman-Diaconis Rule
            def bin (data):
                q75, q25 = np.percentile(data, [75, 25])
                iqr = q75 - q25  # Interquartile range
                bin_width = 2 * iqr / np.cbrt(len(data))  # Freedman-Diaconis rule
                # num_bins = int((data.max() - data.min()) / bin_width)  # Number of bins
                bin_width = 2 * iqr / np.power(len(data), 1 / 3)
                num_bins = np.arange(min(data), max(data), bin_width)
                return num_bins

            sns.histplot(y1, alpha=0.5, kde=True, bins=bin(y1), edgecolor='None', color='#3594cc',
                         label='Branch 1', stat='density')

            sns.histplot(y2, alpha=0.5, kde=True, bins=bin(y2), edgecolor='None', color='#ea801c',
                         label='Branch 2', stat='density')

            sns.histplot(y3, alpha=0.5, kde=True, bins=bin(y3), edgecolor='None', color='#8cc5e3',
                         label='Branch 3', stat='density')

            sns.histplot(y4, alpha=0.5, kde=True, bins=bin(y4), edgecolor='None', color='#f0b077',
                         label='Branch 4', stat='density')

            sns.kdeplot(y1, alpha=1.0, color='#3594cc', linewidth=3, label='KDE Branch 1')
            sns.kdeplot(y2, alpha=1.0, color='#ea801c', linewidth=3, label='KDE Branch 2')
            sns.kdeplot(y3, alpha=1.0, color='#8cc5e3', linewidth=3, label='KDE Branch 3')
            sns.kdeplot(y4, alpha=1.0, color='#f0b077', linewidth=3, label='KDE Branch 4')

            # Set bold and font size for tick labels
            plt.xticks(fontsize=14, weight='bold')  # Bold and font size 20 for x-axis ticks
            plt.yticks(fontsize=14, weight='bold')

            # Add labels, legend, and title
            plt.xlabel(r'Timing Factor ($\alpha$)', fontsize=14, weight='bold')
            plt.ylabel('Density', fontsize=14, weight='bold')
            plt.legend(prop={'size': 12, 'weight': 'bold'}, ncol=2)

            # Display the plot
            plt.savefig(output_file_f, format='pdf', dpi=600)
            # plt.savefig("TUH_Timing_Factor_Highest_Auroc.svg", format='svg', dpi=600)

            plt.show()

        output_file_f = "TUH_Timing_Factor_Highest_Auroc_Layer1.pdf"
        plotting_beta (Acc_beta1_1, Acc_beta1_2,Acc_beta1_3,Acc_beta1_4,output_file_f)
        output_file_f = "TUH_Timing_Factor_Highest_Auroc_Layer2.pdf"
        plotting_beta (Acc_beta2_1, Acc_beta2_2,Acc_beta2_3,Acc_beta2_4,output_file_f)

        def plotting_alpha(alpha):

            plt.figure(figsize=(12, 6))

            y1 = [tensor.cpu().detach().numpy()  for tensor in alpha]
            y1 = np.array(y1)
            y1 = y1[:, :]
            y1 = y1.flatten()

            # Freedman-Diaconis Rule
            def bin(data):
                q75, q25 = np.percentile(data, [75, 25])
                iqr = q75 - q25  # Interquartile range
                bin_width = 2 * iqr / np.cbrt(len(data))  # Freedman-Diaconis rule
                # num_bins = int((data.max() - data.min()) / bin_width)  # Number of bins
                bin_width = 2 * iqr / np.power(len(data), 1 / 3)
                num_bins = np.arange(min(data), max(data), bin_width)
                return num_bins

            sns.histplot(y1, alpha = 0.5, label='Timing', kde=True, bins=bin(y1), color='#a559aa',
                         edgecolor= 'None', stat='density')

            sns.kdeplot(y1, alpha=1.0, color='#a559aa', linewidth=3, label='KDE')

            # Add labels, legend, and title
            plt.xlabel(r'Timing factor ($\beta$)', fontsize=14, weight='bold')
            plt.ylabel('Density', fontsize=14, weight='bold')
            plt.legend()

            # Set bold and font size for tick labels
            plt.xticks(fontsize=14, weight='bold')  # Bold and font size 20 for x-axis ticks
            plt.yticks(fontsize=14, weight='bold')
            plt.legend(prop={'size': 12, 'weight': 'bold'}, ncol=2)

            # # Display the plot
            # plt.savefig("TUH_Tau_Highest_Auroc.pdf", format='pdf', dpi=600)
            # plt.savefig("TUH_Tau_Highest_Auroc.svg", format='svg', dpi=600)

            plt.show()

        # plotting_alpha (Acc_alpha1)

        #LIQUID PLOTTING
        def plot_2d_heatmap(array: np.ndarray, title: str, xlabel: str, ylabel: str, int_x_ticks=True, int_y_ticks=True,
                            plot_colorbar=True, colorbar_y_label='magnitude', x_max=None, figsize=(12, 8), dpi=200):

            if array.ndim != 2:
                raise ValueError(f"Expected 2D array, got {array.ndim}D array instead")

            fig, heatmap = plt.subplots(figsize=figsize, dpi=dpi)
            
            vmax = 0.40
            vmin = -0.4
            cmap = 'coolwarm'

            if x_max is not None:
                im = heatmap.imshow(array.T, aspect='auto', cmap=cmap,
                                    extent=[-0.5, x_max, array.shape[1] - 0.5, -0.5], vmax=vmax, vmin=vmin)
            else:
                im = heatmap.imshow(array.T, aspect='auto', cmap=cmap)

            heatmap.set_title(title, fontsize=16, fontweight='bold')
            heatmap.set_xlabel(xlabel, fontsize=14, fontweight='bold')
            heatmap.set_ylabel(ylabel, fontsize=14, fontweight='bold')

            heatmap.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=int_x_ticks))
            heatmap.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=int_y_ticks))
            heatmap.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
            heatmap.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
            plt.xticks(fontsize=14, weight='bold')  # Bold and font size 14 for x-axis ticks
            plt.yticks(fontsize=14, weight='bold')  # Bold and font size 14 for y-axis ticks

            if plot_colorbar:
                cbar = heatmap.figure.colorbar(im)
                cbar.ax.set_ylabel(colorbar_y_label, rotation=90, va='top', fontsize=12, fontweight='bold')
                cbar.ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
                cbar.ax.tick_params(labelsize=12)  # Change tick label size

                for label in cbar.ax.get_yticklabels():
                    label.set_fontweight('bold')

            plt.savefig("Membrane_Potential_for_0.pdf", format='pdf', dpi=600)

            return fig


        def plot_Liquid_T_Constants(TauM1, TauAdp, mem_ltc_np, spk_ltc_np):

            def plot_spk(spk_list, shape='*', label='spk', baseline=0.):
                spk_t = np.where(spk_list == 1)[0]
                plt.plot(spk_t, np.ones(len(spk_t)) + baseline, shape, label=label)

            plt.figure(figsize=(12, 6))

            y1 = [tensor.cpu().detach().numpy() for tensor in TauM1]
            y1 = np.array(y1)
            tau_ltc_np = y1
            y1 = y1[0:250, :]

            y2 = [tensor.cpu().detach().numpy() for tensor in TauAdp]
            y2 = np.array(y2)
            y2 = y2[0:250, :]

            y1 = y1.flatten()
            y2 = y2.flatten()

            mem_ltc_np = [tensor.cpu().detach().numpy() for tensor in mem_ltc_np]
            mem_ltc_np = np.array(mem_ltc_np)

            spk_ltc_np = [tensor.cpu().detach().numpy() for tensor in spk_ltc_np]
            spk_ltc_np = np.array(spk_ltc_np)

            #Freedman-Diaconis Rule
            def bin (data):
                q75, q25 = np.percentile(data, [75, 25])
                iqr = q75 - q25  # Interquartile range
                # bin_width = 2 * iqr / np.cbrt(len(data))  # Freedman-Diaconis rule
                bin_width = 2 * iqr / np.power(len(data), 1 / 3)
                num_bins = np.arange(min(data), max(data), bin_width)
                return num_bins

            sns.histplot(y1, alpha=0.5, kde=True, bins=bin(y1), edgecolor='None', color='#8cc5e3',
                         label=r'$\mathbf{\tau}_{\mathrm{m}}$', stat='density')

            sns.histplot(y2, alpha=0.5, kde=True, bins=bin(y2), edgecolor='None', color='#ea801c',
                         label=r'$\mathbf{\tau}_{\mathrm{adp}}$', stat='density')

            sns.kdeplot(y1, alpha=1.0, color='#8cc5e3', linewidth=3, label=r'$\mathbf{\tau}_{\mathrm{m}}$')
            sns.kdeplot(y2, alpha=1.0, color='#ea801c', linewidth=3, label=r'$\mathbf{\tau}_{\mathrm{adp}}$')

            # Set bold and font size for tick labels
            plt.xticks(fontsize=14, weight='bold')  # Bold and font size 20 for x-axis ticks
            plt.yticks(fontsize=14, weight='bold')

            # Add labels, legend, and title
            plt.xlabel(r'Liquid Time Factors', fontsize=14, weight='bold')
            plt.ylabel('Density', fontsize=14, weight='bold')
            plt.legend(prop={'size': 12, 'weight': 'bold'}, ncol=2)
            plt.savefig("Liquid_Factors.pdf", format='pdf', dpi=600)
            plt.show()

            # #Behaviour of the spike
            # plt.plot(mem_ltc_np[:, 0, 10], label='mem')
            # plot_spk(spk_ltc_np[:, 0, 10])
            # plt.plot(tau_ltc_np[:, 0, 10], label='tau')
            # plt.legend()
            # plt.show()

            plot_2d_heatmap(array=np.asarray(mem_ltc_np[:,0,:]), title='Membrane Potentials', xlabel='Simulating Step',
                                        ylabel='Neuron Index', int_x_ticks=True, x_max=1500, dpi=600)
            plt.show()

            visualizing.plot_1d_spikes(spikes=np.asarray(spk_ltc_np[:, 0, :]), title='Spike', xlabel='Simulating Step',
                                       ylabel='Neuron Index', dpi=300)

            plt.show()

        plot_Liquid_T_Constants (Acc_TauMLiq, Acc_TauAdpLiq, Mem_ltc_list, Spk_ltc_list)

        output = F.softmax(output/seq_length,dim=1)
        return output

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("device:",device)

model = RNN_test()

# Function to count the number of parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Print the number of parameters
print(f'Total number of parameters: {count_parameters(model)}')

# Function to calculate the memory size of the parameters
def calculate_memory_size(model):
    total_params = count_parameters(model)
    param_size = next(model.parameters()).element_size()  # Size of a single parameter in bytes
    total_size_bytes = total_params * param_size
    total_size_kb = total_size_bytes / 1024
    total_size_mb = total_size_kb / 1024
    return total_size_bytes, total_size_kb, total_size_mb

# Calculate and print the memory size
total_size_bytes, total_size_kb, total_size_mb = calculate_memory_size(model)
print(f'Total memory size: {total_size_bytes:.2f} bytes')
print(f'Total memory size: {total_size_kb:.2f} KB')
print(f'Total memory size: {total_size_mb:.2f} MB')

criterion = nn.CrossEntropyLoss()#nn.NLLLoss()

def save_values (epoch, train_loss_sum, train_acc, valid_acc,val_auroc,val_recall, val_precision, output_file_fol):

    output_file = output_file_fol + str(batch_size) + 'B-' + str(2) + 'L-' + str(
        args.branches) + 'BR-' + str(args.n)+'n-' + str(ICA) + 'ICA-' + str(args.dataset) + '.txt'

    with open(output_file, 'a') as file:
        # Create the formatted string
        output_str = 'epoch: {:7d}, Train Loss: {:.4f}, Train Acc: {:.4f},Valid Acc: {:.4f},Val_AUROC: {:.4f}, Val_Recal: {:.4f}, Val_Precision: {:.4f}'.\
            format(epoch,train_loss_sum/len(train_loader), train_acc,valid_acc,val_auroc,val_recall, val_precision)

        # Write the string to the file
        file.write(output_str + "\n")

    plot_results(output_file,output_file_fol)

def calculate_auroc(labels, predicted_probs):
    return roc_auc_score(labels, predicted_probs)

def calculate_recall(labels, predicted_probs):
    return recall_score(labels, predicted_probs)

def calculate_precision(labels, predicted_probs):
    return precision_score(labels, predicted_probs)

def plot_results(output_file,output_file_fol):
    epochs = []
    valid_acc = []
    val_auroc = []
    val_recall = []
    val_precision = []
    train_loss = []

    # Read the log file and extract metric values

    with open(output_file, 'r') as log_file:
        for line in log_file:
            if line.startswith('epoch'):
                parts = line.strip().split(',')
                epoch = int(parts[0].split(':')[1].strip())
                loss = float(parts[1].split(':')[1].strip())
                acc = float(parts[3].split(':')[1].strip())
                auroc = float(parts[4].split(':')[1].strip())
                recall = float(parts[5].split(':')[1].strip())
                precision = float(parts[6].split(':')[1].strip())

                epochs.append(epoch)
                valid_acc.append(acc)
                val_auroc.append(auroc)
                val_recall.append(recall)
                val_precision.append(precision)
                train_loss.append(loss)

    plt.figure(figsize=(12, 6))
    plt.plot(epochs, valid_acc, label='Valid Acc')
    plt.plot(epochs, val_auroc, label='Val AUROC')
    plt.plot(epochs, val_recall, label='Val Recall')
    plt.plot(epochs, val_precision, label='Val Precision')
    plt.plot(epochs, train_loss, label="train_loss")

    # Add labels and legend
    plt.xlabel('Epoch')
    plt.ylabel('Metric Value')
    plt.title('Validation Metrics Over Epochs')
    plt.legend()

    outputfold = os.path.join(output_file_fol + str(batch_size) + 'B-' + str(2) + 'L-' + str(
        args.branches) + 'BR-' + str(args.n)+'n-' + str(ICA) + 'ICA-' + str(args.dataset) + '.png')

    plt.savefig(outputfold)

def plot_AUROC (target_1,output_1,auroc):

    fpr, tpr, thresholds = metrics.roc_curve(target_1, output_1)
    # Create ROC curve plot
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % auroc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')
    # Save the plot to a file
    output_folder = output_file_fol + '/AUROC/'
    os.makedirs(output_folder, exist_ok=True)
    outputfold = os.path.join(output_folder + str(batch_size) + 'B-' + str(2) + 'L-' + str(
        args.branches) + 'BR-' + str(args.n)+'n-'+ str(ICA) + 'ICA' + str(args.dataset) + '.pdf')

    plt.savefig(outputfold)

def plot_AUPRC (target_1,output_1):

    precision, recall, thresholds = metrics.precision_recall_curve(target_1, output_1)
    auprc = metrics.average_precision_score(target_1, output_1)
    print("AUPRC: ", auprc)

    # Create ROC curve plot
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='darkorange', lw=2, label='AUPRC curve (area = %0.2f)' % auprc)
    plt.plot([0, 1], [0, 0], color='navy', lw=2, linestyle='--')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim([0.0, 1.0])
    plt.ylim([-0.10, 1.0])
    plt.title('Precision-Recall Curve')
    plt.legend(loc='best')
    plt.show()

    # Save the plot to a file
    output_folder = output_file_fol + '/AUPRC/'

    os.makedirs(output_folder, exist_ok=True)
    outputfold = os.path.join(output_folder + str(batch_size) + 'B-' + str(2) + 'L-' + str(
        args.branches) + 'BR-' + str(args.n)+'n-'+ str(ICA) + 'ICA' + str(args.dataset) + '.pdf')

    plt.savefig(outputfold)

def print_message(i, epoch, batch_size, train_loss, start_time):

    elapsed_time = time.time() - start_time
    data_processed = i * batch_size
    total_data = len(train_loader.dataset)
    progress_percentage = 100. * data_processed / total_data
    time_per_iteration = elapsed_time / i if i > 0 else 0
    remaining_iterations = len(train_loader) - i
    remaining_time_seconds = remaining_iterations * time_per_iteration

    # Convert remaining time to hours, minutes, and seconds
    remaining_hours, remaining_seconds = divmod(remaining_time_seconds, 3600)
    remaining_minutes, remaining_seconds = divmod(remaining_seconds, 60)

    message = 'Train Epoch: {} [{}/{} ({:.0f}%)]\t Loss: {:.5f}\tRemaining: {:1}:{:02}:{:02}s'.format(
        epoch, data_processed, total_data, progress_percentage, train_loss, int(remaining_hours),
        int(remaining_minutes), int(remaining_seconds))
    print(message, end='\r', flush=True)

    sys.stdout.flush()

def test(model):

    test_acc = 0.
    sum_sample = 0.

    model.eval()
    model.rnn_1.apply_mask()
    model.rnn_2.apply_mask()

    with torch.no_grad():

        model.rnn_1.apply_mask()
        model.rnn_2.apply_mask()

        predictS= []
        true_labels = []
        predicted1 = []

        for i, (images, labels) in enumerate(test_loader):

            images = images.to(device)
            labels = labels.view((-1)).long().to(device)
            labels1 = labels

            predictions = model(images)

            _, predicted = torch.max(predictions.data, 1)

            output = predictions[:, 1]

            labels = labels.cpu()
            predicted = predicted.cpu().t()
            print ("Predicted_value", predicted)

            model.rnn_1.apply_mask()
            model.rnn_2.apply_mask()

            test_acc += (predicted == labels).sum()
            sum_sample+=predicted.numel()

            predicted1.append(predicted)
            predictS.append(output.squeeze())
            true_labels.append(labels1.squeeze())
            print (batch_size*i)

        valid_acc = test_acc.data.cpu().numpy() / sum_sample

        output_1 = torch.cat(predictS, axis=0)
        target_1 = torch.cat(true_labels, axis=0)
        predicted_1 = torch.cat(predicted1, axis=0)

        val_auroc = calculate_auroc(target_1.cpu(), output_1.cpu())
        print ("val_auroc: ", val_auroc)

        plot_AUROC(target_1.cpu(), output_1.cpu(), val_auroc)
        plot_AUPRC (target_1.cpu(), output_1.cpu())

        val_recall = calculate_recall(target_1.cpu(), predicted_1.cpu())
        print ("val_recall", val_recall)
        val_precision = calculate_precision(target_1.cpu(), predicted_1.cpu())
        print ("val_precision", val_precision)

        # print("classification report", classification_report(target_1.cpu(), predicted_1.cpu()))
        # print ("confusion matrix", confusion_matrix(target_1.cpu(), predicted_1.cpu()))

    return valid_acc, val_auroc, val_recall, val_precision

def train (epochs,criterion,optimizer,scheduler,output_file_fol) :

    acc_list = []
    best_acc = 0
    best_rec = 0


    for epoch in range(epochs):

        start_time = time.time()
        train_acc = 0
        sum_sample = 0
        train_loss_sum = 0
        model.train()
        model.rnn_1.apply_mask()
        model.rnn_2.apply_mask()


        train_loss = 0
        total_clf_loss = 0
        total_regularizaton_loss = 0

        for i, (images, labels) in enumerate(train_loader):

            images = images.to(device)
            labels = labels.view((-1)).long().to(device)
            optimizer.zero_grad()

            predictions = model(images)

            _, predicted = torch.max(predictions.data, 1)

            total_clf_loss = criterion(predictions,labels)

            train_loss = total_clf_loss

            train_loss.backward()  #calculate gradient of current time step
            train_loss_sum += train_loss.item()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1) #that means yes.

            model.rnn_1.apply_mask()  #apply the connection pattern
            model.rnn_2.apply_mask()

            optimizer.step()  #update network

            labels = labels.cpu()  #Labels to cpu
            predicted = predicted.cpu().t() #Value to CPU

            train_acc += (predicted == labels).sum()
            sum_sample+=predicted.numel()

            print_message(i, epoch, batch_size, train_loss, start_time)

        if scheduler:
            scheduler.step()

        train_acc = train_acc.data.cpu().numpy()/sum_sample
        train_loss_sum+= train_loss
        acc_list.append(train_acc)

        valid_acc, val_auroc, val_recall, val_precision = test(model)

        if val_auroc>best_acc:
            best_acc = val_auroc
            print ("saving new model validation at:", best_acc)

            torch.save(model,output_file_fol +"AUC-"+str(best_acc)[:7]+'-'+str(batch_size)+'B-'+str(2)+'L-'+str(args.branches)+'BR-'+str(args.n)+'n-'+ str(ICA) + 'ICA-' +str(args.dataset)+'.pth')

        if val_recall>best_rec:
            best_rec = val_recall
            print ("saving new model validation at:", best_rec)
            torch.save(model,output_file_fol +"REC-"+str(best_rec)[:7]+'-'+str(batch_size)+'B-'+str(2)+'L-'+str(args.branches)+'BR-'+str(args.n)+'n-'+ str(ICA) + 'ICA-' +str(args.dataset)+'.pth')

        print(
            'epoch: {:7d}, Train Loss: {:.4f}, Train Acc: {:.4f}, Valid Acc: {:.4f}, Val_AUROC: {:.4f}, Val_Recall: {:.4f}, Val_Precision: {:.4f}'.format(
                epoch,
                train_loss_sum / len(train_loader),
                train_acc,
                valid_acc,
                val_auroc,
                val_recall,
                val_precision), flush=True)

        save_values (epoch, train_loss_sum, train_acc, valid_acc, val_auroc,val_recall, val_precision, output_file_fol)

        total_clf_loss = 0
        total_regularizaton_loss = 0

    return acc_list

learning_rate = 1e-3#1.2e-2

base_params = [
                    model.dense_1.dense.weight,
                    model.dense_1.dense.bias,
                    model.rnn_1.dense.weight,
                    model.rnn_1.dense.bias,
                    model.rnn_2.dense.weight,
                    model.rnn_2.dense.bias,
                ]

optimizer = torch.optim.Adam([
                              {'params': base_params, 'lr': learning_rate},
                              {'params': model.dense_1.tau_m, 'lr': learning_rate*2},
                              {'params': model.rnn_1.tau_m, 'lr': learning_rate*2},  
                              {'params': model.rnn_1.tau_n, 'lr': learning_rate*2},
                                {'params': model.rnn_2.tau_m, 'lr': learning_rate * 2},
                                {'params': model.rnn_2.tau_n, 'lr': learning_rate * 2},
                              ],
                        lr=learning_rate)

model.to(device)
named_params = get_stats_named_params(model)

# scheduler = StepLR(optimizer, step_size=100, gamma=.1) # 20
scheduler = None
epochs = 200

if len(args.load) > 0:
    model = torch.load(args.load)
    valid_acc, val_auroc, val_recall, val_precision = test(model)
else:
    acc_list = train(epochs, criterion, optimizer, scheduler, output_file_fol)


