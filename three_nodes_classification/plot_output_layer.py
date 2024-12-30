import torch
from models import NonLinearSAGE
from safetensors.torch import load_file
import numpy as np
import matplotlib.pyplot as plt
import argparse

# read from script args
parser = argparse.ArgumentParser(description="Plot the output layer of a model")
parser.add_argument("--readout_name", type=str, default="mlp2-8-lockedw1w2", help="Name of the model")
parser.add_argument("--guid", type=str, default="9c231419-c272-4e1e-93f1-3548446d5073", help="GUID of the model")
args = parser.parse_args()

readout_name = args.readout_name
guid = args.guid

model_weights = load_file(f"./results/{readout_name}/{guid}.safetensors")

model = NonLinearSAGE()

model.load_state_dict(model_weights)
model.eval

x_values = np.linspace(-1.5, 1.5, 400)

y_values = model.activation(torch.tensor([x_values]).float().reshape(-1,1)).detach().numpy()

plt.plot(x_values, y_values, label=f"Output Layer: {readout_name}")

plt.title(f"{guid}")
plt.axhline(0, color='black',linewidth=0.5, ls='--')
plt.axvline(0, color='black',linewidth=0.5, ls='--')
plt.grid()
plt.legend()
plt.savefig(f"output_layer_{readout_name}.{guid}.svg")




