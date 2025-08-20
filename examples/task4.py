import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from iltn import utils
from iltn.events import TrapzEvent
import utils as example_utils

# Setting
task_name = "task4"
trace_length = 15.
x = torch.nn.Parameter(torch.tensor(2.0))
A = TrapzEvent(r"$A$", [6.,6.,9.,9.], trainable=False, beta=1/trace_length)
trainable_variables = [x]
trapz_list = [A]

def plot_function(name):
    fig, ax = plt.subplots(1,1,figsize=utils.plot.set_size(200., subplots=(1, 1)))
    ax.set_yticks([0.,1.])
    utils.plot.plot_events(np.arange(0.,trace_length,0.1), [A], ax)
    ax.plot([x.detach().numpy(),x.detach().numpy()],[0.,1.],label=r"$x$")
    ax.legend()
    plt.savefig(f"figs/{name}.pdf")

plot_function(f"{task_name}_init")

# Constraints
ltn_ops = example_utils.get_default_ltn_operators()
trapz_ops = example_utils.get_default_trapz_operators()
trapz_rel = example_utils.get_default_trapz_relations(trapz_ops, ltn_ops, beta=1/trace_length)
def constraints(training:bool = True):
    cstr1 = trapz_ops.end(A).mf_map_fn(x)
    return ltn_ops.Forall(torch.stack([cstr1]))
constraints()


# Training
optimizer = optim.Adam(trainable_variables, lr=0.1)
for epoch in range(200):
    optimizer.zero_grad()
    for trapz in trapz_list:
        if trapz._trainable:
            trapz.start_optimized_step()
    
    loss = -constraints(training=True)
    loss.backward()
    
    for trapz in trapz_list:
        if trapz._trainable:
            trapz.end_optimized_step()
    
    optimizer.step()
    
    if epoch%100 == 0:
        print("Epoch %d: Sat Level %.3f"
            %(epoch, torch.exp(constraints()).item()))
print("Training finished at Epoch %d with Sat Level %.3f "
    %(epoch, torch.exp(constraints()).item()))

plot_function(f"{task_name}_res")
