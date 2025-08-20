import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from iltn import utils
from iltn.events import TrapzEvent
import utils as example_utils

# Setting
task_name = "task3"
trace_length = 15.
A = TrapzEvent(r"$A$", [0.5,1.5,3,5], trainable=True, beta=1/trace_length)
B = TrapzEvent(r"$B$", [5.,6.,9.,10.], trainable=False, beta=1/trace_length)
trainable_variables = A.trainable_variables
trapz_list = [A,B]

def plot_function(name):
    fig, ax = plt.subplots(1,1,figsize=utils.plot.set_size(200., subplots=(1, 1)))
    ax.set_yticks([0.,1.])
    utils.plot.plot_events(np.arange(0.,trace_length,0.1), [A,B], ax)
    ax.legend()
    plt.savefig(f"figs/{name}.pdf")

plot_function(f"{task_name}_init")

# Constraints
ltn_ops = example_utils.get_default_ltn_operators()
trapz_ops = example_utils.get_default_trapz_operators()
trapz_rel = example_utils.get_default_trapz_relations(trapz_ops, ltn_ops, beta=1/trace_length)
def constraints(training:bool = True):
    cstr1 = trapz_rel.overlaps(A, B, smooth=training)
    cstr2 = torch.tensor(A.mf_map_fn(3., smooth=True), dtype=torch.float32)
    cstr3 = ltn_ops.Not(torch.tensor(A.mf_map_fn(2., smooth=True), dtype=torch.float32))
    return ltn_ops.Forall(torch.stack([cstr1, cstr2, cstr3]))
constraints()


# Training
optimizer = optim.Adam(trainable_variables, lr=0.1)
#optimizer = optim.SGD(trainable_variables, lr=0.01)
for epoch in range(5000):
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
