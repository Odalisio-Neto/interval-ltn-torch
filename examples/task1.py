import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from iltn import utils
from iltn.events import TrapzEvent

# Setting
task_name = "task1"
trace_length = 15.5
A = TrapzEvent(r"$A$", [0.5, 1.5, 3, 5], trainable=False, beta=1/trace_length)
B = TrapzEvent(r"$B$", [1., 2, 2.5, 4.5], trainable=True, beta=1/trace_length)
C = TrapzEvent(r"$C$", [11, 12, 14, 15], trainable=False, beta=1/trace_length)
trainable_variables = [p for p in B.parameters() if p.requires_grad]
trapz_list = [A, B, C]

def plot_function(name):
    fig, ax = plt.subplots(1, 1, figsize=utils.plot.set_size(200., subplots=(1, 1)))
    ax.set_yticks([0., 1.])
    utils.plot.plot_events(np.arange(0., trace_length, 0.1), [A, B, C], ax)
    ax.legend()
    plt.savefig(f"figs/{name}.pdf")

plot_function(f"{task_name}_init")

# Constraints
ltn_ops = utils.get_default_ltn_operators()
trapz_ops = utils.get_default_trapz_operators()
trapz_rel = utils.get_default_trapz_relations(trapz_ops, ltn_ops, beta=1/trace_length)

def constraints(training=True):
    cstr1 = trapz_rel.after(B, A, smooth=training)
    cstr2 = trapz_rel.before(B, C, smooth=training)
    cstr3 = utils.ops.smooth_equal(trapz_ops.duration(B), torch.tensor(3.0))
    return ltn_ops.Forall(torch.stack([cstr1, cstr2, cstr3]))

constraints()

# Training
optimizer = optim.Adam(trainable_variables, lr=0.1)
for epoch in range(50):
    optimizer.zero_grad()
    for trapz in trapz_list:
        trapz.start_optimized_step()

    loss = -constraints(training=True)
    loss.backward()

    for trapz in trapz_list:
        trapz.end_optimized_step()

    optimizer.step()

    if epoch % 10 == 0:
        print("Epoch %d: Sat Level %.3f" % (epoch, torch.exp(constraints()).item()))

print("Training finished at Epoch %d with Sat Level %.3f " % (epoch, torch.exp(constraints()).item()))

plot_function(f"{task_name}_res")
