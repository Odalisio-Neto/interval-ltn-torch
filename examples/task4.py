import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import iltn
import utils

iltn.utils.plot.set_tex_style()

# Setting
task_name = "task4"
trace_length = 15.
x = tf.Variable(2.)
A = iltn.events.TrapzEvent(r"$A$", [6.,6.,9.,9.], trainable=False, beta=1/trace_length)
trainable_variables = [x]
trapz_list = [A]

def plot_function(name):
    fig, ax = plt.subplots(1,1,figsize=iltn.utils.plot.set_size(200., subplots=(1, 1)))
    ax.set_yticks([0.,1.])
    iltn.utils.plot.plot_events(np.arange(0.,trace_length,0.1), [A], ax)
    ax.plot([x.numpy(),x.numpy()],[0.,1.],label=r"$x$")
    ax.legend()
    plt.savefig(f"figs/{name}.pdf")

plot_function(f"{task_name}_init")

# Constraints
ltn_ops = utils.get_default_ltn_operators()
trapz_ops = utils.get_default_trapz_operators()
trapz_rel = utils.get_default_trapz_relations(trapz_ops, ltn_ops, beta=1/trace_length)
def constraints(training:bool = True):
    cstr1 = trapz_ops.end(A).mf_map_fn(x)
    return ltn_ops.Forall(tf.stack([cstr1]))
constraints()


# Training
optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)
for epoch in range(200):
    with tf.GradientTape() as tape:
        for trapz in trapz_list:
            trapz.start_optimized_step(tape=tape)
        loss = - constraints(training=True)
        for trapz in trapz_list:
            trapz.end_optimized_step()
    grads = tape.gradient(loss, trainable_variables)
    optimizer.apply_gradients(zip(grads, trainable_variables))
    if epoch%100 == 0:
        print("Epoch %d: Sat Level %.3f"
            %(epoch, tf.math.exp(constraints())))
print("Training finished at Epoch %d with Sat Level %.3f "
    %(epoch, tf.math.exp(constraints())))

plot_function(f"{task_name}_res")
