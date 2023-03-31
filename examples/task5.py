import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import iltn
import utils

iltn.utils.plot.set_tex_style()

# Setting
task_name = "task5"
trace_length = 15.
A = iltn.events.TrapzEvent(r"$A$", [0.5,1.5,3,5], trainable=True, beta=1/trace_length)
B = iltn.events.TrapzEvent(r"$B$", [5.,6.,9.,10.], trainable=False, beta=1/trace_length)
trainable_variables = A.trainable_variables
trapz_list = [A,B]

def plot_function(name):
    fig, ax = plt.subplots(1,1,figsize=iltn.utils.plot.set_size(200., subplots=(1, 1)))
    ax.set_yticks([0.,1.])
    iltn.utils.plot.plot_events(np.arange(0.,trace_length,0.1), [A,B], ax)
    ax.legend()
    plt.savefig(f"figs/{name}.pdf")

plot_function(f"{task_name}_init")

# Constraints
ltn_ops = utils.get_default_ltn_operators()
trapz_ops = utils.get_default_trapz_operators()
trapz_rel = utils.get_default_trapz_relations(trapz_ops, ltn_ops, beta=1/trace_length)
def constraints(training:bool = True):
    cstr1 = trapz_rel.overlaps(A, B, smooth=training)
    # cstr2 = A.mf_map_fn(3., smooth=True)
    # cstr3 = ltn_ops.Not(A.mf_map_fn(2., smooth=True))
    return ltn_ops.Forall(tf.stack([cstr1]))
constraints()


# Training
optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)
#optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
for epoch in range(1000):
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
