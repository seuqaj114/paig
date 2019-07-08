import os
import logging
import inspect
import tensorflow as tf
from nn.network import physics_models
from nn.utils.misc import classes_in_module
from nn.datasets.iterators import get_iterators
import runners.run_base

tf.app.flags.DEFINE_string("task", "", "Type of task.")
tf.app.flags.DEFINE_string("model", "PhysicsNet", "Model to use.")
tf.app.flags.DEFINE_integer("recurrent_units", 100, "Number of units for each lstm, if using black-box dynamics.")
tf.app.flags.DEFINE_integer("lstm_layers", 1, "Number of lstm cells to use, if using black-box dynamics")
tf.app.flags.DEFINE_string("cell_type", "", "Type of pendulum to use.")
tf.app.flags.DEFINE_string("encoder_type", "conv_encoder", "Type of encoder to use.")
tf.app.flags.DEFINE_string("decoder_type", "conv_st_decoder", "Type of decoder to use.")

tf.app.flags.DEFINE_float("autoencoder_loss", 0.0, "Autoencoder loss weighing.")
tf.app.flags.DEFINE_bool("alt_vel", False, "Whether to use linear velocity computation.")
tf.app.flags.DEFINE_bool("color", False, "Whether images are rbg or grayscale.")
tf.app.flags.DEFINE_integer("datapoints", 0, "How many datapoints from the dataset to use. \
                                              Useful for measuring data efficiency. default=0 uses all data.")

FLAGS = tf.app.flags.FLAGS

model_classes = classes_in_module(physics_models)
Model = model_classes[FLAGS.model]

data_file, test_data_file, cell_type, seq_len, test_seq_len, input_steps, pred_steps, input_size = {
    "bouncing_balls": (
        "balls/color_bounce_vx8_vy8_sl12_r2.npz", 
        "balls/color_bounce_vx8_vy8_sl30_r2.npz", 
        "bouncing_ode_cell",
        12, 30, 4, 6, 32*32),
    "spring_color": (
        "balls/color_spring_vx8_vy8_sl12_r2_k4_e6.npz", 
        "balls/color_spring_vx8_vy8_sl30_r2_k4_e6.npz",
        "spring_ode_cell",
        12, 30, 4, 6, 32*32),
    "spring_color_half": (
        "balls/color_spring_vx4_vy4_sl12_r2_k4_e6_halfpane.npz", 
        "balls/color_spring_vx4_vy4_sl30_r2_k4_e6_halfpane.npz", 
        "spring_ode_cell",
        12, 30, 4, 6, 32*32),
    "3bp_color": (
        "balls/color_3bp_vx2_vy2_sl20_r2_g60_m1_dt05.npz", 
        "balls/color_3bp_vx2_vy2_sl40_r2_g60_m1_dt05.npz", 
        "gravity_ode_cell",
        20, 40, 4, 12, 36*36),
    "mnist_spring_color": (
        "balls/color_mnist_spring_vx8_vy8_sl12_r2_k2_e12.npz", 
        "balls/color_mnist_spring_vx8_vy8_sl30_r2_k2_e12.npz", 
        "spring_ode_cell",
        12, 30, 3, 7, 64*64)
}[FLAGS.task]

if __name__ == "__main__":
    if not FLAGS.test_mode:
        network = Model(FLAGS.task, FLAGS.recurrent_units, FLAGS.lstm_layers, cell_type, 
                        seq_len, input_steps, pred_steps,
                       FLAGS.autoencoder_loss, FLAGS.alt_vel, FLAGS.color, 
                       input_size, FLAGS.encoder_type, FLAGS.decoder_type)

        network.build_graph()
        network.build_optimizer(FLAGS.base_lr, FLAGS.optimizer, FLAGS.anneal_lr)
        network.initialize_graph(FLAGS.save_dir, FLAGS.use_ckpt, FLAGS.ckpt_dir)

        data_iterators = get_iterators(
                              os.path.join(
                                  os.path.dirname(os.path.realpath(__file__)), 
                                  "../data/datasets/%s"%data_file), conv=FLAGS.conv, datapoints=FLAGS.datapoints)
        network.get_data(data_iterators)
        network.train(FLAGS.epochs, FLAGS.batch_size, FLAGS.save_every_n_epochs, FLAGS.eval_every_n_epochs,
                    FLAGS.print_interval, FLAGS.debug)
        
        tf.reset_default_graph()
    
    network = Model(FLAGS.task, FLAGS.recurrent_units, FLAGS.lstm_layers, cell_type, 
                    test_seq_len, input_steps, pred_steps,
                   FLAGS.autoencoder_loss, FLAGS.alt_vel, FLAGS.color, 
                   input_size, FLAGS.encoder_type, FLAGS.decoder_type)

    network.build_graph()
    network.build_optimizer(FLAGS.base_lr, FLAGS.optimizer, FLAGS.anneal_lr)
    network.initialize_graph(FLAGS.save_dir, True, FLAGS.ckpt_dir)

    data_iterators = get_iterators(
                          os.path.join(
                              os.path.dirname(os.path.realpath(__file__)), 
                              "../data/datasets/%s"%test_data_file), conv=FLAGS.conv, datapoints=FLAGS.datapoints)
    network.get_data(data_iterators)
    network.train(0, FLAGS.batch_size, FLAGS.save_every_n_epochs, FLAGS.eval_every_n_epochs,
                FLAGS.print_interval, FLAGS.debug)
