import os
import shutil
import logging
import numpy as np
import tensorflow as tf
from pprint import pprint
import inspect

from nn.network.base import BaseNet, OPTIMIZERS
from nn.network.cells import bouncing_ode_cell, spring_ode_cell, gravity_ode_cell
from nn.network.stn import stn
from nn.network.blocks import unet, shallow_unet, variable_from_network
from nn.utils.misc import log_metrics
from nn.utils.viz import gallery, gif
from nn.utils.math import sigmoid
import matplotlib.pyplot as plt
import matplotlib.cm as cm
plt.switch_backend('agg')

logger = logging.getLogger("tf")

CELLS = {
    "bouncing_ode_cell": bouncing_ode_cell,
    "spring_ode_cell": spring_ode_cell,
    "gravity_ode_cell": gravity_ode_cell,
    "lstm": tf.nn.rnn_cell.LSTMCell
}

# total number of latent units for each datasets
# coord_units = num_objects*num_dimensions*2
COORD_UNITS = {
    "bouncing_balls": 8,
    "spring_color": 8,
    "spring_color_half": 8,
    "3bp_color": 12,
    "mnist_spring_color": 8
}

class PhysicsNet(BaseNet):
    def __init__(self,
                 task="",
                 recurrent_units=128,
                 lstm_layers=1,
                 cell_type="",
                 seq_len=20,
                 input_steps=3,
                 pred_steps=5,
                 autoencoder_loss=0.0,
                 alt_vel=False,
                 color=False,
                 input_size=36*36,
                 encoder_type="conv_encoder",
                 decoder_type="conv_st_decoder"):

        super(PhysicsNet, self).__init__()

        assert task in COORD_UNITS
        self.task = task

        # Only used when using black-box dynamics (baselines)
        self.recurrent_units = recurrent_units
        self.lstm_layers = lstm_layers

        self.cell_type = cell_type
        self.cell = CELLS[self.cell_type]
        self.color = color
        self.conv_ch = 3 if color else 1
        self.input_size = input_size

        self.conv_input_shape = [int(np.sqrt(input_size))]*2+[self.conv_ch]
        self.input_shape = [int(np.sqrt(input_size))]*2+[self.conv_ch] # same as conv_input_shape, just here for backward compatibility

        self.encoder = {name: method for name, method in \
            inspect.getmembers(self, predicate=inspect.ismethod) if "encoder" in name
        }[encoder_type] 
        self.decoder = {name: method for name, method in \
            inspect.getmembers(self, predicate=inspect.ismethod) if "decoder" in name
        }[decoder_type]  

        self.output_shape = self.input_shape

        assert seq_len > input_steps + pred_steps
        assert input_steps >= 1
        assert pred_steps >= 1
        self.seq_len = seq_len
        self.input_steps = input_steps
        self.pred_steps = pred_steps
        self.extrap_steps = self.seq_len-self.input_steps-self.pred_steps

        self.alt_vel = alt_vel
        self.autoencoder_loss = autoencoder_loss

        self.coord_units = COORD_UNITS[self.task]
        self.n_objs = self.coord_units//4

        self.extra_valid_fns.append((self.visualize_sequence,[],{}))
        self.extra_test_fns.append((self.visualize_sequence,[],{}))

    def get_batch(self, batch_size, iterator):
        batch_x, _ = iterator.next_batch(batch_size)
        batch_len = batch_x.shape[1]
        feed_dict = {self.input: batch_x}
        return feed_dict, (batch_x, None)

    def compute_loss(self):

        # Compute reconstruction loss
        recons_target = self.input[:,:self.input_steps+self.pred_steps]
        recons_loss = tf.square(recons_target-self.recons_out)
        #recons_ce_loss = -(recons_target*tf.log(self.recons_out+1e-7) + (1.0-recons_target)*tf.log(1.0-self.recons_out+1e-7))
        recons_loss = tf.reduce_sum(recons_loss, axis=[2,3,4])

        self.recons_loss = tf.reduce_mean(recons_loss)

        target = self.input[:,self.input_steps:]
        #ce_loss = -(target*tf.log(self.output+1e-7) + (1.0-target)*tf.log(1.0-self.output+1e-7))
        loss = tf.square(target-self.output)
        loss = tf.reduce_sum(loss, axis=[2,3,4])

        # Compute prediction losses. pred_loss is used for training, extrap_loss is used for evaluation
        self.pred_loss = tf.reduce_mean(loss[:,:self.pred_steps])
        self.extrap_loss = tf.reduce_mean(loss[:,self.pred_steps:])

        train_loss = self.pred_loss
        if self.autoencoder_loss > 0.0:
            train_loss += self.autoencoder_loss*self.recons_loss

        eval_losses = [self.pred_loss, self.extrap_loss, self.recons_loss]
        return train_loss, eval_losses

    def build_graph(self):
        self.input = tf.placeholder(tf.float32, shape=[None, self.seq_len]+self.input_shape)
        self.output = self.conv_feedforward()

        self.train_loss, self.eval_losses = self.compute_loss()
        self.train_metrics["train_loss"] = self.train_loss
        self.eval_metrics["eval_pred_loss"] = self.eval_losses[0]
        self.eval_metrics["eval_extrap_loss"] = self.eval_losses[1]
        self.eval_metrics["eval_recons_loss"] = self.eval_losses[2]
        self.loss = self.train_loss

    def build_optimizer(self, base_lr, optimizer="rmsprop", anneal_lr=True):
        # Uncomment lines below to have different learning rates for physics and vision components

        self.base_lr = base_lr
        self.anneal_lr = anneal_lr
        self.lr = tf.Variable(base_lr, trainable=False, name="base_lr")
        self.optimizer = OPTIMIZERS[optimizer](self.lr)
        #self.dyn_optimizer = OPTIMIZERS[optimizer](1e-3)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            gvs = self.optimizer.compute_gradients(self.loss, var_list=tf.trainable_variables())
            gvs = [(tf.clip_by_value(grad, -1.0, 1.0), var) for grad, var in gvs if grad is not None]
            self.train_op = self.optimizer.apply_gradients(gvs)

            # self.train_op = self.optimizer.apply_gradients([gv for gv in gvs if "cell" not in gv[1].name])
            # if len([gv for gv in gvs if "cell" in gv[1].name]) > 0:
            #     self.dyn_train_op = self.dyn_optimizer.apply_gradients([gv for gv in gvs if "cell" in gv[1].name])
            #     self.train_op = tf.group(self.train_op, self.dyn_train_op)

    def conv_encoder(self, inp, scope=None, reuse=tf.AUTO_REUSE):
        with tf.variable_scope(scope or tf.get_variable_scope(), reuse=reuse):
            with tf.variable_scope("encoder"):
                rang = tf.range(self.conv_input_shape[0], dtype=tf.float32)
                grid_x, grid_y = tf.meshgrid(rang, rang)
                grid = tf.concat([grid_x[:,:,None], grid_y[:,:,None]], axis=2)
                grid = tf.tile(grid[None,:,:,:], [tf.shape(inp)[0], 1, 1, 1])

                if self.input_shape[0] < 40:
                    h = inp
                    h = shallow_unet(h, 8, self.n_objs, upsamp=True)

                    h = tf.concat([h, tf.ones_like(h[:,:,:,:1])], axis=-1)
                    h = tf.nn.softmax(h, axis=-1)
                    self.enc_masks = h
                    self.masked_objs = [self.enc_masks[:,:,:,i:i+1]*inp for i in range(self.n_objs)]

                    h = tf.concat(self.masked_objs, axis=0)
                    h = tf.reshape(h, [tf.shape(h)[0], self.input_shape[0]*self.input_shape[0]*self.conv_ch])
                    h = tf.layers.dense(h, 200, activation=tf.nn.relu)
                    h = tf.layers.dense(h, 200, activation=tf.nn.relu)
                    h = tf.layers.dense(h, 2, activation=None)
                    h = tf.concat(tf.split(h, self.n_objs, 0), axis=1)
                    h = tf.tanh(h)*(self.conv_input_shape[0]/2)+(self.conv_input_shape[0]/2)
                else:
                    h = inp
                    h = unet(h, 16, self.n_objs, upsamp=True)

                    h = tf.concat([h, tf.ones_like(h[:,:,:,:1])], axis=-1)
                    h = tf.nn.softmax(h, axis=-1)
                    self.enc_masks = h
                    self.masked_objs = [self.enc_masks[:,:,:,i:i+1]*inp for i in range(self.n_objs)]
                    h = tf.concat(self.masked_objs, axis=0)
                    h = tf.layers.average_pooling2d(h, 2, 2)
                    #h = tf.reduce_mean(h, axis=-1)

                    h = tf.layers.flatten(h)
                    h = tf.layers.dense(h, 200, activation=tf.nn.relu)
                    h = tf.layers.dense(h, 200, activation=tf.nn.relu)
                    h = tf.layers.dense(h, 2, activation=None)
                    h = tf.concat(tf.split(h, self.n_objs, 0), axis=1)
                    h = tf.tanh(h)*(self.conv_input_shape[0]/2)+(self.conv_input_shape[0]/2)
                return h

    def vel_encoder(self, inp, scope=None, reuse=tf.AUTO_REUSE):
        with tf.variable_scope(scope or tf.get_variable_scope(), reuse=reuse):
            with tf.variable_scope("init_vel"):
                if self.alt_vel:
                    # Computes velocity as a linear combination of the differences
                    # between previous time-steps
                    h = tf.split(inp, self.input_steps, 1)
                    h = [h[i+1]-h[i] for i in range(self.input_steps-1)]
                    h = tf.concat(h, axis=1)
                    h = tf.split(h, self.n_objs, 2)
                    h = tf.concat(h, axis=0)
                    h = tf.reshape(h, [tf.shape(h)[0], (self.input_steps-1)*2])
                    h = tf.layers.dense(h, 2, activation=None)
                    h = tf.split(h, self.n_objs, 0)
                    h = tf.concat(h, axis=1)
                else:
                    # Computes velocity using an MLP with positions as input
                    h = tf.split(inp, self.n_objs, 2)
                    h = tf.concat(h, axis=0)
                    h = tf.reshape(h, [tf.shape(h)[0], self.input_steps*self.coord_units//self.n_objs//2])
                    h = tf.layers.dense(h, 100, activation=tf.tanh)
                    h = tf.layers.dense(h, 100, activation=tf.tanh)
                    h = tf.layers.dense(h, self.coord_units//self.n_objs//2, activation=None)
                    h = tf.split(h, self.n_objs, 0)
                    h = tf.concat(h, axis=1)
        return h

    def conv_st_decoder(self, inp, scope=None, reuse=tf.AUTO_REUSE):
        with tf.variable_scope(scope or tf.get_variable_scope(), reuse=reuse):
            with tf.variable_scope("decoder"):

                batch_size = tf.shape(inp)[0]
                tmpl_size = self.conv_input_shape[0]//2

                # This parameter can be played with.
                # Setting it to log(2.0) makes the attention window half the size, which might make
                # it easier for the model to discover objects in some cases.
                # I haven't found this to make a consistent difference though. 
                logsigma = tf.get_variable("logsigma", shape=[], initializer=tf.constant_initializer(np.log(1.0)), trainable=True)
                sigma = tf.exp(logsigma)

                template = variable_from_network([self.n_objs, tmpl_size, tmpl_size, 1])
                self.template = template
                template = tf.tile(template, [1,1,1,3])+5

                contents = variable_from_network([self.n_objs, tmpl_size, tmpl_size, self.conv_ch])
                self.contents = contents 
                contents = tf.nn.sigmoid(contents)
                joint = tf.concat([template, contents], axis=-1)

                c2t = tf.convert_to_tensor
                out_temp_cont = []
                for loc, join in zip(tf.split(inp, self.n_objs, -1), tf.split(joint, self.n_objs, 0)):
                    theta0 = tf.tile(c2t([sigma]), [tf.shape(inp)[0]])
                    theta1 = tf.tile(c2t([0.0]), [tf.shape(inp)[0]])
                    theta2 = (self.conv_input_shape[0]/2-loc[:,0])/tmpl_size*sigma
                    theta3 = tf.tile(c2t([0.0]), [tf.shape(inp)[0]])
                    theta4 = tf.tile(c2t([sigma]), [tf.shape(inp)[0]])
                    theta5 = (self.conv_input_shape[0]/2-loc[:,1])/tmpl_size*sigma
                    theta = tf.stack([theta0, theta1, theta2, theta3, theta4, theta5], axis=1)

                    out_join = stn(tf.tile(join, [tf.shape(inp)[0], 1, 1, 1]), theta, self.conv_input_shape[:2])
                    out_temp_cont.append(tf.split(out_join, 2, -1))

                background_content = variable_from_network([1]+self.input_shape)
                self.background_content = tf.nn.sigmoid(background_content)
                background_content = tf.tile(self.background_content, [batch_size, 1, 1, 1])
                contents = [p[1] for p in out_temp_cont]
                contents.append(background_content)
                self.transf_contents = contents

                background_mask = tf.ones_like(out_temp_cont[0][0])
                masks = tf.stack([p[0]-5 for p in out_temp_cont]+[background_mask], axis=-1)
                masks = tf.nn.softmax(masks, axis=-1)
                masks = tf.unstack(masks, axis=-1)
                self.transf_masks = masks

                out = tf.add_n([m*c for m, c in zip(masks, contents)])

        return out

    def conv_feedforward(self):
        with tf.variable_scope("net") as tvs:
            lstms = [tf.nn.rnn_cell.LSTMCell(self.recurrent_units) for i in range(self.lstm_layers)]
            states = [lstm.zero_state(tf.shape(self.input)[0], dtype=tf.float32) for lstm in lstms]
            rollout_cell = self.cell(self.coord_units//2)

            # Encode all the input and train frames
            h = tf.reshape(self.input[:,:self.input_steps+self.pred_steps], [-1]+self.input_shape)
            enc_pos = self.encoder(h, scope=tvs)

            # decode the input and pred frames
            recons_out = self.decoder(enc_pos, scope=tvs)

            self.recons_out = tf.reshape(recons_out, 
                                         [tf.shape(self.input)[0], self.input_steps+self.pred_steps]+self.input_shape)
            self.enc_pos = tf.reshape(enc_pos, 
                                      [tf.shape(self.input)[0], self.input_steps+self.pred_steps, self.coord_units//2])

            if self.input_steps > 1:
                vel = self.vel_encoder(self.enc_pos[:,:self.input_steps], scope=tvs)
            else:
                vel = tf.zeros([tf.shape(self.input)[0], self.coord_units//2])

            pos = self.enc_pos[:,self.input_steps-1]
            output_seq = []
            pos_vel_seq = []
            pos_vel_seq.append(tf.concat([pos, vel], axis=1))

            # rollout ODE and decoder
            for t in range(self.pred_steps+self.extrap_steps):
                # rollout
                pos, vel = rollout_cell(pos, vel)

                # decode
                out = self.decoder(pos, scope=tvs)

                pos_vel_seq.append(tf.concat([pos, vel], axis=1))
                output_seq.append(out)

            current_scope = tf.get_default_graph().get_name_scope()
            self.network_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 
                                                  scope=current_scope)
            logger.info(self.network_vars)

        output_seq = tf.stack(output_seq)
        pos_vel_seq = tf.stack(pos_vel_seq)
        output_seq = tf.transpose(output_seq, (1,0,2,3,4))
        self.pos_vel_seq = tf.transpose(pos_vel_seq, (1,0,2))
        return output_seq

    def visualize_sequence(self):
        batch_size = 5

        feed_dict, (batch_x, _) = self.get_batch(batch_size, self.test_iterator)
        fetches = [self.output, self.recons_out]
        if hasattr(self, 'pos_vel_seq'):
            fetches.append(self.pos_vel_seq)

        res = self.sess.run(fetches, feed_dict=feed_dict)
        output_seq = res[0]
        recons_seq = res[1]
        if hasattr(self, 'pos_vel_seq'):
            pos_vel_seq = res[2]
        output_seq = np.concatenate([batch_x[:,:self.input_steps], output_seq], axis=1)
        recons_seq = np.concatenate([recons_seq, np.zeros((batch_size, self.extrap_steps)+recons_seq.shape[2:])], axis=1)

        # Plot a grid with prediction sequences
        for i in range(batch_x.shape[0]):
            #if hasattr(self, 'pos_vel_seq'):
            #    if i == 0 or i == 1:
            #        logger.info(pos_vel_seq[i])

            to_concat = [output_seq[i],batch_x[i],recons_seq[i]]
            total_seq = np.concatenate(to_concat, axis=0) 

            total_seq = total_seq.reshape([total_seq.shape[0], 
                                           self.input_shape[0], 
                                           self.input_shape[1], self.conv_ch])

            result = gallery(total_seq, ncols=batch_x.shape[1])

            norm = plt.Normalize(0.0, 1.0)

            figsize = (result.shape[1]//self.input_shape[1], result.shape[0]//self.input_shape[0])
            fig, ax = plt.subplots(figsize=figsize)
            ax.imshow(np.squeeze(result), interpolation='nearest', cmap=cm.Greys_r, norm=norm)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            fig.tight_layout()
            fig.savefig(os.path.join(self.save_dir, "example%d.jpg"%i))

        # Make a gif from the sequences
        bordered_output_seq = 0.5*np.ones([batch_size, self.seq_len, 
                                          self.conv_input_shape[0]+2, self.conv_input_shape[1]+2, 3])
        bordered_batch_x = 0.5*np.ones([batch_size, self.seq_len, 
                                          self.conv_input_shape[0]+2, self.conv_input_shape[1]+2, 3])
        output_seq = output_seq.reshape([batch_size, self.seq_len]+self.input_shape)
        batch_x = batch_x.reshape([batch_size, self.seq_len]+self.input_shape)
        bordered_output_seq[:,:,1:-1,1:-1] = output_seq
        bordered_batch_x[:,:,1:-1,1:-1] = batch_x
        output_seq = bordered_output_seq
        batch_x = bordered_batch_x
        output_seq = np.concatenate(np.split(output_seq, batch_size, 0), axis=-2).squeeze()
        batch_x = np.concatenate(np.split(batch_x, batch_size, 0), axis=-2).squeeze()
        frames = np.concatenate([output_seq, batch_x], axis=1)

        gif(os.path.join(self.save_dir, "animation%d.gif"%i), 
            frames*255, fps=7, scale=3)

        # Save extra tensors for visualization
        fetches = {"contents": self.contents,
                   "templates": self.template,
                   "background_content": self.background_content,
                   "transf_contents": self.transf_contents,
                   "transf_masks": self.transf_masks,
                   "enc_masks": self.enc_masks,
                   "masked_objs": self.masked_objs}
        results = self.sess.run(fetches, feed_dict=feed_dict)
        np.savez_compressed(os.path.join(self.save_dir, "extra_outputs.npz"), **results)
        contents = results["contents"]
        templates = results["templates"]
        contents = 1/(1+np.exp(-contents))
        templates = 1/(1+np.exp(-(templates-5)))
        if self.conv_ch == 1:
            contents = np.tile(contents, [1,1,1,3])
        templates = np.tile(templates, [1,1,1,3])
        total_seq = np.concatenate([contents, templates], axis=0)
        result = gallery(total_seq, ncols=self.n_objs)
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(np.squeeze(result), interpolation='nearest', cmap=cm.Greys_r, norm=norm)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        fig.tight_layout()
        fig.savefig(os.path.join(self.save_dir, "templates.jpg"))

        logger.info([(v.name, self.sess.run(v)) for v in tf.trainable_variables() if "ode_cell" in v.name or "sigma" in v.name])

