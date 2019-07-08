import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from itertools import combinations

from nn.utils.viz import gallery
from nn.utils.misc import rgb2gray

def generate_bouncing_ball_dataset(dest,
                                   train_set_size,
                                   valid_set_size,
                                   test_set_size,
                                   seq_len,
                                   box_size):
    np.random.seed(0)

    def verify_collision(x, v):
        if x[0] + v[0] > box_size or x[0] + v[0] < 0.0:
            v[0] = -v[0]
        if x[1] + v[1] > box_size or x[1] + v[1] < 0.0:
            v[1] = -v[1]
        return v

    def generate_trajectory(steps):
        traj = []
        x = np.random.rand(2)*box_size
        speed = np.random.rand()+1
        angle = np.random.rand()*2*np.pi
        v = np.array([speed*np.cos(angle), speed*np.sin(angle)])
        for _ in range(steps):
            traj.append(x)
            v = verify_collision(x, v)
            x = x + v
        return traj

    trajectories = []
    for i in range(train_set_size+valid_set_size+test_set_size):
        trajectories.append(generate_trajectory(seq_len))
    trajectories = np.array(trajectories)

    np.savez_compressed(dest, 
                        train_x=trajectories[:train_set_size],
                        valid_x=trajectories[train_set_size:train_set_size+valid_set_size],
                        test_x=trajectories[train_set_size+valid_set_size:])
    print("Saved to file %s" % dest)


def compute_wall_collision(pos, vel, radius, img_size):
    if pos[1]-radius <= 0:
        vel[1] = -vel[1]
        pos[1] = -(pos[1]-radius)+radius
    if pos[1]+radius >= img_size[1]:
        vel[1] = -vel[1]
        pos[1] = img_size[1]-(pos[1]+radius-img_size[1])-radius  
    if pos[0]-radius <= 0:
        vel[0] = -vel[0]
        pos[0] = -(pos[0]-radius)+radius
    if pos[0]+radius >= img_size[0]:
        vel[0] = -vel[0]
        pos[0] = img_size[0]-(pos[0]+radius-img_size[0])-radius 
    return pos, vel


def verify_wall_collision(pos, vel, radius, img_size):
    if pos[1]-radius <= 0:
        return True
    if pos[1]+radius >= img_size[1]:
        return True 
    if pos[0]-radius <= 0:
        return True
    if pos[0]+radius >= img_size[0]:
        return True
    return False


def verify_object_collision(poss, radius):
    for pos1, pos2 in combinations(poss, 2):
        if np.linalg.norm(pos1-pos2) <= radius:
            return True
    return False


def generate_falling_ball_dataset(dest,
                                  train_set_size,
                                  valid_set_size,
                                  test_set_size,
                                  seq_len,
                                  img_size=None,
                                  radius=3,
                                  dt=0.15,
                                  g=9.8,
                                  ode_steps=10):

    from skimage.draw import circle
    from nn.utils.viz import gallery
    import matplotlib.cm as cm
    if img_size is None:
        img_size = [32,32]

    def generate_sequence():
        seq = []
        # sample initial position, with v=0
        pos = np.random.rand(2)
        pos[0] = radius+(img_size[0]-2*radius)*pos[0]
        pos[1] = radius + (img_size[1]-2*radius)/2*pos[1]
        vel = np.array([0.0,0.0])

        for i in range(seq_len):
            assert pos[1]+radius < img_size[1]

            frame = np.zeros(img_size+[1], dtype=np.int8)
            rr, cc = circle(int(pos[1]), int(pos[0]), radius)
            frame[rr, cc, 0] = 255

            seq.append(frame)

            # rollout physics
            for _ in range(ode_steps):
                vel[1] = vel[1] + dt/ode_steps*g
                pos[1] = pos[1] + dt/ode_steps*vel[1]    

        return seq
    
    sequences = []
    for i in range(train_set_size+valid_set_size+test_set_size):
        if i % 100 == 0:
            print(i)
        sequences.append(generate_sequence())
    sequences = np.array(sequences, dtype=np.uint8)

    np.savez_compressed(dest, 
                        train_x=sequences[:train_set_size],
                        valid_x=sequences[train_set_size:train_set_size+valid_set_size],
                        test_x=sequences[train_set_size+valid_set_size:])
    print("Saved to file %s" % dest)

    # Save 10 samples
    result = gallery(np.concatenate(sequences[:10]/255), ncols=sequences.shape[1])

    norm = plt.Normalize(0.0, 1.0)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(np.squeeze(result), interpolation='nearest', cmap=cm.Greys_r, norm=norm)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    fig.tight_layout()
    fig.savefig(dest.split(".")[0]+"_samples.jpg")


def generate_falling_bouncing_ball_dataset(dest,
                                  train_set_size,
                                  valid_set_size,
                                  test_set_size,
                                  seq_len,
                                  img_size=None,
                                  radius=3,
                                  dt=0.30,
                                  g=9.8,
                                  vx0_max=0.0,
                                  vy0_max=0.0,
                                  cifar_background=False,
                                  ode_steps=10):

    if cifar_background:
        import tensorflow as tf
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    from skimage.draw import circle
    from skimage.transform import resize

    if img_size is None:
        img_size = [32,32]
    scale = 10
    scaled_img_size = [img_size[0]*scale, img_size[1]*scale]

    def generate_sequence():
        seq = []
        # sample initial position, with v=0
        pos = np.random.rand(2)
        pos[0] = radius + (img_size[0]-2*radius)*pos[0]
        if g == 0.0:
            pos[1] = radius + (img_size[1]-2*radius)*pos[1]
        else:
            pos[1] = radius + (img_size[1]-2*radius)/2*pos[1]
        angle = np.random.rand()*2*np.pi
        vel = np.array([np.cos(angle)*vx0_max, 
                        np.sin(angle)*vy0_max])

        if cifar_background:
            cifar_img = x_train[np.random.randint(50000)]

        for i in range(seq_len):
            if cifar_background:
                frame = cifar_img
                frame = rgb2gray(frame)/255
                frame = resize(frame, scaled_img_size)
                frame = np.clip(frame-0.2, 0.0, 1.0) # darken image a bit
            else:
                frame = np.zeros(scaled_img_size, dtype=np.float32)

            rr, cc = circle(int(pos[1]*scale), int(pos[0]*scale), radius*scale, scaled_img_size)
            frame[rr, cc] = 1.0
            frame = resize(frame, img_size, anti_aliasing=True)
            frame = (frame[:,:,None]*255).astype(np.uint8)

            seq.append(frame)

            # rollout physics
            for _ in range(ode_steps):
                vel[1] = vel[1] + dt/ode_steps*g
                pos[1] = pos[1] + dt/ode_steps*vel[1]

                pos[0] = pos[0] + dt/ode_steps*vel[0]

                # verify wall collisions
                pos, vel = compute_wall_collision(pos, vel, radius, img_size)
        return seq
    
    sequences = []
    for i in range(train_set_size+valid_set_size+test_set_size):
        if i % 100 == 0:
            print(i)
        sequences.append(generate_sequence())
    sequences = np.array(sequences, dtype=np.uint8)

    np.savez_compressed(dest, 
                        train_x=sequences[:train_set_size],
                        valid_x=sequences[train_set_size:train_set_size+valid_set_size],
                        test_x=sequences[train_set_size+valid_set_size:])
    print("Saved to file %s" % dest)

    # Save 10 samples
    result = gallery(np.concatenate(sequences[:10]/255), ncols=sequences.shape[1])

    norm = plt.Normalize(0.0, 1.0)
    fig, ax = plt.subplots(figsize=(sequences.shape[1], 10))
    ax.imshow(np.squeeze(result), interpolation='nearest', cmap=cm.Greys_r, norm=norm)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    fig.tight_layout()
    fig.savefig(dest.split(".")[0]+"_samples.jpg")


def generate_spring_balls_dataset(dest,
                                  train_set_size,
                                  valid_set_size,
                                  test_set_size,
                                  seq_len,
                                  img_size=None,
                                  radius=3,
                                  dt=0.3,
                                  k=3,
                                  equil=5,
                                  vx0_max=0.0,
                                  vy0_max=0.0,
                                  color=False,
                                  cifar_background=False,
                                  ode_steps=10):

    if cifar_background:
        import tensorflow as tf
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    from skimage.draw import circle
    from skimage.transform import resize

    if img_size is None:
        img_size = [32,32]
    scale = 10
    scaled_img_size = [img_size[0]*scale, img_size[1]*scale]

    def generate_sequence():
        # sample initial position of the center of mass, then sample
        # position of each object relative to that.

        collision = True
        while collision == True:
            seq = []

            cm_pos = np.random.rand(2)
            cm_pos[0] = radius+equil + (img_size[0]-2*(radius+equil))*cm_pos[0]
            cm_pos[1] = radius+equil + (img_size[1]-2*(radius+equil))*cm_pos[1]

            angle = np.random.rand()*2*np.pi
            # calculate position of both objects
            r = np.random.rand()+0.5
            poss = [[np.cos(angle)*equil*r+cm_pos[0], np.sin(angle)*equil*r+cm_pos[1]],
                   [np.cos(angle+np.pi)*equil*r+cm_pos[0], np.sin(angle+np.pi)*equil*r+cm_pos[1]]]
            poss = np.array(poss)
            angles = np.random.rand(2)*2*np.pi
            vels = [[np.cos(angles[0])*vx0_max, np.sin(angles[0])*vy0_max],
                   [np.cos(angles[1])*vx0_max, np.sin(angles[1])*vy0_max]]
            vels = np.array(vels)

            if cifar_background:
                cifar_img = x_train[np.random.randint(50000)]

            for i in range(seq_len):
                if cifar_background:
                    frame = cifar_img
                    frame = rgb2gray(frame)/255
                    frame = resize(frame, scaled_img_size)
                    frame = np.clip(frame-0.2, 0.0, 1.0) # darken image a bit
                else:
                    if color:
                        frame = np.zeros(scaled_img_size+[3], dtype=np.float32)
                    else:
                        frame = np.zeros(scaled_img_size+[1], dtype=np.float32)


                for j, pos in enumerate(poss):
                    rr, cc = circle(int(pos[1]*scale), int(pos[0]*scale), radius*scale, scaled_img_size)
                    if color:
                        frame[rr, cc, 2-j] = 1.0 
                    else:
                        frame[rr, cc, 0] = 1.0 

                frame = resize(frame, img_size, anti_aliasing=True)
                frame = (frame*255).astype(np.uint8)

                seq.append(frame)

                # rollout physics
                for _ in range(ode_steps):
                    norm = np.linalg.norm(poss[0]-poss[1])
                    direction = (poss[0]-poss[1])/norm
                    F = k*(norm-2*equil)*direction
                    vels[0] = vels[0] - dt/ode_steps*F
                    vels[1] = vels[1] + dt/ode_steps*F
                    poss = poss + dt/ode_steps*vels

                    collision = verify_wall_collision(poss[0], vels[0], radius, img_size) or \
                                verify_wall_collision(poss[1], vels[1], radius, img_size)
                    if collision:
                        break
                    #poss[0], vels[0] = compute_wall_collision(poss[0], vels[0], radius, img_size)
                    #poss[1], vels[1] = compute_wall_collision(poss[1], vels[1], radius, img_size)
                if collision:
                    break

        return seq
    
    sequences = []
    for i in range(train_set_size+valid_set_size+test_set_size):
        if i % 100 == 0:
            print(i)
        sequences.append(generate_sequence())
    sequences = np.array(sequences, dtype=np.uint8)

    np.savez_compressed(dest, 
                        train_x=sequences[:train_set_size],
                        valid_x=sequences[train_set_size:train_set_size+valid_set_size],
                        test_x=sequences[train_set_size+valid_set_size:])
    print("Saved to file %s" % dest)

    # Save 10 samples
    result = gallery(np.concatenate(sequences[:10]/255), ncols=sequences.shape[1])

    norm = plt.Normalize(0.0, 1.0)
    fig, ax = plt.subplots(figsize=(sequences.shape[1], 10))
    ax.imshow(np.squeeze(result), interpolation='nearest', cmap=cm.Greys_r, norm=norm)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    fig.tight_layout()
    fig.savefig(dest.split(".")[0]+"_samples.jpg")


def generate_spring_mnist_dataset(dest,
                                  train_set_size,
                                  valid_set_size,
                                  test_set_size,
                                  seq_len,
                                  img_size=None,
                                  radius=3,
                                  dt=0.3,
                                  k=3,
                                  equil=5,
                                  vx0_max=0.0,
                                  vy0_max=0.0,
                                  color=False,
                                  cifar_background=False,
                                  ode_steps=10):

    # A single CIFAR image is used for background
    # Only 2 mnist digits are used
    import tensorflow as tf
    from skimage.draw import circle
    from skimage.transform import resize

    scale = 5
    if img_size is None:
        img_size = [32,32]    
    scaled_img_size = [img_size[0]*scale, img_size[1]*scale]

    if cifar_background:
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        cifar_img = x_train[1]
        
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    digits = x_train[0:2, 3:-3, 3:-3]/255
    digits = [resize(d, [22*scale, 22*scale]) for d in digits]
    radius = 11

    def generate_sequence():
        # sample initial position of the center of mass, then sample
        # position of each object relative to that.

        collision = True
        while collision == True:
            seq = []

            cm_pos = np.random.rand(2)
            cm_pos[0] = radius+equil + (img_size[0]-2*(radius+equil))*cm_pos[0]
            cm_pos[1] = radius+equil + (img_size[1]-2*(radius+equil))*cm_pos[1]

            angle = np.random.rand()*2*np.pi
            # calculate position of both objects
            r = np.random.rand()+0.5
            poss = [[np.cos(angle)*equil*r+cm_pos[0], np.sin(angle)*equil*r+cm_pos[1]],
                   [np.cos(angle+np.pi)*equil*r+cm_pos[0], np.sin(angle+np.pi)*equil*r+cm_pos[1]]]
            poss = np.array(poss)
            angles = np.random.rand(2)*2*np.pi
            vels = [[np.cos(angles[0])*vx0_max, np.sin(angles[0])*vy0_max],
                   [np.cos(angles[1])*vx0_max, np.sin(angles[1])*vy0_max]]
            vels = np.array(vels)

            for i in range(seq_len):
                if cifar_background:
                    frame = cifar_img
                    if not color:
                        frame = rgb2gray(frame)
                        frame = frame[:,:,None]
                    frame = frame/255
                    frame = resize(frame, scaled_img_size)
                    frame = np.clip(frame-0.2, 0.0, 1.0) # darken image a bit
                else:
                    if color:
                        frame = np.zeros(scaled_img_size+[3], dtype=np.float32)
                    else:
                        frame = np.zeros(scaled_img_size+[1], dtype=np.float32)


                for j, pos in enumerate(poss):
                    rr, cc = circle(int(pos[1]*scale), int(pos[0]*scale), radius*scale, scaled_img_size)
                    frame_coords = np.array([[max(0, (pos[1]-radius)*scale), min(scaled_img_size[1], (pos[1]+radius)*scale)],
                                             [max(0, (pos[0]-radius)*scale), min(scaled_img_size[0], (pos[0]+radius)*scale)]])
                    digit_coords = np.array([[max(0, (radius-pos[1])*scale), min(2*radius*scale, scaled_img_size[1]-(pos[1]-radius)*scale)],
                                             [max(0, (radius-pos[0])*scale), min(2*radius*scale, scaled_img_size[0]-(pos[0]-radius)*scale)]])
                    frame_coords = np.round(frame_coords).astype(np.int32)
                    digit_coords = np.round(digit_coords).astype(np.int32)
                    
                    digit_slice = digits[j][digit_coords[0,0]:digit_coords[0,1], 
                                            digit_coords[1,0]:digit_coords[1,1]]
                    if color:
                        for l in range(3):
                            frame_slice = frame[frame_coords[0,0]:frame_coords[0,1], 
                                                frame_coords[1,0]:frame_coords[1,1], l]
                            c = 1.0 if l == j else 0.0
                            frame[frame_coords[0,0]:frame_coords[0,1], 
                                  frame_coords[1,0]:frame_coords[1,1], l] = digit_slice*c + (1-digit_slice)*frame_slice

                    else:
                        frame_slice = frame[frame_coords[0,0]:frame_coords[0,1], 
                                            frame_coords[1,0]:frame_coords[1,1], 0]
                        frame[frame_coords[0,0]:frame_coords[0,1], 
                              frame_coords[1,0]:frame_coords[1,1], 0] = digit_slice + (1-digit_slice)*frame_slice

                frame = resize(frame, img_size, anti_aliasing=True)
                frame = (frame*255).astype(np.uint8)

                seq.append(frame)

                # rollout physics
                for _ in range(ode_steps):
                    norm = np.linalg.norm(poss[0]-poss[1])
                    direction = (poss[0]-poss[1])/norm
                    F = k*(norm-2*equil)*direction
                    vels[0] = vels[0] - dt/ode_steps*F
                    vels[1] = vels[1] + dt/ode_steps*F
                    poss = poss + dt/ode_steps*vels

                    collision = verify_wall_collision(poss[0], vels[0], 2, img_size) or \
                                verify_wall_collision(poss[1], vels[1], 2, img_size)
                    if collision:
                        break
                    #poss[0], vels[0] = compute_wall_collision(poss[0], vels[0], radius, img_size)
                    #poss[1], vels[1] = compute_wall_collision(poss[1], vels[1], radius, img_size)
                if collision:
                    break

        return seq
    
    sequences = []
    for i in range(train_set_size+valid_set_size+test_set_size):
        if i % 100 == 0:
            print(i)
        sequences.append(generate_sequence())
    sequences = np.array(sequences, dtype=np.uint8)

    np.savez_compressed(dest, 
                        train_x=sequences[:train_set_size],
                        valid_x=sequences[train_set_size:train_set_size+valid_set_size],
                        test_x=sequences[train_set_size+valid_set_size:])
    print("Saved to file %s" % dest)

    # Save 10 samples
    result = gallery(np.concatenate(sequences[:10]/255), ncols=sequences.shape[1])

    norm = plt.Normalize(0.0, 1.0)
    fig, ax = plt.subplots(figsize=(sequences.shape[1], 10))
    ax.imshow(np.squeeze(result), interpolation='nearest', cmap=cm.Greys_r, norm=norm)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    fig.tight_layout()
    fig.savefig(dest.split(".")[0]+"_samples.jpg")


def generate_3_body_problem_dataset(dest,
                                  train_set_size,
                                  valid_set_size,
                                  test_set_size,
                                  seq_len,
                                  img_size=None,
                                  radius=3,
                                  dt=0.3,
                                  g=9.8,
                                  m=1.0,
                                  vx0_max=0.0,
                                  vy0_max=0.0,
                                  color=False,
                                  cifar_background=False,
                                  ode_steps=10):

    if cifar_background:
        import tensorflow as tf
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    from skimage.draw import circle
    from skimage.transform import resize

    if img_size is None:
        img_size = [32,32]
    scale = 10
    scaled_img_size = [img_size[0]*scale, img_size[1]*scale]

    def generate_sequence():
        # sample initial position of the center of mass, then sample
        # position of each object relative to that.

        collision = True
        while collision == True:
            seq = []

            cm_pos = np.random.rand(2)
            cm_pos = np.array(img_size)/2
            angle1 = np.random.rand()*2*np.pi
            angle2 = angle1 + 2*np.pi/3+(np.random.rand()-0.5)/2
            angle3 = angle1 + 4*np.pi/3+(np.random.rand()-0.5)/2

            angles = [angle1, angle2, angle3]
            # calculate position of both objects
            r = (np.random.rand()/2+0.75)*img_size[0]/4
            poss = [[np.cos(angle)*r+cm_pos[0], np.sin(angle)*r+cm_pos[1]] for angle in angles]
            poss = np.array(poss)
            
            #angles = np.random.rand(3)*2*np.pi
            #vels = [[np.cos(angle)*vx0_max, np.sin(angle)*vy0_max] for angle in angles]
            #vels = np.array(vels)
            r = np.random.randint(0,2)*2-1
            angles = [angle+r*np.pi/2 for angle in angles]
            noise = np.random.rand(2)-0.5
            vels = [[np.cos(angle)*vx0_max+noise[0], np.sin(angle)*vy0_max+noise[1]] for angle in angles]
            vels = np.array(vels)

            if cifar_background:
                cifar_img = x_train[np.random.randint(50000)]

            for i in range(seq_len):
                if cifar_background:
                    frame = cifar_img
                    frame = rgb2gray(frame)/255
                    frame = resize(frame, scaled_img_size)
                    frame = np.clip(frame-0.2, 0.0, 1.0) # darken image a bit
                else:
                    if color:
                        frame = np.zeros(scaled_img_size+[3], dtype=np.float32)
                    else:
                        frame = np.zeros(scaled_img_size+[1], dtype=np.float32)

                for j, pos in enumerate(poss):
                    rr, cc = circle(int(pos[1]*scale), int(pos[0]*scale), radius*scale, scaled_img_size)
                    if color:
                        frame[rr, cc, 2-j] = 1.0 
                    else:
                        frame[rr, cc, 0] = 1.0 

                frame = resize(frame, img_size, anti_aliasing=True)
                frame = (frame*255).astype(np.uint8)

                seq.append(frame)

                # rollout physics
                for _ in range(ode_steps):
                    norm01 = np.linalg.norm(poss[0]-poss[1])
                    norm12 = np.linalg.norm(poss[1]-poss[2])
                    norm20 = np.linalg.norm(poss[2]-poss[0])
                    vec01 = (poss[0]-poss[1])
                    vec12 = (poss[1]-poss[2])
                    vec20 = (poss[2]-poss[0])

                    # Compute force vectors
                    F = [vec01/norm01**3-vec20/norm20**3,
                         vec12/norm12**3-vec01/norm01**3,
                         vec20/norm20**3-vec12/norm12**3]
                    F = np.array(F)
                    F = -g*m*m*F

                    vels = vels + dt/ode_steps*F
                    poss = poss + dt/ode_steps*vels

                    collision = any([verify_wall_collision(pos, vel, radius, img_size) for pos, vel in zip(poss, vels)]) or \
                                verify_object_collision(poss, radius+1)
                    if collision:
                        break

                if collision:
                    break

        return seq
    
    sequences = []
    for i in range(train_set_size+valid_set_size+test_set_size):
        if i % 100 == 0:
            print(i)
        sequences.append(generate_sequence())
    sequences = np.array(sequences, dtype=np.uint8)

    np.savez_compressed(dest, 
                        train_x=sequences[:train_set_size],
                        valid_x=sequences[train_set_size:train_set_size+valid_set_size],
                        test_x=sequences[train_set_size+valid_set_size:])
    print("Saved to file %s" % dest)

    # Save 10 samples
    result = gallery(np.concatenate(sequences[:10]/255), ncols=sequences.shape[1])

    norm = plt.Normalize(0.0, 1.0)
    fig, ax = plt.subplots(figsize=(sequences.shape[1], 10))
    ax.imshow(np.squeeze(result), interpolation='nearest', cmap=cm.Greys_r, norm=norm)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    fig.tight_layout()
    fig.savefig(dest.split(".")[0]+"_samples.jpg")
