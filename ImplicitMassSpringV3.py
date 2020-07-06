# V3: Hand in version

# 1: Jacobi and Gauss-Siedel:

# 2: Add operation and Sub operation between 'var' tensors with same shape:
# I am having some trouble with it.

# 3: 'norm' operation to 'var' tensor?

# 4: AssertionError: The 0-th index of a Matrix/Vector must be a compile-time constant integer, got <taichi.lang.expr.Expr object at 0x00000239F998D580>
# 为什么要有这个规定呢？感觉不太方便呀 （横向或者竖向遍历某张量中的最底层的数值元素）


# 5: IndexError: only integers, slices (`:`), ellipsis (`...`), numpy.newaxis (`None`) and integer or boolean arrays are valid indices
# @kernel内用变量指定numpy数组下标

# 6: A s s e r t i o n   f a i l e d :   g e t O p e r a n d ( 0 ) - > g e t T y p e ( )   = =   c a s t < P o i n t e r T y p e > ( g e t O p e r a n d ( 1 ) - > g e t T y p e ( ) ) - > g e t E l e m e n t T y p e ( )   & &   " P t r   m u s t   b e   a   p o i n t e r   t o   V a l   t y p e ! " ,   f i l e   e : \ r e p o s \ l l v m - 8 . 0 . 1 \ l i b \ i r \ i n s t r u c t i o n s . c p p ,   l i n e   1 2 1 0
# Init the parameters in this system:

# V2: Truncate some apparent useless code and optimize 'step' process.

import taichi as ti
import numpy as np

ti.init(arch=ti.gpu)
pixels = ti.var(ti.u8, shape=(512, 512, 3))
max_num_particles = 256
dt = 8e-3

num_particles = ti.var(ti.i32, shape=())
spring_stiffness = ti.var(ti.f32, shape=())
paused = ti.var(ti.i32, shape=())
damping = ti.var(ti.f32, shape=())

particle_mass = 1
bottom_y = 0.05

x = ti.Vector(2, dt=ti.f32, shape=max_num_particles)
v = ti.Vector(2, dt=ti.f32, shape=max_num_particles)
new_v = ti.var(dt=ti.f32, shape=2*max_num_particles)
diff_vec = ti.var(dt=ti.f32, shape=2*max_num_particles)

A = ti.Matrix(2, 2, dt=ti.f32, shape=(max_num_particles, max_num_particles))
b = ti.Vector(2, dt=ti.f32, shape=max_num_particles)

# rest_length[i, j] = 0 means i and j are not connected
rest_length = ti.var(ti.f32, shape=(max_num_particles, max_num_particles))
A_implicit = ti.Matrix(2, 2, dt=ti.f32, shape=(max_num_particles, max_num_particles))
b_implicit = ti.Vector(2, dt=ti.f32, shape=max_num_particles)

connection_radius = 0.15

gravity = [0, -9.8]


@ti.kernel
def createEqn():
    # Construct A_implicit Matrix:
    n = num_particles[None]
    mass_inv = 1.0 / particle_mass
    for tensor_ele_idx in range(n * n):
        # Construct original Jacobian matrix's elements (2x2):
        tensor_ele_idx_row = tensor_ele_idx // n
        tensor_ele_idx_col = tensor_ele_idx - tensor_ele_idx_row * n
        if tensor_ele_idx_row == tensor_ele_idx_col:
            # Populate diagonal elements in the tensor:
            A_implicit[tensor_ele_idx_row, tensor_ele_idx_col][0, 0] = 0
            A_implicit[tensor_ele_idx_row, tensor_ele_idx_col][0, 1] = 0
            A_implicit[tensor_ele_idx_row, tensor_ele_idx_col][1, 0] = 0
            A_implicit[tensor_ele_idx_row, tensor_ele_idx_col][1, 1] = 0
            for j in range(n):
                if rest_length[tensor_ele_idx_row, j] != 0:
                    xdt = x[tensor_ele_idx_row]
                    xjt = x[j]
                    dist = (xdt - xjt).norm()
                    ti_dist_pow_n3 = ti.pow(dist, -3)
                    A_implicit[tensor_ele_idx_row, tensor_ele_idx_col][0, 0] += -spring_stiffness[None] * (
                                rest_length[tensor_ele_idx_row, j] * ti_dist_pow_n3 * (xdt[0] - xjt[0]) * (
                                    xdt[0] - xjt[0]) - rest_length[tensor_ele_idx, j] * ti.pow(dist, -1) + 1)
                    A_implicit[tensor_ele_idx_row, tensor_ele_idx_col][0, 1] += -spring_stiffness[None] * rest_length[
                        tensor_ele_idx_row, j] * ti_dist_pow_n3 * (xdt[0] - xjt[0]) * (xdt[1] - xjt[1])
                    A_implicit[tensor_ele_idx_row, tensor_ele_idx_col][1, 0] += -spring_stiffness[None] * rest_length[
                        tensor_ele_idx_row, j] * ti_dist_pow_n3 * (xdt[0] - xjt[0]) * (xdt[1] - xjt[1])
                    A_implicit[tensor_ele_idx_row, tensor_ele_idx_col][1, 1] += -spring_stiffness[None] * (
                                rest_length[tensor_ele_idx_row, j] * ti_dist_pow_n3 * (xdt[1] - xjt[1]) * (
                                    xdt[1] - xjt[1]) - rest_length[tensor_ele_idx, j] * ti.pow(dist, -1) + 1)
            A_implicit[tensor_ele_idx_row, tensor_ele_idx_col][0, 0] = 1.0 - dt * dt * mass_inv * A_implicit[
                tensor_ele_idx_row, tensor_ele_idx_col][0, 0]
            A_implicit[tensor_ele_idx_row, tensor_ele_idx_col][0, 1] = - dt * dt * mass_inv * A_implicit[
                tensor_ele_idx_row, tensor_ele_idx_col][0, 1]
            A_implicit[tensor_ele_idx_row, tensor_ele_idx_col][1, 0] = - dt * dt * mass_inv * A_implicit[
                tensor_ele_idx_row, tensor_ele_idx_col][1, 0]
            A_implicit[tensor_ele_idx_row, tensor_ele_idx_col][1, 1] = 1.0 - dt * dt * mass_inv * A_implicit[
                tensor_ele_idx_row, tensor_ele_idx_col][1, 1]
        elif rest_length[tensor_ele_idx_row, tensor_ele_idx_col] != 0:
            # Populate elements that have a connection:
            xrt = x[tensor_ele_idx_row]
            xct = x[tensor_ele_idx_col]
            dist = (xrt - xct).norm()
            dist_pow_n3 = ti.pow(dist, -3)
            lrc = rest_length[tensor_ele_idx_row, tensor_ele_idx_col]
            A_implicit[tensor_ele_idx_row, tensor_ele_idx_col][0, 0] = -spring_stiffness[None] * (
                        -lrc * dist_pow_n3 * (xrt[0] - xct[0]) * (xrt[0] - xct[0]) + lrc * ti.pow(dist, -1) - 1)
            A_implicit[tensor_ele_idx_row, tensor_ele_idx_col][0, 1] = spring_stiffness[None] * lrc * dist_pow_n3 * (
                        xrt[0] - xct[0]) * (xrt[1] - xct[1])
            A_implicit[tensor_ele_idx_row, tensor_ele_idx_col][1, 0] = spring_stiffness[None] * lrc * dist_pow_n3 * (
                        xrt[0] - xct[0]) * (xrt[1] - xct[1])
            A_implicit[tensor_ele_idx_row, tensor_ele_idx_col][1, 1] = -spring_stiffness[None] * (
                        -lrc * dist_pow_n3 * (xrt[1] - xct[1]) * (xrt[1] - xct[1]) + lrc * ti.pow(dist, -1) - 1)
            A_implicit[tensor_ele_idx_row, tensor_ele_idx_col][0, 0] = - dt * dt * mass_inv * A_implicit[
                tensor_ele_idx_row, tensor_ele_idx_col][0, 0]
            A_implicit[tensor_ele_idx_row, tensor_ele_idx_col][0, 1] = - dt * dt * mass_inv * A_implicit[
                tensor_ele_idx_row, tensor_ele_idx_col][0, 1]
            A_implicit[tensor_ele_idx_row, tensor_ele_idx_col][1, 0] = - dt * dt * mass_inv * A_implicit[
                tensor_ele_idx_row, tensor_ele_idx_col][1, 0]
            A_implicit[tensor_ele_idx_row, tensor_ele_idx_col][1, 1] = - dt * dt * mass_inv * A_implicit[
                tensor_ele_idx_row, tensor_ele_idx_col][1, 1]
        else:
            # Populate zeros:
            A_implicit[tensor_ele_idx_row, tensor_ele_idx_col][0, 0] = 0
            A_implicit[tensor_ele_idx_row, tensor_ele_idx_col][0, 1] = 0
            A_implicit[tensor_ele_idx_row, tensor_ele_idx_col][1, 0] = 0
            A_implicit[tensor_ele_idx_row, tensor_ele_idx_col][1, 1] = 0
    # Construct b_implicit Vector:
    for i in range(n):
        # Calculate the overall force on this point:
        total_force = ti.Vector(gravity) * particle_mass
        for j in range(n):
            if rest_length[i, j] != 0:
                x_ij = x[i] - x[j]
                total_force += -spring_stiffness[None] * (x_ij.norm() - rest_length[i, j]) * x_ij.normalized()
        b_implicit[i] = v[n] + dt * mass_inv * total_force


@ti.kernel
def iterate(A_implicit_np: ti.ext_arr(), b_implicit_np: ti.ext_arr(), v_np: ti.ext_arr()) -> ti.f32:
    n = num_particles[None]
    residual = 1
    for i in range(2 * n):
        i_tensor_ele_idx_row = i // 2
        i_internal_ele_idx_row = i % 2
        temp = b_implicit_np[i_tensor_ele_idx_row, i_internal_ele_idx_row]
        for j in range(2 * n):
            if i != j:
                j_tensor_ele_idx_col = j // 2
                j_internal_ele_idx_col = j % 2
                temp -= A_implicit_np[i_tensor_ele_idx_row, j_tensor_ele_idx_col, i_internal_ele_idx_row, j_internal_ele_idx_col] * v_np[j_tensor_ele_idx_col, j_internal_ele_idx_col]
        # Divide everything by the coefficient of that unknown
        new_v[i] = temp / A_implicit_np[i_tensor_ele_idx_row, i_tensor_ele_idx_row, i_internal_ele_idx_row, i_internal_ele_idx_row]
    # Calculate the residual of this iteration by using infinite norm:
    norm1 = -1.0
    norm2 = -1.0
    for i in range(2 * n):
        i_tensor_ele_idx = i // 2
        i_internal_ele_idx = i % 2
        diff_vec[i] = v_np[i_tensor_ele_idx, i_internal_ele_idx] - new_v[i]
        if ti.abs(diff_vec[i]) > norm1:
            norm1 = ti.abs(diff_vec[i])
        if ti.abs(v_np[i_tensor_ele_idx, i_internal_ele_idx]) > norm2:
            norm2 = ti.abs(v_np[i_tensor_ele_idx, i_internal_ele_idx])
    residual = norm1 / norm2
    # Update the unknown vector:
    for i in range(n):
        v[i][0] = new_v[2 * i]
        v[i][1] = new_v[2 * i + 1]

    return residual


@ti.kernel
def substep():
    n = num_particles[None]
    # Collide with ground
    for i in range(n):
        if x[i].y < bottom_y:
            x[i].y = bottom_y
            v[i].y = 0

    # Compute new position
    for i in range(num_particles[None]):
        x[i] += v[i] * dt


@ti.kernel
def new_particle(pos_x: ti.f32, pos_y: ti.f32):  # Taichi doesn't support using Matrices as kernel arguments yet
    new_particle_id = num_particles[None]
    x[new_particle_id] = [pos_x, pos_y]
    v[new_particle_id] = [0, 0]
    num_particles[None] += 1

    # Connect with existing particles
    for i in range(new_particle_id):
        dist = (x[new_particle_id] - x[i]).norm()
        if dist < connection_radius:
            rest_length[i, new_particle_id] = 0.1
            rest_length[new_particle_id, i] = 0.1


gui = ti.GUI('Mass Spring System', res=(512, 512), background_color=0xdddddd)
result_dir = "./results"
video_manger = ti.VideoManager(output_dir=result_dir, framerate=24, automatic_build=False)

spring_stiffness[None] = 10000
damping[None] = 20

new_particle(0.3, 0.3)
new_particle(0.3, 0.4)
new_particle(0.4, 0.4)

while True:
    for e in gui.get_events(ti.GUI.PRESS):
        if e.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
            exit()
        elif e.key == gui.SPACE:
            paused[None] = not paused[None]
        elif e.key == ti.GUI.LMB:
            new_particle(e.pos[0], e.pos[1])
        elif e.key == 'c':
            num_particles[None] = 0
            rest_length.fill(0)
        elif e.key == 's':
            if gui.is_pressed('Shift'):
                spring_stiffness[None] /= 1.1
            else:
                spring_stiffness[None] *= 1.1
        elif e.key == 'd':
            if gui.is_pressed('Shift'):
                damping[None] /= 1.1
            else:
                damping[None] *= 1.1

    if not paused[None]:
        # substep:
        for ss in range(8):
            # Construct A, b vectors:
            createEqn()
            b_implicit_np = b_implicit.to_numpy()
            A_implicit_np = A_implicit.to_numpy()
            # Get solution by using Jacobian method. Stop it when its residual is small enough:
            residual = 1000
            while residual > 0.001:
                v_np = v.to_numpy()
                residual = iterate(A_implicit_np, b_implicit_np, v_np)
            # Step forward:
            substep()

    X = x.to_numpy()
    gui.circles(X[:num_particles[None]], color=0xffaa77, radius=5)

    gui.line(begin=(0.0, bottom_y), end=(1.0, bottom_y), color=0x0, radius=1)

    for i in range(num_particles[None]):
        for j in range(i + 1, num_particles[None]):
            if rest_length[i, j] != 0:
                gui.line(begin=X[i], end=X[j], radius=2, color=0x445566)
    gui.text(content=f'C: clear all; Space: pause', pos=(0, 0.95), color=0x0)
    gui.text(content=f'S: Spring stiffness {spring_stiffness[None]:.1f}', pos=(0, 0.9), color=0x0)
    gui.text(content=f'D: damping {damping[None]:.2f}', pos=(0, 0.85), color=0x0)
    gui.show()

