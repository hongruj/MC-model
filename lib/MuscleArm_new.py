"""
originally from https://github.com/neurohari/CorticalDynamics
Change arm parameters and accelerate computation speed

"""
import torch

use_cuda = False
# use_cuda = True
device = torch.device('cuda:0' if use_cuda else 'cpu')


class muscular_arm():
    def __init__(self, dh,num_reach):
        super(muscular_arm, self).__init__()
        
        # fixed Monkey arm parameters (1=shoulder; 2=elbow) (refer to Lillicrap et al 2013, Li&Todorov2007)
        self.i1 = 0.025 # kg*m**2 shoulder inertia
        self.i2 = 0.045 # kg*m**2 elbow inertia
        self.m1 = 1.4 # kg mass of shopulder link
        self.m2 = 1 # kg mass of elbow link
        self.l1 = 0.3 # meter
        self.l2 = 0.33 # meter
        self.s1 = 0.0749 
        self.s2 = 0.16 # center of mass of lower
        self.b11 = 0.05 # Could use low values like 0.05 also
        self.b22 = 0.05 # Could use low values like 0.05 also
        self.b21 = 0.025 # Could use low values like 0.02 - 0.05 also
        self.b12 = 0.025 # Could use low values like 0.02 - 0.05 also

        # inertial matrix tmp vars
        self.a1 = (self.i1 + self.i2) + (self.m2 * self.l1**2)
        self.a2 = self.m2 * self.l1 * self.s2
#         self.a3 = self.i2

        # Moment arm param in centimeters, but it can be directly used with this code...
        # ...as the scaling can be assumed to happen at the output layer (1 a.u = 100N force)
        self.M = torch.tensor([[2.0, -2.0, 0.0, 0.0, 1.50, -2.0], [0.0, 0.0, 2.0, -2.0, 2.0, -1.50]]).to(device) 

        # Muscle properties 
        self.theta0 = 0.0175*torch.tensor([[15.0, 4.88, 0.00, 0.00, 4.5, 2.12], [0.00, 0.00, 80.86, 109.32, 92.96, 91.52]]).to(device)
        self.L0 = torch.tensor([[7.32, 3.26, 6.4, 4.26, 5.95, 4.04]]).to(device) # in centimeters but ( self.M / self.L0 ) ratio will be unaffected as self.M is in centimeters too
        self.beta = 1.55
        self.omega = 0.81
        self.rho = 2.12
        self.Vmax = -7.39
        self.cv0 = -3.21
        self.cv1 = 4.17
        self.bv = 0.62
        self.av0 = -3.12
        self.av1 = 4.21
        self.av2 = -2.67

        # time-step of dynamics
        self.dh = dh
        #shuanq        
        self.cur_j_state = torch.zeros(num_reach, 4).to(device)
        self.FV = torch.zeros(num_reach,6).to(device)
        
    def forward(self, x, mus_inp, mus_inp_d):
        """
        rout is the readout from M1 layer. mact is the muscle activation fcn
        """
        self.cur_j_state = x
        # for non-linear muscle activation - add F-L/V property contribution
        fl_out, fv_out = self.muscleDyn()
        flv_computed = fl_out * fv_out
#         print(fl_out, fv_out)
        mus_out = flv_computed * mus_inp
        mus_out_d = flv_computed * mus_inp_d
        
        
        #muscle-force to joint-torque transformation (using M)
        tor = torch.mm(self.M, mus_out_d.transpose(0,1))

        tor = tor.transpose(0,1)
        
        x = self.armdyn(x, tor)
        
        return x, mus_out, tor
        
        
    def armdyn(self, x, u):
        batch_size = u.size(0)
        
        
        # extract joint angle states
        theta1 = x[:, 0].clone().unsqueeze(1)
        theta2 = x[:, 1].clone().unsqueeze(1)
        theta1_dot = x[:, 2].clone().unsqueeze(1)
        theta2_dot = x[:, 3].clone().unsqueeze(1)
        
    
        # compute inertia matrix
        I11 = self.a1 + (2*self.a2*(torch.cos(theta2)))
        I12 = self.i2 + (self.a2*(torch.cos(theta2)))
        I21 = self.i2 + (self.a2*(torch.cos(theta2)))
        I22 = self.i2*torch.ones(batch_size,1)
        

        # compute determinant of mass matrix [a * b of two tensors is the element-wise product]
        det = (I11 * I22) - (I12 * I21)

        # compute Inverse of inertia matrix
        Irow1 = torch.cat((I22, -I12), 1)
        Irow2 = torch.cat((-I21, I11), 1)

#        Iinv = (1/det.unsqueeze(1)) * torch.cat((Irow1.unsqueeze(1), Irow2.unsqueeze(1)), 1) # WORKING

        
        # compute extra torque H (coriolis, centripetal, friction)
        h1 = ((-theta2_dot) * ((2*theta1_dot) + theta2_dot) * (self.a2 * torch.sin(theta2))) + (self.b11*theta1_dot) + (self.b12*theta2_dot)
        h2 = ((theta1_dot**2) * self.a2 * torch.sin(theta2)) + (self.b21*theta1_dot) + (self.b22*theta2_dot)
        

        H = torch.cat((h1, h2), 1)
        # eq 14  X
        
        # compute xdot = inv(M) * (u - H)
        #torque = u - H
        
        
        #print(torque)
        #torque = torque.unsqueeze(2) # WORKING
        #print(torque)
        # determione the terms in xdot matrix; xdot = [[dq1], [dq2], [ddq1], [ddq2]]
        dq1 = theta1_dot
        dq2 = theta2_dot
        dq = torch.cat((dq1, dq2), 1)
        
        torque = u - H
        
        Irow1 = (1/det) * Irow1
        Irow2 = (1/det) * Irow2
        # terms of Iinv matrix
        Iinv_11 = Irow1[:, 0].unsqueeze(1)
        Iinv_12 = Irow1[:, 1].unsqueeze(1)
        Iinv_21 = Irow2[:, 0].unsqueeze(1)
        Iinv_22 = Irow2[:, 1].unsqueeze(1)
        
        # Update acceleration of shoulder and elbow joints - FWDDYN equations        
        ddq1 = Iinv_11*torque[:, 0].unsqueeze(1) + Iinv_12*torque[:, 1].unsqueeze(1)
        ddq2 = Iinv_21*torque[:, 0].unsqueeze(1) + Iinv_22*torque[:, 1].unsqueeze(1)
        ddq = torch.cat((ddq1, ddq2), 1)
                
        
        #ddq = torch.matmul(Iinv, torque) # matmul is a bit slower than the <einsum> by 1 sec for batch matrix multiplication # WORKING
        #ddq = torch.einsum('ijk,ikl->ijl', [Iinv, torque]) # WORKING
        
        # update xdot
        x_dot = torch.cat((dq, ddq), 1)
        
        # step-update from x to x_next
        x_next = x + (self.dh * x_dot) 
        
        x = x_next
        return x
    
    def armkin(self, x):
        theta1 = x[:, 0].clone().unsqueeze(1)
        theta2 = x[:, 1].clone().unsqueeze(1)
        theta1_dot = x[:, 2].clone().unsqueeze(1)
        theta2_dot = x[:, 3].clone().unsqueeze(1)
        
        g11 = (self.l1 * torch.cos(theta1)) + (self.l2 * torch.cos(theta1+theta2)) 
        g12 = (self.l1*torch.sin(theta1)) + (self.l2*torch.sin(theta1+theta2))
        g13 = -theta1_dot*((self.l1*torch.sin(theta1))+(self.l2*torch.sin(theta1+theta2)))
        g13 = g13-(theta2_dot*(self.l2*torch.sin(theta1+theta2)))
        g14 = theta1_dot*((self.l1*torch.cos(theta1))+(self.l2*torch.cos(theta1+theta2)))
        g14=g14+(theta2_dot*(self.l2*torch.cos(theta1+theta2)))
        y = torch.cat((g11,g12,g13,g14), 1)
        return y


    def muscleDyn(self):
        # F-L/V dependency
        mus_l = 1 + self.M[0,:] * (self.theta0[0,:] - self.cur_j_state[:, 0].unsqueeze(1))/self.L0 + self.M[1,:] * (self.theta0[1,:] - self.cur_j_state[:, 1].unsqueeze(1))/self.L0
        mus_v = self.M[0, :] * self.cur_j_state[:, 2].unsqueeze(1)/self.L0 + self.M[1, :] * self.cur_j_state[:, 3].unsqueeze(1)/self.L0    
        mus_v = -mus_v + 0.0  # mus_v = d (mus_l) / dt
        FL = torch.exp(-torch.abs((mus_l**self.beta - 1)/self.omega)**self.rho)

        vel_neg = mus_v <= 0
        vel_pos = mus_v > 0

        FV = self.FV.clone()
        FV[vel_neg] = (self.Vmax - mus_v[vel_neg]) / (self.Vmax + mus_v[vel_neg] * (self.cv0 + self.cv1 * mus_l[vel_neg]))
        FV[vel_pos] = (self.bv - mus_v[vel_pos] * (self.av0 + self.av1 * mus_l[vel_pos] + self.av2 * mus_l[vel_pos]**2)) / (self.bv + mus_v[vel_pos])

        return FL, FV
