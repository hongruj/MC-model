'''
Movement execution: start from go cue onset to reach the target
'''


import torch
import torch.nn as nn
import numpy as np
from lib.MuscleArm_new import muscular_arm
from lib.gramians import Make
from lib.some import reduct



class mc_model(nn.Module):
    def __init__(self, dh, home_joint_state, home_cart_state):
        super(mc_model, self).__init__()
        self.dh = dh 
        self.num_network_inputs = 10
        self.nobs = self.num_neurons = 400
        self.num_muscle = 6
        self.num_reach_combinations = 8
        self.fb_delay = 25 
        
        self.home_joint_state = home_joint_state*torch.ones(self.num_reach_combinations,2)
        self.home_cart_state = home_cart_state*torch.ones(self.num_reach_combinations,2)
        self.spontaneous = (torch.normal(1, 0.15, size=(self.num_neurons,1)))
        
        # Intantiate the biomechanical arm dynamics at home location
        self.adyn = muscular_arm(dh,self.num_reach_combinations)
        
        # Input layer 
        self.mc_inplayer = nn.Linear(self.num_network_inputs, self.num_neurons, bias=True)
        self.mc_inplayer.weight = nn.Parameter(torch.normal(0,std=1/(self.num_network_inputs**0.5), size=(self.num_neurons,self.num_network_inputs)))   
        self.mc_inplayer.bias = torch.nn.Parameter(torch.zeros(self.num_neurons))

        # MC layer 
        self.w_rec = np.loadtxt('isn_1.2_0.9.txt')                     
        self.W = torch.from_numpy(self.w_rec).float()
        self.mc_act = nn.ReLU() # activation function of layer neurons        
        self.h_inp = (self.spontaneous - self.W@self.spontaneous).T        
        self.xstars_prms = nn.Parameter(torch.normal(0,std=1.5/(self.nobs**0.5), size=(self.nobs,self.num_reach_combinations)))        
        self.top_obs, self.z = self.param_unpack()             
        self.g = torch.randn(1,self.num_neurons)*0.2

        # Output layer
        self.c_prms = nn.Parameter(torch.normal(0,std=0.1/self.num_neurons**0.5, size=(self.num_muscle, self.num_neurons))) 
        self.muscle_act = nn.LeakyReLU(0.4)


    def forward(self, T, des_cart, choice, ptb= None): 
        # reset state MC & arm 
        joint_state = torch.zeros(self.num_reach_combinations, 4)
        joint_state[:, :2] = self.home_joint_state # initial shoulder angle
                
        self.inp_list = []
        self.networkactivity_list = []
        self.torq_list = []
        self.mus_out_list = []
        self.force_list = []
        self.jointstate_list = []
        self.cartesianstate_list = []
       
        tau_x = 10/200
        tau_m = 10/50
        # initial states
        s = torch.zeros(self.num_reach_combinations, self.num_neurons)
        x, self.c, xstars_tar = self.unpack(des_cart-self.home_cart_state)
        u = torch.mm(-xstars_tar, self.W.T) + xstars_tar
        # noise for initial condition
        if ptb is not None:
            x = x + ptb            
        r = self.mc_act(x)        
        muscle = 0

        for i in range(T):   
            self.networkactivity_list.append(r)            
            muscle = self.muscle_act(tau_m * torch.mm(r, self.c.T)) + (1 - tau_m) * muscle
            if i<10:
                joint_state, force, torq = self.adyn.forward(joint_state, muscle, torch.zeros(self.num_reach_combinations, self.num_muscle)) 
            else:
                joint_state, force, torq = self.adyn.forward(joint_state, muscle, self.mus_out_list[i-10]) 

            self.mus_out_list.append(muscle)            
            self.force_list.append(force)
            self.torq_list.append(torq)

            cart_state = self.adyn.armkin(joint_state) 
            self.cartesianstate_list.append(cart_state)   
            
            if choice == 'all':  # baseline model
                sen_inp = torch.cat(((cart_state[:,:2]-self.home_cart_state), cart_state[:,2:], force/2.7), 1)
            elif choice == 'novel':   # without velocity feedback
                sen_inp = torch.cat(((cart_state[:,:2]-self.home_cart_state), torch.zeros(self.num_reach_combinations, 2), force/2.7), 1) 
            elif choice == 'nomus':   # without muscle force feedback
                sen_inp = torch.cat(((cart_state[:,:2]-self.home_cart_state), cart_state[:,2:], torch.zeros(self.num_reach_combinations, 6)), 1)
            elif choice == 'only':    # without velocity and muscle force feedback
                sen_inp = torch.cat(((cart_state[:,:2]-self.home_cart_state), torch.zeros(self.num_reach_combinations, 8)), 1)           
          
            # MC network layers dynamics
            tau_rise=6
            tau_decay=60                       
            cis = (5+self.g)/0.6968*(np.exp(-(i+1) / tau_decay) - np.exp(-(i+1) / tau_rise))
            if choice == 'auto':  # autonomous dynamical system
                input_sum = torch.mm(r, self.W.T) - x + self.h_inp + cis 
            else:
                network_inputs = self.mc_inplayer(sen_inp)*2.7
                self.inp_list.append(network_inputs)  
                input_sum = torch.mm(network_inputs + r, self.W.T) - x - network_inputs  + self.h_inp + cis + u*2.7

            x = tau_x * input_sum + x 
            r = self.mc_act(x) 

        return self.cartesianstate_list


    def param_unpack(self):
        xstars_std = 0.2                                               
        # Get the top n_obs eigenvectors of the observability Gramian                        
        G = Make(self.w_rec - np.eye(self.num_neurons))
        top_obs = torch.Tensor(G.O.top(self.nobs))
        z = self.num_neurons * self.num_reach_combinations * xstars_std ** 2
        return top_obs, z


    def unpack(self, joint_targ):              
        # Calculate xstars 
        xstars_tar = torch.mm(joint_targ, self.mc_inplayer.weight[:,:2].T)
        xstars = torch.mm(self.top_obs, self.xstars_prms) + xstars_tar.T
        xstars = torch.sqrt(self.z / torch.sum(xstars ** 2)) * xstars 
        xstars_motor = torch.cat((xstars + self.spontaneous, self.spontaneous), axis=1)
        h = torch.linalg.solve(torch.mm(xstars_motor.T, xstars_motor), xstars_motor.T)
        c = self.c_prms - torch.mm(self.c_prms, torch.mm(xstars_motor, h))               
        return (self.spontaneous + xstars).T, c, xstars_tar     
                
                         

def costCriterionReaching(reach_sim, des_tar, act_tar):
    num_reach_combinations = reach_sim.num_reach_combinations
    T = act_tar.shape[0]
    loss = (1/30)*torch.norm((act_tar[-30:,:,:2] - des_tar))**2
    # movement stop
    loss += (1/30)*torch.norm(act_tar[-30:,:,2:])**2  
    loss += (1/20)*torch.norm(torch.stack(reach_sim.mus_out_list[-20:]))**2
    # minimal torque
    loss += (1/T)*torch.norm(torch.stack(reach_sim.torq_list[1:])-torch.stack(reach_sim.torq_list[:-1]))**2
    
    md_fr = 20*torch.stack(reach_sim.networkactivity_list[:28*3])
    md_pca = 10*reduct(md_fr[::3],10)
    x = md_pca.reshape(8,28,10)     
    
    loss += (1e-2/15)*(torch.norm(x[:,:14,0] - x[:,14:14+14,0].flip(1))**2 + torch.norm(x[:,:14,1] - x[:,14:,1].flip(1))**2)
    loss += (1e-4/15)*reach_sim.mc_act(torch.norm(x[:,:14,-4:])**2 - torch.norm(x[:,14:,-4:])**2)
    return 10*loss/num_reach_combinations  