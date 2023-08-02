import numpy as np
import gymnasium as gym
import random

def null_cube():
  return np.zeros((6,6,3,3)) # 6 colors, 6 sides, 3x3 sides

def init_solved_cube(cube = None):
  if cube is None:
    cube = null_cube()
  for i in range(6):
    cube[i,i,:,:] = 1
  return cube

def sanity_check(cube):
  bool_ = True
  for i in range(6):
    bool_ = bool_ and np.sum(cube[i,:,:,:]) == 9 # each color has only 9
  return bool_ and np.sum(cube) == 54

def rotate_n_times_right(cube,face,n):
  return np.rot90(cube[:,face,:,:],k=n,axes=(1,2))

def move_r(cube):
  new_cube = null_cube()
  #print(cube[:,0,:,:],rotate_face_left(cube,0))
  new_cube[:,0,:,:] = rotate_n_times_right(cube,0,1)
  new_cube[:,1,:,:] = rotate_n_times_right(cube,1,3)
  new_cube[:,2,:,:] = cube[:,5,:,:]
  new_cube[:,3,:,:] = cube[:,2,:,:]
  new_cube[:,4,:,:] = cube[:,3,:,:]
  new_cube[:,5,:,:] = cube[:,4,:,:]
  return new_cube

def move_l(cube):
  new_cube = null_cube()
  #print(cube[:,0,:,:],rotate_face_left(cube,0))
  new_cube[:,0,:,:] = rotate_n_times_right(cube,0,3)
  new_cube[:,1,:,:] = rotate_n_times_right(cube,1,1)
  new_cube[:,5,:,:] = cube[:,2,:,:]
  new_cube[:,2,:,:] = cube[:,3,:,:]
  new_cube[:,3,:,:] = cube[:,4,:,:]
  new_cube[:,4,:,:] = cube[:,5,:,:]
  return new_cube

def move_d(cube):
  new_cube = null_cube()
  #print(cube[:,0,:,:],rotate_face_left(cube,0))
  new_cube[:,2,:,:] = cube[:,0,:,:]
  new_cube[:,1,:,:] = cube[:,2,:,:]
  new_cube[:,4,:,:] = cube[:,1,:,:]
  new_cube[:,0,:,:] = cube[:,4,:,:]
  new_cube[:,3,:,:] = rotate_n_times_right(cube,3,1)
  new_cube[:,5,:,:] = rotate_n_times_right(cube,5,3)
  return new_cube

def move_u(cube):
  new_cube = null_cube()
  #print(cube[:,0,:,:],rotate_face_left(cube,0))
  new_cube[:,0,:,:] = cube[:,2,:,:]
  new_cube[:,2,:,:] = cube[:,1,:,:]
  new_cube[:,1,:,:] = cube[:,4,:,:]
  new_cube[:,4,:,:] = cube[:,0,:,:]
  new_cube[:,3,:,:] = rotate_n_times_right(cube,3,3)
  new_cube[:,5,:,:] = rotate_n_times_right(cube,5,1)
  return new_cube

def move_f(cube):
  new_cube = null_cube()
  #print(cube[:,0,:,:],rotate_face_left(cube,0))
  new_cube[:,1,:,:] = cube[:,3,:,:]
  new_cube[:,0,:,:] = cube[:,5,:,:]
  new_cube[:,5,:,:] = cube[:,1,:,:]
  new_cube[:,3,:,:] = cube[:,0,:,:]
  new_cube[:,2,:,:] = rotate_n_times_right(cube,2,3)
  new_cube[:,4,:,:] = rotate_n_times_right(cube,4,1)
  return new_cube

def move_fp(cube):
  new_cube = null_cube()
  #print(cube[:,0,:,:],rotate_face_left(cube,0))
  new_cube[:,3,:,:] = cube[:,1,:,:]
  new_cube[:,5,:,:] = cube[:,0,:,:]
  new_cube[:,1,:,:] = cube[:,5,:,:]
  new_cube[:,0,:,:] = cube[:,3,:,:]
  new_cube[:,2,:,:] = rotate_n_times_right(cube,2,1)
  new_cube[:,4,:,:] = rotate_n_times_right(cube,4,3)
  return new_cube

def move_R(cube):
  new_cube = cube.copy()
  #print(cube[:,0,:,:],rotate_face_left(cube,0))
  new_cube[:,0,:,2] = cube[:,2,:,2]
  new_cube[:,2,:,2] = cube[:,1,:,2]
  new_cube[:,1,:,2] = cube[:,4,:,2]
  new_cube[:,4,:,2] = cube[:,0,:,2]
  new_cube[:,3,:,:] = rotate_n_times_right(cube,3,3)
  return new_cube

def move_Rp(cube):
  new_cube = cube.copy()
  #print(cube[:,0,:,:],rotate_face_left(cube,0))
  new_cube[:,2,:,2] = cube[:,0,:,2]
  new_cube[:,1,:,2] = cube[:,2,:,2]
  new_cube[:,4,:,2] = cube[:,1,:,2]
  new_cube[:,0,:,2] = cube[:,4,:,2]
  new_cube[:,3,:,:] = rotate_n_times_right(cube,3,1)
  return new_cube

def move_L(cube):
  new_cube = cube.copy()
  #print(cube[:,0,:,:],rotate_face_left(cube,0))
  new_cube[:,0,:,0] = cube[:,2,:,0]
  new_cube[:,2,:,0] = cube[:,1,:,0]
  new_cube[:,1,:,0] = cube[:,4,:,0]
  new_cube[:,4,:,0] = cube[:,0,:,0]
  new_cube[:,5,:,:] = rotate_n_times_right(cube,5,1)
  return new_cube

def move_Lp(cube):
  new_cube = cube.copy()
  #print(cube[:,0,:,:],rotate_face_left(cube,0))
  new_cube[:,2,:,0] = cube[:,0,:,0]
  new_cube[:,1,:,0] = cube[:,2,:,0]
  new_cube[:,4,:,0] = cube[:,1,:,0]
  new_cube[:,0,:,0] = cube[:,4,:,0]
  new_cube[:,5,:,:] = rotate_n_times_right(cube,5,3)
  return new_cube

def move_U(cube):
  new_cube = cube.copy()
  #print(cube[:,0,:,:],rotate_face_left(cube,0))
  new_cube[:,2,2,:] = cube[:,5,2,:]
  new_cube[:,3,2,:] = cube[:,2,2,:]
  new_cube[:,4,2,:] = cube[:,3,2,:]
  new_cube[:,5,2,:] = cube[:,4,2,:]
  new_cube[:,0,:,:] = rotate_n_times_right(cube,0,1)
  return new_cube

def move_Up(cube):
  new_cube = cube.copy()
  #print(cube[:,0,:,:],rotate_face_left(cube,0))
  new_cube[:,2,2,:] = cube[:,3,2,:]
  new_cube[:,3,2,:] = cube[:,4,2,:]
  new_cube[:,4,2,:] = cube[:,5,2,:]
  new_cube[:,5,2,:] = cube[:,2,2,:]
  new_cube[:,0,:,:] = rotate_n_times_right(cube,0,3)
  return new_cube

def move_D(cube):
  new_cube = cube.copy()
  #print(cube[:,0,:,:],rotate_face_left(cube,0))
  new_cube[:,2,0,:] = cube[:,5,0,:]
  new_cube[:,3,0,:] = cube[:,2,0,:]
  new_cube[:,4,0,:] = cube[:,3,0,:]
  new_cube[:,5,0,:] = cube[:,4,0,:]
  new_cube[:,1,:,:] = rotate_n_times_right(cube,1,3)
  return new_cube

def move_Dp(cube):
  new_cube = cube.copy()
  #print(cube[:,0,:,:],rotate_face_left(cube,0))
  new_cube[:,2,0,:] = cube[:,3,0,:]
  new_cube[:,3,0,:] = cube[:,4,0,:]
  new_cube[:,4,0,:] = cube[:,5,0,:]
  new_cube[:,5,0,:] = cube[:,2,0,:]
  new_cube[:,1,:,:] = rotate_n_times_right(cube,1,1)
  return new_cube

def move_F(cube):
  new_cube = cube.copy()
  #print(cube[:,0,:,:],rotate_face_left(cube,0))
  new_cube[:,1,0,:] = cube[:,3,:,0]
  new_cube[:,0,0,:] = cube[:,5,:,2]
  new_cube[:,5,:,2] = cube[:,1,0,:]
  new_cube[:,3,:,0] = cube[:,0,0,:]
  new_cube[:,2,:,:] = rotate_n_times_right(cube,2,3)
  return new_cube

def move_Fp(cube):
  new_cube = cube.copy()
  #print(cube[:,0,:,:],rotate_face_left(cube,0))
  new_cube[:,3,:,0] = cube[:,1,0,:]
  new_cube[:,5,:,2] = cube[:,0,0,:]
  new_cube[:,1,0,:] = cube[:,5,:,2]
  new_cube[:,0,0,:] = cube[:,3,:,0]
  new_cube[:,2,:,:] = rotate_n_times_right(cube,2,1)
  return new_cube

def move_B(cube):
  new_cube = cube.copy()
  #print(cube[:,0,:,:],rotate_face_left(cube,0))
  new_cube[:,1,2,:] = cube[:,3,:,2]
  new_cube[:,0,2,:] = cube[:,5,:,0]
  new_cube[:,5,:,0] = cube[:,1,2,:]
  new_cube[:,3,:,2] = cube[:,0,2,:]
  new_cube[:,4,:,:] = rotate_n_times_right(cube,4,1)
  return new_cube

def move_Bp(cube):
  new_cube = cube.copy()
  #print(cube[:,0,:,:],rotate_face_left(cube,0))
  new_cube[:,3,:,2] = cube[:,1,2,:]
  new_cube[:,5,:,0] = cube[:,0,2,:]
  new_cube[:,1,2,:] = cube[:,5,:,0]
  new_cube[:,0,2,:] = cube[:,3,:,2]
  new_cube[:,4,:,:] = rotate_n_times_right(cube,4,3)
  return new_cube

def str_to_move(move,cube):
  if move == "r":
    new_cube = move_r(cube)
  elif move == "l":
    new_cube = move_l(cube)
  elif move == "u":
    new_cube = move_u(cube)
  elif move == "d":
    new_cube = move_d(cube)
  elif move == "f":
    new_cube = move_f(cube)
  elif move == "b":
    new_cube = move_fp(cube)
  elif move == "R":
    new_cube = move_R(cube)
  elif move == "Rp":
    new_cube = move_Rp(cube)
  elif move == "L":
    new_cube = move_L(cube)
  elif move == "Lp":
    new_cube = move_Lp(cube)
  elif move == "U":
    new_cube = move_U(cube)
  elif move == "Up":
    new_cube = move_Up(cube)
  elif move == "D":
    new_cube = move_D(cube)
  elif move == "Dp":
    new_cube = move_Dp(cube)
  elif move == "F":
    new_cube = move_F(cube)
  elif move == "Fp":
    new_cube = move_Fp(cube)
  elif move == "B":
    new_cube = move_B(cube)
  elif move == "Bp":
    new_cube = move_Bp(cube)
  return new_cube

def check_cube_solved(cube):
  solved_cube = init_solved_cube()
  return np.array_equal(solved_cube,cube)


class RubiksEnv(gym.Env):
    """Mock environment for testing purposes.

    Observation=0, reward=1.0, episode-len is configurable.
    Actions are ignored.
    """

    def __init__(self, episode_length, config=None):
        self.episode_length = episode_length
        self.config = config
        self.i = 0
        self.action_dic = {0:"R",1:"Rp",2:"L",3:"Lp",4:"U",5:"Up",6:"D",7:"Dp",8:"F",9:"Fp",10:"B",11:"Bp",12:"f",13:"b",14:"r",15:"l",16:"u",17:"d"}
        self.action_list = [v for k,v in self.action_dic.items()]
        self.observation_space = gym.spaces.Box(low = 0, high = 1, shape = (6,6,3,3))
        self.action_space = gym.spaces.Discrete(18)
        self.random_init = 5
        self.cube = init_solved_cube()

    def reset(self, *, seed=None, options=None):
        self.i = 0
        self.cube = init_solved_cube()
        if self.random_init > 0:
            for i in range(self.random_init):
                move = random.choice(self.action_list)
                new_cube = str_to_move(move,self.cube)
                self.cube = new_cube.copy() # potential for improvement
        return self.cube, {}

    def action_to_move(self,action):
        move = self.action_dic[np.argmax(action)]
        new_cube = str_to_move(move,self.cube)
        self.cube = new_cube.copy() # potential for improvement

    def objective_eval(self):
        return self.i/self.random_init

    def step(self, action):
        self.i += 1
        truncated = False
        self.action_to_move(action)
        if check_cube_solved(self.cube):
          terminated = True
          return self.cube, self.objective_eval(), terminated, truncated, {}
        else:
          terminated = truncated = self.i >= self.episode_length
          return self.cube, 0.0, terminated, truncated, {}

