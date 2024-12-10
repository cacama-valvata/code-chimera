# -*- coding: utf-8 -*-
"""
@author: maestre
"""
from __future__ import print_function

#import matplotlib.pyplot as plt
#from matplotlib.patches import Rectangle, Circle
#from matplotlib import colors, colorbar
#from matplotlib import gridspec
#from math import sin,cos
import numpy as np
import random
import pyAgrum as agrum
from collections import OrderedDict
#from scipy import spatial

import os, sys
run_path = os.path.abspath(os.path.join('..', '..'))
sys.path.append(run_path)
lib_path = os.path.abspath(os.path.join('..', 'a2l_core_lib'))
sys.path.append(lib_path)
import environment_dynamics as env
import environment_setup as setup
import environment_delta as delta
import ros_services
import inf_traj_series_dataset_classes as dataset_series_classes
import simulation_parameters as sim_param

import discretize_move as discr_move
if sim_param.discr_hand_coded:
    import discretize_orientation_hand_coded as discr_orien
    import discretize_distance_hand_coded as discr_dist
else:
    import discretize_orientation_sections as discr_orien
    import discretize_inclination_sections as discr_inclin
    import discretize_distance_sections as discr_dist
    
#import roslib
#roslib.load_manifest('std_msgs')
import rospy
from std_msgs.msg import Float64MultiArray

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#from matplotlib.patches import Circle
import mpl_toolkits.mplot3d.art3d as art3d
from scipy.linalg import norm

from scipy.spatial import distance as d
from math import sqrt

import copy

from collections import OrderedDict

'''
callback class
'''
class Traj_callback():
    def __init__(self, 
                 simulated_traj, 
#                 simulated_obj_pos,
#                 current_orien, current_inclin, current_dist,
#                 eef_traj_vector, obj_traj_vector, delta_vector, inf_dicr,
#                 delta_class_vector, traj_res,
#                 expected_effect, obtained_effect,
#                 nb_executed_deltas,
                 below_box):
#        self.simulated_obj_pos = simulated_obj_pos
#        self.current_orien = current_orien
#        self.current_inclin = current_inclin
#        self.current_dist = current_dist
#                     
#        self.eef_traj_vector = eef_traj_vector ## [[eef_x, eef_y, eef_z]]
#        self.obj_traj_vector = obj_traj_vector ## [[obj_x, obj_y, obj_z]]
#        self.delta_vector = delta_vector ## [[orientation, inclionation, distance, next_mov_discr]]
#        self.traj_inferred_discr_delta_vector = inf_dicr ## [[effect, idem]]
#        self.delta_class_vector = delta_class_vector
#        self.traj_res = traj_res ##True or False
#        self.expected_effect = expected_effect
#        self.obtained_effect = obtained_effect
        self.nb_executed_deltas = 0
        self.traj_res = False
        
        ## subscribe to feedback topic
        rospy.init_node('listener', anonymous=True)
        self.sub = rospy.Subscriber(sim_param.feedback_topic, 
                                   Float64MultiArray, 
                                   self.execute_traj_callback,
                                   queue_size=1)
        self.sub_up = True        
        
        
        ## execute trajectory
#        simulated_traj_vector = [i for el in simulated_traj for i in el] # [float]
#        ros_services.call_trajectory_motion(sim_param.feedback_window, 
#                                            simulated_traj_vector)
#        print("FINISHED EXECUTION IN ROS !! ")
        ## avoid touching table in REAL ROBOT
        if not sim_param.real_robot or (sim_param.real_robot and not below_box):    
            ## execute trajectory
            simulated_traj_vector = [i for el in simulated_traj for i in el] ## [float]
            res_exec = ros_services.call_trajectory_motion(sim_param.feedback_window, 
                                                           simulated_traj_vector)
#            rospy.spin()
            if not res_exec:
                print("FAILED EXECUTION IN ROS !! ")
            else:
                print("FINISHED EXECUTION IN ROS !! ")       
        else:
            print("-----------------> TRAJECTORY NOT EXECUTED !! TOUCHING THE TABLE!!! ")         

    def execute_traj_callback(self, feedback_data):
        print("\ntraj_callback ---> execute_traj_callback function", feedback_data.data)
        print('traj_callback ---> feedback_data size', len(feedback_data.data))
        
        ''' Return if trajectory stopped '''
        stop_bool = rospy.get_param("/stop_traj")
        print("traj_callback ---> stop_bool", stop_bool)
        if stop_bool:
            print("traj_callback ---> traj stopped !!!")
            return
        
        ''' Read feedback '''
        self.nb_executed_deltas = len(feedback_data.data)/6
        pos = 0
        batch_size = 6 * sim_param.feedback_window
        for pos in range(0, len(feedback_data.data), batch_size):
            feedback_wp_vector = [] 
            feedback_obj_vector = []
            nb_wp = sim_param.feedback_window
            for i in range(nb_wp):
                eef_pos_x = feedback_data.data[6*i+0]
                eef_pos_y = feedback_data.data[6*i+1]
                eef_pos_z = feedback_data.data[6*i+2]
                feedback_wp_vector.append([eef_pos_x,
                                           eef_pos_y,
                                           eef_pos_z])
#                self.eef_traj_vector = \
#                    self.eef_traj_vector + feedback_wp_vector                
                
                obj_pos_x = feedback_data.data[6*i+3]
                obj_pos_y = feedback_data.data[6*i+4]
                obj_pos_z = feedback_data.data[6*i+5]                
                feedback_obj_vector.append([obj_pos_x,
                                            obj_pos_y,
                                            obj_pos_z])                
#                self.obj_traj_vector = \
#                    self.obj_traj_vector + feedback_obj_vector
                    
                print("traj_callback ---> current eef pos", self.eef_traj_vector[-1])
                print("traj_callback ---> obj_traj_vector", self.obj_traj_vector[-1])
                
#        ''' Check if object moved'''
#        obj_moved = False
#        thrs = sim_param.obj_moved_threshold
#        initial_obj_pos = self.simulated_obj_pos
##        print("\ninitial_obj_pos", initial_obj_pos)
#        for obj_pos in feedback_obj_vector:
#            obj_moved = \
#                (abs(initial_obj_pos[0] - obj_pos[0]) > thrs) or \
#                (abs(initial_obj_pos[1] - obj_pos[1]) > thrs) or \
#                (abs(initial_obj_pos[2] - obj_pos[2]) > thrs)
#            if obj_moved:
#                break
        
#        ''' Store delta info '''
#        self.compute_traj_info(feedback_wp_vector, 
#                               feedback_obj_vector,
#                               obj_moved)
            
#        if obj_moved :                                                
#            ''' If object touched, compute result '''
#            self.obtained_effect = 'left' ######################################## TBD 
#            obtained_effect = env.identify_effect_3d(initial_obj_pos,
#                                                     current_obj_pos)
            
#            if self.expected_effect == self.obtained_effect:
#                self.res = 'success'
#            else:
#                self.res = 'false_pos'
                    
#            for tmp_delta in self.delta_vector: #[orientation, inclination, distance, move]
#                ## in discr dataset file : 
#                ## effect,orientation,inclination,move,distance
#                self.traj_inferred_discr_delta_vector.append(
#                    ['', #self.obtained_effect,  ## effect
#                     tmp_delta[0],  ## orientation
#                     tmp_delta[1],  ## inclination
#                     tmp_delta[3],  ## move
#                     tmp_delta[2]]) ## distance
            
#            if sim_param.debug_infer:
#                print(self.expected_effect, self.obtained_effect)
#                print('Result :', self.res.upper())               
                
                
#            ''' If object position changes alone then
#                update trajectory'''   
#            stop_bool = rospy.get_param("/stop_traj")
#            if stop_bool:
#            self.sub.unregister() # close subscriber
#            self.sub_up = False
##            else:
##                rospy.set_param("/stop_traj", "true") # stop execution 
#        
#        elif self.nb_executed_deltas > sim_param.max_nb_executed_deltas:
#            self.sub.unregister() # close subscriber
#            self.sub_up = False
#
##        print("nb executed deltas", self.nb_executed_deltas)
       

        ''' Check if object moved''' 
        if self.nb_executed_deltas == 1: #not possible in one mov
            obj_moved = False
        else:
            previous_eef = self.eef_traj_vector[-2]
            current_eef = self.eef_traj_vector[-1]
            obj_pos = self.obj_traj_vector[-1]
                  
            obj_moved, obj_moved_pose = \
                env.compute_obj_pos(
                    obj_pos,
                    [previous_eef[0],previous_eef[1],previous_eef[2]], 
                    [current_eef[0],current_eef[1],current_eef[2]])
            print("traj_callback ---> obj_moved:", obj_moved)
            
            if obj_moved:                      
                ''' If obj_moved was due to the robot '''
                if self.nb_executed_deltas == len(simulate_traj):
                    self.traj_res = True
                    print("traj_callback ---> The robot touched the box !!!!!!")                
                    self.sub.unregister() # close subscriber
                    self.sub_up = False            
                
                else:
                    ''' If not, moved due to external change, and then stop traj
                    to recalculate it '''                
                    rospy.set_param("/stop_traj", "true") # stop execution
                    print("traj_callback ---> object externally moved")
            
            else: ## not moved
                if self.nb_executed_deltas > sim_param.max_nb_executed_deltas:
                    self.traj_res = False
                    print("traj_callback ---> Trajectory does not interact with object !!!!!!")                
                    self.sub.unregister() # close subscriber
                    self.sub_up = False                  

                
#    '''
#    a
#    '''
#    def compute_traj_info(self, 
#                          feedback_wp_vector, 
#                          feedback_obj_vector,
#                          obj_moved):
#        print("executing compute_traj_info function")  
#        for pos in range(len(feedback_wp_vector)-1):
#            ## store delta also as delta class              
#            current_delta_class = delta.Delta(
#                        self.expected_effect,
#                        feedback_wp_vector[pos][0],
#                        feedback_wp_vector[pos][1],
#                        feedback_wp_vector[pos][2],
#                        feedback_wp_vector[pos+1][0],
#                        feedback_wp_vector[pos+1][1],
#                        feedback_wp_vector[pos+1][2],
#                        feedback_obj_vector[pos][0],
#                        feedback_obj_vector[pos][1],
#                        feedback_obj_vector[pos][2],
#                        feedback_obj_vector[pos+1][0],
#                        feedback_obj_vector[pos+1][1],
#                        feedback_obj_vector[pos+1][2],
#                        obj_moved)
#            self.delta_class_vector.append(current_delta_class)
#            
#            ## compute and store discretized values
#            move = discr_move.compute_move_discr(
#                        [feedback_wp_vector[pos][0],
#                        feedback_wp_vector[pos][1],
#                        feedback_wp_vector[pos][2]],
#                        [feedback_wp_vector[pos+1][0],
#                        feedback_wp_vector[pos+1][1],
#                        feedback_wp_vector[pos+1][2]])
#    
#            orientation = discr_orien.compute_orientation_discr(
#                        [feedback_wp_vector[pos][0],
#                        feedback_wp_vector[pos][1]],
#                        [feedback_wp_vector[pos+1][0],
#                        feedback_wp_vector[pos+1][1]],
#                        self.current_orien)
#                             
#            inclination = discr_inclin.compute_inclination_discr(
#                        [feedback_wp_vector[pos][0],
#                        feedback_wp_vector[pos][1],
#                        feedback_wp_vector[pos][2]],
#                        [feedback_wp_vector[pos+1][0],
#                        feedback_wp_vector[pos+1][1],
#                        feedback_wp_vector[pos+1][2]],
#                        self.current_inclin)
#
#            distance = discr_dist.compute_distance(
#                        [feedback_wp_vector[pos][0],
#                        feedback_wp_vector[pos][1],
#                        feedback_wp_vector[pos][2]],
#                        [feedback_wp_vector[pos+1][0],
#                        feedback_wp_vector[pos+1][1],
#                        feedback_wp_vector[pos+1][2]],
#                        self.current_dist)            
#            
#            self.delta_vector.append([orientation, inclination, distance, move])  
##            print("delta_discr", [orientation, inclination, distance, move])

'''
Infere trajs for a dataset, algo and nb_init_pos, 
for each initial position of the eef and for each goal
'''
def infere_trajectories(current_results_folder,
                        bn, 
                        alg,
                        dataset_size_class,
                        dataset_string,
                        nb_initial_pos,
                        curr_nb_infere_trajs,
                        current_orien,
                        current_inclin,
                        current_dist):   
                            
    print('\n\n\n////////////////////////////////////////////////////////////////')
    print('////////////////////////////////////////////////////////////////')
    print('////////////////////////////////////////////////////////////////')
    print('////////////////////////////////////////////////////////////////')                            
    print('///////////////////// TRAJECTORY', curr_nb_infere_trajs + 1)
                            
    ## to infer next move
    ie = agrum.LazyPropagation(bn) 
    
    ## create initial positions of eef
    _list_x_axis, _list_y_axis, _list_z_axis = \
        setup.gen_init_eef(nb_initial_pos)

    if len(_list_x_axis) > 1:
        init_pos_vector = list(
            zip(_list_x_axis, 
                _list_y_axis, 
                _list_z_axis))
    else:
        init_pos_vector = [[_list_x_axis[0],
                           _list_y_axis[0],
                           _list_z_axis[0]]]
    ## restart statistics
    nb_success_r = 0
    nb_success_l = 0
    nb_success_u = 0
    nb_success_d = 0
    nb_false_positive_r = 0
    nb_false_positive_l = 0
    nb_false_positive_u = 0
    nb_false_positive_d = 0
    nb_fail_r = 0
    nb_fail_l = 0
    nb_fail_u = 0
    nb_fail_d = 0
    
    mean_prob = 0
    
    print(' ')    
    
    ''' Restart env '''
#    if sim_param.exec_traj:
    if sim_param.experiment_type != 'a2l_reproduce_dataset':
        ''' Restart scenario '''         
        success = ros_services.call_restart_world("all")
        if not success:
            print("ERROR - restart_world failed")
    else:
        if curr_nb_infere_trajs == 0:
            ''' Restart scenario '''         
            success = ros_services.call_restart_world("all")
            if not success:
                print("ERROR - restart_world failed")    

    ''' Get all object pos'''
    obj_pos_dict = OrderedDict()
    for obj_name in sim_param.obj_name_vector:
        obj_pos = ros_services.call_get_model_state(obj_name)
        obj_pos = [round(pos, sim_param.round_value) for pos in obj_pos[0:3]]
        if obj_name == 'cube':
            obj_pos[2] = -0.15
        else:
            obj_pos[2] = -0.14
        obj_pos_dict[obj_name] = obj_pos
        
    ''' Get first obj pos to compute eef initial pos'''
    first_obj_pos = obj_pos_dict[sim_param.obj_name_vector[0]]
#    print(first_obj_pos, ':', first_obj_pos)
    for name,pos in obj_pos_dict.items():
        print(name, '-->', pos)
    

    ######################################################################################################3
    ## TODO THIS SHOULD BE IN A DIFFERENT FUNCTION
    ## From second iteration
    if sim_param.experiment_type == 'a2l_reproduce_dataset' and curr_nb_infere_trajs > 0:
        ''' Update initial eef pos '''
        eef_init_pos = ros_services.call_get_eef_pose('left')
        eef_init_pos = [round(pos, sim_param.round_value) for pos in eef_init_pos[0:3]]          
    
        ## eef init pos is related to obj pos    
        eef_init_pos = copy.copy(init_pos_vector[0])
        eef_init_pos[0] = first_obj_pos[0] + random.uniform(-sim_param.new_obj_pos_dist*2,
                                           sim_param.new_obj_pos_dist)
        if first_obj_pos[1] > eef_init_pos[1]:
            eef_init_pos[1] = eef_init_pos[1] + random.uniform(0,sim_param.new_obj_pos_dist)
        elif first_obj_pos[1] < eef_init_pos[1]:
            eef_init_pos[1] = eef_init_pos[1] + random.uniform(-sim_param.new_obj_pos_dist/2,0)
        eef_init_pos[2] = eef_init_pos[2] + random.uniform(-sim_param.new_obj_pos_dist/2,0)

        res_init_pos = ros_services.call_move_to_initial_position(eef_init_pos)        
        ## TODO check result        
        print("eef_init_pos", eef_init_pos)
        
        ''' Change pos first object close to eef '''
        ## if obj too far from end_effector put it around the center of the table
        ## and the eef close 
#        ''' Get obj pos '''
#        obj_pos_dict = dict()
#        for obj_name in sim_param.obj_name_vector:
#            obj_pos = ros_services.call_get_model_state(obj_name)    
#            obj_pos = [round(pos, sim_param.round_value) for pos in obj_pos[0:3]]    
#            obj_pos[2] = -0.13
#            obj_pos_dict[obj_name] = obj_pos
#            print(obj_name, ':', obj_pos)
        first_obj_pos = obj_pos_dict[sim_param.obj_name_vector[0]]
        print('euclidean(obj_pos, eef_init_pos)', 
              d.euclidean(first_obj_pos, eef_init_pos))
              
        if d.euclidean(first_obj_pos, init_pos_vector[0]) > sim_param.obj_too_far_distance:
            print('-------------> UPDATING OBJECT POSITIONS')
            ## compute new obj pos
            new_obj_pos = sim_param.first_obj_pos
            new_obj_pos = [new_obj_pos[0] + random.uniform(-sim_param.new_obj_pos_dist,
                                                           sim_param.new_obj_pos_dist),
                           new_obj_pos[1] + random.uniform(-sim_param.new_obj_pos_dist,
                                                           sim_param.new_obj_pos_dist),
#                           -0.14]
                           new_obj_pos[2]]
            new_obj_pos = [round(pos, sim_param.round_value) for pos in new_obj_pos]
            obj_pos = copy.copy(new_obj_pos)
            
            ## move obj to new pos    
            success = ros_services.call_restart_world("object",
                                                      sim_param.obj_name_vector[0],
                                                      new_obj_pos)                                                                          
            if not success:
                print("ERROR - restart_world failed for object", sim_param.obj_name_vector[0])
                
            ## move rest of the objects, related to first object pos
            for obj_name in sim_param.obj_name_vector[1:]:
                if obj_name == 'cylinder':
                    obj_pos_dict[obj_name] = \
                        [new_obj_pos[0] + sim_param.new_obj_pos_dist,
                         new_obj_pos[1] - sim_param.new_obj_pos_dist,
                         -0.14]

#                         new_obj_pos[2]]      
                    ## move obj to new pos    
                    success = ros_services.call_restart_world("object",
                                                      obj_name,
                                                      obj_pos_dict[obj_name])
            if not success:
                print("ERROR - restart_world failed for object", obj_pos_dict[obj_name])
        
        for name,pos in obj_pos_dict.items():
            print(name,':', obj_pos)
       
    ######################################################################################################3    

    ## for each initial position infere / plot / save a traj
    nb_init_pos = len(init_pos_vector)
    total_inferred_discr_delta_vector = [] ## delta knowledge created during evaluations
    for curr_init_pos in range(nb_init_pos):
        nb_effect = 0
        init_pos_coord = init_pos_vector[curr_init_pos]
        
        ## for each effect
        while nb_effect < len(sim_param.effect_values):          
            desired_effect = sim_param.effect_values[nb_effect]
            
            print('NEW TRAJ for init_pos', init_pos_coord,
                  'effect', desired_effect.upper())            
  
            ## infere traj
            res, eef_traj, obj_traj, \
            delta_class_vector, obtained_effect, \
            tmp_inferred_discr_delta_vector = \
                infere_traj(bn, ie,
                            init_pos_coord, ## [X, Y, Z]
                            init_pos_vector,
                            obj_pos_dict,
                            desired_effect, 
                            current_orien, current_inclin, current_dist,
                            curr_nb_infere_trajs)

            ## store current inferred traj
            tmp_infer_traj_class = \
                dataset_series_classes.Traj_info(eef_traj,
                                                 delta_class_vector,
                                                 res, 
                                                 desired_effect, 
                                                 obtained_effect)
            dataset_size_class.add_inferred_traj(curr_init_pos, desired_effect, 
                                                 tmp_infer_traj_class)

            ## store delta know created during iteration
#            print("Inferred deltas for (", nb_effect, effect, ") : ", 
#                  len(tmp_inferred_discr_delta_vector))
            total_inferred_discr_delta_vector = \
                total_inferred_discr_delta_vector + tmp_inferred_discr_delta_vector

            ## compute current mean prob
#            traj_mean_prob = 0
#            for traj in eef_traj:
#                traj_mean_prob += traj[3]
#            traj_mean_prob = traj_mean_prob / len(eef_traj)
#            mean_prob += traj_mean_prob
            mean_prob = 0 ## TBDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD
            
            ## update statistics
            if res == 'success' and desired_effect == 'right':
                nb_success_r += 1
            elif res == 'success' and desired_effect == 'left':
                nb_success_l += 1
            elif res == 'success' and desired_effect == 'far':
                nb_success_u += 1
            elif res == 'success' and desired_effect == 'close':
                nb_success_d += 1
            elif res == 'false_pos' and desired_effect == 'right':
                nb_false_positive_r += 1
            elif res == 'false_pos' and desired_effect == 'left':
                nb_false_positive_l += 1                
            elif res == 'false_pos' and desired_effect == 'far':
                nb_false_positive_u += 1
            elif res == 'false_pos' and desired_effect == 'close':
                nb_false_positive_d += 1
            elif res == 'fail' and desired_effect == 'right':
                nb_fail_r += 1
            elif res == 'fail' and desired_effect == 'left':
                nb_fail_l += 1                
            elif res == 'fail' and desired_effect == 'far':
                nb_fail_u += 1
            elif res == 'fail' and desired_effect == 'close':
                nb_fail_d += 1
            nb_effect += 1

        curr_init_pos += 1
    
    if sim_param.print_stats:
        print('\nNumber of trajectories inferred :',  len(sim_param.effect_values) 
                                                * len(init_pos_vector))
        
        print('\nSuccess trajectories :', nb_success_r + nb_success_l +
                                                nb_success_u + nb_success_d)     
        
        print('\nFalse positive trajectories :', nb_false_positive_r + 
                                                nb_false_positive_l +
                                                nb_false_positive_u +
                                                nb_false_positive_d)
        
        print('\nFailed trajectories :', nb_fail_r + nb_fail_l +
                                                nb_fail_u + nb_fail_d)
    
    succ_value = sum([nb_success_r, nb_success_l, nb_success_u, nb_success_d]) 
    false_pos_value = sum([nb_false_positive_r, nb_false_positive_l,
                        nb_false_positive_u, nb_false_positive_d])
    fail_value = sum([nb_fail_r, nb_fail_l, nb_fail_u, nb_fail_d])
    
    mean_prob = mean_prob / (succ_value + false_pos_value + fail_value)
    dataset_size_class.set_inference_res([succ_value, 
                                          false_pos_value, 
                                          fail_value])
    dataset_size_class.set_mean_prob_move(mean_prob)
    
    return total_inferred_discr_delta_vector

'''
Infere traj to get a desired effect
'''
def infere_traj(bn, ie,
                init_pos_coord,
                init_pos_vector,
                initial_obj_pos_dict,
                expected_effect, 
                current_orien, current_inclin, current_dist,
                curr_nb_infere_trajs):
    
    eef_pos = ros_services.call_get_eef_pose('left')
    eef_pos = [round(pos, sim_param.round_value) for pos in eef_pos[0:3]]
    print("eef_pos", eef_pos)  
    
    eef_traj_vector = []
    obj_traj_vector = []
#    delta_vector = []
    inf_dicr = []
    delta_class_vector = []
    traj_res = ''
    obtained_effect = ''
    obj_moved = False
    
#    nb_executed_deltas = 0
    
#    while execution_active:
   
    ''' Simulate trajectory '''
    traj, obj_moved, final_obj_pos_dict = simulate_traj(bn, ie, 
                                   eef_pos,
                                   initial_obj_pos_dict,
                                   expected_effect, 
                                   current_orien,
                                   current_inclin,
                                   current_dist)

    ''' Plot simulated traj '''
    below_box = plot_traj_3d(init_pos_vector,
                             traj,
                             initial_obj_pos_dict,
                             init_pos_coord,
                             expected_effect)                                  
    print('below_box:', below_box)

    ''' Identify object positions'''
#    for obj_name in sim_param.obj_name_vector:
#        print('positions', initial_obj_pos_dict[obj_name], '->', final_obj_pos_dict[obj_name])
    #        obtained_effect = env.identify_effect(initial_obj_pos,
#                                              final_obj_pos)                                          
    ''' Identify effect '''
    print('expected_effect: ------------->', expected_effect.upper())
#        print('obtained_effect: ------------->', obtained_effect.upper())



    execution_active = True ## still trying to touch object
            
#    if expected_effect == obtained_effect and sim_param.exec_traj:
    if sim_param.exec_traj:
#        ''' Execute trajectory, listening to feedback '''
#        tc = Traj_callback(
#                traj, 
##                initial_obj_pos,
##                current_orien, current_inclin, current_dist,
##                eef_traj_vector, obj_traj_vector, delta_vector, inf_dicr,
##                delta_class_vector, traj_res,
##                expected_effect, obtained_effect,
##                nb_executed_deltas,
#                below_box)
    
        correct_traj = True    
    
        ## avoid touching table in REAL ROBOT
        if sim_param.real_robot and below_box:
            correct_traj = False
            print("-----------------> TRAJECTORY NOT EXECUTED !! TOUCHING THE TABLE!!! ")       

        if correct_traj:            
            ## execute trajectory
            simulated_traj_vector = [i for el in traj for i in el] ## [float]
            res_exec = ros_services.call_trajectory_motion(sim_param.feedback_window, 
                                                           simulated_traj_vector)
            if not res_exec:
                print("FAILED EXECUTION IN ROS !! ")
            else:
                print("FINISHED EXECUTION IN ROS !! ")                   
         
#    while tc.sub_up :
#        print('Nb of deltas executed :', tc.nb_executed_deltas)
            
            
#        import time
#        time.sleep(3)
#        sys.exit()
            
        ''' Values once trajectory execution stopped (but it can continue) '''
#        execution_active = (tc.traj_res == '')
#        nb_executed_deltas += tc.nb_executed_deltas
#        print("after exec", execution_active, nb_executed_deltas)
#        print ("Result obtained.", tc.traj_res)
        
#        eef_traj_vector = tc.eef_traj_vector
#        obj_traj_vector = tc.obj_traj_vector
#        delta_vector = tc.obj_traj_vector
#        inf_dicr = tc.traj_inferred_discr_delta_vector
#        delta_class_vector = tc.delta_class_vector
#        traj_res = tc.traj_res
#        expected_effect = tc.expected_effect
#        obtained_effect = tc.obtained_effect     
            
#        ## if max nb of deltas executed, fail
#        if traj_res == '':
#            traj_res = 'fail'
        
#        execution_active == tc.sub_up

    else: ## only simulate traj
#        eef_traj_vector = []
#        obj_traj_vector = []
#        delta_vector = []
#        inf_dicr = []
#        delta_class_vector = []
#        if expected_effect in obtained_effect:
#            traj_res = 'success'
#        elif obj_moved:
#            traj_res = 'false_pos'
#        else:
#            traj_res = 'fail'           
        execution_active == False
                 
    traj_res = 'fail'
    eef_traj_vector = []
    obj_traj_vector = []
    inf_dicr = []
    delta_class_vector = []

    return traj_res, \
            eef_traj_vector, obj_traj_vector, \
            delta_class_vector, \
            obtained_effect, \
            inf_dicr

'''
Simulate traj to get a desired effect
'''
def simulate_traj(bn, ie, 
                eef_pos,
                obj_pos_dict,
                expected_effect, 
                current_orien,
                current_inclin,
                current_dist,
                accurate_level = 1):
    
    if accurate_level > 1:
        print('ACCURATE TRAJECTORY:',accurate_level)
    
    eef_traj = [[round(eef_pos[0],2),
                 round(eef_pos[1], 2), 
                 round(eef_pos[2], 2)]]
    delta_vector = [] ## [[(orientation, inclination, distance)], next_mov_discr]
    all_obj_moved = False
    prev_mov_delta = [0,0,0]
    i = 0
    while not all_obj_moved and i < sim_param.inferred_max_moves*accurate_level:
        print('\nInferred delta ', i)
        
        current_eef_x = eef_traj[-1][0] ## last value added
        current_eef_y = eef_traj[-1][1]
        current_eef_z = eef_traj[-1][2]
        
        ## compute neighbours virtual positions
        nn_pos_vector =  [[current_eef_x,
                           current_eef_y,
                           current_eef_z]]        
        add_dist = sim_param.step_length/2*accurate_level
        for j in range(6):
            tmp_pos = [current_eef_x,
                       current_eef_y,
                       current_eef_z]
            if j == 0: ## front
                tmp_pos[0] = current_eef_x + add_dist
            elif j == 1: ## back
                tmp_pos[0] = current_eef_x - add_dist
            elif j == 2: ## right
                tmp_pos[1] = current_eef_y - add_dist                
            elif j == 3: ## left
                tmp_pos[1] = current_eef_y + add_dist                
            elif j == 4: ## up
                tmp_pos[2] = current_eef_z + add_dist
            elif j == 5: ## down
                tmp_pos[2] = current_eef_z - add_dist
            tmp_pos = [round(curr_pos, sim_param.round_value) 
                             for curr_pos in tmp_pos]
#            print(j, tmp_pos)
            nn_pos_vector.append(tmp_pos)
        
        ## compute their prob for next move
        virt_inference_vector = []
        for curr_virt_pos in nn_pos_vector:        
            current_eef_x_tmp = curr_virt_pos[0]
            current_eef_y_tmp = curr_virt_pos[1]
            current_eef_z_tmp = curr_virt_pos[2]
            
            ## compute current variables value
            node_names = ['effect']
            node_values = [expected_effect]
            obj_id = 0
            for obj_name in sim_param.obj_name_vector:                                
                distance = discr_dist.compute_distance(
                    [current_eef_x_tmp,current_eef_y_tmp], 
                    obj_pos_dict[obj_name],
                    current_dist)
                
                orientation = discr_orien.compute_orientation_discr(
                    [current_eef_x_tmp,current_eef_y_tmp], 
                    obj_pos_dict[obj_name],
                    current_orien)
                    
                inclination = discr_inclin.compute_inclination_discr(
                    obj_pos_dict[obj_name],
                    [current_eef_x_tmp,current_eef_y_tmp,current_eef_z_tmp],             
                    current_inclin)                    
                    
                node_names += ['distance'+str(obj_id),
                               'orientation'+str(obj_id),
                               'inclination'+str(obj_id)]
                node_values += [distance, orientation, inclination]
                
                obj_id += 1
                
            ## infere next move            
            try:
                next_mov_discr, prob_value, delta_nb_var, same_prob = \
                    infere_mov(bn, ie,                                        
                               node_names,
                               node_values)
            except Exception as e: 
                if sim_param.debug_infer:
                    print('-------------------------------> UNKNOWN LABEL WHILE INFERRING!!!', e)
                    print([curr_virt_pos, 
                           node_values])
                virt_inference_vector.append([curr_virt_pos, 
                                             '',
                                             '', 
                                             0]) ## to avoid being selected
                continue
            
            if sim_param.debug_infer:
                print([curr_virt_pos, 
                       node_values,
                       next_mov_discr, 
                       prob_value])
            virt_inference_vector.append([curr_virt_pos, 
                                         node_values,
                                         next_mov_discr, 
                                         prob_value])                
        
        ## check if same prob for all close points
        prob_found = all([x[-1]==virt_inference_vector[0][-1] for x in virt_inference_vector])
        if prob_found:
            print('-------------------------------> SAME PROB', 
                  virt_inference_vector[0][-1])
            if virt_inference_vector[0][-1] == 0:
                print('-------------------------------> NO MOVE INFERRED')
                print(node_values)
                return eef_traj, False, ''

        print(node_values)
                
        ## MOVE 
        ## with higher prob    
        max_prob = virt_inference_vector[0][3]
        max_next_move = virt_inference_vector[0][2]
        max_pos = 0
        tmp_i = 1
        for tmp_infer_values in virt_inference_vector[1:]:
            if tmp_infer_values[3] > max_prob:
                max_prob = tmp_infer_values[3]
                max_next_move = tmp_infer_values[2]
                max_pos = tmp_i
            tmp_i += 1
        print('Next move for pos', max_next_move.upper(), max_prob, 'in pose', max_pos)
        next_mov_discr = [max_next_move]

#        if sim_param.debug_infer:
#            print(expected_effect.upper(),
#                  orientation.upper(),
#                  inclination.upper(),
#                  distance.upper(), 
#                  "-> ", 
#                  next_mov_discr.upper(), 
#                  "with probability", prob_value)
        
        ## compute displacement regarding move or moves        
#        print('next_mov_discr', next_mov_discr)
        delta_x = 0
        delta_y = 0
        delta_z = 0
        for curr_move in next_mov_discr:
            move_coord = 1 ## if move in this coord
            if 'far' in curr_move:
                delta_x = move_coord
            elif 'close' in curr_move:
                delta_x = -move_coord
            if 'right' in curr_move:
                delta_y = -move_coord
            elif 'left' in curr_move:
                delta_y = move_coord
            if 'up' in curr_move:
                delta_z = move_coord
            elif 'down' in curr_move:
                delta_z = -move_coord
                
            ## length of the movement must be equal to step_length
            dims_vector = abs(delta_x) + abs(delta_y) + abs(delta_z)
            if dims_vector == 0: ## to avoid div by 0 ??? TODOOOOOOOOOOOOOOOOOOOOOOO
                dims_vector = 1
            mov_step = round(sim_param.step_length/sqrt(dims_vector), sim_param.round_value)
            if delta_x > 0: #== 1 :
                delta_x = mov_step
            elif delta_x < 0: #== 1 : == -1:
                delta_x = -mov_step
            if delta_y > 0: #== 1 : == 1 :
                delta_y = mov_step
            elif delta_y < 0: #== 1 : == -1:
                delta_y = -mov_step
            if delta_z > 0: #== 1 : == 1 :
                delta_z = mov_step
            elif delta_z < 0: #== 1 : == -1:
                delta_z = -mov_step            
                
        print("Delta :", delta_x, delta_y, delta_z, 
              'Norm:', round(d.euclidean([0, 0, 0],
                                          [delta_x, delta_y, delta_z]), 3))
                                          
        ## move eef to new position
        next_eef_x = round(current_eef_x + delta_x/len(next_mov_discr), sim_param.round_value)# + prev_mov_delta[0]/2, sim_param.round_value)
        next_eef_y = round(current_eef_y + delta_y/len(next_mov_discr), sim_param.round_value)# + prev_mov_delta[1]/2, sim_param.round_value)
        next_eef_z = round(current_eef_z + delta_z/len(next_mov_discr), sim_param.round_value)# + prev_mov_delta[2]/2, sim_param.round_value)
#        print('Prev mov delta', prev_mov_delta)
        print('Previous EEF pos :', 
              current_eef_x, current_eef_y, current_eef_z)
        print('New EEF pos :', 
              next_eef_x, next_eef_y, next_eef_z)
        
        eef_traj.append([next_eef_x, next_eef_y, next_eef_z])
#        delta_vector.append([orientation, inclination, distance, 
#                             next_mov_discr])
        delta_vector.append(node_values[1:] + [next_mov_discr])        

        ## check if obj was moved
#        obj_moved = env.check_obj_moved(
#                        obj_pos,
#                        [current_eef_x,current_eef_y,current_eef_z], 
#                        [next_eef_x,next_eef_y,next_eef_z])
        obj_moved_pose_dict = OrderedDict()
        for obj_name in sim_param.moved_obj_name_vector:
            obj_pos = obj_pos_dict[obj_name]
            obj_moved, obj_moved_pose = \
                env.compute_obj_pos(
                    obj_pos,
                    [current_eef_x,current_eef_y,current_eef_z], 
                    [next_eef_x,next_eef_y,next_eef_z])
            obj_moved_pose_dict[obj_name] = (obj_moved,obj_pos)

        tmp_all_obj_moved = True
        for name, moved in obj_moved_pose_dict.items():
            print(name, "moved ? :", moved[0])
            tmp_all_obj_moved = tmp_all_obj_moved and moved[0]
        all_obj_moved = tmp_all_obj_moved
#        prev_mov_delta = [delta_x,
#                          delta_y,
#                          delta_z]
        i += 1
        
        ## end delta

    return eef_traj, all_obj_moved, obj_moved_pose


'''
a
'''
def plot_traj_3d(init_pos_vector,                 
                 traj,
                 obj_pos_dict,
                 init_pos_coord,
                 effect):

    # plot figure
    fig = plt.figure(figsize=(7,7))
    fig.clf()
#    fig.canvas.set_window_title(str(curr_init_pos) + '->' + effect)
#    fig.canvas.set_window_title('-> ' + effect)
    ax = Axes3D(fig)

#    # table        
#    ax.bar3d([.3], [-.8], [-.1-obj_side/2], 
#             [.8], [1.6], [.0011], 
#             color='brown', alpha=0.2,
#             edgecolor = 'lightgrey')                  
    
    # box init_pos
    obj_pos = obj_pos_dict['cube']
    ax.bar3d(obj_pos[0] - 0.085/2, 
             obj_pos[1] - 0.07/2, 
             obj_pos[2] - 0.08/2 + 0.02, 
             [.085], [.07], [.08], 
             color='green',
             alpha=0.2,
             edgecolor='none')             
             
#             
#    # box final_pos
#    obj_final_pos = traj.get_obj_final_pos()
#    ax.bar3d(obj_final_pos[0] - obj_side/2, 
#             obj_final_pos[1] - obj_side/2, 
#             obj_final_pos[2] - obj_side/2, 
#             [.06], [.06], [.075], 
#             color=traj_color,
#             alpha=0.2,
#             edgecolor='none')                     

    # limits
    lim = .3
    ax.set_xlim3d([obj_pos[0]-lim, obj_pos[0]+lim])
    ax.set_ylim3d([obj_pos[1]-lim, obj_pos[1]+lim])
    ax.set_zlim3d([obj_pos[2]-lim, obj_pos[2]+lim])

    if len(sim_param.obj_name_vector) > 1:
        # cylinder
        ## http://stackoverflow.com/questions/39822480/plotting-a-solid-cylinder-centered-on-a-plane-in-matplotlib
        height = 0.09
        obj_pos = obj_pos_dict['cylinder']
        p0 = np.array([obj_pos[0], obj_pos[1], obj_pos[2] - height/2]) #point at one end
        p1 = np.array([obj_pos[0], obj_pos[1], obj_pos[2] + height/2]) #point at other end
        R = 0.035
        v = p1 - p0
        mag = norm(v)
        v = v / mag
        not_v = np.array([1, 0, 0])
        if (v == not_v).all():
            not_v = np.array([0, 1, 0])
        n1 = np.cross(v, not_v)
        n1 /= norm(n1)
        n2 = np.cross(v, n1)
        t = np.linspace(0, mag, 2)
        theta = np.linspace(0, 2 * np.pi, 100)
        rsample = np.linspace(0, R, 2)
        t, theta2 = np.meshgrid(t, theta)
        rsample,theta = np.meshgrid(rsample, theta)
        # "Tube"
        X, Y, Z = [p0[i] + v[i] * t + R * np.sin(theta2) * n1[i] + R * np.cos(theta2) * n2[i] for i in [0, 1, 2]]
        # "Bottom"
        X2, Y2, Z2 = [p0[i] + rsample[i] * np.sin(theta) * n1[i] + rsample[i] * np.cos(theta) * n2[i] for i in [0, 1, 2]]
        # "Top"
        X3, Y3, Z3 = [p0[i] + v[i]*mag + rsample[i] * np.sin(theta) * n1[i] + rsample[i] * np.cos(theta) * n2[i] for i in [0, 1, 2]]        
        ax.plot_surface(X, Y, Z, color='blue', linewidth=0, alpha=0.2)
        ax.plot_surface(X2, Y2, Z2, color='blue', linewidth=0, alpha=0.2)
        ax.plot_surface(X3, Y3, Z3, color='blue', linewidth=0, alpha=0.2)           

    # robot
    robot_width = .2
    robot_height = .6
    robot_length = .4
    ax.bar3d(-robot_width/2, 
             -robot_length/2, 
             -robot_height/2, 
             robot_width, robot_length, robot_height, 
             color='red',
             alpha=0.2,
             edgecolor='none')
    
    # labels
    plt.xlabel('x-axis')
    plt.ylabel('y-axis')
#            plt.zlabel('zlabel')
    
    ## view
    ## All commented = diagonal view
#    ax.view_init(90,180) # top view
#    ax.view_init(0,0) # front view
    ax.view_init(0,270) # left view

    # plot initial position
    list_x_axis = [pos[0] for pos in init_pos_vector]
    list_y_axis = [pos[1] for pos in init_pos_vector]
    list_z_axis = [pos[2] for pos in init_pos_vector]
    ax.plot(list_x_axis,
            list_y_axis,
            list_z_axis,
            'o',
            color='red',
            markersize = 5)

    ## plot traj from init pos        
    eef_pos_vector_x = [pos[0] for pos in traj]
    eef_pos_vector_y = [pos[1] for pos in traj]
    eef_pos_vector_z = [pos[2] for pos in traj]    
    
    ax.plot(eef_pos_vector_x, 
            eef_pos_vector_y, 
            eef_pos_vector_z,
            '-',
            color = 'blue')
    ax.plot(eef_pos_vector_x, 
            eef_pos_vector_y, 
            eef_pos_vector_z,
            '*',                                     
            markersize=3,
            c='grey')
        
    plt.show()
    
    ## check if trajectory under table
    z_under_object = False
    pos = 0
    while not z_under_object and pos < len(eef_pos_vector_z):
        z_under_object = eef_pos_vector_z[pos] < -0.17
        pos += 1    

    return z_under_object

'''
Find pos of value given label 
Output: pos
'''
def findPosLabel(rand_var,label):
    label_found = False
    pos = 0
    while not label_found and pos < rand_var.domainSize():
        if rand_var.label(pos) == label:
            label_found = True
            return pos
        pos += 1

    if not label_found:
        return -1
        
'''
Infere next movement

bn = current bn
ie = inference setup
discr_values = 

'''
def infere_mov(bn, ie, node_names, node_values):    

    same_prob = False

    ''' Add soft evidence '''        
    ## generate dict with current node values to apply
    values_dict = OrderedDict()
    for i in range(len(node_names)): 
        current_node = node_names[i]
        current_value = node_values[i]
        values_dict[current_node] = current_value

    if sim_param.debug_infer:
        for i in values_dict:
            print (i, values_dict[i])

    ## transform each node values for the correspondant array 
    ## with 1 in the position of the current node value
    ## ej : if the possible node values are a b c, and we want to set c
    ## then it is 0 0 1
    for key in values_dict:        
        random_var = bn.variable(bn.idFromName(key))
        pos = findPosLabel(random_var,values_dict[key])
        if pos==-1:
            if sim_param.debug_infrt:
                print("Label not found for",str(key),values_dict[key])
        else:
            tmp=np.zeros(random_var.domainSize())
            for p in range(len(tmp)):
                if p==pos:
                    tmp[p] = 1
                else:
                    tmp[p] = 0
            values_dict[key] = tmp.tolist()
        
    ie.setEvidence(values_dict)
    
    ''' Infere next movement to be done '''    
    ie.makeInference()
    
    ## get posterior    
    posterior = ie.posterior(bn.idFromName('move'))
#    if sim_param.debug:
#    print('\n')
    print(posterior)
#    print(posterior[0])
#    print(posterior[-1])
#    print(type(posterior))
#    print(type(posterior[0]))
#    print(posterior.variablesSequence()[0])
#    print(type(posterior.variablesSequence()[0]))
#    print(posterior.variablesSequence()[0].labels())
    

#    ## check if all values of 'move' have been generated
    move_var = bn.variable(bn.idFromName('move'))    
    move_nb_var = move_var.domainSize()
#    if move_nb_var != len(sim_param.move_values):
#        print('ERROR - infere_mov- Some move value is not available in the dataset')
#        sys.exit()

    ## get used moves
    move_list = []
    for pos in range(move_nb_var):
        move_list.append((posterior.variablesSequence())[0].label(pos))
        pos += 1

    ## select value of next move
    ## check if some prob is very high, or if 2 variables have a similar value
    posterior_list = []
    for i in range(len(move_list)):
        posterior_list.append(posterior[i])
        
    if all(x==posterior_list[0] for x in posterior_list):## no prob inferred
        posterior_pos = random.randint(0,len(posterior_list)-1)
        posterior_value = posterior_list[posterior_pos]
        same_prob = True
    else:
        max_value = max(posterior_list)
        posterior_pos = posterior_list.index(max_value)
        posterior_value = posterior_list[posterior_pos]
        max_pos = posterior_list.index(max_value)
        
        ## choose posterior value
        if max_value > 0.9: ## very high
            posterior_pos = max_pos              
        else: ## several probs
            posterior_list[max_pos] = 0
            second_max_value = max(posterior_list)
            second_max_pos = posterior_list.index(max(posterior_list))

#            if max_value > 2*second_max_value: ## far probs
#                posterior_pos = max_pos
#            else: ## close probs
#                posterior_pos = random.choice([max_pos, second_max_pos])

            if max_value == second_max_value:
                posterior_pos = random.choice([max_pos, second_max_pos])
            else:
                posterior_pos = max_pos

        ## assign posterior value            
        if posterior_pos == max_pos:
            posterior_value = max_value
        else:
            posterior_value = second_max_value
            
    move_value = (posterior.variablesSequence())[0].label(posterior_pos)    
    if sim_param.debug:
        print("Best posterior of move is", move_value.upper(),\
            "with prob", posterior_value, "in position", posterior_pos, '\n')      
    
    return move_value, round(posterior_value, 5), int(move_nb_var), same_prob
    
''' 
Compute score for each algorithm of a dataset type
Score = mean value of the perfomance value of an algorithm for a dataset
Example : for extended, hand-coded, mean perf_value when nb_init_pos = 4, 8, etc.
'''
def compute_score_norm(dataset_stat,
                  learn_algo_vector):
    dataset_score = 0
    for current_algo in learn_algo_vector:
        added_norm_perf = 0
        current_algo_class = dataset_stat.get_algo_results(current_algo)
        current_traj_dict = current_algo_class.get_results_dict()        
        for nb_init_pos, dataset_size_res_class in current_traj_dict.items():
            nb_trajs = sum(dataset_size_res_class.get_inference_res())
            max_performance = sim_param.perf_success_value*nb_trajs
            added_norm_perf += \
                (float(dataset_size_res_class.get_global_performance_value())/max_performance)*100
        algo_score = (added_norm_perf/len(current_traj_dict.keys()))
        current_algo_class.set_algo_score(algo_score)
        dataset_score += algo_score
    
    dataset_stat.set_dataset_score(float(dataset_score)/len(learn_algo_vector))

'''
Check if the posterior score is better than the previous one
'''
def check_better_score_norm(curr_score_vector, 
                       prev_score_vector):
            
    ## global score higher 
    if curr_score_vector[0] == prev_score_vector[0]:
        return 'equal'
    elif curr_score_vector[0] > prev_score_vector[0]:
        return 'bigger'
    else:
        return 'smaller'


'''
Compute cumulated performance value of each init position
given a dataset and a dicretization configuration
Store vector representing for each dataset, for each (algo, nb_init_pos) 
each position of the vector the perf_value of a init_pos, in range (0,nb_init_pos)
'''
def compute_perf_value_for_init_pos(current_dataset,
                                   current_algo,
                                   nb_init_pos_vector):

    max_added_perf_value = sim_param.perf_success_value * \
                            len(sim_param.effect_values)

    dataset_cumulated_perf_value_dict = OrderedDict()
    current_infer_algo_class = current_dataset.get_algo_results(current_algo)
    current_infer_algo_dict = current_infer_algo_class.get_results_dict()
    for current_nb_init_pos in nb_init_pos_vector: ## 8, 16, 32
        if sim_param.debug:
            print("current_dataset", current_dataset.get_dataset_name())
            print("current_nb_init_pos", current_nb_init_pos)
        current_dataset_size_class = \
            current_infer_algo_dict[current_nb_init_pos]
        current_dataset_size_dict = \
            current_dataset_size_class.get_indiv_traj_info()
        cumulated_perf_value_vector = []
        for current_init_pos in range(current_nb_init_pos):
            if sim_param.debug:
                print("current_init_pos", current_init_pos)
            added_perf_value = 0
            for current_effect in sim_param.effect_values:
                current_pos_effect_perf = \
                    current_dataset_size_dict[(current_init_pos, 
                                         current_effect)].get_perf_value()
                added_perf_value += current_pos_effect_perf
            added_perf_value = (float(added_perf_value)/max_added_perf_value)*100            
            if sim_param.debug:                
                print("Cumulated perf value", added_perf_value)
            cumulated_perf_value_vector.append(added_perf_value)
            
        dataset_cumulated_perf_value_dict[current_nb_init_pos] = \
            cumulated_perf_value_vector
            
    return dataset_cumulated_perf_value_dict