# -*- coding: utf-8 -*-
"""
@author: maestre
"""
from __future__ import print_function

import numpy as np
import random
import pyAgrum as agrum
from collections import OrderedDict

import os, sys
run_path = os.path.realpath(os.path.abspath(os.path.join('..', '..')))
sys.path.append(run_path)
lib_path = os.path.realpath(os.path.abspath(os.path.join('..', 'a2l_core_lib')))
sys.path.append(lib_path)
import environment_dynamics as env
import environment_setup as setup
import ros_services
import inf_traj_series_dataset_classes as dataset_series_classes
import simulation_parameters as sim_param
import traj_callback as real_exec
import inference as infer

if sim_param.discr_hand_coded:
    import discretize_orientation_hand_coded as discr_orien
    import discretize_distance_hand_coded as discr_dist
else:
    import discretize_orientation_sections as discr_orien
    import discretize_inclination_sections as discr_inclin
    import discretize_distance_sections as discr_dist

import rospy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#from matplotlib.patches import Circle
import mpl_toolkits.mplot3d.art3d as art3d
from scipy.linalg import norm
import copy

from scipy.spatial import distance as d
from math import sqrt

import operator

'''
Infere trajs for a dataset, algo and nb_init_pos, 
for each initial position of the eef and for each goal
'''
def affordance_validation(current_results_folder,
                        bn, 
                        alg,
                        dataset_size_class,
                        dataset_string,
                        nb_initial_pos,
                        current_iteration,
                        current_orien,
                        current_inclin,
                        current_dist,
                        left_gripper_interface):   
                            
    ''' To infer next move '''
    ie = agrum.LazyPropagation(bn) 
    
    ''' Initiqlize statistics '''
    nb_success = 0
    nb_false_positive = 0
    nb_fail = 0    
    mean_iter_prob = 0    
    
    print(' ')    
    
    ''' Restart env '''
    if sim_param.experiment_type == 'a2l_reproduce_dataset': ## LbD
        if current_iteration == 0:
            sim_param.radio = sim_param.untucked_left_eef_pos[0] - sim_param.first_obj_pos[0]     
            
            ''' Restart scenario '''
            if sim_param.real_robot:
                l_eef_init_pos_x = rospy.get_param("/eef_left_init_pos_x")
                l_eef_init_pos_y = rospy.get_param("/eef_left_init_pos_y")
                l_eef_init_pos_z = rospy.get_param("/eef_left_init_pos_z")
                success = ros_services.call_move_to_initial_position(
                            [l_eef_init_pos_x,
                            l_eef_init_pos_y,
                            l_eef_init_pos_z]) 
                if not success:
                    print("ERROR - call_move_to_initial_position failed")    
            else:            
                success = ros_services.call_restart_world("all")
                if not success:
                    print("ERROR - restart_world failed")  
    else: ## A2L
        if current_iteration == 0:
            ''' Restart scenario '''         
            success = ros_services.call_restart_world("all")
            if not success:
                print("ERROR - restart_world failed")
                
    ''' Get all object pos'''
    obj_pos_dict = OrderedDict()
    for obj_name in sim_param.obj_name_vector:
        obj_pos = ros_services.call_get_model_state(obj_name)
        obj_pos = [round(pos, sim_param.round_value) for pos in obj_pos[0:3]]
        if sim_param.real_robot:
            if obj_name == 'cube':
                obj_pos[2] = -0.09
            else:
                obj_pos[2] = -0.08
        obj_pos_dict[obj_name] = obj_pos
        
    ''' Get first obj pos to compute eef initial pos'''
#    first_obj_pos = obj_pos_dict[sim_param.obj_name_vector[0]]
    for name,pos in obj_pos_dict.items():
        if sim_param.debug_infer:
            print(name, '-->', pos)
    
    ## create initial positions of eef
    _list_x_axis, _list_y_axis, _list_z_axis = \
        setup.gen_init_eef(
            nb_initial_pos,
            sim_param.radio,
            obj_pos_dict[sim_param.obj_name_vector[0]])

    if len(_list_x_axis) > 1:
        init_pos_vector = list(
            zip(_list_x_axis, 
                _list_y_axis, 
                _list_z_axis))
    else:
        init_pos_vector = [[_list_x_axis[0],
                           _list_y_axis[0],
                           _list_z_axis[0]]]

#    ######################################################################################################3
#    ## TODO THIS SHOULD BE IN A DIFFERENT FUNCTION
#    ## From second iteration
#    if sim_param.experiment_type == 'a2l_reproduce_dataset' and current_iteration > 0:
#        ''' Update initial eef pos '''
#        eef_init_pos = ros_services.call_get_eef_pose('left')
#        eef_init_pos = [round(pos, sim_param.round_value) for pos in eef_init_pos[0:3]]          
#    
#        ## eef init pos is related to obj pos    
#        eef_init_pos = copy.copy(init_pos_vector[0])
#        eef_init_pos[0] = first_obj_pos[0] + random.uniform(-sim_param.new_obj_pos_dist*2,
#                                           sim_param.new_obj_pos_dist)
#        if first_obj_pos[1] > eef_init_pos[1]:
#            eef_init_pos[1] = eef_init_pos[1] + random.uniform(0,sim_param.new_obj_pos_dist)
#        elif first_obj_pos[1] < eef_init_pos[1]:
#            eef_init_pos[1] = eef_init_pos[1] + random.uniform(-sim_param.new_obj_pos_dist/2,0)
#        eef_init_pos[2] = eef_init_pos[2] + random.uniform(-sim_param.new_obj_pos_dist/2,0)
#
#        res_init_pos = ros_services.call_move_to_initial_position(eef_init_pos)        
#        ## TODO check result        
#        if sim_param.debug_infer:
#            print("eef_init_pos", eef_init_pos)
#        
#        ''' Change pos first object close to eef '''
#        ## if obj too far from end_effector put it around the center of the table
#        ## and the eef close 
##        ''' Get obj pos '''
##        obj_pos_dict = dict()
##        for obj_name in sim_param.obj_name_vector:
##            obj_pos = ros_services.call_get_model_state(obj_name)    
##            obj_pos = [round(pos, sim_param.round_value) for pos in obj_pos[0:3]]    
##            obj_pos[2] = -0.13
##            obj_pos_dict[obj_name] = obj_pos
##            print(obj_name, ':', obj_pos)
#        first_obj_pos = obj_pos_dict[sim_param.obj_name_vector[0]]
#        if sim_param.debug_infer:
#            print('euclidean(obj_pos, eef_init_pos)', 
#                  d.euclidean(first_obj_pos, eef_init_pos))
#              
#        if d.euclidean(first_obj_pos, init_pos_vector[0]) > sim_param.obj_too_far_distance:
#            if sim_param.debug_infer:
#                print('-------------> UPDATING OBJECT POSITIONS')
#            ## compute new obj pos
#            new_obj_pos = sim_param.first_obj_pos
#            new_obj_pos = [new_obj_pos[0] + random.uniform(-sim_param.new_obj_pos_dist,
#                                                           sim_param.new_obj_pos_dist),
#                           new_obj_pos[1] + random.uniform(-sim_param.new_obj_pos_dist,
#                                                           sim_param.new_obj_pos_dist),
##                           -0.14]
#                           new_obj_pos[2]]
#            new_obj_pos = [round(pos, sim_param.round_value) for pos in new_obj_pos]
#            obj_pos = copy.copy(new_obj_pos)
#            
#            ## move obj to new pos    
#            success = ros_services.call_restart_world("object",
#                                                      sim_param.obj_name_vector[0],
#                                                      new_obj_pos)                                                                          
#            if not success:
#                print("ERROR - restart_world failed for object", sim_param.obj_name_vector[0])
#                
#            ## move rest of the objects, related to first object pos
#            for obj_name in sim_param.obj_name_vector[1:]:
#                if obj_name == 'cylinder':
#                    obj_pos_dict[obj_name] = \
#                        [new_obj_pos[0] + sim_param.new_obj_pos_dist,
#                         new_obj_pos[1] - sim_param.new_obj_pos_dist,
##                         -0.14]
#                         new_obj_pos[2]]      
#                    ## move obj to new pos    
#                    success = ros_services.call_restart_world("object",
#                                                      obj_name,
#                                                      obj_pos_dict[obj_name])
#            if not success:
#                print("ERROR - restart_world failed for object", obj_pos_dict[obj_name])
#        
#        for name,pos in obj_pos_dict.items():
#            if sim_param.debug_infer:
#                print(name,':', obj_pos)
#       
#    ######################################################################################################   

    ## for each initial position infere / plot / save a traj
    nb_init_pos = len(init_pos_vector)
    total_inferred_discr_delta_vector = [] ## delta knowledge created during evaluations

    for curr_init_pos in range(nb_init_pos):
        init_pos_coord = init_pos_vector[curr_init_pos]
        
        ## for each effect
        for desired_effect in sim_param.effect_values:

            print('\n////////////////////')
            print('NEW TRAJ for init_pos', curr_init_pos, #init_pos_coord,
                  'effect', desired_effect.upper())            
                         
            ''' Update all object pos'''
            obj_pos_dict = OrderedDict()
            for obj_name in sim_param.obj_name_vector:
                obj_pos = ros_services.call_get_model_state(obj_name)
                obj_pos = [round(pos, sim_param.round_value) for pos in obj_pos[0:3]]
                if sim_param.real_robot:
                    if obj_name == 'cube':
                        obj_pos[2] = -0.09
                    else:
                        obj_pos[2] = -0.08            
                obj_pos_dict[obj_name] = obj_pos
            
            ''' Update init pos list '''
            _list_x_axis, _list_y_axis, _list_z_axis = \
                setup.gen_init_eef(
                    nb_initial_pos,
                    sim_param.radio,
                    obj_pos_dict[sim_param.obj_name_vector[0]])            
            if len(_list_x_axis) > 1:
                init_pos_vector = list(
                    zip(_list_x_axis, 
                        _list_y_axis, 
                        _list_z_axis))
            else:
                init_pos_vector = [[_list_x_axis[0],
                                   _list_y_axis[0],
                                   _list_z_axis[0]]]
            init_pos_coord = init_pos_vector[curr_init_pos]  

            ''' Update eef pos'''            
            ros_services.call_move_to_initial_position([0.65,0.1,0.1])
            success = ros_services.call_move_to_initial_position(init_pos_coord) 
            if not success:
                print("ERROR - call_move_to_initial_position failed")   
                        
            eef_pos = ros_services.call_get_eef_pose('left')
            eef_pos = [round(pos, sim_param.round_value) for pos in eef_pos[0:3]]
#            left_gripper_interface.close()
            
            ## Folder path to save traj
            if sim_param.save_trajs or \
                sim_param.save_some_trajs or \
                sim_param.plot_trajs or \
                sim_param.plot_some_trajs:
                tmp_plot_bool = True
                
                if sim_param.plot_some_trajs or sim_param.save_some_trajs:
                    current_pos = \
                        round((float(nb_init_pos)/len(init_pos_vector))*100)
                    if not current_pos == 0 and \
                        not current_pos == 75:
                            tmp_plot_bool = False
                
                if tmp_plot_bool:
                    filepath = current_results_folder + \
                                dataset_string + '_' + \
                                str(current_iteration) + '_' + \
                                alg + '_' + \
                                str(curr_init_pos) + '_' + \
                                desired_effect
            ## infere traj
            res, eef_traj, obj_traj, \
            delta_class_vector, obtained_effect, \
            tmp_inferred_discr_delta_vector, \
            mean_traj_prob = \
                check_affordance(bn, ie,
                                init_pos_coord, ## [X, Y, Z]
                                init_pos_vector,
                                obj_pos_dict,
                                desired_effect, 
                                current_orien, current_inclin, current_dist,
                                filepath)

            ## store current inferred traj
            tmp_infer_traj_class = \
                dataset_series_classes.Traj_info(eef_traj,
                                                 delta_class_vector,
                                                 res, 
                                                 desired_effect, 
                                                 obtained_effect)
            dataset_size_class.add_inferred_traj(curr_init_pos, desired_effect, 
                                                 tmp_infer_traj_class)

            ## store delta knowledge created during iteration
            total_inferred_discr_delta_vector = \
                total_inferred_discr_delta_vector + tmp_inferred_discr_delta_vector        
            
            ## update statistics
            mean_iter_prob += mean_traj_prob
            if res == 'success':
                nb_success += 1
            elif res == 'false_pos':
                nb_false_positive += 1
            elif res == 'fail':
                nb_fail += 1

        curr_init_pos += 1
    
    dataset_size_class.set_inference_res([nb_success, 
                                          nb_false_positive, 
                                          nb_fail])
    dataset_size_class.set_mean_prob_move(mean_iter_prob)
    
    return total_inferred_discr_delta_vector, \
            mean_iter_prob

'''
Infere traj to get a desired effect
'''
def check_affordance(bn, ie,
                    init_pos_coord,
                    init_pos_vector,
                    obj_pos_dict,
                    expected_effect, 
                    current_orien, current_inclin, current_dist,
                    filepath):
 
    eef_traj_vector = []
    obj_traj_vector = []
    inf_dicr = []
    delta_class_vector = []
    traj_res = ''
    obtained_effect = ''
   
    ''' Simulate trajectory '''
    traj, sim_obj_moved, mean_traj_prob = \
        simulate_traj(bn, ie, 
                       init_pos_coord,
                       obj_pos_dict,
                       expected_effect, 
                       current_orien,
                       current_inclin,
                       current_dist)

    ''' Plot simulated traj '''
    if sim_param.plot_trajs or sim_param.save_trajs:
        plot_save_traj_3d(init_pos_vector,
                          traj,
                          obj_pos_dict,
                          init_pos_coord,
                          expected_effect,
                          filepath,
                          sim_obj_moved)                                                  
        
    ''' Check if under box '''
    ## check if trajectory under table
    below_box = False
    pos = 0
    while not below_box and pos < len(traj):
        below_box = traj[pos] < -0.17
        pos += 1  
    if sim_param.debug_infer:
        print('\nbelow_box:', below_box)        

    ''' Execute trajectory, listening to feedback '''
    tc = real_exec.Traj_callback(
            traj,
            sim_obj_moved,
            current_orien, current_inclin, current_dist,
            below_box)
        
    eef_traj_vector = tc.eef_traj_vector
    obj_traj_vector = tc.obj_traj_vector
    inf_dicr = tc.traj_inferred_discr_delta_vector
    delta_class_vector = tc.delta_class_vector

    obtained_effect = tc.obtained_effect     

    ''' Identify effect '''
    if obtained_effect != "" and obtained_effect !=  None and obtained_effect != '':
        print('real obtained_effect: ------------->', obtained_effect.upper())
        
#        if expected_effect in obtained_effect:
        if expected_effect == obtained_effect:
            traj_res = 'success'
        else:
            if sim_obj_moved:
                traj_res = 'false_pos'
            else:
                traj_res = 'fail'
    else:
        traj_res = 'fail'
        print('real obtained_effect not moved')

    ''' Update scenario '''
    curr_obj_pos = obj_pos_dict['cube']
    if sim_param.debug_infer:
        print('euclidean dist:', d.euclidean(curr_obj_pos, sim_param.first_obj_pos), \
                                '<', sim_param.obj_too_far_distance)                     
    if d.euclidean(curr_obj_pos, 
                   sim_param.first_obj_pos) > sim_param.obj_too_far_distance:        
        print('-------------> UPDATING CUBE POSITION!')

        ## move arm up
        eef_pos = ros_services.call_get_eef_pose('left')
        eef_pos = [round(poss, sim_param.round_value) for poss in eef_pos[0:3]]
        pos_up = copy.copy(eef_pos)
        pos_up[2] += 0.2
        success = ros_services.call_move_to_initial_position(pos_up) 
        if not success:
            print("ERROR - extend dataset failed")
        
        ## compute new obj pos
        new_obj_pos = sim_param.first_obj_pos
        new_obj_pos = [new_obj_pos[0] + random.uniform(-sim_param.new_obj_pos_dist,
                                                       sim_param.new_obj_pos_dist),
                       new_obj_pos[1] + random.uniform(-sim_param.new_obj_pos_dist,
                                                       sim_param.new_obj_pos_dist),
                       new_obj_pos[2]]
        new_obj_pos = [round(poss, sim_param.round_value) for poss in new_obj_pos]
        
        ## move obj to new pos    
        success = ros_services.call_restart_world("object",
                                                  sim_param.obj_name_vector[0],
                                                  new_obj_pos)        
        if not success:
            print("ERROR - restart_world failed for object", sim_param.obj_name_vector[0])    

    return traj_res, \
            eef_traj_vector, obj_traj_vector, \
            delta_class_vector, \
            obtained_effect, \
            inf_dicr, \
            mean_traj_prob

'''
Simulate traj to get a desired effect
'''
def simulate_traj(bn, ie, 
                eef_pos,
                obj_pos_dict,
                expected_effect, 
                current_orien,
                current_inclin,
                current_dist):
    
    eef_traj = [[round(eef_pos[0], sim_param.round_value),
                 round(eef_pos[1], sim_param.round_value), 
                 round(eef_pos[2], sim_param.round_value)]]
    all_obj_moved = False
    mean_traj_prob = 0
    i = 0
    while not all_obj_moved and i < sim_param.inferred_max_moves:
        if sim_param.debug_infer:
            print('\nInferred delta ', i)
        
        current_eef_x = eef_traj[-1][0] ## last value added
        current_eef_y = eef_traj[-1][1]
        current_eef_z = eef_traj[-1][2]
        
        virt_inference_vector = []
        if sim_param.experiment_type != 'a2l_reproduce_dataset':

            ## compute current variables value
            node_names = ['effect']
            node_values = [expected_effect]
            obj_id = 0
            for obj_name in sim_param.obj_name_vector:
#                print('obj pos inferring -->', obj_name, obj_pos_dict[obj_name])                            
                distance = discr_dist.compute_distance(
                    [current_eef_x,current_eef_y], 
                    obj_pos_dict[obj_name],
                    current_dist)
                
                orientation = discr_orien.compute_orientation_discr(
                    [current_eef_x,current_eef_y], 
                    obj_pos_dict[obj_name],
                    current_orien)
                    
                if sim_param.inclination_param:
                    inclination = discr_inclin.compute_inclination_discr(
                        obj_pos_dict[obj_name],
                        [current_eef_x,current_eef_y,current_eef_z],             
                        current_inclin)                      
                    node_names += ['distance'+str(obj_id),
                                   'orientation'+str(obj_id),
                                   'inclination'+str(obj_id)]
                    node_values += [distance, orientation, inclination]
                else:
                    node_names += ['distance'+str(obj_id),
                                   'orientation'+str(obj_id)]
                    node_values += [distance, orientation]                    
                                
                obj_id += 1
                
            ## infere next move            
            try:                
                next_mov_discr, prob_value, delta_nb_var, same_prob = \
                    infer.infere_mov(bn, ie,                                        
                                     node_names,
                                     node_values)
                mean_traj_prob += prob_value
#                if sim_param.debug_infer:                                        
                print(node_values, next_mov_discr, prob_value)
                print('Next move :', next_mov_discr.upper(),
                      'for', node_values,
                      'with prob value', prob_value)
                if same_prob:
                    print('-------------------------------> UNKNOWN PROBABILITY, STOP INFERENCE')
                    return eef_traj, False, 0    
                        
                next_mov_discr = [next_mov_discr] ## to compute move later
            except Exception as e: 
                if sim_param.debug_infer:
                    print('-------------------------------> UNKNOWN LABEL WHILE INFERRING!!!', e)
                return eef_traj, False, 0

        else:
            ## compute neighbours virtual positions
            nn_pos_vector =  [[current_eef_x,
                               current_eef_y,
                               current_eef_z]]        
            add_dist = sim_param.step_length/2
            
            for j in range(sim_param.nb_NN):
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
            for curr_virt_pos in nn_pos_vector:        
                current_eef_x_tmp = curr_virt_pos[0]
                current_eef_y_tmp = curr_virt_pos[1]
                current_eef_z_tmp = curr_virt_pos[2]
                
                ## compute current variables value
                node_names = ['effect']
                node_values = [expected_effect]
                obj_id = 0
                for obj_name in sim_param.obj_name_vector:
    #                print('obj pos inferring -->', obj_name, obj_pos_dict[obj_name])                            
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
                       infer.infere_mov(bn, ie,                                        
                                        node_names,
                                        node_values)
                    if sim_param.debug_infer:
                        print(node_values, next_mov_discr, prob_value)
                    
                except Exception as e: 
                    if sim_param.debug_infer:
                        print('-------------------------------> UNKNOWN LABEL WHILE INFERRING!!!', e)
                        print([curr_virt_pos, 
                               node_values])
                    virt_inference_vector.append([curr_virt_pos, 
                                                 '',
                                                 '', 
                                                 0]) ## to avoid being selected
                    continue ## dont do next steps
                        
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
            if prob_found and len(virt_inference_vector) > 2 and sim_param.debug_infer:
                print('-------------------------------> SAME PROB', 
                      virt_inference_vector[0][-1])
                if virt_inference_vector[0][-1] == 0:
                    print('-------------------------------> NO MOVE INFERRED')
                    print(node_values)
                    return eef_traj, False
    
            if sim_param.debug_infer:
                print(node_values)
                

#            ## MOVE 
#            ## with higher prob    
#            max_prob = virt_inference_vector[0][3]
#            max_next_move = virt_inference_vector[0][2]
#            max_pos = 0
#            tmp_i = 1
#            for tmp_infer_values in virt_inference_vector[1:]:
#                if tmp_infer_values[3] > max_prob:
#                    max_prob = tmp_infer_values[3]
#                    max_next_move = tmp_infer_values[2]
#                    max_pos = tmp_i
#                tmp_i += 1
#            print('Next move for pos', max_next_move.upper(), max_prob, 'in pose', max_pos)
#            next_mov_discr = [max_next_move]
                    
            ## get 2 higher values
            ## if close, we select the one more repeated
            ## else, the higher value
            cumulated_prob_dict = OrderedDict()
            for tmp_infer_values in virt_inference_vector:
                if not tmp_infer_values[2] in cumulated_prob_dict.keys():
                    cumulated_prob_dict[tmp_infer_values[2]] = 0 ## cumulated prob
                cumulated_prob_dict[tmp_infer_values[2]] += tmp_infer_values[3]
                    
    #        counter_dict = Counter(cumulated_prob_dict) ## { key1:n times, key2:p times etc }
            
            keys_found_list = []
            for curr_res in virt_inference_vector:
                keys_found_list.append(curr_res[2])
                
            counter_dict = OrderedDict()
            for key in keys_found_list:
                if key == '':
                    continue
                
                if not key in counter_dict:
                    counter_dict[key] = 1
                else:
                    counter_dict[key] += 1
        
            mean_prob_dict = OrderedDict()
            for move,repetitions in counter_dict.items():
                print(move,repetitions)
                mean_prob_dict[move] = round(cumulated_prob_dict[move]/repetitions, 
                                             sim_param.round_value) ## mean value
    
            sorted_mean_list = sorted(mean_prob_dict.items(), key=operator.itemgetter(1), reverse=True)
            print(sorted_mean_list)
            max_next_move = ''
            if len(sorted_mean_list) == 1 or sorted_mean_list[0][1] > 0.95:
                max_next_move = sorted_mean_list[0][0]
                max_prob = sorted_mean_list[0][1]
            ## if 2 elems check if are close
            elif len(sorted_mean_list) > 1:
                ## if are close, get the more repeated one
                if round(sorted_mean_list[1][1] / sorted_mean_list[0][1],2) >= 0.75:
                    if counter_dict[sorted_mean_list[0][0]] >= \
                       counter_dict[sorted_mean_list[1][0]]:
                        max_next_move = sorted_mean_list[0][0]
                        max_prob = sorted_mean_list[0][1]
                    else:
                        max_next_move = sorted_mean_list[1][0]
                        max_prob = sorted_mean_list[1][1]
                else:
                    max_next_move = sorted_mean_list[0][0]
                    max_prob = sorted_mean_list[0][1]
            
            print('Next move for pos', max_next_move.upper(), 'with mean prob', max_prob)
            next_mov_discr = [max_next_move]

#        if sim_param.debug_infer:
#            print(expected_effect.upper(),
#                  orientation.upper(),
#                  inclination.upper(),
#                  distance.upper(), 
#                  "-> ", 
#                  next_mov_discr.upper(), 
#                  "with probability", prob_value)
        
        ## COMMON CODE AGAIN ! 
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
#            dims_vector = abs(delta_x) + abs(delta_y) + abs(delta_z)
#            if dims_vector == 0: ## to avoid div by 0 ??? TODOOOOOOOOOOOOOOOOOOOOOOO
#                dims_vector = 1
            dims_vector = 1
            mov_step = round(sim_param.step_length/sqrt(dims_vector), sim_param.round_value)
            if delta_x > 0:
                delta_x = mov_step
            elif delta_x < 0:
                delta_x = -mov_step
            if delta_y > 0: 
                delta_y = mov_step
            elif delta_y < 0: 
                delta_y = -mov_step
            if delta_z > 0: 
                delta_z = mov_step
            elif delta_z < 0:
                delta_z = -mov_step            
                
        if sim_param.debug_infer:
            print("Delta :", delta_x, delta_y, delta_z, 
                  'Norm:', round(d.euclidean([0, 0, 0],
                                [delta_x, delta_y, delta_z]), sim_param.round_value))
                                              
        ## move eef to new position
#        next_eef_x = round(current_eef_x + delta_x/len(next_mov_discr), sim_param.round_value)
#        next_eef_y = round(current_eef_y + delta_y/len(next_mov_discr), sim_param.round_value)
#        next_eef_z = round(current_eef_z + delta_z/len(next_mov_discr), sim_param.round_value)   
        next_eef_x = round(current_eef_x + delta_x, sim_param.round_value)
        next_eef_y = round(current_eef_y + delta_y, sim_param.round_value)
        next_eef_z = current_eef_z
        if sim_param.debug_infer:
            print('Previous EEF pos :', 
                  current_eef_x, current_eef_y, current_eef_z)
            print('New EEF pos :', 
                  next_eef_x, next_eef_y, next_eef_z)
            
        eef_traj.append([next_eef_x, next_eef_y, next_eef_z])  

        ## check if obj was moved
        obj_moved_pose_dict = OrderedDict()
        for obj_name in sim_param.moved_obj_name_vector:
            obj_pos = obj_pos_dict[obj_name]
            obj_moved, obj_moved_pose = \
                env.compute_obj_pos(
                    obj_pos,
                    [current_eef_x,current_eef_y,current_eef_z], 
                    [next_eef_x,next_eef_y,next_eef_z])
            obj_moved_pose_dict[obj_name] = (obj_moved,obj_pos)

        all_obj_moved = True
        for name, moved in obj_moved_pose_dict.items():
            if sim_param.debug_infer:
                print('simulated', name.upper(), "MOVED ? :", moved[0])
            all_obj_moved = all_obj_moved and moved[0]

        i += 1
        ## end delta
        
    ## repeat last move to improve effect identification
    if all_obj_moved:
        current_eef_x = next_eef_x
        current_eef_y = next_eef_y
        current_eef_z = next_eef_z
        next_eef_x = round(current_eef_x + delta_x/len(next_mov_discr), sim_param.round_value)
        next_eef_y = round(current_eef_y + delta_y/len(next_mov_discr), sim_param.round_value)
        next_eef_z = round(current_eef_z + delta_z/len(next_mov_discr), sim_param.round_value)
        eef_traj.append([next_eef_x, next_eef_y, next_eef_z])
        
    mean_traj_prob = mean_traj_prob/len(eef_traj)

    return eef_traj, all_obj_moved, mean_traj_prob

'''
a
'''
def plot_save_traj_3d(init_pos_vector,                 
                 traj,
                 obj_pos_dict,
                 init_pos_coord,
                 effect,
                 filepath,
                 sim_obj_moved):

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
             obj_pos[2] - 0.08/2, 
             [sim_param.cube_y], [sim_param.cube_x], [sim_param.cube_z], 
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
        R = 0.035
        obj_pos = obj_pos_dict['cylinder']
        p0 = np.array([obj_pos[0], obj_pos[1], obj_pos[2]]) #point at one end
        p1 = np.array([obj_pos[0], obj_pos[1], obj_pos[2] - height]) #point at other end        
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
#    ax.view_init(90,0) # top view
#    ax.view_init(0,0) # front view
#    ax.view_init(0,270) # left view

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
            linewidth = 8,
            color = 'blue')
    ax.plot(eef_pos_vector_x, 
            eef_pos_vector_y, 
            eef_pos_vector_z,
            '*',                                     
            markersize=3,
            c='grey')
    
    if sim_param.save_trajs: # and sim_obj_moved:  
        plt.savefig(filepath)
        
    if sim_param.plot_trajs: 
        plt.show()
        
    plt.close()

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
