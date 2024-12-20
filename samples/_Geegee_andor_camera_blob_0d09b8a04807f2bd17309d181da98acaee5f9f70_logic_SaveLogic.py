# -*- coding: utf-8 -*-
from logic.GenericLogic import GenericLogic
from pyqtgraph.Qt import QtCore
from core.util.Mutex import Mutex
from collections import OrderedDict
import os
import sys
import inspect
import time
import numpy as np

class FunctionImplementationError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)


class SaveLogic(GenericLogic):
    """
    UNSTABLE: Alexander Stark
    A general class which saves all kind of data in a general sense.
    """
    
    def __init__(self, manager, name, config, **kwargs):
        
        state_actions = {'onactivate': self.activation}
        GenericLogic.__init__(self, manager, name, config, state_actions, **kwargs)
        self._modclass = 'savelogic'
        self._modtype = 'logic'

        ## declare connectors
        
        self.connector['out']['savelogic'] = OrderedDict()
        self.connector['out']['savelogic']['class'] = 'SaveLogic'
        
        #locking for thread safety
        self.lock = Mutex()

        self.logMsg('The following configuration was found.', 
                    msgType='status')
                  
        # Some default variables concerning the operating system:
        self.os_system = None                  
        self.default_unix_data_dir = '$HOME/Data'
        self.default_win_data_dir = 'C:/Data/'                        
                  
        # Chech which operation system is used and include a case if the 
        # directory was not found in the config:
        if 'linux' in sys.platform or sys.platform == 'darwin':
            self.os_system = 'unix'
            self.dir_slash = '/'
            if 'unix_data_directory' in config.keys():
                self.data_dir = config['unix_data_directory']
            else:
                self.data_dir = self.default_unix_data_dir
            
        elif 'win32' in sys.platform or 'AMD64' in sys.platform :
            self.os_system = 'win'
            self.dir_slash = '\\'
            if 'win_data_directory' in config.keys():
                self.data_dir = config['win_data_directory'] 
            else:
                self.data_dir = self.default_win_data_dir                      
        else:
            self.logMsg('Identify the operating system.', 
                    msgType='error')
                            
                            
        # checking for the right configuration
        for key in config.keys():
            self.logMsg('{}: {}'.format(key,config[key]), 
                        msgType='status')
                        
                             
    def activation(self,e=None):
        pass

    
    
    def save_data(self, data, filepath, parameters=None, filename=None, 
                  as_text=True, as_xml=False, precision=':.3f', delimiter='\t'):
            """ General save routine for data.

            @param dict or OrderedDict data: 
                                      Any dictonary with a keyword of the data 
                                      and a corresponding list, where the data 
                                      are situated. E.g. like:
                                      
                         data = {'Frequency (MHz)':[1,2,4,5,6]}
                         data = {'Frequency':[1,2,4,5,6],'Counts':[234,894,743,423,235]}
                         data = {'Frequency (MHz),Counts':[ [1,234],[2,894],...[30,504] ]}

            @param string filepath: The path to the directory, where the data
                                      will be saved.
                                      If filepath is corrupt, the saving routine 
                                      will retrieve the basic filepath for the 
                                      data from the inherited base module 
                                      'get_data_dir' and saves the data in the
                                      directory .../UNSPECIFIED_<module_name>/
            @param dict or OrderedDict parameters: 
                                      optional, a dictionary 
                                      with all parameters you want to pass to 
                                      the saving routine.
            @parem string filename: optional, if you really want to fix an own
                                      filename, otherwise an unique filename 
                                      will be generated from the class which is
                                      calling the save method with a timestamp.  
                                      The filename will be looking like:

                                        <calling-class>_JJJJ-MM-DD_HHh-MMm.dat
            @param bool as_text: specify how the saved data are saved to file.
            @param bool as_xml: specify how the saved data are saved to file.
            
            @param int precision: optional, specifies the number of degits
                                  after the comma for the saving precision. All
                                  number, which follows afterwards are cut off.
                                  A c-like format should be used.
                                  For 'precision=3' a number like
                                       '323.423842' is saved as '323.423'.
                                       Default is precision = 3.

            @param string delimiter: optional, insert here the delimiter, like
                                     \n for new line,  \t for tab, , for a 
                                     comma, ect.

            This method should be called from the modules and it will call all 
            the needed methods for the saving routine. This module guarentees 
            that if the passing of the data is correct, the data are saved 
            always.

            1D data
            =======
            1D data should be passed in a dictionary where the data trace should be
            assigned to one identifier like
            
                {'<identifier>':[list of values]}
                {'Numbers of counts':[1.4, 4.2, 5, 2.0, 5.9 , ... , 9.5, 6.4]}
            
            You can also pass as much 1D arrays as you want:
                {'Frequency (MHz)':list1, 'signal':list2, 'correlations': list3, ...}
               
            YOU ARE RESPONSIBLE FOR THE IDENTIFIER! DO NOT FORGET THE UNITS FOR THE
            SAVED TIME TRACE/MATRIX.
            
            2D data
            =======
            
            
            """

            frm = inspect.stack()[1]    # try to trace back the functioncall to
                                        # the class which was calling it.
            mod = inspect.getmodule(frm[0]) # this will get the object, which 
                                            # called the save_data function.
            module_name =  mod.__name__.split('.')[-1]  # that will extract the 
                                                        # name of the class.

            # check whether the given directory path does exist. If not, the
            # file will be saved anyway in the unspecified directory.

            if not os.path.exists(filepath):
                filepath = self.get_daily_directory('UNSPECIFIED_'+str(module_name))
                self.logMsg('No Module name specified! Please correct this! '
                            'Data are saved in the \'UNSPECIFIED_<module_name>\' folder.', 
                            msgType='warning', importance=7)


            # create a unique name for the file, if no name was passed:
            if filename == None:
                filename = time.strftime('%Hh%Mm%Ss') + module_name + '.dat'
                

            # open the file
            textfile = open(filepath+self.dir_slash+filename,'w')


            # write the paramters if specified:
            textfile.write('# Saved Data from the class ' +module_name+ ' on '
                           + time.strftime('%d.%m.%Y at %Hh%Mm%Ss.\n') )
            textfile.write('#\n')
            textfile.write('# Parameters:\n')
            textfile.write('# ===========\n')
            textfile.write('#\n')
            
            
            if parameters != None:
                
                # check whether the format for the parameters have a dict type:
                if type(parameters) is dict or OrderedDict:
                    for entry in parameters:
                        textfile.write('# '+str(entry)+':'+delimiter+str(parameters[entry])+'\n')
                
                
                # make a hardcore string convertion and try to save the 
                # parameters directly:
                else:
                    self.logMsg('The parameters are not passed as a dictionary! '
                                'The SaveLogic will try to save the paramters '
                                'directely.', msgType='error', importance=9)
                    textfile.write('# not specified parameters: '+str(parameters)+'\n')
                
                
            textfile.write('#\n')
            textfile.write('# Data:\n')
            textfile.write('# =====\n')
            # check the input data:

            
            # go through each data in t
            if len(data)==1:
                key_name = list(data.keys())[0]
                
                # check whether the data is only a 1d trace
                if len(np.shape(data[key_name])) == 1:
                        
                    self.save_1d_trace_as_text(trace_data = data[key_name], 
                                                trace_name=key_name,
                                                opened_file = textfile,
                                                precision=precision)
                                                    
                # check whether the data is only a 2d array                                
                elif len(np.shape(data[key_name])) == 2:
                    
                    key_name_array = key_name.split(',')                    
                    
                    self.save_2d_points_as_text(trace_data = data[key_name],
                                                trace_name = key_name_array,
                                                opened_file=textfile,
                                                precision=precision,
                                                delimiter=delimiter)
                elif len(np.shape(data[key_name])) == 3:
                    
                    self.logMsg('Savelogic has no implementation for 3 '
                                'dimensional arrays. The data is saved in a '
                                'raw fashion.', msgType='warning', importance=7)
                    textfile.write(str(data[key_name]))
                
                else:
                    
                    self.logMsg('Savelogic has no implementation for 4 '
                                'dimensional arrays. The data is saved in a '
                                'raw fashion.', msgType='warning', importance=7)
                    textfile.write(+str(data[key_name]))
                    
                    
            else:
                key_list = list(data)

                trace_1d_flag = True                
                
                data_traces = []
                for entry in key_list:
                    data_traces.append(data[entry])
                    if len(np.shape(data[entry])) > 1:
                        trace_1d_flag = False
                    
                    
                if trace_1d_flag:
                    
                    self.save_N_1d_traces_as_text(trace_data = data_traces,
                                                  trace_name = key_list,
                                                  opened_file=textfile,
                                                  precision=precision,
                                                  delimiter=delimiter)
                else:
                    # go through each passed element again and treat them as 
                    # independant, i.e. each element is saved in an extra file.
                    # That is an recursive procedure:
                
                    for entry in key_list:
                        self.save_data(data = {entry:data[entry]}, 
                                       filepath = filepath, 
                                       parameters=parameters, 
                                       filename=filename[:-4]+'_'+entry+'.dat',
                                       as_text=True, as_xml=False, 
                                       precision=precision, delimiter=delimiter)                            


            textfile.close()
            





    def save_1d_trace_as_text(self, trace_data, trace_name, opened_file=None,
                              filepath=None, filename=None, precision=':.3f'):
        """An independant method, which can save a 1d trace. 
        
        If you call this method but you are respondible, that the passed 
        optional parameters are correct."""
        
        close_file_flag = False        
        
        if opened_file == None:
            opened_file = open(filepath+self.dir_slash+filename+'.dat','wb') 
            close_file_flag = True
            
            
        opened_file.write('# '+str(trace_name)+'\n')
        
        for entry in trace_data:
            opened_file.write(str('{0'+precision+'}\n').format(entry))
            
            
        if close_file_flag:
            opened_file.close()
        

    def save_N_1d_traces_as_text(self, trace_data, trace_name, opened_file=None,
                              filepath=None, filename=None, precision=':.3f',
                              delimiter='\t'):
        
        close_file_flag = False        
        
        if opened_file == None:
            opened_file = open(filepath+self.dir_slash+filename+'.dat','wb') 
            close_file_flag = True
        
        if trace_name != None:
            opened_file.write('# ')
            for name in trace_name:
                opened_file.write(name + delimiter )                          
            opened_file.write('\n') 
        
        max_trace_length = max(np.shape(trace_data))   

        for row in range(max_trace_length):
            for column in range(len(trace_data)):
                try:
                    opened_file.write(str('{0'+precision+'}'+delimiter).format(trace_data[column][row]))
                except:
                    opened_file.write(str('{0}'+delimiter).format('NaN'))
            opened_file.write('\n')
            
        if close_file_flag:
            opened_file.close()  

    def save_2d_points_as_text(self,trace_data, trace_name=None, opened_file=None,
                              filepath=None, filename=None, precision=':.3f',
                              delimiter='\t'):
        
        close_file_flag = False        
        
        if opened_file == None:
            opened_file = open(filepath+self.dir_slash+filename+'.dat','wb') 
            close_file_flag = True
            
        # write the trace names:
        if trace_name != None:
            opened_file.write('# ')
            for name in trace_name:
                opened_file.write(name + delimiter )                          
            opened_file.write('\n')  

        for row in trace_data:
            for entry in row:
                opened_file.write(str('{0'+precision+'}'+delimiter).format(entry)) 
            opened_file.write('\n')    
            
        if close_file_flag:
            opened_file.close()      



    def _save_1d_traces_as_xml():
        

        if as_xml:        
        
            root = ET.Element(module_name)  # which class wanted to access the save
                                            # function
            
            para = ET.SubElement(root, 'Parameters')
            
            if parameters != None:
                for element in parameters:
                    ET.SubElement(para, element).text = parameters[element]
            
            data_xml = ET.SubElement(root, 'data')
            
            for entry in data:
                
                dimension_data_array = len(np.shape(data[entry]))   
                
                # filter out the events which has only a single trace:  
                if dimension_data_array == 1:
                    
                    value = ET.SubElement(data_xml, entry)
                                     
                    for list_element in data[entry]:

                        ET.SubElement(value, 'value').text = str(list_element)
                    
                elif dimension_data_array == 2:
                    
                    dim_list_entry = len(np.shape(data[entry][0])) 
                    length_list_entry = np.shape(data[entry][0])[0]
                    if (dim_list_entry == 1) and (np.shape(data[entry][0])[0] == 2):
                    
                        # get from the keyword, which should be within the string
                        # separated by the delimiter ',' the description for the
                        # values:
                        try:
                            axis1 = entry.split(',')[0] 
                            axis2 = entry.split(',')[1] 
                        except:
                            print('Enter a commaseparated description for the given values!!!')
                            print('like:  dict_data[\'Frequency (MHz), Signal (arb. u.)\'] = 2d_list ')
                            print('But your data will be saved.')
                            
                            axis1 = str(entry)  
                            axis2 = 'value2'
                            
                        for list_element in data[entry]:
                                                                         
                            element = ET.SubElement(data_xml, 'value' ).text = str(list_element)
#                                
#                            ET.SubElement(element, str(axis1)).text = str(list_element[0])
#                            ET.SubElement(element, str(axis2)).text = str(list_element[1])
                        
                    elif (dim_list_entry == 1):
                        
                        for list_element in data[entry]:
                        
                            row = ET.SubElement(data_xml, 'row')
                            
                            for sub_element in list_element:
                                
                                ET.SubElement(row, 'value').text = str(sub_element)
                        
                        
                        
            #write to file:
            tree = ET.ElementTree(root)
            tree.write('output.xml', pretty_print=True, xml_declaration=True)



    def _save_2d_data_as_xml():
        pass





    def get_daily_directory(self):
        """
        Creates the daily directory.

          @return string: path to the daily directory.       
        
        If the daily directory does not exits in the specified <root_dir> path
        in the config file, then it is created according to the following scheme:
        
            <root_dir>\<year>\<month>\<day>
            
        and the filepath is returned. There should be always a filepath 
        returned.
        """
        
        # First check if the directory exists and if not then the default 
        # directory is taken.
        if not os.path.exists(self.data_dir):
                if self.data_dir != '':
                    print('The specified Data Directory in the config file '
                          'does not exist. Using default instead.')                         
                if self.os_system == 'unix':                    
                    self.data_dir = self.default_unix_data_dir
                elif self.os_system == 'win':
                    self.data_dir = self.default_win_data_dir
                else:
                    self.logMsg('Identify the operating system.', 
                                msgType='error')
                                
                # Check if the default directory does exist. If yes, there is 
                # no need to create it, since it will overwrite the existing
                # data there.
                if not os.path.exists(self.data_dir):
                    os.makedirs(self.data_dir)
                    self.logMsg('The specified Data Directory in the config '
                                'file does not exist. Using default for {0} '
                                'system instead. The directory\n{1} was '
                                'created'.format(self.os_system,self.data_dir), 
                                msgType='status', importance=3)
                                
        # That is now the current directory:
        current_dir = self.data_dir + self.dir_slash + time.strftime("%Y") + self.dir_slash + time.strftime("%m")


        
        folder_exists = False   # Flag to indicate that the folder does not exist.
        if os.path.exists(current_dir):
            
            # Get only the folders without the files there:
            folderlist = [d for d in os.listdir(current_dir) if os.path.isdir(os.path.join(current_dir, d))]
            # Search if there is a folder which starts with the current date:
            for entry in folderlist:
                if (time.strftime("%Y%m%d") in (entry[:2])):
                    current_dir = current_dir +self.dir_slash+ str(entry)
                    folder_exists = True
                    break
            
        if not folder_exists:
            current_dir = current_dir + self.dir_slash + time.strftime("%Y%m%d")
            self.logMsg('Creating directory for today\'s data in \n'+current_dir, 
                                msgType='status', importance=5)

            # The exist_ok=True is necessary here to prevent Error 17 "File Exists"
            # Details at http://stackoverflow.com/questions/12468022/python-fileexists-error-when-making-directory
            os.makedirs(current_dir,exist_ok=True)
        
        return current_dir
                
    def get_path_for_module(self,module_name=None):
        """
        Method that creates a path for 'module_name' where data are stored.

          @param string module_name: Specify the folder, which should be 
                                     created in the daily directory. The
                                     module_name can be e.g. 'Confocal'.
          @retun string: absolute path to the module name
          
        This method should be called directly in the saving routine and NOT in
        the init method of the specified module! This prevents to create empty
        folders! 
        
        """
        if module_name == None:
            self.logMsg('No Module name specified! Please correct this! Data '
                        'are saved in the \'UNSPECIFIED_<module_name>\' folder.', 
                        msgType='warning', importance=7)
                  
            frm = inspect.stack()[1]    # try to trace back the functioncall to
                                        # the class which was calling it.
            mod = inspect.getmodule(frm[0]) # this will get the object, which 
                                            # called the save_data function.
            module_name =  mod.__name__.split('.')[-1]  # that will extract the 
                                                        # name of the class.  
            module_name = 'UNSPECIFIED_'+module_name
            
        dir_path = self.get_daily_directory() +self.dir_slash+ module_name
        
        if not os.path.exists(dir_path):        
            os.makedirs(dir_path)
        return dir_path
 
