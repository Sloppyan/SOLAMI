import os
import csv
import copy
import yaml
import logging
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# logging.basicConfig(level=logging.INFO)


def load_yaml(file_path):
    with open(file_path, 'r', encoding='utf-8') as yaml_file:
        return yaml.safe_load(yaml_file)
    

def read_csv(file_path):
    with open(file_path, 'r', encoding='utf-8-sig') as csv_file:
        csv_reader = csv.reader(csv_file)
        return list(csv_reader)

def convert_to_int(s):
    try:
        return int(s)
    except ValueError:
        return -1


def load_npz(file_path):
    with np.load(file_path, allow_pickle=True) as data:
        filename = os.path.splitext(os.path.basename(file_path))[0]
        return filename, data['smplx']

def load_npz_files_from_directory(directory, load_npz=True):
    file_paths = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith('.npz')]
    if load_npz:
        with ThreadPoolExecutor(max_workers=8) as executor:
            results = dict(executor.map(load_npz, file_paths))
    else:
    # TODO UNIT TEST HERE!
        results = {os.path.splitext(os.path.basename(file_path))[0]: None for file_path in file_paths}
    return results


class Action:
    def __init__(self, script, config, subject_id):
        self.last_self_action = None
        self.last_other_action = None
        self.next_other_action = None
        self.next_self_action = None
        self.last_action_type = None 
        self.next_action_type = None
        self.dataset_name = 'dlp'
        self.index = None
        pass

    def get_current_smplx_sequences(self, smplx_dicts_name_to_ndarray=None):
        # for smplx data
        data = np.load(self.file_path_smplx_sub, allow_pickle=True)
        smplx_data = smplx_dicts_name_to_ndarray[self.sub_smplx_npz_file_name]
        # get synchronized smplx data
        start_frame = int(self.sub_start_frame / self.fbx_ratio)
        end_frame = int(self.sub_end_frame / self.fbx_ratio)
        smplx_info = {}
        for key, value in smplx_data.item().items():
            if key in ['transl', 'poses', 'global_orient', 'betas']:
                smplx_info[key] = value[start_frame:end_frame]
            else:
                smplx_info[key] = value
        return smplx_info

    

class AtomicAction(Action):
    def __init__(self, script, config, subject_id):
        super().__init__(script, config, subject_id)
        self.subject_id = subject_id
        self.init_common_info(script, config)
        self.init_subject_info(subject_id)
        self.index = self.file_name + '---' + str(self.script_index) + '---' + self.subject_id + '---' + str(0)
        pass

    # set detailed info for this line of script
    # _cn means Chinese
    # _both means both A and B

    def init_common_info(self, script, config):
        # line_index is the index of the line in one script
        self.line_index = script[-1]
        self.file_name = script[0]
        self.script_index = convert_to_int(script[1]) # TODO script[1] must exists
        self.action_description_brief_both = script[2]
        self.action_variation = script[3]
        self.action_description_detailed_both = script[4]
        self.dialog_scripts = script[5]
        self.A_action_description = script[6]
        self.B_action_description = script[7]
        self.A_emotion = script[8]
        self.B_emotion = script[9]
        self.area = script[10]
        self.positions_and_props = script[11]

        self.action_description_brief_both_cn = script[12]
        self.action_variation_cn = script[13]
        self.action_description_detailed_both_cn = script[14]
        self.dialog_scripts_cn = script[15]
        self.A_action_description_cn = script[16]
        self.B_action_description_cn = script[17]
        self.A_emotion_cn = script[27]
        self.B_emotion_cn = script[28]
        self.area_cn = script[29]
        self.positions_and_props_cn = script[30]

        self.A_start_frame = convert_to_int(script[18])
        self.A_end_frame = convert_to_int(script[19])
        self.B_start_frame = convert_to_int(script[20])
        self.B_end_frame = convert_to_int(script[21])
        self.A_B_interact_start_frame = convert_to_int(script[22])
        self.A_B_interact_end_frame = convert_to_int(script[23])

        self.gender_of_A = script[24]
        self.script_issue = script[25]
        self.hand_issue = script[26]

        self.fbx_ratio = convert_to_int(script[32])
        self.gender_of_0 = script[33]


        # TODO show the dir format
        # depends on the format of the data
        self.file_dir = os.path.join(config['database']['motion_data_dir'], '2023' + self.file_name[1:5], self.file_name)
        fbx_name = 'Take_' + self.file_name[-3:] + '_merged'
        self.file_path_fbx = os.path.join(self.file_dir, fbx_name + '.fbx')
        self.file_path_video = os.path.join(self.file_dir, self.file_name + '.mp4')
        self.file_path_smplx_female = os.path.join(config['database']['smplx_data_dir'], '2023' + self.file_name[1:5], self.file_name + '_00.npz')
        self.file_path_smplx_male = os.path.join(config['database']['smplx_data_dir'], '2023' + self.file_name[1:5], self.file_name + '_01.npz')
    
    # set subject info for this line of script
    def init_subject_info(self, subject_id):
        if subject_id not in ['A', 'B']:
            raise ValueError('flag must be A or B.')
        for attr in ['action_description', 'emotion', 'action_description_cn', 'emotion_cn']:
            setattr(self, 'sub_' + attr, getattr(self, subject_id + '_' + attr))
        setattr(self, 'sub_start_frame', getattr(self, subject_id + '_start_frame'))
        setattr(self, 'sub_end_frame', getattr(self, subject_id + '_end_frame'))
        if subject_id == 'A' and self.gender_of_A == 'Female' or subject_id == 'B' and self.gender_of_A == 'Man':
            self.sub_gender = 'Female'
            self.file_path_smplx_sub = self.file_path_smplx_female
            self.sub_smplx_npz_file_name = self.file_name + '_00'
        else:
            self.sub_gender = 'Male'
            self.file_path_smplx_sub = self.file_path_smplx_male
            self.sub_smplx_npz_file_name = self.file_name + '_01'
    
    def __str__(self):
        return 'AtomicAction: []' + self.file_name + '] [subject ' + self.subject_id + '] [script index ' + str(self.script_index) + '] [start_frame ' + str(self.sub_start_frame) + ']'

    def __eq__(self, other):
        return self.file_name == other.file_name and self.script_index == other.script_index \
            and self.subject_id == other.subject_id and self.sub_start_frame == other.sub_start_frame 
    
    def __hash__(self):
        return hash('AtomicAction', self.file_name, self.script_index, self.subject_id, self.sub_start_frame)

            

class ShortAction(Action):
    def __init__(self, script, config, subject_id):
        super().__init__(script, config, subject_id)
        self.subject_id = subject_id
        self.init_common_info(script, config)
        self.init_subject_info(subject_id)
        self.index = self.file_name + '---' + str(self.script_index) + '---' + self.subject_id + '---' + str(self.line_index)
        pass

    def init_common_info(self, script, config):
        self.line_index = script[-1]
        self.file_name = script[0]
        self.ori_script_index = convert_to_int(script[1]) # TODO script[1] must exists

        self.script_table = script[2]
        self.script_theme = script[3]
        self.dialog_scripts = script[4]

        ## TODO
        # the short scripts are different from the atomic scripts
        if self.script_table == 'SHORT_V1':
            self.script_index = self.ori_script_index + 100000
        elif self.script_table == 'SHORT_V2':
            self.script_index = self.ori_script_index + 100000 + 100

        self.action_description_brief_both = script[6]
        self.action_description_detailed_both = script[5]
        self.A_action_description = script[7]
        self.B_action_description = script[8]
        self.A_emotion = script[9]
        self.B_emotion = script[10]
        self.area = script[11]
        self.positions_and_props = ''

        self.action_variation = ''

        self.script_theme_cn = script[12]

        self.action_description_brief_both_cn = script[15]
        self.action_variation_cn = ''
        self.action_description_detailed_both_cn = script[14]
        self.dialog_scripts_cn = script[13]
        self.A_action_description_cn = script[16]
        self.B_action_description_cn = script[17]
        self.A_emotion_cn = script[27]
        self.B_emotion_cn = script[28]
        self.area_cn = script[29]
        self.positions_and_props_cn = script[30]

        self.A_start_frame = convert_to_int(script[18])
        self.A_end_frame = convert_to_int(script[19])
        self.B_start_frame = convert_to_int(script[20])
        self.B_end_frame = convert_to_int(script[21])
        self.A_B_interact_start_frame = convert_to_int(script[22])
        self.A_B_interact_end_frame = convert_to_int(script[23])

        self.gender_of_A = script[24]
        self.script_issue = script[25]
        self.hand_issue = script[26]

        self.fbx_ratio = convert_to_int(script[32])
        self.gender_of_0 = script[33]

        # TODO show the dir format
        self.file_dir = os.path.join(config['database']['motion_data_dir'], '2023' + self.file_name[1:5], self.file_name)
        fbx_name = 'Take_' + self.file_name[-3:] + '_merged'
        self.file_path_fbx = os.path.join(self.file_dir, fbx_name + '.fbx')
        self.file_path_video = os.path.join(self.file_dir, self.file_name + '.mp4')
        self.file_path_smplx_female = os.path.join(config['database']['smplx_data_dir'], '2023' + self.file_name[1:5], self.file_name + '_00.npz')
        self.file_path_smplx_male = os.path.join(config['database']['smplx_data_dir'], '2023' + self.file_name[1:5], self.file_name + '_01.npz')
        # TODO
        pass

    def init_subject_info(self, subject_id):
        if subject_id not in ['A', 'B']:
            raise ValueError('flag must be A or B.')
        for attr in ['action_description', 'emotion']:
            setattr(self, 'sub_' + attr, getattr(self, subject_id + '_' + attr))
        setattr(self, 'sub_start_frame', getattr(self, subject_id + '_start_frame'))
        setattr(self, 'sub_end_frame', getattr(self, subject_id + '_end_frame'))
        if subject_id == 'A' and self.gender_of_A == 'Female' or subject_id == 'B' and self.gender_of_A == 'Man':
            self.sub_gender = 'Female'
            self.file_path_smplx_sub = self.file_path_smplx_female
            self.sub_smplx_npz_file_name = self.file_name + '_00'
        else:
            self.sub_gender = 'Male'
            self.file_path_smplx_sub = self.file_path_smplx_male
            self.sub_smplx_npz_file_name = self.file_name + '_01'


    def __str__(self):
        return 'ShortAction: [' + self.file_name + '] [subject ' + self.subject_id \
              + '] [script index ' + str(self.script_index) + '] [line index ' + str(self.line_index) + ']'

    def __eq__(self, other):
        return self.file_name == other.file_name and self.script_index == other.script_index \
            and self.subject_id == other.subject_id and self.line_index == other.line_index

    def __hash__(self):
        return hash('ShortAction', self.file_name, self.script_index, self.subject_id, self.line_index)


class DLPMoCap_ActionDatabase:
    def __init__(self, config):
        self.config = copy.deepcopy(config)
        self.actions = []
        self.get_smplx_dicts_name_to_ndarray()
        self.process_atomic_scripts()
        self.process_short_scripts()
        self.filter_illegal_actions()
        self.build_action_temporal_interact()
        self.check_action_type()


    def process_atomic_scripts(self):
        self.atomic_scripts_list = read_csv(self.config['database']['annotated_atomic_path'])
        line_index = 0
        last_file_name = None
        for script in self.atomic_scripts_list[1:]:
            if script[0] == '':
                continue
            else:
                script_tmp = copy.deepcopy(script)
                if script_tmp[0] == last_file_name:
                    line_index += 1
                else:
                    last_file_name = script_tmp[0]
                    line_index = 0
                script_tmp.append(line_index)
                # one atomic script will be processed twice, one for A and one for B
                self.actions.append(AtomicAction(script=script_tmp, config=self.config, subject_id='A'))
                self.actions.append(AtomicAction(script=script_tmp, config=self.config, subject_id='B'))
        logging.info('[DLP] Atomic scripts processed. Totally ' + str(len(self.actions)) + ' actions.')
        pass

    def process_short_scripts(self):
        self.short_scripts_list = read_csv(self.config['database']['annotated_short_path'])
        # depends on the format of the csv file
        tmp_index_table_theme = self.short_scripts_list[1][1:4]
        line_index = 0
        for script in self.short_scripts_list[1:]:
            if script[0] == '':
                continue
            else:
                script_tmp = copy.deepcopy(script)
                if script_tmp[1] == '':
                    script_tmp[1:4] = tmp_index_table_theme
                    line_index += 1
                else:
                    tmp_index_table_theme = copy.deepcopy(script_tmp[1:4])
                    line_index = 0
                script_tmp.append(line_index)
                # one line in short script will be processed twice, one for A and one for B
                self.actions.append(ShortAction(script=script_tmp, config=self.config, subject_id='A'))
                self.actions.append(ShortAction(script=script_tmp, config=self.config, subject_id='B'))
        logging.info('[DLP] Short scripts processed. Totally ' + str(len(self.actions)) + ' actions.')
        pass

    def filter_illegal_actions(self):
        # filter illegal actions
        for i, action in enumerate(self.actions[:]):
            if action.sub_start_frame == -1 or action.sub_end_frame == -1 \
                or action.script_issue == 'Lost' or action.hand_issue == 'Issue 0' \
                    or action.sub_action_description == '' \
                    or action.hand_issue == 'Issue 1' or action.sub_smplx_npz_file_name not in self.smplx_dicts_name_to_ndarray.keys():
                # if using os.path.exists to query the NAS server, it will be very slow
                self.actions.remove(action)
        logging.info('[DLP] Illegal actions filtered. Totally ' + str(len(self.actions)) + ' actions.')
        pass

    def build_action_temporal_interact(self):
        # build temporal interact relationship for each action
        dicts_of_file_name_to_action = {}
        for action in self.actions:
            if action.file_name not in dicts_of_file_name_to_action:
                dicts_of_file_name_to_action[action.file_name] = [action]
            else:
                dicts_of_file_name_to_action[action.file_name].append(action)
        for file_name, actions in dicts_of_file_name_to_action.items():
            actions.sort(key=lambda x: (x.line_index, x.subject_id))
            for id, action in enumerate(actions):
                # find last self action
                for id_p in range(id-1, -1, -1):
                    if actions[id_p].subject_id == action.subject_id:
                        action.last_self_action = actions[id_p]
                        logging.debug(str(action) + ' last self action => ' + str(action.last_self_action))
                        break
                # find last other action
                for id_p in range(id-1, -1, -1):
                    if actions[id_p].subject_id != action.subject_id:
                        action.last_other_action = actions[id_p]
                        logging.debug(str(action) + ' last other action => ' + str(action.last_other_action))
                        break
                # find next other action
                for id_n in range(id+1, len(actions)):
                    if actions[id_n].subject_id != action.subject_id:
                        action.next_other_action = actions[id_n]
                        logging.debug(str(action) + ' next other action => ' + str(action.next_other_action))
                        break
                # find next self action
                for id_n in range(id+1, len(actions)):
                    if actions[id_n].subject_id == action.subject_id:
                        action.next_self_action = actions[id_n]
                        logging.debug(str(action) + ' next self action => ' + str(action.next_self_action))
                        break
            pass

    def check_action_type(self):
        for action in self.actions:
            if action.last_other_action is None and action.next_other_action is None:
                action.action_type = 'action_solo'
            elif action.A_B_interact_start_frame > 0:
                action.action_type = 'interaction_simultaneous'
            else:
                action.action_type = 'interaction_turn'
        pass

    def __str__(self):
        return 'DLPMoCap_ActionDatabase for Digital Characters'

    def get_attrs_and_actions_lists(self, attr_name):
        # attrs must be a common attribute of all actions
        assert all(hasattr(action, attr_name) for action in self.actions)
        attrs = []
        actions = []
        for action in self.actions:
            attrs.append(getattr(action, attr_name))
            actions.append(action)
        return attrs, actions
    
    def get_attr_to_actions_dict(self, attr_name):
        assert all(hasattr(action, attr_name) for action in self.actions)
        dicts = {}
        for action in self.actions:
            if dicts.get(getattr(action, attr_name)):
                dicts[getattr(action, attr_name)].append(action)
            else:
                dicts[getattr(action, attr_name)] = [action]
        return dicts

    def get_smplx_dicts_name_to_ndarray(self):
        # if the smplx data is already saved in a big npz file, load it
        if self.config['database']['smplx_data_npz_path'] is not None and os.path.exists(self.config['database']['smplx_data_npz_path']):
            self.smplx_dicts_name_to_ndarray = np.load(self.config['database']['smplx_data_npz_path'], allow_pickle=True)['data'].item()
        else:
            dates = os.listdir(self.config['database']['smplx_data_dir'])
            dicts_name_to_ndarray = {}
            for date in dates:
                date_dir = os.path.join(self.config['database']['smplx_data_dir'], date)
                # multiprocessing to load many small npz files
                if os.path.isdir(date_dir):
                    folder_data = load_npz_files_from_directory(date_dir, load_npz=self.config['database']['load_npz'])
                    dicts_name_to_ndarray.update(folder_data)
            self.smplx_dicts_name_to_ndarray = dicts_name_to_ndarray
        pass

    def get_text_to_smplx_sequences(self):
        # get the smplx sequences of the action descriptions
        text_to_smplx_dicts = {}
        action_description, actions = self.get_attrs_and_actions_lists('sub_action_description')
        for i in range(len(action_description)):
            text = action_description[i]
            smplx = actions[i].get_current_smplx_sequences(self.smplx_dicts_name_to_ndarray)
            motion_list = [smplx["transl"], smplx["global_orient"], smplx["poses"][:, 3:66], smplx["poses"][:, -90:]]
            if text_to_smplx_dicts.get(text):
                text_to_smplx_dicts[text].append(motion_list)
            else:
                text_to_smplx_dicts[text] = [motion_list]
        return text_to_smplx_dicts
    
    

if __name__ == '__main__':
    # for test
    # build motion database
    config_path = r'.\digital_life_project\motion_database\config.yaml'
    config = load_yaml(config_path)
#     config = {
#     "database": {
#     "annotated_atomic_path": r'C:\Users\jiangjianping\Projects\motion database\data\scripts_annotation_v0\Script_Annotation_V0_atmoic.csv',
#     "annotated_short_path": r'C:\Users\jiangjianping\Projects\motion database\data\scripts_annotation_v0\Script_Annotation_V0_short.csv',
#     "motion_data_dir": r'\Zoehuman\DL\mocap',
#     "smplx_data_dir": r'C:\Users\jiangjianping\Projects\motion database\data\smplx_h', 
#     "text_to_smplx_data_path": None,
#     "smplx_data_npz_path": None, #'.\digital_life_project\motion_database\smplx_data.npz'
#     },
# }

    actionset = DLPMoCap_ActionDatabase(config=config)

    # get the attributes and cooresponding actions
    print(10*'#')
    subject_actions = actionset.get_attrs_and_actions_lists('sub_action_description')
    print(subject_actions[0][15:20], subject_actions[1][15:20])
    subject_emotions = actionset.get_attrs_and_actions_lists('sub_emotion')
    print(subject_emotions[0][15:20], subject_emotions[1][15:20])
    print(10*'#')

    # get the smplx sequences of the actions
    smplx = subject_actions[1][0].get_current_smplx_sequences(actionset.smplx_dicts_name_to_ndarray)
    print(smplx)

    # get text_to_smplx_dicts
    text_to_smplx_dicts = actionset.get_text_to_smplx_sequences()
    print(text_to_smplx_dicts)

    pass
