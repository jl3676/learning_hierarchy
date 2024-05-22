import numpy as np
import copy

import warnings

def extract_stage2_info(data, experiment):
  '''
  Extracts and re-organize stage2 data from preprocessed data. Organize trials by iteration for each
  (stage1, stage2) stimuli combination, rather than the raw trial number.

  Args:
    - data[dict]: the data dictionary
    - experiment[str]: one of 'V1-V1', 'V1-V2', 'V1-V3', 'V2-V1', 'V2-V2', 'V3-V1', 'V3-V3'

  Returns:
    - stage2_info: reorganized stage2 data
  '''
  nsubjects = data['tr'].shape[0]
  nblocks = 12
  ntrials = 32

  # the 2*2 is F1 or F2 by S1 or S2
  stage2_info = np.empty((nsubjects,nblocks,2,2,int(ntrials/(2*2))),dtype='object')

  tr = data['tr']
  a = data['a']
  a_12_12 = data['a_12_12']
  s1 = data['s1']
  s2 = data['s2']
  s_12_12 = data['s_12_12'] # subject * block * stage * trial
  population_counter1 = data['counter1']
  population_counter2 = data['counter2']
  population_counter12_12 = data['counter12_12']

  for subject in range(nsubjects):
    all_actions = tr[subject,4:]

    for block in range(nblocks):
      index = np.zeros((2,2),dtype=int)

      for trial in range(ntrials):
        if block < 2:
          if np.isnan(s_12_12[subject, block, 0, trial]) or np.isnan(s_12_12[subject, block, 1, trial]):
            continue

          sf = int(s_12_12[subject, block, 0, trial])
          ss = int(s_12_12[subject, block, 1, trial])

          temp_counter1 = population_counter12_12[subject, block, 0, trial]
          temp_counter2 = population_counter12_12[subject, block, 1, trial]
          if ~np.isnan(temp_counter1):
            temp_counter1 = int(temp_counter1)
          if ~np.isnan(temp_counter2):
            temp_counter2 = int(temp_counter2)

          this_a = a_12_12[subject, block, trial]
        else:
          if np.isnan(s1[subject,block-2,trial]) or np.isnan(s2[subject,block-2,trial]):
            continue

          # obtain stage1 stimulus and stage2 stimulus
          sf = int(s1[subject,block-2,trial])
          ss = int(s2[subject,block-2,trial])

          # get numbers of key presses in stage1 and stage2
          temp_counter1 = population_counter1[subject, block-2, trial]
          temp_counter2 = population_counter2[subject, block-2, trial]
          if ~np.isnan(temp_counter1):
            temp_counter1 = int(temp_counter1)
          if ~np.isnan(temp_counter2):
            temp_counter2 = int(temp_counter2)

          this_a = a[subject,block-2,trial]
          
        if len(this_a.ravel()) == 0 or np.isnan(temp_counter1) or np.isnan(temp_counter2):
          result = []
        else:
          temp_a = this_a[1,temp_counter1-1]
          temp_RT = np.nan 
          
          if temp_a == all_actions[0]:
            choice_identity = 1
          elif temp_a == all_actions[1]:
            choice_identity = 2
          elif temp_a == all_actions[2]:
            choice_identity = 3
          elif temp_a == all_actions[3]:
            choice_identity = 4
          else:
            choice_identity = 5

          if temp_counter2 > 1:
            temp_a = this_a[1,temp_counter1]
            if temp_a == all_actions[0]:
              choice_identity_2 = 1
            elif temp_a == all_actions[1]:
              choice_identity_2 = 2
            elif temp_a == all_actions[2]:
              choice_identity_2 = 3
            elif temp_a == all_actions[3]:
              choice_identity_2 = 4
            else:
              choice_identity_2 = 5
          else:
            choice_identity_2 = 5

          if block == 10 and experiment == 'V1-V2':
            choice_type = get_choice_type_stage2_V2(6,sf,ss,temp_a,all_actions) 
          elif block == 10 and experiment == 'V2-V2':
            choice_type = get_choice_type_stage2_V2(6,sf,ss,temp_a,all_actions)
          elif block == 10 and experiment[-2:] == 'V3':
            choice_type = get_choice_type_stage2_V3(6,sf,ss,temp_a,all_actions)
          elif experiment[:2] == 'V1':
            choice_type = get_choice_type_stage2_V1(block,sf,ss,temp_a,all_actions) 
          elif experiment[:2] == 'V2':
            choice_type = get_choice_type_stage2_V2(block,sf,ss,temp_a,all_actions)
          elif experiment[:2] == 'V3':
            choice_type = get_choice_type_stage2_V3(block,sf,ss,temp_a,all_actions)
          elif experiment == 'All':
            choice_type = get_choice_type_stage2_V1(block,sf,ss,temp_a,all_actions) 
          result = np.array([choice_identity, temp_RT, choice_type, temp_counter1, temp_counter2, choice_identity_2, trial])

        if index[sf-1,ss-1] == 8:
          continue

        stage2_info[subject,block,sf-1,ss-1,index[sf-1,ss-1]] = result # -1 for 0-indexing
        index[sf-1,ss-1] = index[sf-1,ss-1] + 1

  return stage2_info


def get_choice_type_stage2_V1(block, sf, ss, temp_a, actions):
  '''
  This function outputs the choice type in stage2 for V1, which are defined as:

  1 = correct
  2 = compression error over stage 1
  3 = compression error over stage 2
  4 = other error
  5 = all else

  Args:
    - block[int]: the block number (indexing starts at 0)
    - sf[int]: stage1 stimulus
    - ss[int]: stage2 stimulus
    - temp_a[int]: first action of the stage2
    - actions[list]: all actions in stage1, ordered

  Returns:
    - choice_type[int]: the choice type for the given trial
  '''
  a1 = actions[0]
  a2 = actions[1]
  a3 = actions[2]
  a4 = actions[3]

  if block == 0 or block == 2 or block == 4:
    if sf == 1 and ss == 1:
      if temp_a == a4:
        choice_type = 1
      elif temp_a == a2:
        choice_type = 2
      elif temp_a == a3:
        choice_type = 3
      elif temp_a == a1:
        choice_type = 4
      else:
        choice_type = 5
    elif sf == 1 and ss == 2:
      if temp_a == a2:
        choice_type = 1
      elif temp_a == a4:
        choice_type = 2
      elif temp_a == a3:
        choice_type = 3
      elif temp_a == a1:
        choice_type = 4
      else:
        choice_type = 5
    elif sf == 2 and ss == 1:
      if temp_a == a1:
        choice_type = 1
      elif temp_a == a3:
        choice_type = 2
      elif temp_a == a4:
        choice_type = 3
      elif temp_a == a2:
        choice_type = 4
      else:
        choice_type = 5
    elif sf == 2 and ss == 2:
      if temp_a == a3:
        choice_type = 1
      elif temp_a == a1:
        choice_type = 2
      elif temp_a == a4:
        choice_type = 3
      elif temp_a == a2:
        choice_type = 4
      else:
        choice_type = 5
  elif block == 1 or block == 3 or block == 5 or block == 7 or block == 9 or block== 11:
    if sf == 1 and ss == 1:
      if temp_a == a2:
        choice_type = 1
      elif temp_a == a4:
        choice_type = 2
      elif temp_a == a1:
        choice_type = 3
      elif temp_a == a3:
        choice_type = 4
      else:
        choice_type = 5
    elif sf == 1 and ss == 2:
      if temp_a == a4:
        choice_type = 1
      elif temp_a == a2:
        choice_type = 2
      elif temp_a == a1:
        choice_type = 3
      elif temp_a == a3:
        choice_type = 4
      else:
        choice_type = 5
    elif sf == 2 and ss == 1:
      if temp_a == a3:
        choice_type = 1
      elif temp_a == a1:
        choice_type = 2
      elif temp_a == a2:
        choice_type = 3
      elif temp_a == a4:
        choice_type = 4
      else:
        choice_type = 5
    elif sf == 2 and ss == 2:
      if temp_a == a1:
        choice_type = 1
      elif temp_a == a3:
        choice_type = 2
      elif temp_a == a2:
        choice_type = 3
      elif temp_a == a4:
        choice_type = 4
      else:
        choice_type = 5
  elif block == 6 or block == 10:
    if sf == 1 and ss == 1:
      if temp_a == a3:
        choice_type = 1
      elif temp_a == a4:
        choice_type = 2
      elif temp_a == a2:
        choice_type = 3
      elif temp_a == a1:
        choice_type = 4
      else:
        choice_type = 5
    elif sf == 1 and ss == 2:
      if temp_a == a2:
        choice_type = 1
      elif temp_a == a4:
        choice_type = 2
      elif temp_a == a3:
        choice_type = 3
      elif temp_a == a1:
        choice_type = 4
      else:
        choice_type = 5
    elif sf == 2 and ss == 1:
      if temp_a == a1:
        choice_type = 1
      elif temp_a == a3:
        choice_type = 2
      elif temp_a == a4:
        choice_type = 3
      elif temp_a == a2:
        choice_type = 4
      else:
        choice_type = 5
    elif sf == 2 and ss == 2:
      if temp_a == a4:
        choice_type = 1
      elif temp_a == a3:
        choice_type = 2
      elif temp_a == a1:
        choice_type = 3
      elif temp_a == a2:
        choice_type = 4
      else:
        choice_type = 5
  elif block == 8:
    if sf == 1 and ss == 1:
      if temp_a == a4:
        choice_type = 1
      elif temp_a == a3:
        choice_type = 2
      elif temp_a == a2:
        choice_type = 3
      elif temp_a == a1:
        choice_type = 4
      else:
        choice_type = 5
    elif sf == 1 and ss == 2:
      if temp_a == a2:
        choice_type = 1
      elif temp_a == a3:
        choice_type = 2
      elif temp_a == a4:
        choice_type = 3
      elif temp_a == a1:
        choice_type = 4
      else:
        choice_type = 5
    elif sf == 2 and ss == 1:
      if temp_a == a1:
        choice_type = 1
      elif temp_a == a4:
        choice_type = 2
      elif temp_a == a3:
        choice_type = 3
      elif temp_a == a2:
        choice_type = 4
      else:
        choice_type = 5
    elif sf == 2 and ss == 2:
      if temp_a == a3:
        choice_type = 1
      elif temp_a == a4:
        choice_type = 2
      elif temp_a == a1:
        choice_type = 3
      elif temp_a == a2:
        choice_type = 4
      else:
        choice_type = 5

  return choice_type


def get_choice_type_stage2_V2(block, sf, ss, temp_a, actions):
  '''
  This function outputs the choice type in stage2 for V2, which are defined as:

  1 = correct
  2 = compression error over stage 1
  3 = compression error over stage 2
  4 = other error
  5 = all else

  Args:
    - block[int]: the block number (indexing starts at 0)
    - sf[int]: stage1 stimulus
    - ss[int]: stage2 stimulus
    - temp_a[int]: first action of the stage2
    - actions[list]: all actions in stage1, ordered

  Returns:
    - choice_type[int]: the choice type for the given trial
  '''
  a1 = actions[0]
  a2 = actions[1]
  a3 = actions[2]
  a4 = actions[3]

  if block == 0 or block == 2 or block == 4:
    if sf == 1 and ss == 1:
      if temp_a == a4:
        choice_type = 1
      elif temp_a == a2:
        choice_type = 2
      elif temp_a == a3:
        choice_type = 3
      elif temp_a == a1:
        choice_type = 4
      else:
        choice_type = 5
    elif sf == 1 and ss == 2:
      if temp_a == a2:
        choice_type = 1
      elif temp_a == a4:
        choice_type = 2
      elif temp_a == a3:
        choice_type = 3
      elif temp_a == a1:
        choice_type = 4
      else:
        choice_type = 5
    elif sf == 2 and ss == 1:
      if temp_a == a1:
        choice_type = 1
      elif temp_a == a3:
        choice_type = 2
      elif temp_a == a4:
        choice_type = 3
      elif temp_a == a2:
        choice_type = 4
      else:
        choice_type = 5
    elif sf == 2 and ss == 2:
      if temp_a == a3:
        choice_type = 1
      elif temp_a == a1:
        choice_type = 2
      elif temp_a == a4:
        choice_type = 3
      elif temp_a == a2:
        choice_type = 4
      else:
        choice_type = 5
  elif block == 1 or block == 3 or block == 5 or block == 7 or block == 9 or block == 11:
    if sf == 1 and ss == 1:
      if temp_a == a2:
        choice_type = 1
      elif temp_a == a4:
        choice_type = 2
      elif temp_a == a1:
        choice_type = 3
      elif temp_a == a3:
        choice_type = 4
      else:
        choice_type = 5
    elif sf == 1 and ss == 2:
      if temp_a == a4:
        choice_type = 1
      elif temp_a == a2:
        choice_type = 2
      elif temp_a == a1:
        choice_type = 3
      elif temp_a == a3:
        choice_type = 4
      else:
        choice_type = 5
    elif sf == 2 and ss == 1:
      if temp_a == a3:
        choice_type = 1
      elif temp_a == a1:
        choice_type = 2
      elif temp_a == a2:
        choice_type = 3
      elif temp_a == a4:
        choice_type = 4
      else:
        choice_type = 5
    elif sf == 2 and ss == 2:
      if temp_a == a1:
        choice_type = 1
      elif temp_a == a3:
        choice_type = 2
      elif temp_a == a2:
        choice_type = 3
      elif temp_a == a4:
        choice_type = 4
      else:
        choice_type = 5
  elif block == 6:
    if sf == 1 and ss == 1:
      if temp_a == a2:
        choice_type = 1
      elif temp_a == a4:
        choice_type = 2
      elif temp_a == a3:
        choice_type = 3
      elif temp_a == a1:
        choice_type = 4
      else:
        choice_type = 5
    elif sf == 1 and ss == 2:
      if temp_a == a3:
        choice_type = 1
      elif temp_a == a2:
        choice_type = 2
      elif temp_a == a4:
        choice_type = 3
      elif temp_a == a1:
        choice_type = 4
      else:
        choice_type = 5
    elif sf == 2 and ss == 1:
      if temp_a == a4:
        choice_type = 1
      elif temp_a == a1:
        choice_type = 2
      elif temp_a == a3:
        choice_type = 3
      elif temp_a == a2:
        choice_type = 4
      else:
        choice_type = 5
    elif sf == 2 and ss == 2:
      if temp_a == a1:
        choice_type = 1
      elif temp_a == a3:
        choice_type = 2
      elif temp_a == a4:
        choice_type = 3
      elif temp_a == a2:
        choice_type = 4
      else:
        choice_type = 5
  elif block == 8:
    if sf == 1 and ss == 1:
      if temp_a == a4:
        choice_type = 1
      elif temp_a == a2:
        choice_type = 2
      elif temp_a == a3:
        choice_type = 3
      elif temp_a == a1:
        choice_type = 4
      else:
        choice_type = 5
    elif sf == 1 and ss == 2:
      if temp_a == a2:
        choice_type = 1
      elif temp_a == a3:
        choice_type = 2
      elif temp_a == a4:
        choice_type = 3
      elif temp_a == a1:
        choice_type = 4
      else:
        choice_type = 5
    elif sf == 2 and ss == 1:
      if temp_a == a1:
        choice_type = 1
      elif temp_a == a4:
        choice_type = 2
      elif temp_a == a3:
        choice_type = 3
      elif temp_a == a2:
        choice_type = 4
      else:
        choice_type = 5
    elif sf == 2 and ss == 2:
      if temp_a == a3:
        choice_type = 1
      elif temp_a == a1:
        choice_type = 2
      elif temp_a == a4:
        choice_type = 3
      elif temp_a == a2:
        choice_type = 4
      else:
        choice_type = 5
  elif block == 10:
    if sf == 1 and ss == 1:
      if temp_a == a3:
        choice_type = 1
      elif temp_a == a4:
        choice_type = 2
      elif temp_a == a2:
        choice_type = 3
      elif temp_a == a1:
        choice_type = 4
      else:
        choice_type = 5
    elif sf == 1 and ss == 2:
      if temp_a == a2:
        choice_type = 1
      elif temp_a == a4:
        choice_type = 2
      elif temp_a == a3:
        choice_type = 3
      elif temp_a == a1:
        choice_type = 4
      else:
        choice_type = 5
    elif sf == 2 and ss == 1:
      if temp_a == a1:
        choice_type = 1
      elif temp_a == a3:
        choice_type = 2
      elif temp_a == a4:
        choice_type = 3
      elif temp_a == a2:
        choice_type = 4
      else:
        choice_type = 5
    elif sf == 2 and ss == 2:
      if temp_a == a4:
        choice_type = 1
      elif temp_a == a3:
        choice_type = 2
      elif temp_a == a1:
        choice_type = 3
      elif temp_a == a2:
        choice_type = 4
      else:
        choice_type = 5

  return choice_type


def get_choice_type_stage2_V3(block, sf, ss, temp_a, actions):
  '''
  This function outputs the choice type in stage2 for V3, which are defined as:

  1 = correct
  2 = compression error over stage 1
  3 = compression error over stage 2
  4 = other error
  5 = all else

  Args:
    - block[int]: the block number (indexing starts at 0)
    - sf[int]: stage1 stimulus
    - ss[int]: stage2 stimulus
    - temp_a[int]: first action of the stage2
    - actions[list]: all actions in stage1, ordered

  Returns:
    - choice_type[int]: the choice type for the given trial
  '''
  a1 = actions[0]
  a2 = actions[1]
  a3 = actions[2]
  a4 = actions[3]

  if block == 0 or block == 2 or block == 4:
    if sf == 1 and ss == 1:
      if temp_a == a4:
        choice_type = 1
      elif temp_a == a2:
        choice_type = 2
      elif temp_a == a3:
        choice_type = 3
      elif temp_a == a1:
        choice_type = 4
      else:
        choice_type = 5
    elif sf == 1 and ss == 2:
      if temp_a == a2:
        choice_type = 1
      elif temp_a == a4:
        choice_type = 2
      elif temp_a == a3:
        choice_type = 3
      elif temp_a == a1:
        choice_type = 4
      else:
        choice_type = 5
    elif sf == 2 and ss == 1:
      if temp_a == a1:
        choice_type = 1
      elif temp_a == a3:
        choice_type = 2
      elif temp_a == a4:
        choice_type = 3
      elif temp_a == a2:
        choice_type = 4
      else:
        choice_type = 5
    elif sf == 2 and ss == 2:
      if temp_a == a3:
        choice_type = 1
      elif temp_a == a1:
        choice_type = 2
      elif temp_a == a4:
        choice_type = 3
      elif temp_a == a2:
        choice_type = 4
      else:
        choice_type = 5
  elif block == 1 or block == 3 or block == 5 or block == 7 or block == 9 or block == 11:
    if sf == 1 and ss == 1:
      if temp_a == a2:
        choice_type = 1
      elif temp_a == a4:
        choice_type = 2
      elif temp_a == a1:
        choice_type = 3
      elif temp_a == a3:
        choice_type = 4
      else:
        choice_type = 5
    elif sf == 1 and ss == 2:
      if temp_a == a4:
        choice_type = 1
      elif temp_a == a2:
        choice_type = 2
      elif temp_a == a1:
        choice_type = 3
      elif temp_a == a3:
        choice_type = 4
      else:
        choice_type = 5
    elif sf == 2 and ss == 1:
      if temp_a == a3:
        choice_type = 1
      elif temp_a == a1:
        choice_type = 2
      elif temp_a == a2:
        choice_type = 3
      elif temp_a == a4:
        choice_type = 4
      else:
        choice_type = 5
    elif sf == 2 and ss == 2:
      if temp_a == a1:
        choice_type = 1
      elif temp_a == a3:
        choice_type = 2
      elif temp_a == a2:
        choice_type = 3
      elif temp_a == a4:
        choice_type = 4
      else:
        choice_type = 5
  elif block == 6 or block == 10:
    if sf == 1 and ss == 1:
      if temp_a == a2:
        choice_type = 1
      elif temp_a == a4:
        choice_type = 2
      elif temp_a == a3:
        choice_type = 3
      elif temp_a == a1:
        choice_type = 4
      else:
        choice_type = 5
    elif sf == 1 and ss == 2:
      if temp_a == a1:
        choice_type = 1
      elif temp_a == a2:
        choice_type = 2
      elif temp_a == a4:
        choice_type = 3
      elif temp_a == a3:
        choice_type = 4
      else:
        choice_type = 5
    elif sf == 2 and ss == 1:
      if temp_a == a3:
        choice_type = 1
      elif temp_a == a1:
        choice_type = 2
      elif temp_a == a2:
        choice_type = 3
      elif temp_a == a4:
        choice_type = 4
      else:
        choice_type = 5
    elif sf == 2 and ss == 2:
      if temp_a == a4:
        choice_type = 1
      elif temp_a == a3:
        choice_type = 2
      elif temp_a == a1:
        choice_type = 3
      elif temp_a == a2:
        choice_type = 4
      else:
        choice_type = 5
  elif block == 8:
    if sf == 1 and ss == 1:
      if temp_a == a4:
        choice_type = 1
      elif temp_a == a2:
        choice_type = 2
      elif temp_a == a3:
        choice_type = 3
      elif temp_a == a1:
        choice_type = 4
      else:
        choice_type = 5
    elif sf == 1 and ss == 2:
      if temp_a == a2:
        choice_type = 1
      elif temp_a == a3:
        choice_type = 2
      elif temp_a == a4:
        choice_type = 3
      elif temp_a == a1:
        choice_type = 4
      else:
        choice_type = 5
    elif sf == 2 and ss == 1:
      if temp_a == a1:
        choice_type = 1
      elif temp_a == a4:
        choice_type = 2
      elif temp_a == a3:
        choice_type = 3
      elif temp_a == a2:
        choice_type = 4
      else:
        choice_type = 5
    elif sf == 2 and ss == 2:
      if temp_a == a3:
        choice_type = 1
      elif temp_a == a1:
        choice_type = 2
      elif temp_a == a4:
        choice_type = 3
      elif temp_a == a2:
        choice_type = 4
      else:
        choice_type = 5

  return choice_type


def calc_mean(data, start_trial=0, trials_to_probe=10, first_press_accuracy=False):
  '''
  Calculates the mean number of key presses for each subject in each stage.

  Args:
    - data[dict]: the preprocessed data dictionary
    - start_trial[int]: index of first trial
    - trials_to_probe[int]: the number of trials to include since start_trial
    - first_press_accuracy[bool]: True if plotting first press accuracy, False if plotting number of presses

  Returns:
    - mean_population_counter1[np.array]: the means of all subjects in the first stage
    - mean_population_counter2[np.array]: the means of all subjects in the second stage
  '''
  data_counter12_12 = copy.deepcopy(data['counter12_12'])
  data_counter1 = copy.deepcopy(data['counter1'])
  data_counter2 = copy.deepcopy(data['counter2'])

  nsubjects = data_counter2.shape[0]

  mean_population_counter1_12 = np.full((nsubjects,2),np.nan)
  mean_population_counter2_12 = np.full((nsubjects,2),np.nan)

  for sub in range(nsubjects):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning) # skip warnings for nanmean over all nan values
        if first_press_accuracy:
          mean_population_counter1_12[sub,0] = np.nanmean(data_counter12_12[sub,0,0,start_trial:int(start_trial+trials_to_probe)]==1)
          mean_population_counter1_12[sub,1] = np.nanmean(data_counter12_12[sub,1,0,start_trial:int(start_trial+trials_to_probe)]==1)
          mean_population_counter2_12[sub,0] = np.nanmean(data_counter12_12[sub,0,1,start_trial:int(start_trial+trials_to_probe)]==1) 
          mean_population_counter2_12[sub,1] = np.nanmean(data_counter12_12[sub,1,1,start_trial:int(start_trial+trials_to_probe)]==1) 

          mean_population_counter1 = np.hstack((mean_population_counter1_12, np.nanmean(data_counter1[:,:,start_trial:int(start_trial+trials_to_probe)]==1,axis=2)))
          mean_population_counter2 = np.hstack((mean_population_counter2_12, np.nanmean(data_counter2[:,:,start_trial:int(start_trial+trials_to_probe)]==1,axis=2)))
        else:
          mean_population_counter1_12[sub,0] = np.nanmean(data_counter12_12[sub,0,0,start_trial:int(start_trial+trials_to_probe)])
          mean_population_counter1_12[sub,1] = np.nanmean(data_counter12_12[sub,1,0,start_trial:int(start_trial+trials_to_probe)]) 
          mean_population_counter2_12[sub,0] = np.nanmean(data_counter12_12[sub,0,1,start_trial:int(start_trial+trials_to_probe)])
          mean_population_counter2_12[sub,1] = np.nanmean(data_counter12_12[sub,1,1,start_trial:int(start_trial+trials_to_probe)]) 

          mean_population_counter1 = np.hstack((mean_population_counter1_12, np.nanmean(data_counter1[:,:,start_trial:int(start_trial+trials_to_probe)],axis=2)))
          mean_population_counter2 = np.hstack((mean_population_counter2_12, np.nanmean(data_counter2[:,:,start_trial:int(start_trial+trials_to_probe)],axis=2)))

  return mean_population_counter1, mean_population_counter2


def get_model_fit_data(data, num_block):
  '''
  Reorganizes the data into a format used for model fitting.

  Args:
    - data[dict]: the data dictionary
    - num_block[int]: the number of blocks to get data for

  Returns:
    - D[np.array]: model fitting data
  '''

  s1 = data['s1']
  s2 = data['s2']
  s_12_12 = data['s_12_12'] # nsubjects * block * stage * trial
  a = data['a']
  a_12_12 = data['a_12_12'] # nsubjects * block * trial
  r = data['r']
  r_12_12 = data['r_12_12']
  counter1 = data['counter1']
  counter2 = data['counter2']
  counter12_12 = data['counter12_12']

  num_subject = a.shape[0]
  num_trial_12 = 60
  num_trial_else = 32

  ntrials = num_trial_12 * num_block if num_block < 2 else num_trial_12 * 2 + (num_block - 2) * num_trial_else
  D = np.full((num_subject * ntrials * 10, 6), np.nan)
  k = 0

  for sub in range(num_subject):
    # first stage
    for b in range(min(num_block,2)):
      b_1 = 1
      for trial in range(num_trial_12):
        s_1 = s_12_12[sub,b,0,trial] - 1
        if np.isnan(counter12_12[sub,b,0,trial]):
          continue
        c1 = int(counter12_12[sub,b,0,trial])
        for t in range(c1):
          a_1 = a_12_12[sub,b,trial][0,t]
          r_1 = r_12_12[sub,b,trial][0,t]
          temp = np.array([sub, 1, s_1, a_1, r_1, b_1])
          if np.sum(np.isnan(temp)) > 0:
            continue
          D[k,:] = temp
          b_1 = 0
          k += 1

        s_2 = (s_12_12[sub,b,1,trial] - 1)
        if np.isnan(counter12_12[sub,b,1,trial]):
          continue
        c2 = int(counter12_12[sub,b,1,trial])
        for t in range(c2):
          a_2 = a_12_12[sub,b,trial][1,c1-1+t]
          r_2 = r_12_12[sub,b,trial][1,c1-1+t]
          D[k,:] = [sub, 2, s_2, a_2, r_2, 0]
          k += 1

    if num_block > 2:
      for b in range(num_block-2):
        b_1 = 1
        for trial in range(num_trial_else):
          s_1 = s1[sub,b,trial] - 1
          if np.isnan(counter1[sub,b,trial]):
            continue
          c1 = int(counter1[sub,b,trial])
          for t in range(c1):
            a_1 = a[sub,b,trial][0,t]
            r_1 = r[sub,b,trial][0,t]
            temp = np.array([sub, 1, s_1, a_1, r_1, b_1])
            if np.sum(np.isnan(temp)) > 0:
              continue
            D[k,:] = temp
            b_1 = 0
            k += 1

          s_2 = (s2[sub,b,trial] - 1)
          if np.isnan(counter2[sub,b,trial]):
            continue
          c2 = int(counter2[sub,b,trial])
          for t in range(c2):
            a_2 = a[sub,b,trial][1,c1-1+t]
            r_2 = r[sub,b,trial][1,c1-1+t]
            D[k,:] = [sub, 2, s_2, a_2, r_2, 0]
            k += 1

  D = D[~np.isnan(D).any(axis=1),:]

  return D


def slice_data(data, meta_data, condition, exp, cluster):
    '''
    Slices the data based on the given condition, experiment, and cluster.

    Args:
      - data[dict]: the data dictionary
      - meta_data[pd.DataFrame]: the meta data
      - condition[str]: the condition to slice the data, such as 'V1-V1'
      - exp[list]: the experiments to slice the data, such as [1] or [1, 2]
      - cluster[int]: the cluster to slice the data, such as 0

    Returns:
      - sliced_data[dict]: the sliced data
    '''
    keys = data.keys()
    sliced_data = {}
    inds = (meta_data['Experiment'].isin(exp)) & (meta_data['Cluster'] == cluster)
    if condition != 'All':
        if len(condition) == 2:
          inds = inds & (meta_data['Condition'].str.slice(0,2) == condition)
        elif len(condition) == 5:
          inds = inds & (meta_data['Condition'] == condition)
    for key in keys:
        sliced_data[key] = data[key][inds]
    return sliced_data


def concatenate_data(this_data, all_data):
  '''
  Concatenates the data.

  Args:
    - this_data[dict]: the data to concatenate
    - all_data[dict]: the data to concatenate to

  Returns:
    - all_data[dict]: the concatenated data
  '''
  keys = this_data.keys()
  
  if all_data == {}:
    for key in keys:
      all_data[key] = this_data[key].copy()
  else:
    for key in keys:
      all_data[key] = np.concatenate((all_data[key], this_data[key]), axis=0)
      
  return all_data


def aggregate_type_stage2_b8(stage2_info,trials_to_probe,start_trial=0,block=8):
  '''
  Calculates the proportions of all 4 types of choices in stage2 of the second training structure (e.g., Block 8) for all types of trials. 
    1 = Correct
    2 = Compression over stage 1
    3 = Compression over stage 2
    4 = Other

  Args:
    - stage2_info[np.array]: the preprocessed stage2 data
    - trials_to_probe[int]: the number of trials to include from start_trial
    - start_trial[int]: the trial at which to start aggregation
    - block[int]: the block number (indexing starts at 0)

  Return:
    - aggregate_type[np.array]: nsubject x 4 array containing choice type counts per subject
  '''
  # calculate data
  nsubjects = stage2_info.shape[0]
  aggregate_type = np.zeros((nsubjects,4))
  choice_types = [(0,0),(0,1),(1,0),(1,1)]

  counter = 0
  for i in range(nsubjects):
      for sf,ss in choice_types:
          for t in range(start_trial,start_trial+trials_to_probe):
              try:
                  if stage2_info[i,block-1,sf,ss,t] is None or \
                  len(stage2_info[i,block-1,sf,ss,t]) == 0 or \
                  len(stage2_info[i,block-1,sf,ss,t].ravel()) == 0: # skip if no data
                      continue
                  choice_identity = stage2_info[i,block-1,sf,ss,t][0]
                  choice_type = 5
                  if sf == 0 and ss == 0:
                    if choice_identity == 1:
                      choice_type = 4
                    elif choice_identity == 2:
                      choice_type = 2
                    elif choice_identity == 3:
                      choice_type = 3
                    elif choice_identity == 4:
                      choice_type = 1
                  elif sf == 0 and ss == 1:
                    if choice_identity == 1:
                      choice_type = 2
                    elif choice_identity == 2:
                      choice_type = 4
                    elif choice_identity == 3:
                      choice_type = 1
                    elif choice_identity == 4:
                      choice_type = 3
                  elif sf == 1 and ss == 0:
                    if choice_identity == 1:
                      choice_type = 3
                    elif choice_identity == 2:
                      choice_type = 1
                    elif choice_identity == 3:
                      choice_type = 4
                    elif choice_identity == 4:
                      choice_type = 2
                  elif sf == 1 and ss == 1:
                    if choice_identity == 1:
                      choice_type = 1
                    elif choice_identity == 2:
                      choice_type = 3
                    elif choice_identity == 3:
                      choice_type = 2
                    elif choice_identity == 4:
                      choice_type = 4

                  if choice_type < 5:
                      aggregate_type[counter,int(choice_type-1)] += 1
              except IndexError:
                  continue
      counter += 1
      
  # calculate frequencies
  with np.errstate(divide='ignore', invalid='ignore'):
    aggregate_type = (aggregate_type.T / np.sum(aggregate_type,axis=1)).T

  return aggregate_type


def aggregate_type_stage2_b9(stage2_info,trials_to_probe,start_trial=0,block=9,V2=False,V3=False):
  '''
  Calculates the proportions of all 4 types of choices in stage2 of the second training structure (e.g., Block 8) for all types of trials. 
    1 = Correct
    2 = Compression over stage 1
    3 = Compression over stage 2
    4 = Other

  Args:
    - stage2_info[np.array]: the preprocessed stage2 data
    - trials_to_probe[int]: the number of trials to include from start_trial
    - start_trial[int]: the trial at which to start aggregation
    - block[int]: the block number (indexing starts at 0)
    - V2[bool]: True if V2, False if not
    - V3[bool]: True if V3, False if not

  Return:
    - aggregate_type[np.array]: nsubject x 4 array containing choice type counts per subject
  '''
  # calculate data
  nsubjects = stage2_info.shape[0]
  aggregate_type = np.zeros((nsubjects,4))
  choice_types = [(0,0),(0,1),(1,0),(1,1)]

  counter = 0
  for i in range(nsubjects):
      for sf,ss in choice_types: 
          for t in range(start_trial,start_trial+trials_to_probe):
              try:
                  if stage2_info[i,block-1,sf,ss,t] is None or \
                  len(stage2_info[i,block-1,sf,ss,t]) == 0 or \
                  len(stage2_info[i,block-1,sf,ss,t].ravel()) == 0: # skip if no data
                      continue
                  choice_identity = stage2_info[i,block-1,sf,ss,t][0]
                  choice_type = 6

                  if (block == 7 or block == 11) and not V3 and not V2: # V1
                    if sf == 0 and ss == 0:
                      if choice_identity == 1:
                        choice_type = 4
                      elif choice_identity == 2:
                        choice_type = 3
                      elif choice_identity == 3:
                        choice_type = 2
                      elif choice_identity == 4:
                        choice_type = 1
                    elif sf == 0 and ss == 1:
                      if choice_identity == 1:
                        choice_type = 2
                      elif choice_identity == 2:
                        choice_type = 1
                      elif choice_identity == 3:
                        choice_type = 4
                      elif choice_identity == 4:
                        choice_type = 3
                    elif sf == 1 and ss == 0:
                      if choice_identity == 1:
                        choice_type = 3
                      elif choice_identity == 2:
                        choice_type = 4
                      elif choice_identity == 3:
                        choice_type = 1
                      elif choice_identity == 4:
                        choice_type = 2
                    elif sf == 1 and ss == 1:
                      if choice_identity == 1:
                        choice_type = 1
                      elif choice_identity == 2:
                        choice_type = 2
                      elif choice_identity == 3:
                        choice_type = 3
                      elif choice_identity == 4:
                        choice_type = 4
                  elif (block == 7 or block == 11) and V2: 
                    if sf == 0 and ss == 0:
                      if choice_identity == 1:
                        choice_type = 2
                      elif choice_identity == 2:
                        choice_type = 1
                      elif choice_identity == 3:
                        choice_type = 4
                      elif choice_identity == 4:
                        choice_type = 3
                    elif sf == 0 and ss == 1:
                      if choice_identity == 1:
                        choice_type = 3
                      elif choice_identity == 2:
                        choice_type = 4
                      elif choice_identity == 3:
                        choice_type = 2
                      elif choice_identity == 4:
                        choice_type = 1
                    elif sf == 1 and ss == 0:
                      if choice_identity == 1:
                        choice_type = 1
                      elif choice_identity == 2:
                        choice_type = 2
                      elif choice_identity == 3:
                        choice_type = 3
                      elif choice_identity == 4:
                        choice_type = 4
                    elif sf == 1 and ss == 1:
                      if choice_identity == 1:
                        choice_type = 3
                      elif choice_identity == 2:
                        choice_type = 4
                      elif choice_identity == 3:
                        choice_type = 1
                      elif choice_identity == 4:
                        choice_type = 2
                  elif (block == 7 or block == 11) and V3: 
                    if sf == 0 and ss == 0:
                      if choice_identity == 1:
                        choice_type = 4
                      elif choice_identity == 2:
                        choice_type = 1
                      elif choice_identity == 3:
                        choice_type = 3
                      elif choice_identity == 4:
                        choice_type = 2
                    elif sf == 0 and ss == 1:
                      if choice_identity == 1:
                        choice_type = 2
                      elif choice_identity == 2:
                        choice_type = 3
                      elif choice_identity == 3:
                        choice_type = 1
                      elif choice_identity == 4:
                        choice_type = 4
                    elif sf == 1 and ss == 0:
                      if choice_identity == 1:
                        choice_type = 3
                      elif choice_identity == 2:
                        choice_type = 2
                      elif choice_identity == 3:
                        choice_type = 4
                      elif choice_identity == 4:
                        choice_type = 1
                    elif sf == 1 and ss == 1:
                      if choice_identity == 1:
                        choice_type = 1
                      elif choice_identity == 2:
                        choice_type = 4
                      elif choice_identity == 3:
                        choice_type = 2
                      elif choice_identity == 4:
                        choice_type = 3
                  else:
                    if sf == 0 and ss == 0:
                      if choice_identity == 1:
                        choice_type = 1
                      elif choice_identity == 2:
                        choice_type = 3
                      elif choice_identity == 3:
                        choice_type = 2
                      elif choice_identity == 4:
                        choice_type = 4
                    elif sf == 0 and ss == 1:
                      if choice_identity == 1:
                        choice_type = 3
                      elif choice_identity == 2:
                        choice_type = 1
                      elif choice_identity == 3:
                        choice_type = 4
                      elif choice_identity == 4:
                        choice_type = 2
                    elif sf == 1 and ss == 0:
                      if choice_identity == 1:
                        choice_type = 2
                      elif choice_identity == 2:
                        choice_type = 4
                      elif choice_identity == 3:
                        choice_type = 1
                      elif choice_identity == 4:
                        choice_type = 3
                    elif sf == 1 and ss == 1:
                      if choice_identity == 1:
                        choice_type = 4
                      elif choice_identity == 2:
                        choice_type = 2
                      elif choice_identity == 3:
                        choice_type = 3
                      elif choice_identity == 4:
                        choice_type = 1

                  if choice_type < 5:
                      aggregate_type[counter,int(choice_type-1)] += 1
              except IndexError:
                  continue
      counter += 1

  # calculate frequencies
  with np.errstate(divide='ignore', invalid='ignore'):
    aggregate_type = (aggregate_type.T / np.sum(aggregate_type,axis=1)).T

  return aggregate_type