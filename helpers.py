import numpy as np
import copy

def extract_stage2_info(data,experiment,same_same=True,same_diff=True,diff_same=True,diff_diff=True):
  '''
  Extracts stage2 info from preprocessed data. In particular, reorganize data
  in F1 or F2 by S1 or S2.

  Args:
    - data: the data dictionary returned by restructure_data_ca
    - experiment: 'V1-V1', 'V1-V2', 'V1-V3', 'V2-V1', 'V2-V2', 'V3-V1', 'V3-V3'
    - prev_same: True, False, or None; indicating whether the stimulus of the previous trial was the same as the current stim

  Returns:
    - stage2_info: reorganized stage2 data
  '''
  nsubjects = data['tr'].shape[0]
  nblocks =12
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
    # find the permutation for A1-4
    all_actions = tr[subject,4:]

    for block in range(nblocks):
      index = np.zeros((2,2),dtype=int)
      prev_stim_1 = None
      prev_stim_2 = None

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
          this_s1 = s_12_12[subject, block, 0, trial]
          this_s2 = s_12_12[subject, block, 1, trial]
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
          this_s1 = s1[subject,block-2,trial]
          this_s2 = s2[subject,block-2,trial]

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
          result = np.array([choice_identity, temp_RT, choice_type, temp_counter1, temp_counter2, choice_identity_2, trial])

        if index[sf-1,ss-1] == 8:
          continue

        if (not same_same and prev_stim_1 == this_s1 and prev_stim_2 == this_s2) or \
        (not same_diff and prev_stim_1 == this_s1 and prev_stim_2 != this_s2) or \
        (not diff_same and prev_stim_1 != this_s1 and prev_stim_2 == this_s2) or \
        (not diff_diff and prev_stim_1 != this_s1 and prev_stim_2 != this_s2):
          proceed = False
        else:
          proceed = True 

        prev_stim_1 = this_s1
        prev_stim_2 = this_s2

        if not proceed:
          continue

        stage2_info[subject,block,sf-1,ss-1,index[sf-1,ss-1]] = result # -1 for 0-indexing
        index[sf-1,ss-1] = index[sf-1,ss-1] + 1

  return stage2_info


def get_choice_type_stage2_V1(block, sf, ss, temp_a, actions):
  '''
  This function outputs the choice type in the second stage of OT-CA1. The
  choice types are defined as follows:

  In Blocks 1-6, 8, 10 and 12
  1 = correct
  2 = sequence
  3 = non-sequence
  4 = f-choice
  5 = all else

  In Blocks 7, 9 and 11 (F1, S1) and (F2, S2)
  1 = correct
  2 = option transfer
  3 = other
  4 = f-choice
  5 = all else

  In all blocks (F1, S2) and (F2, S1)

  Args:
    - block: the block number (indexing starts at 0)
    - sf: stage1 stimulus
    - ss: stage2 stimulus
    - temp_a: first action of the stage2
    - actions: all actions in stage1, ordered

  Returns:
    - choice_type
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
  This function outputs the choice type in the second stage of OT-CA1. The
  choice types are defined as follows:

  In Blocks 1-6, 8, 10 and 12
  1 = correct
  2 = sequence
  3 = non-sequence
  4 = f-choice
  5 = all else

  In Blocks 7, 9 and 11 (F1, S1) and (F2, S2)
  1 = correct
  2 = option transfer
  3 = other
  4 = f-choice
  5 = all else

  In all blocks (F1, S2) and (F2, S1)

  Args:
    - block: the block number (indexing starts at 0)
    - sf: stage1 stimulus
    - ss: stage2 stimulus
    - temp_a: first action of the stage2
    - actions: all actions in stage1, ordered

  Returns:
    - choice_type
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
  This function outputs the choice type in the second stage of OT-CA1. The
  choice types are defined as follows:

  In Blocks 1-6, 8, 10 and 12
  1 = correct
  2 = sequence
  3 = non-sequence
  4 = f-choice
  5 = all else

  In Blocks 7, 9 and 11 (F1, S1) and (F2, S2)
  1 = correct
  2 = option transfer
  3 = other
  4 = f-choice
  5 = all else

  In all blocks (F1, S2) and (F2, S1)

  Args:
    - block: the block number (indexing starts at 0)
    - sf: stage1 stimulus
    - ss: stage2 stimulus
    - temp_a: first action of the stage2
    - actions: all actions in stage1, ordered

  Returns:
    - choice_type
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


def calc_mean(data, start_trial=0, trials_to_probe=10, exclude_repeated_trials=False, stage1_correct=False, first_press_accuracy=False):
  '''
  Calculates the mean number of key presses for each subject for both stages.

  Args:
    - data: the preprocessed data dictionary
    - blocks: a list of block numbers to calculate; start indexing at 0
    - start_trial: index of first trial
    - trials_to_probe: the number of trials to include
    - inds_to_include: a list of subjects to include
    - exclude_repeated_trials: if True, exclude a trial that has the exact same stimuli as the previous one

  Returns:
    - mean_population_counter1: the means of all subjects in the first stage
    - mean_population_counter2: the means of all subjects in the second stage
  '''
  data_counter12_12 = copy.deepcopy(data['counter12_12'])
  data_counter1 = copy.deepcopy(data['counter1'])
  data_counter2 = copy.deepcopy(data['counter2'])

  if stage1_correct:
    data_counter12_12[:,:,1,:][np.where(data_counter12_12[:,:,1,:]>1)] = np.nan 
    data_counter2[np.where(data_counter1>1)] = np.nan 

  nsubjects = data_counter2.shape[0]

  mean_population_counter1_12 = np.full((nsubjects,2),np.nan)
  mean_population_counter2_12 = np.full((nsubjects,2),np.nan)

  for sub in range(nsubjects):
    if exclude_repeated_trials: # set repeated trials to nan
      stims_12_12 = data['s_12_12'][sub,:,0,:] * 2 + data['s_12_12'][sub,:,1,:]
      stims = data['s1'][sub,:,:] * 2 + data['s2'][sub,:,:]

      rep_inds_12 = np.ones_like(stims_12_12)
      rep_inds_12[:,1:] = stims_12_12[:,1:] - stims_12_12[:,:-1]
      rep_inds_12 = np.argwhere(np.abs(rep_inds_12)<0.1)

      rep_inds = np.ones_like(stims) 
      rep_inds[:,1:] = stims[:,1:] - stims[:,:-1]
      rep_inds = np.argwhere(np.abs(rep_inds)<0.1)

      data_counter12_12[sub,rep_inds_12[:,0],:,rep_inds_12[:,1]] = np.nan
      data_counter1[sub,rep_inds[:,0],rep_inds[:,1]] = np.nan 
      data_counter2[sub,rep_inds[:,0],rep_inds[:,1]] = np.nan 

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


def get_model_fit_data(data,num_block):
  '''
  Gets the data for fitting model.

  Args:
    - data: the data dictionary
    - naive: True if naive model, False otherwise

  Returns:
    - D_1: first stage model fitting data
    - D_2: second stage model fitting data
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

  D = np.full((num_subject * (2 * num_trial_12 + (num_block - 2) * num_trial_else) * 10, 6), np.nan)
  k = 0

  for sub in range(num_subject):
    # first stage
    for b in range(2):
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
    keys = data.keys()
    sliced_data = {}
    inds = (meta_data['Condition'] == condition) & (meta_data['Experiment'].isin(exp)) & (meta_data['Cluster'] == cluster)
    for key in keys:
        sliced_data[key] = data[key][inds]
    return sliced_data


def concatenate_data(this_data, all_data):
  keys = this_data.keys()
  
  if all_data == {}:
    for key in keys:
      all_data[key] = this_data[key].copy()
  else:
    for key in keys:
      all_data[key] = np.concatenate((all_data[key], this_data[key]), axis=0)
      
  return all_data