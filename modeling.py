import numpy as np
from scipy.special import softmax
from scipy.optimize import differential_evolution
import progressbar
import matplotlib.pyplot as plt

def option_model_nllh(params, D, structure, meta_learning=True):
	'''
	Computes the negative log likelihood of the data D given the option model.
	'''
	[alpha_2] = params
	beta_2 = 5
	concentration_2 = 0.2

	llh = 0
	num_block = 12
	s_2 = a_2 = -1
	block = -1

	nTS_2 = 2 # initialize the number of task-set in the second stage
	TS_2s = np.ones((nTS_2,2,4)) / 4
	nC_2 = 2 * num_block
	PTS_2 = np.zeros((nTS_2,nC_2)) 
	# PTS_2[0] = 1
	PTS_2[0,0::2] = 1
	PTS_2[1,1::2] = 1
	encounter_matrix_2 = np.zeros(nC_2)
	encounter_matrix_2[:nTS_2] = 1

	for t in range(D.shape[0]):	
		stage = int(D[t,1])

		if int(D[t,5]) == 1: # new block
			block += 1

		if stage == 1:
			s_1 = int(D[t, 2])
		elif stage == 2:
			s_2 = int(D[t, 2])
			a_2 = int(D[t, 3]) - 4
			r_2 = int(D[t, 4])

			# (v) Second stage starts
			if structure == 'backward':
				cue = s_2
				state = s_1
			elif structure == 'forward':
				cue = s_1
				state = s_2
			c_2 = block * 2 + cue # The context of the second stage
			c_2_alt = block * 2 + (1 - cue)
			for this_c_2 in sorted([c_2, c_2_alt]):
				if encounter_matrix_2[this_c_2] == 0:
					PTS_2 = new_SS_update_option(PTS_2, this_c_2, concentration_2)
					TS_2s = np.vstack((TS_2s, [np.ones((2,4)) / 4])) # initialize Q-values for new TS creation
					nTS_2 += 1
					encounter_matrix_2[this_c_2] = 1

			Q_full = TS_2s[:, state]
			pchoice_2_full = softmax(beta_2 * Q_full, axis=-1)
			pchoice_2 = np.sum(pchoice_2_full * PTS_2[:,c_2].reshape(-1,1), axis=0)
			llh += np.log(pchoice_2[a_2-1])

			if r_2 == 0:
				PTS_2[:,c_2] *= (1 - TS_2s[:,state,a_2-1])
			else:
				PTS_2[:,c_2] *= TS_2s[:,state,a_2-1] 
			PTS_2[:,c_2] += 1e-6
			PTS_2[:,c_2] /= np.sum(PTS_2[:,c_2])

			TS_2s[:,state,a_2-1] += alpha_2 * (r_2 - TS_2s[:,state,a_2-1]) * PTS_2[:,c_2]

	return -llh


def option_model(num_subject, params, experiment, structure, meta_learning=True):
	[alpha_2] = params

	num_block = 6 if experiment == 'All' else 12
	num_trial_12 = 60
	num_trial_else = 32
	beta_2 = 5
	# concentration_2 = 10**concentration_2
	concentration_2 = 0.2

	nC_2 = 2 * num_block

	population_counter1 = np.zeros((num_subject,num_block-2,num_trial_else))
	population_counter2 = np.zeros_like(population_counter1)
	s_12_12 = np.zeros((num_subject,2,2,num_trial_12))
	s1 = np.zeros_like(population_counter1)
	s2 = np.zeros_like(population_counter1)
	r_12_12 = np.empty((num_subject,2,num_trial_12), dtype='object')
	r = np.empty_like(population_counter1,dtype='object')
	a_12_12 = np.empty((num_subject,2,num_trial_12),dtype='object')
	a = np.empty_like(population_counter1,dtype='object')
	tr = np.zeros((num_subject,8))
	population_counter1_12 = np.zeros((num_subject,2,num_trial_12))
	population_counter2_12 = np.zeros((num_subject,2,num_trial_12))
	p_policies_history = np.zeros((num_subject,num_block,num_trial_12,3))
	TS_2_history = np.full((num_subject,num_block*2,num_trial_12), np.nan)

	# run the model
	for sub in range(num_subject):

		# 1. set transitions
		transition_step1, transition_step2 = set_contingency()

		transition_train1_step1 = transition_step1[:2]
		transition_train1_step2 = transition_step2[:2,:]
		transition_train2_step1 = transition_step1[2:]
		transition_train2_step2 = transition_step2[2:,:]

		transition_ca1_step2 = np.array([[transition_step2[1][1], transition_step2[0][1]], [transition_step2[1][0], transition_step2[0][0]]])
		transition_ca2_step2 = np.array([[transition_step2[0][1], transition_step2[1][1]], [transition_step2[0][0], transition_step2[1][0]]])
		transition_ca3_step2 = np.array([[transition_step2[0][1], transition_step2[1][0]], [transition_step2[1][1], transition_step2[0][0]]])

		tr[sub,:] = [1,2,3,4,8,6,5,7]

		# 2. initialize other subject-specific task variables
		s_1_all = np.zeros((num_block-2,num_trial_else))
		s_2_all = np.zeros_like(s_1_all)
		counter_1_all = np.zeros_like(s_1_all)
		counter_2_all = np.zeros_like(s_1_all)
		a_all = np.empty((num_block-2,num_trial_else),dtype='object')
		r_all = np.empty_like(a_all,dtype='object')

		nTS_2 = 2 # initialize the number of task-set in the second stage
		TS_2s = np.ones((nTS_2,2,4)) / 4
		PTS_2 = np.zeros((nTS_2,nC_2)) 
		# PTS_2[0] = 1
		PTS_2[0,0::2] = 1
		PTS_2[1,1::2] = 1
		encounter_matrix_2 = np.zeros(nC_2)
		encounter_matrix_2[:nTS_2] = 1

		# 3. start looping over all blocks
		for block in range(num_block):
			num_trial = num_trial_12 if block < 2 else num_trial_else

			# initialize stimuli sequence
			stimulus_1, stimulus_2, _ = prepare_train_stim_sequence(num_trial / 2)
			if block < 2:
				counter_1_temp = np.full(60, np.nan)
				counter_2_temp = np.full(60, np.nan)
			else:
				counter_1_temp = np.full(32, np.nan)
				counter_2_temp = np.full(32, np.nan)

			# 4. start looping over all trials
			for trial in range(num_trial):	

				# (i) present stimulus
				s_1 = int(stimulus_1[trial])
				s_2 = int(stimulus_2[trial])

				# (ii) initialize trial-specific variables
				correct_1 = 0 # keep track of whether correct or not in the first stage
				correct_2 = 0 # keep track of whether correct or not in the second stage
				a_1_temp = [] # action holder at trial level for the first stage
				a_2_temp = [] # action holder at trial level for the second stage
				counter_1 = 0 # counter holder at trial level for the first stage
				counter_2 = 0 # counter holder at trial level for the second stage
				# correct answer for both stages
				correct_action_1 = find_correct_action(s_1, s_2, transition_train1_step1, transition_train1_step2, transition_train2_step1, transition_train2_step2, transition_ca1_step2, transition_ca2_step2, transition_ca3_step2, block, 1, experiment) 
				correct_action_2 = find_correct_action(s_1, s_2, transition_train1_step1, transition_train1_step2, transition_train2_step1, transition_train2_step2, transition_ca1_step2, transition_ca2_step2, transition_ca3_step2, block, 2, experiment) 

				# (iv) first stage starts
				a_1 = correct_action_1
				a_1_temp.append(a_1) # append the action taken to the list of actions in the first stage
				counter_1 += 1

				# (v) Second stage starts
				actions_tried = set()
				if structure == 'backward':
					cue = s_2 
					state = s_1
				elif structure == 'forward':
					cue = s_1
					state = s_2
				c_2 = block * 2 + cue # The context of the second stage
				c_2_alt = block * 2 + (1 - cue)
				while correct_2 == 0 and counter_2 < 10:
					for this_c_2 in sorted([c_2, c_2_alt]):
						if encounter_matrix_2[this_c_2] == 0:
							PTS_2 = new_SS_update_option(PTS_2, this_c_2, concentration_2)
							TS_2s = np.vstack((TS_2s, [np.ones((2,4)) / 4])) # initialize Q-values for new TS creation
							nTS_2 += 1
							encounter_matrix_2[this_c_2] = 1

					Q_full = TS_2s[:, state]
					pchoice_2_full = softmax(beta_2 * Q_full, axis=-1)
					pchoice_2 = np.sum(pchoice_2_full * PTS_2[:,c_2].reshape(-1,1), axis=0)

					a_2 = np.random.choice(np.arange(1,5), 1, p=pchoice_2)[0]
					actions_tried.add(a_2-1)
					a_2_temp.append(a_2+4) # append the action taken to the list of actions in the second stage
					counter_2 += 1
					correct_2 = int(a_2 + 4 == correct_action_2)

					# Use the result to update PTS_2 with Bayes Rule
					if correct_2 == 0:
						PTS_2[:,c_2] *= (1 - TS_2s[:,state,a_2-1])
					else:
						PTS_2[:,c_2] *= TS_2s[:,state,a_2-1] 
					PTS_2[:,c_2] += 1e-6
					PTS_2[:,c_2] /= np.sum(PTS_2[:,c_2])

					TS_2s[:,state,a_2-1] += alpha_2 * (correct_2 - TS_2s[:,state,a_2-1]) * PTS_2[:,c_2]

				# Record variables per trial
				counter_1_temp[trial] = counter_1
				counter_2_temp[trial] = counter_2
				a_1_temp = np.array(a_1_temp).ravel()
				a_2_temp = np.array(a_2_temp).ravel()

				if block < 2:
					a_12_12[sub,block,trial] = join_actions(a_1_temp, a_2_temp)
					s_12_12[sub,block,0,trial] = s_1 + 1
					s_12_12[sub,block,1,trial] = s_2 + 1
					# make reward structure
					r_temp = np.full((2, counter_1 + max(0, counter_2-1)), np.nan)
					if counter_1 > 1:
						# pad some 0 in reward for incorrect presses in the first stage
						r_temp[0,:counter_1-1] = 0 
					r_temp[0,counter_1-1] = 1 
					if counter_2 > 1:
						r_temp[1, counter_1-1:(counter_1+counter_2-3)] = 0
					r_temp[1, -1] = 1
					r_12_12[sub,block,trial] = r_temp
				else: # record action, RT and reward only for Blocks 3-8
					a_all[block-2,trial] = join_actions(a_1_temp, a_2_temp)
					# Come up with reward
					r_temp = np.empty((2, counter_1 + max(0,counter_2-1)))
					r_temp[:] = np.nan 
					if counter_1 > 1:
						# pad some 0 in reward for incorrect presses in the first stage
						r_temp[0,:counter_1-1] = 0 
					r_temp[0,counter_1-1] = 1 
					if counter_2 > 1:
						r_temp[1,counter_1-1:(counter_1+counter_2-3)] = 0 
					r_temp[1,-1] = 1 
					r_all[block-2,trial] = r_temp # record reward for this trial

				# skip to the next block if performance is good enough 
				if block < 2 and trial >= 32 and np.mean(counter_1_temp[trial-10:trial]) < 1.5 and np.mean(counter_2_temp[trial-10:trial]) < 1.5: 
					trial = 59

			# Recording variables per block
			if block < 2: # record only counter for Blocks 1 and 2
				population_counter1_12[sub,block,:] = counter_1_temp
				population_counter2_12[sub,block,:] = counter_2_temp
			else: # % record stimulus sequence and counter for Blocks 3-8
				s_1_all[block-2,:] = stimulus_1
				s_2_all[block-2,:] = stimulus_2
				counter_1_all[block-2,:] = counter_1_temp
				counter_2_all[block-2,:] = counter_2_temp

		s1[sub,:,:] = s_1_all + 1
		s2[sub,:,:] = s_2_all + 1
		population_counter1[sub,:,:] = counter_1_all
		population_counter2[sub,:,:] = counter_2_all 
		a[sub,:,:] = a_all
		r[sub,:,:] = r_all

	counter12_12 = np.zeros((num_subject,2,2,num_trial_12))
	counter12_12[:,:,0,:] = population_counter1_12
	counter12_12[:,:,1,:] = population_counter2_12

	# Package the output
	data = {'tr': tr, 'a':a, 'r':r, 's1':s1, 's2':s2, 'counter1':population_counter1, 'counter2':population_counter2, \
		 'counter12_12':counter12_12, 'a_12_12':a_12_12, 's_12_12':s_12_12, 'r_12_12':r_12_12, \
			'p_policies_history': p_policies_history, 'TS_2_history': TS_2_history} 	
	return data


def option_model_nllh_backup(params, D, structure, meta_learning=True):
	'''
	Computes the negative log likelihood of the data D given the option model.
	'''
	[alpha_2] = params
	# alpha_2 = 1
	beta_2 = 5
	prior = 0.25
	epsilon = 0.0
	# eps_meta = 10**eps_meta if meta_learning else 0.0
	# concentration_2 = 10**concentration_2
	concentration_2 = 0.2
	eps_meta = 0.01

	fit_all_actions = True

	llh = 0
	num_block = 12
	s_2 = a_2 = -1
	last_stage = 2
	block = -1
	trial = -1

	nTS = 1 # initialize the number of task-set in the first stage
	nTS_2 = 1 # initialize the number of task-set in the second stage
	TSs = np.empty((nTS,2,4)) # each cell is the policy for a new TS
	TS_2s = np.empty((nTS_2,2,4)) 
	TSs[0,:,:] = np.ones((2,4)) / 4 # the first built-in TS is just random
	TS_2s[0,:,:] = np.ones((2,4)) / 4
	nC_2 = 2 * num_block
	PTS_2 = np.ones((nTS_2,nC_2))
	p_policies = np.array([1-eps_meta-prior, prior, eps_meta])
	p_policies_softmax = softmax(beta_2 * p_policies)
	encounter_matrix_2 = np.zeros(nC_2)

	for t in range(D.shape[0]):	
		stage = int(D[t,1])
        
		b_1 = int(D[t,5])
		if b_1 == 1: # new block
			block += 1
			block_trial = 0

		# Reset actions_tried when starting a new stage
		if last_stage != stage:
			last_stage = stage
			actions_tried = set()
			if stage == 1:
				trial += 1
				block_trial += 1

		if stage == 1:
			s_1 = int(D[t, 2])
		elif stage == 2:
			s_2 = int(D[t, 2])
			a_2 = int(D[t, 3]) - 4
			r_2 = int(D[t, 4])

			# (v) Second stage starts
			if structure == 'backward':
				cue = s_2
				state = s_1
			elif structure == 'forward':
				cue = s_1
				state = s_2
			c_2 = block * 2 + cue # The context of the second stage
			c_2_alt = block * 2 + (1 - cue)
			for this_c_2 in sorted([c_2, c_2_alt]):
				if encounter_matrix_2[this_c_2] == 0:
					if this_c_2 > 0:
						PTS_2 = new_SS_update_option(PTS_2, this_c_2, concentration_2)
						TS_2s = np.vstack((TS_2s, [np.ones((2,4)) / 4])) # initialize Q-values for new TS creation
						nTS_2 += 1
					# And finally mark this context as encountered
					encounter_matrix_2[this_c_2] = 1
			# biases = np.zeros((nTS_2, nTS_2))
			# for i in range(block):
			# 	this_TS_1 = np.argmax(PTS_2[:-2,i*2])
			# 	this_TS_2 = np.argmax(PTS_2[:-2,i*2+1])
			# 	biases[this_TS_1, this_TS_2] += 1
			# 	biases[this_TS_2, this_TS_1] += 1

			# # Now we have picked the second stage TS, just pick an action based on the policy of this TS
			# TS_2_alt = np.argmax(PTS_2[:,c_2_alt])
			# if block > 0:
			# 	bias = biases[TS_2_alt].copy()
			# 	b = np.max(PTS_2[:,c_2_alt])
			# 	if np.sum(bias) > 0 and np.max(PTS_2[:,c_2]) < 0.7:
			# 		bias /= np.sum(bias)
			# 		PTS_2[:,c_2] = PTS_2[:,c_2] * (1 - b) + bias * b

			lt_2 = 0
			# if structure == 'backward':
			# 	Q_compress_1 = np.mean(TS_2s * PTS_2[:,c_2].reshape(-1, 1, 1), axis=(0,1))
			# 	Q_compress_2 = np.mean(TS_2s * (PTS_2[:,c_2]/2 + PTS_2[:,c_2_alt]/2).reshape(-1, 1, 1), axis=0)[state]
			# elif structure == 'forward':
			# 	Q_compress_1 = np.mean(TS_2s * (PTS_2[:,c_2]/2 + PTS_2[:,c_2_alt]/2).reshape(-1, 1, 1), axis=0)[state]
			# 	Q_compress_2 = np.mean(TS_2s * PTS_2[:,c_2].reshape(-1, 1, 1), axis=(0,1))
			# Q_full = np.mean(TS_2s * PTS_2[:,c_2].reshape(-1, 1, 1), axis=0)[state]
			
			# if len(actions_tried) > 0:
			# 	Q_compress_1[list(actions_tried)] = -1e20
			# 	Q_compress_2[list(actions_tried)] = -1e20
			# 	Q_full[list(actions_tried)] = -1e20

			# pchoice_2_compress_1 = softmax(beta_2 * Q_compress_1) * (1-epsilon) + epsilon / 4
			# pchoice_2_full = softmax(beta_2 * Q_full) * (1-epsilon) + epsilon / 4
			# pchoice_2_compress_2 = softmax(beta_2 * Q_compress_2) * (1-epsilon) + epsilon / 4
			# p_policies_softmax = softmax(beta_2 * p_policies)
			# pchoice_2 = p_policies_softmax[0] * pchoice_2_compress_1 \
			# 			+ p_policies_softmax[1] * pchoice_2_compress_2 \
            #             + p_policies_softmax[2] * pchoice_2_full
			
			# lt_2 = pchoice_2[a_2-1]
			# actions_tried.add(a_2-1)
			# llh += np.log(lt_2)

			# Use the result to update PTS_2 with Bayes Rule
			# if meta_learning:
			# 	for TS_2 in range(PTS_2.shape[0]):
			# 		Q_full = TS_2s[TS_2,state].copy()
			# 		# if len(actions_tried) > 0:
			# 		# 	Q_full[list(actions_tried)] = 0
			# 		pchoice_2_full = softmax(beta_2 * Q_full)

			# 		if not fit_all_actions and len(actions_tried) > 0:
			# 			continue

			# 		if structure == 'backward':
			# 			Q_compress_1 = np.mean(TS_2s[TS_2], axis=(0))
			# 			pchoice_2_compress_1 = softmax(beta_2 * Q_compress_1)

			# 			for TS_2_alt in range(PTS_2.shape[0]):
			# 				Q_compress_2 = (TS_2s[TS_2]/2 + TS_2s[TS_2_alt]/2)[state]
			# 				pchoice_2_compress_2 = softmax(beta_2 * Q_compress_2)
			# 				pchoice_2 = p_policies_softmax[0] * pchoice_2_compress_1 \
			# 							+ p_policies_softmax[1] * pchoice_2_compress_2 \
			# 							+ p_policies_softmax[2] * pchoice_2_full
			# 				lt_2 += pchoice_2[a_2-1] * PTS_2[TS_2,c_2] * PTS_2[TS_2_alt,c_2] 

			Q_full = TS_2s[:, state]
			pchoice_2_full = softmax(beta_2 * Q_full, axis=-1)
			lt_2 = np.sum(pchoice_2_full[:, a_2-1] * PTS_2[:, c_2])
			# TS_2s[:,state,a_2-1] += alpha_2 * (r_2 - TS_2s[:,state,a_2-1]) * PTS_2[:,c_2]

			if fit_all_actions or len(actions_tried) == 0:
				llh += np.log(lt_2 * (1 - epsilon) + epsilon / 4)

			if r_2 == 0:
				PTS_2[:,c_2] *= (1 - TS_2s[:,state,a_2-1])
			else:
				PTS_2[:,c_2] *= TS_2s[:,state,a_2-1] 
			PTS_2[:,c_2] += 1e-6
			PTS_2[:,c_2] /= np.sum(PTS_2[:,c_2])

			# TS_2 = np.random.choice(np.arange(PTS_2.shape[0]), 1, p=PTS_2[:,c_2])[0] # np.argmax(PTS_2[:,c_2])
			# TS_2 = np.argmax(PTS_2[:,c_2])
			# TS_2s[TS_2,state,a_2-1] += alpha_2 * (r_2 - TS_2s[TS_2,state,a_2-1])
			TS_2s[:,state,a_2-1] += alpha_2 * (r_2 - TS_2s[:,state,a_2-1]) * PTS_2[:,c_2]

			# if meta_learning:
			# 	p_policies[0] *= pchoice_2_compress_1[a_2-1]
			# 	p_policies[1] *= pchoice_2_compress_2[a_2-1]
			# 	p_policies[2] *= pchoice_2_full[a_2-1]
			# 	p_policies /= np.sum(p_policies)
			# 	if np.min(p_policies) < eps_meta:
			# 		p_policies += eps_meta
			# 	p_policies /= np.sum(p_policies)
			# 	p_policies_softmax = softmax(beta_2 * p_policies)

			actions_tried.add(a_2-1)

	return -llh


def option_model_backup(num_subject, params, experiment, structure, meta_learning=True):
	'''
	Fits the option model to the data of the OT-CA1-CA1 task.

	Args:
		- num_subjects: number of subjects
		- alpha_1, alpha_2: the Q learning rates of stages 1 and 2
		- beta_1, beta_2: the betas of softmax for stages 1 and 2 
		- forget_1, forget_2: the forget rates of stages 1 and 2 
		- alpha_S1, alpha_S2: alpha in the chinese restaurant process for stages 1 and 2
		- experiment: 'CA1', 'CA1-CA1', 'CA2', 'CA2-CA2', 'CA1-CA2', or 'CA2-CA2'
		- version: 1, 2, 3 and 4
		- debug: whether to print things for debugging

	Returns:
		- data: the preprocessed data dictionary 
		- mean_counter1, mean_counter2: the mean of number of key presses over trials per subject per block 
		- se_counter1, se_counter2: the sem corresponding to the above mean
		- num_trial_finished: the sample sizes of the above means
	'''
	[alpha_2] = params

	num_block = 6 if experiment == 'All' else 12
	num_trial_12 = 60
	num_trial_else = 32
	alpha_1 = 1
	beta_1 = beta_2 = 5
	concentration_1 = 0.2
	prior = 0.25
	epsilon = 0.0

	eps_meta = 0.01 
	# concentration_2 = 10**concentration_2
	concentration_2 = 0.2
	nC = num_block
	nC_2 = 2 * num_block

	population_counter1 = np.zeros((num_subject,num_block-2,num_trial_else))
	population_counter2 = np.zeros_like(population_counter1)
	s_12_12 = np.zeros((num_subject,2,2,num_trial_12))
	s1 = np.zeros_like(population_counter1)
	s2 = np.zeros_like(population_counter1)
	r_12_12 = np.empty((num_subject,2,num_trial_12), dtype='object')
	r = np.empty_like(population_counter1,dtype='object')
	a_12_12 = np.empty((num_subject,2,num_trial_12),dtype='object')
	a = np.empty_like(population_counter1,dtype='object')
	tr = np.zeros((num_subject,8))
	population_counter1_12 = np.zeros((num_subject,2,num_trial_12))
	population_counter2_12 = np.zeros((num_subject,2,num_trial_12))
	p_policies_history = np.zeros((num_subject,num_block,num_trial_12,3))
	TS_2_history = np.full((num_subject,num_block*2,num_trial_12), np.nan)

	# run the model
	for sub in range(num_subject):

		# 1. set transitions
		transition_step1, transition_step2 = set_contingency()

		transition_train1_step1 = transition_step1[:2]
		transition_train1_step2 = transition_step2[:2,:]
		transition_train2_step1 = transition_step1[2:]
		transition_train2_step2 = transition_step2[2:,:]

		transition_ca1_step2 = np.array([[transition_step2[1][1], transition_step2[0][1]], [transition_step2[1][0], transition_step2[0][0]]])
		transition_ca2_step2 = np.array([[transition_step2[0][1], transition_step2[1][1]], [transition_step2[0][0], transition_step2[1][0]]])
		transition_ca3_step2 = np.array([[transition_step2[0][1], transition_step2[1][0]], [transition_step2[1][1], transition_step2[0][0]]])

		tr[sub,:] = [1,2,3,4,8,6,5,7]

		# 2. initialize other subject-specific task variables
		s_1_all = np.zeros((num_block-2,num_trial_else))
		s_2_all = np.zeros_like(s_1_all)
		counter_1_all = np.zeros_like(s_1_all)
		counter_2_all = np.zeros_like(s_1_all)
		a_all = np.empty((num_block-2,num_trial_else),dtype='object')
		r_all = np.empty_like(a_all,dtype='object')

		nTS = 1 # initialize the number of task-set in the first stage
		nTS_2 = 1 # initialize the number of task-set in the second stage
		TSs = np.empty((nTS,2,4)) # each cell is the policy for a new TS
		TS_2s = np.empty((nTS,2,4)) 
		TSs[0,:,:] = np.ones((2,4)) / 4 # the first built-in TS is just random
		TS_2s[0,:,:] = np.ones((2,4)) / 4
		PTS = np.ones((nTS,nC_2))
		PTS_2 = np.ones((nTS_2,nC_2))
		
		# compression over stage 1, compression over stage 2, full hierarchical
		p_policies = np.array([1-eps_meta-prior, prior, eps_meta]) 
		p_policies_softmax = softmax(beta_2 * p_policies)
		encounter_matrix = np.zeros(nC)
		encounter_matrix_2 = np.zeros(nC_2)

		# 3. start looping over all blocks
		for block in range(num_block):
			if block < 2:
				num_trial = num_trial_12
			else:
				num_trial = num_trial_else

			# initialize stimuli sequence
			stimulus_1, stimulus_2, _ = prepare_train_stim_sequence(num_trial / 2)
			if block < 2:
				counter_1_temp = np.full(60, np.nan)
				counter_2_temp = np.full(60, np.nan)
			else:
				counter_1_temp = np.full(32, np.nan)
				counter_2_temp = np.full(32, np.nan)

			# 4. start looping over all trials
			for trial in range(num_trial_12):	
				if block > 1 and trial >= num_trial_else:
					continue

				# (i) present stimulus
				s_1 = int(stimulus_1[trial])
				s_2 = int(stimulus_2[trial])

				# (ii) initialize trial-specific variables
				correct_1 = 0 # keep track of whether correct or not in the first stage
				correct_2 = 0 # keep track of whether correct or not in the second stage
				a_1_temp = [] # action holder at trial level for the first stage
				a_2_temp = [] # action holder at trial level for the second stage
				counter_1 = 0 # counter holder at trial level for the first stage
				counter_2 = 0 # counter holder at trial level for the second stage
				# correct answer for both stages
				correct_action_1 = find_correct_action(s_1, s_2, transition_train1_step1, transition_train1_step2, transition_train2_step1, transition_train2_step2, transition_ca1_step2, transition_ca2_step2, transition_ca3_step2, block, 1, experiment) 
				correct_action_2 = find_correct_action(s_1, s_2, transition_train1_step1, transition_train1_step2, transition_train2_step1, transition_train2_step2, transition_ca1_step2, transition_ca2_step2, transition_ca3_step2, block, 2, experiment) 

				# (iv) first stage starts
				actions_tried = set()
				while correct_1 == 0 and counter_1 < 10:
					c = block # The block is the temporal context
					if block == 0 and trial == 0:
						TS = 0
						encounter_matrix[c] = 1 
					else:
						if encounter_matrix[c] == 0:
							PTS = new_SS_update_option(PTS, c, concentration_1)
							TSs = np.vstack((TSs, [np.ones((2,4)) / 4])) # initialize Q-values for new TS creation
							nTS += 1
							PTS[:,c] /= np.sum(PTS[:,c])
							TS = np.random.choice(np.arange(PTS.shape[0]), 1, p=PTS[:,c])[0]
							encounter_matrix[c] = 1
						else:
							# We have seen this context before, just sample
							# first stage TS based on PTS
							PTS[:,c] /= np.sum(PTS[:,c])
							TS = np.random.choice(np.arange(PTS.shape[0]), 1, p=PTS[:,c])[0]

					a_1 = sample_action_by_policy(TSs[TS,:,:], s_1+1, beta_1, actions_tried)
					a_1_temp.append(a_1) # append the action taken to the list of actions in the first stage
					actions_tried.add(a_1-1)
					counter_1 += 1
					correct_1 = int(a_1 == correct_action_1)
					# Use the result to update PTS with Bayes Rule
					if correct_1 == 0:
						reg = PTS[:,c] * (1 - TSs[:,s_1,a_1-1]) + 1e-6
					else:
						reg = PTS[:,c] * TSs[:,s_1,a_1-1] + 1e-6
					PTS[:,c] = reg / np.sum(reg)

					# Use the result observed to infer the current TS again
					# TS = np.argmax(PTS[:,c])
					TS = np.random.choice(np.arange(PTS.shape[0]), 1, p=PTS[:,c])[0]
					TSs[TS,s_1,a_1-1] += alpha_1 * (correct_1 - TSs[TS,s_1,a_1-1])

				# (v) Second stage starts
				actions_tried = set()
				if structure == 'backward':
					cue = s_2 
					state = s_1
				elif structure == 'forward':
					cue = s_1
					state = s_2
				c_2 = block * 2 + cue # The context of the second stage
				c_2_alt = block * 2 + (1 - cue)
				while correct_2 == 0 and counter_2 < 10:
					for this_c_2 in sorted([c_2, c_2_alt]):
						if encounter_matrix_2[this_c_2] == 0:
							if this_c_2 > 0:
								PTS_2 = new_SS_update_option(PTS_2, this_c_2, concentration_2)
								TS_2s = np.vstack((TS_2s, [np.ones((2,4)) / 4])) # initialize Q-values for new TS creation
								nTS_2 += 1
							encounter_matrix_2[this_c_2] = 1
					# biases = np.zeros((nTS_2, nTS_2))
					# for i in range(block):
					# 	this_TS_1 = np.argmax(PTS_2[:-2,i*2])
					# 	this_TS_2 = np.argmax(PTS_2[:-2,i*2+1])
					# 	biases[this_TS_1, this_TS_2] += 1
					# 	biases[this_TS_2, this_TS_1] += 1

					# # Now we have picked the second stage TS, just pick an action based on the policy of this TS
					# TS_2_alt = np.argmax(PTS_2[:,c_2_alt])
					# if block > 0:
					# 	bias = biases[TS_2_alt].copy()
					# 	b = np.max(PTS_2[:,c_2_alt])
					# 	if np.sum(bias) > 0 and np.max(PTS_2[:,c_2]) < 0.7:
					# 		bias /= np.sum(bias)
					# 		PTS_2[:,c_2] = PTS_2[:,c_2] * (1 - b) + bias * b
					# TS_2 = np.random.choice(np.arange(PTS_2.shape[0]), 1, p=PTS_2[:,c_2])[0]
					# TS_2_alt = np.random.choice(np.arange(PTS_2.shape[0]), 1, p=PTS_2[:,c_2_alt])[0]
					# TS_2_history[sub,c_2,trial] = TS_2 

					# Compute meta policy
					# Q_full = TS_2s[TS_2,state,:].copy()
					# pchoice_2_full = softmax(beta_2 * Q_full)
					# if meta_learning:
					# 	if structure == 'backward':
					# 		Q_compress_1 = np.mean(TS_2s[TS_2], axis=0)
					# 		Q_compress_2 = (TS_2s[TS_2,state,:] + TS_2s[TS_2_alt,state,:]) / 2
					# 	elif structure == 'forward':
					# 		Q_compress_1 = (TS_2s[TS_2,state,:] + TS_2s[TS_2_alt,state,:]) / 2
					# 		Q_compress_2 = np.mean(TS_2s[TS_2], axis=0)
					# 	pchoice_2_compress_1 = softmax(beta_2 * Q_compress_1) 
					# 	pchoice_2_compress_2 = softmax(beta_2 * Q_compress_2) 
					# 	pchoice_2 = p_policies_softmax[0] * pchoice_2_compress_1 \
					# 			+ p_policies_softmax[1] * pchoice_2_compress_2 \
					# 			+ p_policies_softmax[2] * pchoice_2_full	
					# 	pchoice_2 = pchoice_2 * (1 - epsilon) + epsilon / 4
					# else:
					# 	# pchoice_2 = pchoice_2_full * (1 - epsilon) + epsilon / 4
					Q_full = TS_2s[:, state]
					pchoice_2_full = softmax(beta_2 * Q_full, axis=-1)
					pchoice_2 = np.sum(pchoice_2_full * PTS_2[:,c_2].reshape(-1,1), axis=0)
					
					# if len(actions_tried) > 0:
					# 	Q_full[list(actions_tried)] = 0
					# 	Q_compress_1[list(actions_tried)] = -1e20
					# 	Q_compress_2[list(actions_tried)] = -1e20

					p_policies_history[sub, block, trial] = p_policies

					a_2 = np.random.choice(np.arange(1,5), 1, p=pchoice_2)[0]
					actions_tried.add(a_2-1)
					a_2_temp.append(a_2+4) # append the action taken to the list of actions in the second stage
					# Increment counter_2 (just made a choice in the second stage)
					counter_2 += 1
					# Check if the action is correct
					correct_2 = int(a_2 + 4 == correct_action_2)

					# Use the result to update PTS_2 with Bayes Rule
					if correct_2 == 0:
						PTS_2[:,c_2] *= (1 - TS_2s[:,state,a_2-1])
					else:
						PTS_2[:,c_2] *= TS_2s[:,state,a_2-1] 
					PTS_2[:,c_2] += 1e-6
					PTS_2[:,c_2] /= np.sum(PTS_2[:,c_2])

					# Use the result observed to infer the current TS again
					# TS_2 = np.random.choice(np.arange(PTS_2.shape[0]), 1, p=PTS_2[:,c_2])[0] # 
					# TS_2 = np.argmax(PTS_2[:,c_2])
					TS_2s[:,state,a_2-1] += alpha_2 * (correct_2 - TS_2s[:,state,a_2-1]) * PTS_2[:,c_2]

					# if meta_learning:
					# 	p_policies[0] *= pchoice_2_compress_1[a_2-1]
					# 	p_policies[1] *= pchoice_2_compress_2[a_2-1]
					# 	p_policies[2] *= pchoice_2_full[a_2-1]
					# 	p_policies /= np.sum(p_policies)
					# 	if np.min(p_policies) < eps_meta:
					# 		p_policies += eps_meta
					# 	p_policies /= np.sum(p_policies)
					# 	p_policies_softmax = softmax(beta_2 * p_policies)

				# Record variables per trial
				counter_1_temp[trial] = counter_1
				counter_2_temp[trial] = counter_2
				a_1_temp = np.array(a_1_temp).ravel()
				a_2_temp = np.array(a_2_temp).ravel()

				if block < 2:
					a_12_12[sub,block,trial] = join_actions(a_1_temp, a_2_temp)
					s_12_12[sub,block,0,trial] = s_1 + 1
					s_12_12[sub,block,1,trial] = s_2 + 1
					# make reward structure
					r_temp = np.full((2, counter_1 + max(0, counter_2-1)), np.nan)
					if counter_1 > 1:
						# pad some 0 in reward for incorrect presses in the first stage
						r_temp[0,:counter_1-1] = 0 
					r_temp[0,counter_1-1] = 1 
					if counter_2 > 1:
						r_temp[1, counter_1-1:(counter_1+counter_2-3)] = 0
					r_temp[1, -1] = 1
					r_12_12[sub,block,trial] = r_temp
				else: # record action, RT and reward only for Blocks 3-8
					a_all[block-2,trial] = join_actions(a_1_temp, a_2_temp)
					# Come up with reward
					r_temp = np.empty((2, counter_1 + max(0,counter_2-1)))
					r_temp[:] = np.nan 
					if counter_1 > 1:
						# pad some 0 in reward for incorrect presses in the first stage
						r_temp[0,:counter_1-1] = 0 
					r_temp[0,counter_1-1] = 1 
					if counter_2 > 1:
						r_temp[1,counter_1-1:(counter_1+counter_2-3)] = 0 
					r_temp[1,-1] = 1 
					r_all[block-2,trial] = r_temp # record reward for this trial

				# skip to the next block if performance is good enough 
				if block < 2 and trial >= 32 and np.mean(counter_1_temp[trial-10:trial]) < 1.5 and np.mean(counter_2_temp[trial-10:trial]) < 1.5: 
					trial = 59

			# Recording variables per block
			if block < 2: # record only counter for Blocks 1 and 2
				population_counter1_12[sub,block,:] = counter_1_temp
				population_counter2_12[sub,block,:] = counter_2_temp
			else: # % record stimulus sequence and counter for Blocks 3-8
				s_1_all[block-2,:] = stimulus_1
				s_2_all[block-2,:] = stimulus_2
				counter_1_all[block-2,:] = counter_1_temp
				counter_2_all[block-2,:] = counter_2_temp

		s1[sub,:,:] = s_1_all + 1
		s2[sub,:,:] = s_2_all + 1
		population_counter1[sub,:,:] = counter_1_all
		population_counter2[sub,:,:] = counter_2_all 
		a[sub,:,:] = a_all
		r[sub,:,:] = r_all

	counter12_12 = np.zeros((num_subject,2,2,num_trial_12))
	counter12_12[:,:,0,:] = population_counter1_12
	counter12_12[:,:,1,:] = population_counter2_12

	# Package the output
	data = {'tr': tr, 'a':a, 'r':r, 's1':s1, 's2':s2, 'counter1':population_counter1, 'counter2':population_counter2, \
		 'counter12_12':counter12_12, 'a_12_12':a_12_12, 's_12_12':s_12_12, 'r_12_12':r_12_12, \
			'p_policies_history': p_policies_history, 'TS_2_history': TS_2_history} 	
	return data


def set_contingency():
	'''
	Sets the correct action to take under each condition in stage 2. 
	There are two stages of decision making, 
		in stage 1,choose from 1-4; 
		in stage 2, there are two possible stimuli, and in each scenario choose from 5-8.

	Returns:
		- transition_step1: the transitions in step 1 (1 x 4)
		- transition_step2: the transitions in step 2 (4 x 2)
	'''
	transition_step1 = np.arange(1,5)
	transition_step2 = np.array([[8,6],[5,7],[7,5],[6,8]])
	return transition_step1, transition_step2


def prepare_train_stim_sequence(num_present_per_stim):
	'''
	Generates a sequence of stimuli for training. 

	Arg:
		- num_present_per_stim: number of presentations per stimulus type (2 types of stimuli in each stage)

	Returns:
		- stimulus_1: the sequence of stimuli in stage 1
		- stimulus_2: the sequence of stimuli in stage 2
		- num_trial: the number of trials (= 2 * num_present_per_stim)
	'''
	num_present_per_stim = int(num_present_per_stim)
	a = np.zeros(2*num_present_per_stim)
	a[num_present_per_stim:] += 1

	while True:
		a = np.random.permutation(a)
		if check_seq(a):
			break
	stimulus_1 = a.copy()

	# prepare stage 2 stim sequence
	num_present_per_stim_stage_2 = int(num_present_per_stim / 2)
	a = np.zeros(2*num_present_per_stim_stage_2)
	a[num_present_per_stim_stage_2:] += 1
	stimulus_2_pre = []

	while len(stimulus_2_pre) < 2:
		a = np.random.permutation(a)
		if check_seq(a):
			stimulus_2_pre.append(a.copy())

	I_1 = np.where(stimulus_1 == 0)[0]
	I_2 = np.where(stimulus_1 == 1)[0]

	stimulus_2 = np.zeros_like(stimulus_1)
	stimulus_2[I_1] = stimulus_2_pre[0]
	stimulus_2[I_2] = stimulus_2_pre[1]
	num_trial = len(stimulus_1)

	return stimulus_1, stimulus_2, num_trial


def check_seq(sequence):
	'''
	Check if the sequence of stimulus is good. 

	Return:
		- y: 1 if good, 0 otherwise
	'''
	temp = np.ones(len(sequence) + 1)
	temp[1:-1] = sequence[1:] - sequence[:-1]
	idx = np.where(temp==0)[0]
	consecutive = idx[1:] - idx[:-1]

	l = len(sequence)
	one_two = 0 
	same = 0 

	for j in range(1,l):
		if sequence[j] != sequence[j-1]:
			one_two += 1
		else:
			same += 1

	if abs(one_two - same) <= 2:
		same_transition = True
	else:
		same_transition = False

	# print(sequence)
	if (len(consecutive) == 0 or max(consecutive) < 4) and same_transition:
		return True
	else:
		return False


def find_correct_action(s_1, s_2, transition_train1_step1, transition_train1_step2, transition_train2_step1, transition_train2_step2, transition_ca1_step2, transition_ca2_step2, transition_ca3_step2, block, stage, experiment):
	'''
	Finds the correct action given the experiment design.

	Args:
		- s_1: stage 1 stimulus type index
		- s_2: stage 2 stimulus type index
		- transition_train1_step1, transition_train1_step2, transition_train2_step1, transition_train2_step2, transition_ca1_step2, transition_ca2_step2: transitions
		- block: the block number
		- stage: the stage number, 1 or 2
		- experiment: the experiment name, 'CA1' (including CA1 and CA1-CA1) or 'CA2' (including CA2 and CA2-CA1)

	Returns:
		- correct_action: the correct action for this combination of stimuli in the block at the stage in the experiment
	'''
	if stage == 1:
		# We need the correct action for the first stage
		if block % 2 == 1:
			correct_action = transition_train2_step1[s_1]
		else:
			correct_action = transition_train1_step1[s_1]
	elif stage == 2:
		if block % 2 == 1:
			correct_action = transition_train2_step2[s_1,s_2]
		elif block == 6:
			if experiment[:2] == 'V1':
				correct_action = transition_ca1_step2[s_1,s_2]
			elif experiment[:2] == 'V2':
				correct_action = transition_ca2_step2[s_1,s_2]
			elif experiment[:2] == 'V3':
				correct_action = transition_ca3_step2[s_1,s_2]
		elif block == 10:
			if experiment[-2:] == 'V1':
				correct_action = transition_ca1_step2[s_1,s_2]
			elif experiment[-2:] == 'V2':
				correct_action = transition_ca2_step2[s_1,s_2]
			elif experiment[-2:] == 'V3':
				correct_action = transition_ca3_step2[s_1,s_2]
		else:
			correct_action = transition_train1_step2[s_1,s_2]

	return correct_action


def new_SS_update_option(PTS, c, alpha):
	'''
	Updates the PTS matrix with new option in context c.

	Args:
		- PTS: the probability matrix for choosing TS (Q values).
		- c: context
		- alpha: the clustering coefficient of the CRP

	Returns:
		- new_PTS: the updated PTS matrix
	'''
	specs = PTS.shape
	PTS[:,c] = np.sum(PTS[:,:(c//2)*2], axis=1)
	if np.sum(PTS[:,c]) > 0:
		PTS[:,c] /= np.sum(PTS[:,c])
	new_PTS = np.zeros((specs[0]+1,specs[1]))
	new_PTS[:-1,:] = PTS 
	new_PTS[-1,c] = alpha # create new task set
	new_PTS[:,c] /= np.sum(new_PTS[:,c]) # normalize the probability distribution over the set of TS
	return new_PTS


def sample_action_by_policy(Qs, s, beta, actions_tried=set()):
	'''
	Samples an action based on a policy.

	Args:
		- Qs: the Q table, n x 4
		- s: the identity of the stimulus
		- actions_tried: a set of actions that have been tried

	Returns:
		- the chosen action
	'''
	s = (s - 1) 
	actions = np.array(list(set(np.arange(4)) - actions_tried))
	p_choices = np.exp(beta*Qs[s,actions]) / np.sum(np.exp(beta*Qs[s,actions]))
	a = np.random.choice(actions, 1, p=p_choices)[0] + 1
	return a


def join_actions(a_1, a_2):
	'''
	Join actions a_1 and a_2 with nan paddings.

	Args:
		- a_1: the sequence of actions taken by the agent in stage 1
		- a_2: the sequence of actions taken by the agent in stage 2

	Returns:
		- a_both: 2 x (n_a_1 + n_a_2 - 1) array containing actions in both stages with nan padding
	'''
	l_1 = len(a_1)
	l_2 = len(a_2)

	if l_2 == 0:
		# This means the agent exhaust all 10 chances in the first stage and never get to try the second stage
		nan_array = np.empty(l_1)
		nan_array[:] = np.nan
		a_both = np.vstack((a_1,nan_array))
	else:
		# This means the agent did enter the second stage and tried
		nan_array_1 = np.empty(l_2-1)
		nan_array_1[:] = np.nan
		nan_array_2 = np.empty(l_1-1)
		nan_array_2[:] = np.nan
		a_both = np.vstack((np.insert(nan_array_1, 0, a_1), np.append(nan_array_2, a_2)))

	return a_both


def optimize(fname, bounds, D, structure, meta_learning):
    result = differential_evolution(func=fname, bounds=bounds, args=(D, structure, meta_learning))
    x = result.x
    bestllh = -fname(x, D, structure, meta_learning)
    bestparameters = list(x)
    
    return(bestparameters, bestllh)


def parallel_worker(args):
    fit_model_name, data, structure, i, param_bounds, meta_learning = args
    best_params, best_llh = optimize(globals()[fit_model_name+'_nllh'], param_bounds, data, structure, meta_learning)
    return i, best_params, best_llh


def parallel_simulator(args):
    this_model, i, niters_sim, params, exp, structure, meta_learning = args
    this_data = globals()[this_model](niters_sim, params, exp, structure, meta_learning)
    return i, this_data