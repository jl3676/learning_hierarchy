import numpy as np
from scipy.special import softmax
from scipy.optimize import shgo, differential_evolution, LinearConstraint, Bounds

def abstraction_model_nllh(params, D, structure, meta_learning=True):
	'''
	Computes the negative log likelihood of the data D given the abstraction model parameters.

	Args:
		- params[list]: the parameters of the model
		- D[np.array]: the choice data
		- structure[str]: the structure of the model, 'forward' or 'backward'
		- meta_learning[bool]: whether to use the meta-learning mechanism

	Returns:
		- the negative log likelihood of the data given the model parameters
	'''
	# unpack the parameters
	[alpha_2, beta, beta_meta, concentration_2, epsilon, prior_1, prior_2] = params
	if alpha_2 < 0 or alpha_2 > 1 or epsilon < 0 or epsilon > 1 or prior_1 < 0 or prior_1 > 1 or prior_2 < 0 or prior_2 > 1:
		print(f'Subject: {D[0,0]}')
		print(f"alpha_2: {alpha_2}, beta: {beta}, beta_meta: {beta_meta}, concentration_2: {concentration_2}, epsilon: {epsilon}, prior_1: {prior_1}, prior_2: {prior_2}")
		return np.inf
	beta_2 = beta
	beta_policies = 5
	beta_meta = 10**beta_meta
	concentration_2 = 10**concentration_2
	
	# initialize variables
	llh = 0
	num_block = 12
	s_2 = a_2 = -1
	block = -1

	nTS_2 = 2 # number of task-sets in stage2
	TS_2s = np.ones((nTS_2,2,4)) / 4 # Q-values for each TS in stage2
	nC_2 = 2 * num_block # number of contexts in stage2
	PTS_2 = np.zeros((nTS_2,nC_2)) # probability of choosing each TS in each context
	PTS_2[0,0::2] = 1
	PTS_2[1,1::2] = 1
	encounter_matrix_2 = np.zeros(nC_2) # whether a context has been encountered
	encounter_matrix_2[:nTS_2] = 1
	if meta_learning:
		p_policies = np.array([prior_1, prior_2, 1-prior_1-prior_2]) # probability of sampling each policy
		assert prior_1 + prior_2 <= 1 and prior_1 >= 0 and prior_2 >= 0
		p_policies_softmax = softmax(beta_policies * p_policies) # softmax transform of the policy probabilities

	for t in range(D.shape[0]):	# loop over all trials
		stage = int(D[t,1])

		if int(D[t,5]) == 1: # new block
			block += 1

		if stage == 1: # skip stage1
			s_1 = int(D[t, 2])
			actions_tried = set()
		elif stage == 2: # stage2
			# get stim, action, reward info
			s_2 = int(D[t, 2])
			a_2 = int(D[t, 3]) - 4
			r_2 = int(D[t, 4])

			# get the context and state, determined by model structure
			if structure == 'backward':
				cue = s_2
				state = s_1
			elif structure == 'forward':
				cue = s_1
				state = s_2
			c_2 = block * 2 + cue # The context of stage2
			c_2_alt = block * 2 + (1 - cue) # The alternative context of stage2
			# update the PTS matrix with newly encountered context, if any
			for this_c_2 in sorted([c_2, c_2_alt]):
				if encounter_matrix_2[this_c_2] == 0:
					if this_c_2 > 0:
						PTS_2 = new_SS_update_TS(PTS_2, this_c_2, concentration_2)
						TS_2s = np.vstack((TS_2s, [np.ones((2,4)) / 4])) # initialize Q-values for new TS creation
						nTS_2 += 1
					encounter_matrix_2[this_c_2] = 1

			# update the biases matrix for context-pairing
			biases = np.zeros((nTS_2, nTS_2))
			for i in range(block):
				this_TS_1 = np.argmax(PTS_2[:-2,i*2])
				this_TS_2 = np.argmax(PTS_2[:-2,i*2+1])
				biases[this_TS_1, this_TS_2] += 1
				biases[this_TS_2, this_TS_1] += 1

			# incorporate the biases into task-set probabilities
			TS_2_alt = np.argmax(PTS_2[:,c_2_alt])
			if block > 0:
				bias = biases[TS_2_alt].copy()
				b = np.max(PTS_2[:,c_2_alt])
				if np.sum(bias) > 0 and np.max(PTS_2[:,c_2]) < 0.5:
					bias /= np.sum(bias)
					PTS_2[:,c_2] = PTS_2[:,c_2] * (1 - b) + bias * b

			# sample a task-set based on the TS probabilities given the context
			Q_full = TS_2s[:, state].copy()
			if len(actions_tried) > 0:
				Q_full[:,list(actions_tried)] = -1e20
			pchoice_2_full = softmax(beta_2 * Q_full, axis=-1)
			pchoice_2_full = np.sum(pchoice_2_full[:,a_2-1] * PTS_2[:,c_2]) * (1-epsilon) + epsilon / 4

			# compute the choice policy
			if meta_learning:
				if structure == 'backward':
					Q_compress_1 = np.mean(TS_2s, axis=(1)) # compressed policy over stage1
					Q_compress_2 = (TS_2s + np.sum(TS_2s * PTS_2[:,c_2_alt].reshape(-1,1,1),axis=0))[:,state] / 2 # compressed policy over stage2
				elif structure == 'forward': 
					Q_compress_1 = (TS_2s + np.sum(TS_2s * PTS_2[:,c_2_alt].reshape(-1,1,1),axis=0))[:,state] / 2 # compressed policy over stage1
					Q_compress_2 = np.mean(TS_2s, axis=(1)) # compressed policy over stage2
				
				# avoid choosing the same action in the same stage of the same trial
				if len(actions_tried) > 0:
					Q_compress_1[:,list(actions_tried)] = -1e20
					Q_compress_2[:,list(actions_tried)] = -1e20
				# compute the choice policies for compressed policies
				pchoice_2_compress_1 = softmax(beta_2 * Q_compress_1, axis=-1)
				pchoice_2_compress_1 = np.sum(pchoice_2_compress_1[:,a_2-1] * PTS_2[:,c_2]) * (1-epsilon) + epsilon / 4
				pchoice_2_compress_2 = softmax(beta_2 * Q_compress_2, axis=-1) 
				pchoice_2_compress_2 = np.sum(pchoice_2_compress_2[:,a_2-1] * PTS_2[:,c_2]) * (1-epsilon) + epsilon / 4
				# take a weighted sum of the two compressed policies and the fully hierarchical policy
				pchoice_2 = p_policies_softmax[0] * pchoice_2_compress_1 \
							+ p_policies_softmax[1] * pchoice_2_compress_2 \
							+ p_policies_softmax[2] * pchoice_2_full
			else:
				# the choice policy is the fully hierarchical policy
				pchoice_2 = pchoice_2_full 

			# compute the negative log likelihood of the choice based on the choice policy
			if np.isnan(pchoice_2) or pchoice_2 <= 0:
				print(f"pchoice_2: {pchoice_2}")
				print(f"alpha_2: {alpha_2}, prior_1: {prior_1}, prior_2: {prior_2}, epsilon: {epsilon}")
				print(f"PTS_2[:,c_2]: {PTS_2[:,c_2]}")
				print(f"pchoice_2_compress_1: {pchoice_2_compress_1}")
				print(f"pchoice_2_compress_2: {pchoice_2_compress_2}")
				print(f"pchoice_2_full: {pchoice_2_full}")
			llh += np.log(pchoice_2)
			correct_2 = r_2

			# update the task-set probabilities based on the choice and the reward using Bayes Rule
			PTS_2[:,c_2] *= (1 - correct_2 - (-1)**correct_2 * TS_2s[:, state, a_2-1])
			PTS_2[:,c_2] += 1e-6
			PTS_2[:,c_2] /= np.sum(PTS_2[:,c_2])

			# update the Q-values of the task-sets based on the reward prediction error
			RPE = (r_2 - TS_2s[:,state,a_2-1]) * PTS_2[:,c_2]
			TS_2s[:,state,a_2-1] += alpha_2 * RPE

			# update the policy probabilities using Bayes Rule
			if meta_learning:
				likelihoods = np.array([pchoice_2_compress_1, pchoice_2_compress_2, pchoice_2_full])
				likelihoods = softmax(beta_meta * likelihoods)
				p_policies *= (1 - correct_2 - (-1)**correct_2 * likelihoods)
				if np.min(p_policies) < 1e-6:
					p_policies += 1e-6
				p_policies /= np.sum(p_policies)
				p_policies_softmax = softmax(beta_policies * p_policies)

			actions_tried.add(a_2-1)

	return -llh


def abstraction_model(num_subject, params, experiment, structure, meta_learning=True):
	'''
	Simulates behavior using the abstraction model.

	Args:
		- num_subject[int]: the number of subjects to simulate
		- params[list]: the parameters of the model
		- experiment[str]: the experimental condition, 'All' or something like 'V1-V1'
		- structure[str]: the structure of the model, 'forward' or 'backward'
		- meta_learning[bool]: whether to use the meta-learning mechanism

	Returns:
		- data[dict]: the data of the model
	'''
	# unpack the parameters
	[alpha_2, beta, beta_meta, concentration_2, epsilon, prior_1, prior_2] = params
	beta_2 = beta
	beta_policies = 5
	beta_meta = 10**beta_meta
	concentration_2 = 10**concentration_2

	# initialize variables
	num_block = 6 if experiment == 'All' else 12
	num_trial_12 = 60 # number of trials in blocks 1 and 2
	num_trial_else = 32 # number of trials in blocks 3 and forward
	nC_2 = 2 * num_block # number of contexts in stage2

	# initialize variables for data storage
	population_counter1 = np.zeros((num_subject,num_block-2,num_trial_else)) # number of key presses for stage1
	population_counter2 = np.zeros_like(population_counter1) # number of key presses for stage2
	s_12_12 = np.zeros((num_subject,2,2,num_trial_12)) # stimuli for blocks 1 and 2, both stages
	s1 = np.zeros_like(population_counter1) # stimuli for blocks 3 and forward, stage1
	s2 = np.zeros_like(population_counter1) # stimuli for blocks 3 and forward, stage2
	r_12_12 = np.empty((num_subject,2,num_trial_12), dtype='object') # rewards for blocks 1 and 2
	r = np.empty_like(population_counter1,dtype='object') # rewards for blocks 3 and forward
	a_12_12 = np.empty((num_subject,2,num_trial_12),dtype='object') # actions for blocks 1 and 2
	a = np.empty_like(population_counter1,dtype='object') # actions for blocks 3 and forward
	tr = np.zeros((num_subject,8)) # transition matrix
	population_counter1_12 = np.zeros((num_subject,2,num_trial_12)) # number of key presses for stage1 in blocks 1 and 2
	population_counter2_12 = np.zeros((num_subject,2,num_trial_12)) # number of key presses for stage2 in blocks 1 and 2
	p_policies_history = np.full((num_subject,num_block,num_trial_12,3),np.nan) # history of policy probabilities
	TS_2_history = np.full((num_subject,num_block*2,num_trial_12), np.nan) # history of task-set choices

	# loop over all subjects
	for sub in range(num_subject):

		# set transitions
		transition_step1, transition_step2 = set_contingency()

		transition_train1_step1 = transition_step1[:2]
		transition_train1_step2 = transition_step2[:2,:]
		transition_train2_step1 = transition_step1[2:]
		transition_train2_step2 = transition_step2[2:,:]

		transition_V1_step2 = np.array([[transition_step2[1][1], transition_step2[0][1]], [transition_step2[1][0], transition_step2[0][0]]]) # V1
		transition_V2_step2 = np.array([[transition_step2[0][1], transition_step2[1][1]], [transition_step2[0][0], transition_step2[1][0]]]) # V2
		transition_V3_step2 = np.array([[transition_step2[0][1], transition_step2[1][0]], [transition_step2[1][1], transition_step2[0][0]]]) # V3

		tr[sub,:] = [1,2,3,4,8,6,5,7]

		# initialize other subject-specific task variables
		s_1_all = np.zeros((num_block-2,num_trial_else))
		s_2_all = np.zeros_like(s_1_all)
		counter_1_all = np.zeros_like(s_1_all)
		counter_2_all = np.zeros_like(s_1_all)
		a_all = np.empty((num_block-2,num_trial_else),dtype='object')
		r_all = np.empty_like(a_all,dtype='object')

		nTS_2 = 2 # initialize the number of task-set in stage2
		TS_2s = np.ones((nTS_2,2,4)) / 4 # initialize Q-values for each TS in stage2
		PTS_2 = np.zeros((nTS_2,nC_2)) # initialize the probability of choosing each TS in each context
		PTS_2[0,0::2] = 1 
		PTS_2[1,1::2] = 1
		encounter_matrix_2 = np.zeros(nC_2) # initialize the matrix of whether a context has been encountered
		encounter_matrix_2[:nTS_2] = 1
		if meta_learning:
			p_policies = np.array([prior_1, prior_2, 1-prior_1-prior_2]) # initialize the probability of sampling each policy
			p_policies_softmax = softmax(beta_policies * p_policies) # initialize the softmax transformation of the policy probabilities

		# loop over all blocks
		for block in range(num_block):
			num_trial = num_trial_12 if block < 2 else num_trial_else # number of trials in this block

			# initialize stimuli sequence
			stimulus_1, stimulus_2, _ = prepare_train_stim_sequence(num_trial / 2)
			if block < 2:
				counter_1_temp = np.full(60, np.nan)
				counter_2_temp = np.full(60, np.nan)
			else:
				counter_1_temp = np.full(32, np.nan)
				counter_2_temp = np.full(32, np.nan)

			# loop over all trials
			for trial in range(num_trial):	

				# get stimuli for this trial
				s_1 = int(stimulus_1[trial])
				s_2 = int(stimulus_2[trial])
				# s_2 = 0

				# initialize variables for this trial
				correct_2 = 0 # keep track of whether correct or not in stage2
				a_1_temp = [] # action holder at trial level for stage1
				a_2_temp = [] # action holder at trial level for stage2
				counter_1 = 0 # counter holder at trial level for stage1
				counter_2 = 0 # counter holder at trial level for stage2
				# determine correct answers in both stages
				correct_action_1 = find_correct_action(s_1, s_2, transition_train1_step1, transition_train1_step2, transition_train2_step1, transition_train2_step2, transition_V1_step2, transition_V2_step2, transition_V3_step2, block, 1, experiment) 
				correct_action_2 = find_correct_action(s_1, s_2, transition_train1_step1, transition_train1_step2, transition_train2_step1, transition_train2_step2, transition_V1_step2, transition_V2_step2, transition_V3_step2, block, 2, experiment) 

				# first stage starts
				a_1 = correct_action_1 # make the first stage trivial, since the current work is focused on the second stage
				a_1_temp.append(a_1) # append the action taken to the list of actions in the first stage
				counter_1 += 1

				actions_tried = set()

				# second stage starts
				# determine the context and state based on the temporal structure
				if structure == 'backward':
					cue = s_2 
					state = s_1
				elif structure == 'forward':
					cue = s_1
					state = s_2
				c_2 = block * 2 + cue # The context of the second stage
				c_2_alt = block * 2 + (1 - cue) # The alternative context of the second stage

				# update the PTS matrix with newly encountered context, if any
				for this_c_2 in sorted([c_2, c_2_alt]):
					if encounter_matrix_2[this_c_2] == 0:
						if this_c_2 > 0:
							PTS_2 = new_SS_update_TS(PTS_2, this_c_2, concentration_2)
							TS_2s = np.vstack((TS_2s, [np.ones((2,4)) / 4])) # initialize Q-values for new TS creation
							nTS_2 += 1
						encounter_matrix_2[this_c_2] = 1
						
				# keep choosing in the second stage until the correct action is taken
				while correct_2 == 0 and counter_2 < 10:
					# update the biases matrix for context-pairing
					biases = np.zeros((nTS_2, nTS_2))
					for i in range(block):
						this_TS_1 = np.argmax(PTS_2[:-2,i*2])
						this_TS_2 = np.argmax(PTS_2[:-2,i*2+1])
						biases[this_TS_1, this_TS_2] += 1
						biases[this_TS_2, this_TS_1] += 1

					# incorporate the biases into task-set probabilities
					TS_2_alt = np.argmax(PTS_2[:,c_2_alt])
					if block > 0:
						bias = biases[TS_2_alt].copy()
						b = np.max(PTS_2[:,c_2_alt])
						if np.sum(bias) > 0 and np.max(PTS_2[:,c_2]) < 0.5:
							bias /= np.sum(bias)
							PTS_2[:,c_2] = PTS_2[:,c_2] * (1 - b) + bias * b

					# sample a task-set based on the TS probabilities given the context
					TS_2 = np.random.choice(np.arange(PTS_2.shape[0]), 1, p=PTS_2[:,c_2])[0]
					TS_2_history[sub,c_2,trial] = TS_2
					# compute fully hierarchical policy
					Q_full = TS_2s[TS_2, state].copy()
					if len(actions_tried) > 0:
						Q_full[list(actions_tried)] = -1e20
					pchoice_2_full = softmax(beta_2 * Q_full) * (1-epsilon) + epsilon / 4
					
					# compute the choice policy
					if meta_learning:
						TS_2_alt = np.random.choice(np.arange(PTS_2.shape[0]), 1, p=PTS_2[:,c_2_alt])[0] # sample the alternative TS
						if structure == 'backward':
							Q_compress_1 = np.mean(TS_2s[TS_2], axis=(0)) # compressed policy over stage1
							Q_compress_2 = (TS_2s[TS_2]/2 + TS_2s[TS_2_alt]/2)[state] # compressed policy over stage2
						elif structure == 'forward':
							Q_compress_1 = (TS_2s[TS_2]/2 + TS_2s[TS_2_alt]/2)[state] # compressed policy over stage1
							Q_compress_2 = np.mean(TS_2s[TS_2], axis=(0)) # compressed policy over stage2
						# avoid choosing the same action in the same stage of the same trial
						if len(actions_tried) > 0:
							Q_compress_1[list(actions_tried)] = -1e20
							Q_compress_2[list(actions_tried)] = -1e20
						# compute the choice policies for compressed policies
						pchoice_2_compress_1 = softmax(beta_2 * Q_compress_1) * (1-epsilon) + epsilon / 4
						pchoice_2_compress_2 = softmax(beta_2 * Q_compress_2) * (1-epsilon) + epsilon / 4
						# take a weighted sum of the two compressed policies and the fully hierarchical policy
						pchoice_2 = p_policies_softmax[0] * pchoice_2_compress_1 \
									+ p_policies_softmax[1] * pchoice_2_compress_2 \
									+ p_policies_softmax[2] * pchoice_2_full
					else:
						# the choice policy is the fully hierarchical policy
						pchoice_2 = pchoice_2_full 

					# sample an action based on the choice policy
					a_2 = np.random.choice(np.arange(1,5), 1, p=pchoice_2)[0]
					actions_tried.add(a_2-1)
					a_2_temp.append(a_2+4) # append the action taken to the list of actions in the second stage
					counter_2 += 1
					correct_2 = int((a_2 + 4) == correct_action_2)

					# Use the result to update task-set probabilities with Bayes Rule
					PTS_2[:,c_2] *= (1 - correct_2 - (-1)**correct_2 * TS_2s[:, state, a_2-1])
					PTS_2[:,c_2] += 1e-6
					PTS_2[:,c_2] /= np.sum(PTS_2[:,c_2])

					# Update the Q-values of the resampled task-set based on the reward prediction error
					TS_2 = np.random.choice(np.arange(PTS_2.shape[0]), 1, p=PTS_2[:,c_2])[0]
					RPE = correct_2 - TS_2s[TS_2, state, a_2-1]
					TS_2s[TS_2, state, a_2-1] += alpha_2 * RPE

					# Update the policy probabilities using Bayes Rule
					if meta_learning:
						if len(actions_tried) == 1:
							p_policies_history[sub,block,trial] = p_policies
						likelihoods = np.array([pchoice_2_compress_1[a_2-1], pchoice_2_compress_2[a_2-1], pchoice_2_full[a_2-1]])
						likelihoods = softmax(beta_meta * likelihoods)
						p_policies *= (1 - correct_2 - (-1)**correct_2 * likelihoods)
						if np.min(p_policies) < 1e-6:
							p_policies += 1e-6
						p_policies /= np.sum(p_policies)
						p_policies_softmax = softmax(beta_policies * p_policies)

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
						r_temp[1, counter_1-1:] = 0
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

				# skip to the next block if performance is good enough, like in the human experiment
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

		# Record variables per subject
		s1[sub,:,:] = s_1_all + 1
		s2[sub,:,:] = s_2_all + 1
		population_counter1[sub,:,:] = counter_1_all
		population_counter2[sub,:,:] = counter_2_all 
		a[sub,:,:] = a_all
		r[sub,:,:] = r_all

	counter12_12 = np.zeros((num_subject,2,2,num_trial_12))
	counter12_12[:,:,0,:] = population_counter1_12
	counter12_12[:,:,1,:] = population_counter2_12

	# Package the output and return
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


def find_correct_action(s_1, s_2, transition_train1_step1, transition_train1_step2, transition_train2_step1, transition_train2_step2, transition_V1_step2, transition_V2_step2, transition_V3_step2, block, stage, experiment):
	'''
	Finds the correct action given the experimental design.

	Args:
		- s_1: stage 1 stimulus type index
		- s_2: stage 2 stimulus type index
		- transition_train1_step1, transition_train1_step2, transition_train2_step1, transition_train2_step2, transition_V1_step2, transition_V2_step2, transition_V3_step2: transitions
		- block: the block number
		- stage: the stage number, 1 or 2
		- experiment: the experiment name, 'V1', 'V2' or 'V3'

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
				correct_action = transition_V1_step2[s_1,s_2]
			elif experiment[:2] == 'V2':
				correct_action = transition_V2_step2[s_1,s_2]
			elif experiment[:2] == 'V3':
				correct_action = transition_V3_step2[s_1,s_2]
		elif block == 10:
			if experiment[-2:] == 'V1':
				correct_action = transition_V1_step2[s_1,s_2]
			elif experiment[-2:] == 'V2':
				correct_action = transition_V2_step2[s_1,s_2]
			elif experiment[-2:] == 'V3':
				correct_action = transition_V3_step2[s_1,s_2]
		else:
			correct_action = transition_train1_step2[s_1,s_2]

	return correct_action


def new_SS_update_TS(PTS, c, alpha):
	'''
	Updates the PTS matrix with new task-set in context c.

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
	'''
	Optimizes the model using the global optimization algorithm differential evolution.

	Args:
		- fname: the function name to be optimized
		- bounds: the list of bounds of the parameters
		- D: the data
		- structure: the structure of the model, 'forward' or 'backward'
		- meta_learning: whether to use the meta-learning mechanism

	Returns:
		- bestparameters: the best parameters found
		- bestllh: the best log-likelihood found
	'''
	constraints = LinearConstraint([[0, 0, 0, 0, 0, 1, 1]], lb=2e-6, ub=1-1e-6)
	bounds = Bounds(lb=[b[0] for b in bounds], ub=[b[1] for b in bounds])
	# result = differential_evolution(func=fname, bounds=bounds, constraints=constraints, args=(D, structure, meta_learning))
	def c1(x):
		return x[-2] + x[-1]
	def c2(x):
		return 1 - x[-2] - x[-1]
	cons = ({'type': 'ineq', 'fun': c1},
			{'type': 'ineq', 'fun': c2})
	result = shgo(func=fname, bounds=bounds, constraints=cons, args=(D, structure, meta_learning), options={"infty_constraints": False})
	x = result.x
	bestllh = -fname(x, D, structure, meta_learning)
	bestparameters = list(x)

	return bestparameters, bestllh


def parallel_worker(args):
	'''
	The parallel worker for optimization.

	Args:
		- args: the arguments for optimization

	Returns:
		- i: the index of the worker
		- best_params: the best-fit parameters found
		- best_llh: the best-fit log-likelihood found
	'''
	fit_model_name, data, structure, i, param_bounds, meta_learning = args
	best_params, best_llh = optimize(globals()[fit_model_name+'_nllh'], param_bounds, data, structure, meta_learning)
	return i, best_params, best_llh


def parallel_simulator(args):
	'''
	The parallel worker for simulation.

	Args:
		- args: the arguments for simulation
	
	Returns:
		- i: the index of the worker
		- this_data: the data generated by the model
	'''
	this_model, i, niters_sim, params, exp, structure, meta_learning = args
	this_data = globals()[this_model](niters_sim, params, exp, structure, meta_learning)
	return i, this_data