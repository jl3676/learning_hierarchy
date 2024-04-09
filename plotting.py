import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

import helpers 
from scipy import stats 


def plot_validation_n_presses(data, sim_data_m1, sim_data_m2, condition, cluster, start_trial=0, trials_to_probe=10, m1='Top-down', m2='Bottom-up', nblocks=12, first_press_accuracy=False, save_vector=False, normalize=False):
    num_subjects = data['tr'].shape[0]
    
    _, n_presses_stage_2 = helpers.calc_mean(data, start_trial=start_trial, trials_to_probe=trials_to_probe, first_press_accuracy=first_press_accuracy)
    if normalize:
        n_presses_stage_2 -= np.nanmean(n_presses_stage_2[:,4:6], axis=1).reshape(-1,1)
    n_presses_stage_2_mean = np.nanmean(n_presses_stage_2,axis=0)
    n_presses_stage_2_sem = stats.sem(n_presses_stage_2,axis=0,nan_policy='omit')

    _, n_presses_stage_2_sim_m1 = helpers.calc_mean(sim_data_m1, start_trial=0, trials_to_probe=trials_to_probe, first_press_accuracy=first_press_accuracy)
    if normalize:
        n_presses_stage_2_sim_m1 -= np.nanmean(n_presses_stage_2_sim_m1[:,4:6], axis=1).reshape(-1,1)
    n_presses_stage_2_sim_m1_mean = np.mean(n_presses_stage_2_sim_m1,axis=0)

    _, n_presses_stage_2_sim_m2 = helpers.calc_mean(sim_data_m2, start_trial=0, trials_to_probe=trials_to_probe, first_press_accuracy=first_press_accuracy)
    if normalize:
        n_presses_stage_2_sim_m2 -= np.nanmean(n_presses_stage_2_sim_m2[:,4:6], axis=1).reshape(-1,1)
    n_presses_stage_2_sim_m2_mean = np.mean(n_presses_stage_2_sim_m2,axis=0)

    blocks = range(1,nblocks+1)
    if nblocks > 6:
        plt.fill_between([6.5,7.5],[-0.5,-0.5],[3,3],color='gray',alpha=0.15,edgecolor=None)
        plt.fill_between([10.5,11.5],[-0.5,-0.5],[3,3],color='gray',alpha=0.15,edgecolor=None)
    plt.plot(blocks,np.ones(nblocks)*2.5,'--',color='gray',linewidth=2,alpha=0.6,label='Chance')
    plt.errorbar(blocks,n_presses_stage_2_mean[:nblocks],n_presses_stage_2_sem[:nblocks],fmt='-',capsize=2,color='k',alpha=0.75,label='Human')
    plt.plot(blocks,n_presses_stage_2_sim_m1_mean[:nblocks],'--',color='lightcoral',alpha=0.75,label=f'{m1} model')
    plt.plot(blocks,n_presses_stage_2_sim_m2_mean[:nblocks],'--',color='cornflowerblue',alpha=0.75,label=f'{m2} model')
    plt.xlim([0,nblocks+1])
    if normalize:
        plt.ylim([-0.5,0.8])
        plt.ylabel(r'$\Delta$ number of key presses')
    else:
        if first_press_accuracy:
            plt.ylim([0,1])
            plt.ylabel('First press accuracy')
        else:
            plt.ylim([1,2.7])
            plt.ylabel('Number of key presses')
    plt.xlabel('Block')
    plt.title(f'{condition}, Cluster {cluster}, Trials 1-10 (n={num_subjects})')
    plt.legend()

    if save_vector:
        plt.savefig(f'plots/validation_n_presses_{condition}_cluster{cluster}.svg', format='svg', dpi=1200)
    else:
        plt.show()


def plot_validation_error_types(data, sim_data_m1, sim_data_m2, condition, cluster, m1='Top-down', m2='Bottom-up', nblocks=12, save_vector=False):
    trials_to_probe = 1
    error_types = ['Correct', 'Compression over stage 1 error', 'Compression over stage 2 error', 'Other error']
    colors = ['k', 'lightcoral', 'cornflowerblue']
    num_subject = data['tr'].shape[0]

    stage2_info = helpers.extract_stage2_info(data, condition)
    stage2_info_m1 = helpers.extract_stage2_info(sim_data_m1, condition)
    stage2_info_m2 = helpers.extract_stage2_info(sim_data_m2, condition)

    plt.figure(figsize=(10,10))
    for ind_to_plot in [0, 1, 2, 3]:

        block_trial_data_ca1 = np.zeros((8*nblocks)//trials_to_probe)
        block_trial_data_ca2 = np.zeros((8*nblocks)//trials_to_probe)
        block_trial_data_ca3 = np.zeros((8*nblocks)//trials_to_probe)
        block_trial_data_ca1_se = np.zeros_like(block_trial_data_ca1)
        block_trial_data_ca2_se = np.zeros_like(block_trial_data_ca2)
        block_trial_data_ca3_se = np.zeros_like(block_trial_data_ca3)

        # compute data
        pointer = 0
        for block in range(nblocks):
            this_n_trials = 8
            for start_trial in range(0,this_n_trials,trials_to_probe):
                if block % 2 == 0:
                    V2 = (condition[:2] == 'V2' and block == 6) or (condition[-2:] == 'V2' and block == 10)
                    V3 = (condition[:2] == 'V3' and block == 6) or (condition[-2:] == 'V3' and block == 10)
                    mean_population_counter2_ca1 = helpers.aggregate_type_stage2_b9(stage2_info,trials_to_probe,start_trial=start_trial,block=block+1,V2=V2,V3=V3)
                    mean_population_counter2_ca2 = helpers.aggregate_type_stage2_b9(stage2_info_m1,trials_to_probe,start_trial=start_trial,block=block+1,V2=V2,V3=V3)
                    mean_population_counter2_ca3 = helpers.aggregate_type_stage2_b9(stage2_info_m2,trials_to_probe,start_trial=start_trial,block=block+1,V2=V2,V3=V3)
                else:
                    mean_population_counter2_ca1 = helpers.aggregate_type_stage2_b8(stage2_info,trials_to_probe,start_trial=start_trial,block=block+1)
                    mean_population_counter2_ca2 = helpers.aggregate_type_stage2_b8(stage2_info_m1,trials_to_probe,start_trial=start_trial,block=block+1)
                    mean_population_counter2_ca3 = helpers.aggregate_type_stage2_b8(stage2_info_m2,trials_to_probe,start_trial=start_trial,block=block+1)
                
                block_trial_data_ca1[pointer] = np.nanmean(mean_population_counter2_ca1[:,ind_to_plot])
                block_trial_data_ca1_se[pointer] = stats.sem(mean_population_counter2_ca1[:,ind_to_plot],nan_policy='omit')
                block_trial_data_ca2[pointer] = np.nanmean(mean_population_counter2_ca2[:,ind_to_plot])
                block_trial_data_ca2_se[pointer] = stats.sem(mean_population_counter2_ca2[:,ind_to_plot],nan_policy='omit')
                block_trial_data_ca3[pointer] = np.nanmean(mean_population_counter2_ca3[:,ind_to_plot])
                block_trial_data_ca3_se[pointer] = stats.sem(mean_population_counter2_ca3[:,ind_to_plot],nan_policy='omit')
                
                pointer += 1

        # plot
        nticks = (8*nblocks)//trials_to_probe
        blocks = np.arange(nticks) * trials_to_probe + 1
        plt.subplot(4,1,ind_to_plot+1)
        counter = 0
        for a, b in zip(np.arange(0, nticks, 8)+0.5, np.arange(0, nticks, 8)+8.5):
            if counter % 2 == 0:
                if counter == 6 or counter == 10:
                    plt.fill_between([a,a,b],[1,3,3],color='darkorange',alpha=0.1,edgecolor=None)
                else:
                    plt.fill_between([a,a,b],[1,3,3],color='grey',alpha=0.1,edgecolor=None)
            counter += 1
            
        plt.plot(blocks,np.ones(nticks)*.25,'--',color='gray',linewidth=2,alpha=0.6,label='Chance')
        plt.plot(blocks,block_trial_data_ca1,color=colors[0],label=f'Human (n={num_subject})')
        plt.fill_between(blocks,block_trial_data_ca1-block_trial_data_ca1_se,block_trial_data_ca1+block_trial_data_ca1_se,color=colors[0],alpha=0.1)
        plt.plot(blocks,block_trial_data_ca2,'--',linewidth=2,color=colors[1],label=f'{m1} model')
        plt.plot(blocks,block_trial_data_ca3,'--',linewidth=2,color=colors[2],label=f'{m2} model')

        plt.xlim([0, nticks])
        if ind_to_plot == 0:
            plt.ylim([0,1])
        else:
            plt.ylim([0,0.5])
        plt.ylabel('Proportion')
        plt.xlabel('Block')
        plt.xticks(np.arange(0, nticks, 8)+4, np.arange(1, 1+nblocks))
        plt.title(f'{error_types[ind_to_plot]}')
        plt.legend()

    plt.suptitle(f'Choice types, Cluster {cluster}, stage 2')
    plt.tight_layout()

    if save_vector:
        plt.savefig(f'plots/validation_error_types_{condition}_cluster{cluster}.svg', format='svg', dpi=1200)
    else:
        plt.show()


def plot_validation_PTS(data_sim, m, cond, ntrials=1, save_vector=False, pallette=None):
    p_policies = data_sim['p_policies_history']
    last_trials = data_sim['TS_2_history'][:,:,32-ntrials:32]
    last_trials[:,:2,:] = data_sim['TS_2_history'][:,:2,-ntrials:]
    max_value = int(np.nanmax(last_trials)) + 1

    probabilities_per_subject = []

    for subject_data in last_trials:
        # Initialize a matrix to store probabilities for this subject (blocks x TS_2 values)
        subject_probabilities = np.zeros((subject_data.shape[0], max_value))
        
        for block_index, block_data in enumerate(subject_data):
            # Count occurrences of each value of TS_2 in this block
            value_counts = np.bincount(block_data[np.where(~np.isnan(block_data))].astype(int), minlength=max_value)
            
            # Convert counts to probabilities
            probabilities = value_counts.astype('float')
            if np.sum(~np.isnan(block_data)) > 0:
                probabilities /= np.sum(~np.isnan(block_data))
            
            # Store the probabilities for this block
            subject_probabilities[block_index] = probabilities
        
        # Add the probabilities for this subject to the list
        probabilities_per_subject.append(subject_probabilities)

    # Step 4: Average these probabilities across all subjects
    average_probabilities = np.mean(probabilities_per_subject, axis=0).T

    plt.figure(figsize=(7,8))
    gs = gridspec.GridSpec(3, 1, height_ratios=[1, 2, 2])

    plt.subplot(gs[0,0])
    mean_policies = np.nanmean(p_policies.reshape(p_policies.shape[0],-1,p_policies.shape[-1]),axis=0)
    mean_policies = mean_policies[mean_policies[:,0]>0,:]
    plt.plot(mean_policies[:,0], color='cornflowerblue', label='Compressed over stage 1')
    plt.plot(mean_policies[:,1], color='lightcoral', label='Compressed over stage 2')
    plt.plot(mean_policies[:,2], color='k', label='Hierarchical')
    plt.xticks([30, 90]+list(np.arange(10)*32+16+120), np.arange(1,13), rotation=0)
    plt.xlabel('Block')
    plt.ylim([0,1])
    plt.ylabel('p(policy)')
    plt.legend()

    plt.subplot(gs[1:,0])
    vmin = 0
    # vmax = np.max(average_probabilities) 
    vmax = 1
    
    if pallette is None:
        c = 180
        cmap = sns.diverging_palette(c, c+180, s=100, as_cmap=True)
    else:
        cmap = sns.color_palette('Greys', as_cmap=True)
    # elif pallette == 'V1':
    #     cmap = sns.light_palette("#5da845", as_cmap=True)
    # elif pallette == 'V2':
    #     cmap = sns.light_palette("#2f93c4", as_cmap=True)
    # elif pallette == 'V3':
    #     cmap = sns.light_palette("#f58223", as_cmap=True)
    sns.heatmap(average_probabilities,vmin=vmin,vmax=vmax,cmap=cmap,square=True)
    plt.xticks(np.arange(12)*2+1, np.arange(1,13), rotation=0)
    plt.yticks(np.arange(12)*2+0.5, np.arange(12)*2+1)
    plt.xlabel('Block')
    plt.ylabel('TS_2')
    plt.suptitle(f'{m}')
    plt.tight_layout()

    if save_vector:
        plt.savefig(f'plots/validation_PTS_{m}_{cond}.svg', format='svg', dpi=1200)
    else:
        plt.show()


def plot_transfer_n_presses(data, sim_data_m1, sim_data_m2, condition, cluster, start_trial=0, trials_to_probe=10, m1='Top-down', m2='Bottom-up', first_press_accuracy=False):
    num_subjects = data['tr'].shape[0]

    _, n_presses_stage_2 = helpers.calc_mean(data, start_trial=start_trial, trials_to_probe=trials_to_probe, first_press_accuracy=first_press_accuracy)
    n_presses_stage_2_mean = np.nanmean(n_presses_stage_2,axis=0)
    n_presses_stage_2_sem = stats.sem(n_presses_stage_2,axis=0,nan_policy='omit')
    transfer_human = n_presses_stage_2_mean[6] - n_presses_stage_2_mean[10]
    transfer_human_sem = np.sqrt(n_presses_stage_2_sem[6]**2 + n_presses_stage_2_sem[10]**2)

    _, n_presses_stage_2_sim_m1 = helpers.calc_mean(sim_data_m1, start_trial=0, trials_to_probe=trials_to_probe, first_press_accuracy=first_press_accuracy)
    n_presses_stage_2_sim_m1_mean = np.mean(n_presses_stage_2_sim_m1,axis=0)
    transfer_m1 = n_presses_stage_2_sim_m1_mean[6] - n_presses_stage_2_sim_m1_mean[10]

    _, n_presses_stage_2_sim_m2 = helpers.calc_mean(sim_data_m2, start_trial=0, trials_to_probe=trials_to_probe, first_press_accuracy=first_press_accuracy)
    n_presses_stage_2_sim_m2_mean = np.mean(n_presses_stage_2_sim_m2,axis=0)
    transfer_m2 = n_presses_stage_2_sim_m2_mean[6] - n_presses_stage_2_sim_m2_mean[10]

    plt.figure(figsize=(6,3))

    plt.bar([1,2,3],[transfer_human,transfer_m1,transfer_m2],color=['k','lightcoral','cornflowerblue'],alpha=0.75)
    plt.errorbar([1],[transfer_human], [transfer_human_sem],fmt='-',capsize=4,color='k',alpha=0.75)
    plt.xticks([1,2,3],['Human',m1,m2])
    plt.xlim([0,4])
    plt.ylim(-0.1,0.5)
    plt.ylabel('Number of key presses')
    plt.title(f'Amount of transfer between test blocks, {condition}, Cluster {cluster} (n={num_subjects})')
    plt.show()

def plot_transfer_learning_curves(data, meta_data, cond1, cond2, cond3, exp, cluster, start_trial=0, trials_to_probe=10, first_press_accuracy=False, save_vector=False):
    data_1 = helpers.slice_data(data, meta_data, cond1, exp=exp, cluster=cluster)
    num_subjects_1 = data_1['tr'].shape[0]
    data_2 = helpers.slice_data(data, meta_data, cond2, exp=exp, cluster=cluster)
    num_subjects_2 = data_2['tr'].shape[0]
    if cond3 != '':
        data_3 = helpers.slice_data(data, meta_data, cond3, exp=exp, cluster=cluster)
        num_subjects_3 = data_3['tr'].shape[0]

    _, n_presses_stage_2_1 = helpers.calc_mean(data_1, start_trial=start_trial, trials_to_probe=trials_to_probe, first_press_accuracy=first_press_accuracy)
    n_presses_stage_2_1 -= np.nanmean(n_presses_stage_2_1[:,4:6], axis=1).reshape(-1,1)
    n_presses_stage_2_1_mean = np.nanmean(n_presses_stage_2_1,axis=0)
    n_presses_stage_2_1_sem = stats.sem(n_presses_stage_2_1,axis=0,nan_policy='omit')

    alternative = 'greater' if not first_press_accuracy else 'less'
    print(f'One-sided paired t-test between Blocks 7 and 11 for {cond1}:')
    print(stats.ttest_rel(n_presses_stage_2_1[:,6], n_presses_stage_2_1[:,10], alternative=alternative, nan_policy='omit'))
    print()

    _, n_presses_stage_2_2 = helpers.calc_mean(data_2, start_trial=start_trial, trials_to_probe=trials_to_probe, first_press_accuracy=first_press_accuracy)
    n_presses_stage_2_2 -= np.nanmean(n_presses_stage_2_2[:,4:6], axis=1).reshape(-1,1)
    n_presses_stage_2_2_mean = np.nanmean(n_presses_stage_2_2,axis=0)
    n_presses_stage_2_2_sem = stats.sem(n_presses_stage_2_2,axis=0,nan_policy='omit')

    print(f'One-sided paired t-test between Blocks 7 and 11 for {cond2}:')
    print(stats.ttest_rel(n_presses_stage_2_2[:,6], n_presses_stage_2_2[:,10], alternative=alternative, nan_policy='omit'))
    print()

    print(f'One-sided t-test between {cond1} and {cond2} on Block 11:')
    print(stats.ttest_ind(n_presses_stage_2_1[:,10], n_presses_stage_2_2[:,10], alternative=alternative, nan_policy='omit'))
    print()

    if cond3 != '':
        _, n_presses_stage_2_3 = helpers.calc_mean(data_3, start_trial=start_trial, trials_to_probe=trials_to_probe, first_press_accuracy=first_press_accuracy)
        n_presses_stage_2_3 -= np.nanmean(n_presses_stage_2_3[:,4:6], axis=1).reshape(-1,1)
        n_presses_stage_2_3_mean = np.nanmean(n_presses_stage_2_3,axis=0)
        n_presses_stage_2_3_sem = stats.sem(n_presses_stage_2_3,axis=0,nan_policy='omit')
        print(f'One-sided paired t-test between Blocks 7 and 11 for {cond3}:')
        print(stats.ttest_rel(n_presses_stage_2_3[:,6], n_presses_stage_2_3[:,10], alternative=alternative, nan_policy='omit'))
        print()
        print(f'One-sided t-test between {cond2} and {cond3} on Block 11:')
        print(stats.ttest_ind(n_presses_stage_2_3[:,10], n_presses_stage_2_2[:,10], alternative=alternative, nan_policy='omit'))
        
    blocks = range(7,13)
    plt.errorbar(blocks,n_presses_stage_2_1_mean[6:],n_presses_stage_2_1_sem[6:],fmt='-',capsize=2,alpha=0.75,label=cond1+f' (n={num_subjects_1})')
    plt.errorbar(blocks,n_presses_stage_2_2_mean[6:],n_presses_stage_2_2_sem[6:],fmt='-',capsize=2,alpha=0.75,label=cond2+f' (n={num_subjects_2})')
    if cond3 != '':
        plt.errorbar(blocks,n_presses_stage_2_3_mean[6:],n_presses_stage_2_3_sem[6:],fmt='-',capsize=2,alpha=0.75,label=cond3+f' (n={num_subjects_3})')
    plt.xlim([blocks[0]-0.5,blocks[-1]+0.5])
    plt.ylim([-0.3, 0.5])
    plt.ylabel(r'$\Delta$ number of key presses')
    plt.xlabel('Block')
    plt.title(f'Transfer performance, Cluster {cluster}, Trials {start_trial+1}-{start_trial+trials_to_probe}')
    plt.legend()

    if save_vector:
        plt.savefig(f'plots/transfer_learning_curves_{cond1}_{cond2}_{cond3}_cluster{cluster}.svg', format='svg', dpi=1200)
    else:
        plt.show()


def plot_validation_p_policies(data_sim, m, cond, ntrials=1, save_vector=False, pallette=None):
    p_policies = data_sim['p_policies_history']
    last_trials = data_sim['TS_2_history'][:,:,32-ntrials:32]
    last_trials[:,:2,:] = data_sim['TS_2_history'][:,:2,-ntrials:]
    max_value = int(np.nanmax(last_trials)) + 1

    plt.figure(figsize=(12,3))
    
    for i in range(p_policies.shape[0]):
        y = p_policies[i,:,:,2].reshape(-1)
        print(y.shape)
        y = y[~np.isnan(y)]
        plt.plot(y, color='k', alpha=0.01)
    # plt.plot(mean_policies[:,0], color='cornflowerblue', label='Compressed over stage 1')
    # plt.plot(mean_policies[:,1], color='lightcoral', label='Compressed over stage 2')
    # plt.plot(mean_policies[:,2], color='k', label='Hierarchical')
    plt.xticks([30, 90]+list(np.arange(10)*32+16+120), np.arange(1,13), rotation=0)
    plt.xlabel('Block')
    plt.ylim([-0.05,1.05])
    plt.ylabel('p(policy)')
    plt.legend()