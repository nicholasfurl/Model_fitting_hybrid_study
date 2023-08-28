
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [] = all_hybrid_studies_javascript_ratings(all_draws_set,subjective_vals, payoff_scheme);

%all_hybrid_studies_javascript_ratings.m computes correlations between the two
%ratings in phase 1 for manuscript-named Studies 2 (aka NEW) and 3 (aka seqLen)
%for purposes of answering a reviewer question (Communications Psychology). 
%These two studies were hosted in javascript, instead of Gorilla, hence the
%name *javescript*.m. The equivalent code for Sahira's pilot studies and
%Study 1 (aka hybrid)is all_hybrid_studies_sahira_ratings. The current code
%is adapted from all_hybrid_studies_model_space_NEW.m and might have some
%legacy obsolete code in it, though I've tried to delete the stuff
%irrelevant for computing these correlations.

%I originally wrote all_hybrid_studies_model_space.m to work with the NEW
%study (N=150, seq length 12). Now I'm splitting this code so one version,
%this one, will be called all_hybrid_studies_model_space_NEW.m and
%continue to be the one for NEW. Another one will be
%all_hybrid_studies_model_space_seqLen.m and will work with the study in
%which sequence length is manipulated 10 versus 14 options.

%Adapts new_fit_code_hybrid_prior_subs_v6_betafix_io to work with new
%post-Sahira datasets NEW (big N, sequence length 12, continuous reward,
%phase 1) and a study manipulating sequence length (10 and 14 options).
%Also cleaned up the model code, and verified the correct reward functions.
%Also Sahira's raw datafiles are accessed and processed from scratch.

%study = 1 is manuscript's Study 2 (aka NEW)
%study = 2 is manuscript's Study 3 (aka seqLen) 10 option condition
%study = 3 is manuscript's Study 3 (aka seqLen) 14 option condition
study = 3;

get_sub_data(study);


disp('audi5000')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%now returns (preserving legacy variable names):
%"mean ratings" which is actually 90(i.e., number of rated stimuli)*num_subs lists of phase 1 ratings
%seq_vals, which is 6(number of sequences)*12(number of options)*num_subs(151) lists of sequence values and
%output, which is now 6(number of sequences)*num_subs(N=151), and holds the number of subject draws for each sequence
function get_sub_data(study);

if study == 1;
    
    %N=150 NEW study (irresponsibly preserving some legacy variable names
    %lifted from other code)
    
    % % %STUDY batch 1 (aka NEW): N=20, options=12, num_seqs=6, rating phase, continuous reward
    all_data4_temp = readtable('NEW\batch01.csv');
    all_data4 = all_data4_temp(:,{'name','run_id','response','price','sequence','option','rank','num_options','seq_cond','num_seqs','reward_cond','array'});
    all_data4.run_id = all_data4.run_id + 400;  %ensure no overlapping sub numbers in different pilots
    
    %STUDY batch 2: N=80, options=12, num_seqs=6, rating phase, continuous reward.
    all_data5_temp = readtable('NEW\batch02.csv');
    all_data5 = all_data5_temp(:,{'name','run_id','response','price','sequence','option','rank','num_options','seq_cond','num_seqs','reward_cond','array'});
    all_data5.run_id = all_data5.run_id + 500;  %ensure no overlapping sub numbers in different pilots
    
    %STUDY batch 3: N=50, options=12, num_seqs=6, rating phase, continuous reward.
    all_data6_temp = readtable('NEW\batch03.csv');
    all_data6 = all_data6_temp(:,{'name','run_id','response','price','sequence','option','rank','num_options','seq_cond','num_seqs','reward_cond','array'});
    all_data6.run_id = all_data6.run_id + 600;  %ensure no overlapping sub numbers in different pilots
    
    all_data = vertcat(all_data4, all_data5, all_data6);
    
elseif study == 2;
    
    
    all_data10_temp = readtable('C:\matlab_files\fiance\parameter_recovery\beta_fixed_code\Model_fitting_hybrid_study\sequence_length\seq-len-study-pt-1-n70-10-v-14-options-5-seqs_oneNoncompletedSubRemoved.csv');
    all_data10 = all_data10_temp(:,{'name','run_id','response','price','sequence','option','rank','num_options','seq_cond','num_seqs','reward_cond','array'});
    all_data10.run_id = all_data10.run_id + 1000;  %ensure no overlapping sub numbers in different pilots
    
    all_data = vertcat(all_data10);
    
    
elseif study == 3;
    
    all_data11_temp = readtable('C:\matlab_files\fiance\parameter_recovery\beta_fixed_code\Model_fitting_hybrid_study\sequence_length\seq-len-study-pt-2-n70-10-v-14-options-5-seqs.csv');
    all_data11 = all_data11_temp(:,{'name','run_id','response','price','sequence','option','rank','num_options','seq_cond','num_seqs','reward_cond','array'});
    all_data11.run_id = all_data11.run_id + 1100;  %ensure no overlapping sub numbers in different pilots
    
    all_data = vertcat(all_data11);
   
end;    %which study?



%split off ratings data
ratings_data = all_data(strcmp(all_data.name,'rating_trial'),{'run_id' 'price' 'response'});


%split off phase 2 data where participant said stop
phase2_data = all_data(strcmp(all_data.name,'response_prompt') & all_data.response == 0,{'run_id','response','price','sequence','option','rank','num_options','seq_cond','num_seqs','reward_cond','array'});


%average the two ratingsper subject
group_vars = {'run_id', 'price'};
mean_ratings = grpstats(ratings_data, group_vars, 'mean');

%get correlations between the two ratings
%divide up
subs = table2array(unique(ratings_data(:,'run_id')));
prices = table2array(unique(ratings_data(:,'price')));
%process this subject
for sub=1:numel(subs)
    this_sub_data = ratings_data(table2array(ratings_data(:,'run_id'))==subs(sub,:),:);
    %process each price in this subject
    clear this_sub_two_prices;
    for price = 1:numel(prices);
        temp00 = table2array(this_sub_data(table2array(this_sub_data(:,'price'))==prices(price), 'response'))';
        if size(temp00,2) == 3;
            disp(sprintf('study %d subject %d price %d has three ratings',study,subs(sub),prices(price)));
            temp00 = temp00(1,1:2);
        end;
        this_sub_two_prices(price,:) = temp00;
    end;    %prices loop
    %get this subject's correlation
    if size(this_sub_two_prices,2) == 3;
        fprintf(' ');
    end;
    temp = corr(this_sub_two_prices);
    ratings_corr(sub,1) = temp(1,2);
    
end;    %subjects loop

%output mean and CI over subjects
mean_corr = nanmean(ratings_corr);
pd = fitdist(ratings_corr,'Normal');
disp(sprintf('study NEW: mean ratings correlation: %0.2f, SD ratings correlation: %0.2f',pd.mu,pd.sigma));











