
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [] = all_hybrid_studies_sahira_ratings(all_draws_set,subjective_vals, payoff_scheme, study);

% all_hybrid_studies_sahira_ratings, have modified all_hybrid_studies_model_space_sahira_v2.m
%so that I can just compute correlations between two ratings in phase 1
%without a lot of overhead. So processing stops after get_sub_data. There
%may be a fair amount of legacy code leading up to get_sub_data() that is
%never used that I didn't have time to remove and recheck.

%all_hybrid_studies_model_space_sahira_v2.m. v1 ran all the model fitting
%for the paper. v2 is modified slightly to clean up the code a bit for
%GitHub and also to compute correlations between the two ratings to satisfy
%a reviewer request. Use new_fit_code_make_plots_2022_v3.m on the output
%file *.mat if you want a sneak preview of results or
%hybrid_figures_CP_revisions_v4.m if you want them to appear in
%gifureworthy versions.

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

%input argument defaults
subjective_vals = 0;        %Run models using subjective values (ratings) or objective values (prices)?
payoff_scheme = 2;          %1 means continuous reward, 2 means 3-rank (5:3:1), 3 means 3-rank in Sahira's monetary proportion
study = 5;  %1: baseline pilot, 2: full pilot, 3: baseline, 4: full, 5: ratings phase, 6: squares 7: timing 8:payoff

%now returns (preserving legacy variable names):
%"mean ratings" which is actually 90*num_subs lists of phase 1 ratings
%seq_vals, which is 6*8*num_subs lists of sequence values and
%output, which is now 6*num_subs number of subject draws for each sequence
get_sub_data(subjective_vals, payoff_scheme, study);


disp('audi5000')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%







%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%now returns (preserving legacy variable names):
%"mean ratings" which is actually 90(i.e., number of rated stimuli)*num_subs lists of phase 1 ratings
%seq_vals, which is 6(number of sequences)*12(number of options)*num_subs(151) lists of sequence values and
%output, which is now 6(number of sequences)*num_subs(N=151), and holds the number of subject draws for each sequence
function get_sub_data(subjective_vals, payoff_scheme, study);



%initialise some things that may be changed below depending on study particulars.

%So some conditions (full pilot, full, ratings) have a preceding ratings
%phase that needs to be processed but other conditions don't
ratings_phase = 0;

%Some conditions (those where phase 2 is modelled after Costa & Averbeck:
%pilot baseline, baseline, squares, ratings, timing) have slightly
%differently formatted data. Part of this is that the columns have
%different names as shown here. But also these have two display screens per
%draw (as the participant views the option for fixed time, followed by
%response promot screen, while full and full pilot see option and make
%response on one screen). So I will create a switch header_format, which
%will be 1 (and so will access first header and will divide screen numbers
%by 2) or 2 (and so will access second header row#)
header_names{1} = {'ParticipantPrivateID','Correct','ScreenNumber','TrialNumber','Option1','Option2','Option3','Option4',	'Option5',	'Option6',	'Option7',	'Option8',	'Option9',	'Option10',	'Option11',	'Option12'};
header_names{2} = {'ParticipantPrivateID','Correct','ScreenNumber','TrialNumber','price1a','price2a','price3a','price4a',	'price5a',	'price6a',	'price7a',	'price8a',	'price9a',	'price10a',	'price11a',	'price12a'};
%initialise
header_format = 1;

%0 if header_format == 1, 2 if full pilot and 1 if full. This is determined
%by the string format of the options columns (containing all 12 prices used for the modeling).
%For some reason this varies in the raw data from study to study
option_chars = 0;

%Payoff scheme in this version is to be set manual as an argument to
%all_hybrid_studies_model_space_sahira.m or change the switch at the top of
%the script, which is passed into get_sub_data. 1 is continuous, 2 means 3-rank (5:3:1), 1 means 3-rank in Sahira's monetary proportion
% payoff_scheme = 0;

if study == 1;  %baseline pilot
    data_folder = 'pilot_baseline';
elseif study == 2;  %full pilot
    data_folder = 'pilot_full';
    ratings_phase = 1;
    header_format = 2;
    option_chars = 2;
    %     payoff_scheme = 1;
elseif study == 3;  %baseline
    data_folder = 'baseline';
elseif study == 4;  %full
    data_folder = 'full';
    ratings_phase = 1;
    header_format = 2;
    option_chars = 1;
    %     payoff_scheme = 1;
elseif study == 5;  %rating phase
    data_folder = 'rating_phase';
    ratings_phase = 1;
elseif study == 6;  %squares
    data_folder = 'squares';
elseif study == 7;  %timimg
    data_folder = 'timing';
elseif study == 8;  %timimg
    data_folder = 'payoff';
end;

%set correct headers for this study
sequence_file_headers = header_names{header_format};

%find available data
datafiles = dir( [data_folder filesep '*.csv'] );

%The studies differ in how many sequence files and whether there is a
%ratings file. I renamed the ratings file so it should always be last (the
%initial string ratings_ comes before data_).
num_sequence_files = size(datafiles,1); %assuming all files detected are sequence files

%get ratings data (if applicable)
warning('off'); %otherwise it complains about reformatting spaces in header names for every single file it opens
if ratings_phase == 1;  %if full pilot, full or rating phase
    
    ratings_data_temp = readtable([data_folder filesep datafiles(end).name]);
    ratings_data = ratings_data_temp(strcmp(ratings_data_temp.ZoneType,'response_slider_endValue'),{'ParticipantPrivateID','Response','phone_price'});
    
    num_sequence_files = num_sequence_files - 1;    %one of those the last one) was actually a ratings file
    
    %convert weird html strings in price column to proper numbers
    ratings_data.phone_price = cell2mat(cellfun(@(x) str2double(x(6:end-5)), ratings_data.phone_price , 'UniformOutput', false));
    
    %average the two ratings per subject
    group_vars = {'ParticipantPrivateID', 'phone_price'};
    mean_ratings = grpstats(ratings_data, group_vars, 'mean');
    
    %get correlations between the two ratings
    %divide up
    subs = table2array(unique(ratings_data(:,'ParticipantPrivateID')));
    prices = table2array(unique(ratings_data(:,'phone_price')));
    %process this subject
    for sub=1:numel(subs)
        this_sub_data = ratings_data(table2array(ratings_data(:,'ParticipantPrivateID'))==subs(sub,:),:);
        %process each price in this subject
        clear this_sub_two_prices;
        for price = 1:numel(prices);
            temp00 = table2array(this_sub_data(table2array(this_sub_data(:,'phone_price'))==prices(price), 'Response'))';
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
    disp(sprintf('study %d: mean ratings correlation: %0.2f, SD ratings correlation: %0.2f',study,pd.mu,pd.sigma));
    
end;    %ratings phase?
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%












