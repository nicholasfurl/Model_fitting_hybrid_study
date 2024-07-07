
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [] = all_hybrid_studies_model_space_sahira_v3(subjective_vals, payoff_scheme, study);

%v3: I've gone through and double checked and simplified things a bit for
%preparation for github

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

tic

addpath(genpath('C:\matlab_files\fiance\parameter_recovery\beta_fixed_code\Model_fitting_imageTasks\FMINSEARCHBND'));

%input argument defaults
if ~exist("subjective_vals","var"); %if arguments have not been specified in a function call
    subjective_vals = 0;        %Run models using subjective values (ratings) or objective values (prices)?
    payoff_scheme = 3;          %1 means continuous reward, 2 means 3-rank (5:3:1), 3 means 3-rank in Sahira's monetary proportion
    study = 1;  %1: baseline pilot, 2: full pilot, 3: baseline, 4: full, 5: ratings phase, 6: squares 7: timing 8:payoff
end;    %have arguments already been specified?

%What model fitting / viz steps should the programme run?
check_params = 1;       %fit the same model that created the data and output estimated parameters
make_est_model_data = 1;
use_file = 0; %Set the above to zero and this to 1 and it'll read in a file you specify (See filename_for_plots variable below) and make plots of whatever analyses are in the Generate_params structure in that file;
do_io = 1;  %If a 1, will add io performance as a final model field when make_est_model_data is switched to 1.

%What kinds of models / fits should be run?
%1 cutoff, 2 Cs, 3 IO, (beta only), 4 BV, 5 BR, 6 BPM, 7 Opt, 8 BPV
% do_models = [1 2 7 4 5];    %These are now what v2 called model identifiers - Applies at the moment to both make_model_data and check_params;
do_models =  [2 1 6];    %These are now what v2 called model identifiers - Applies at the moment to both make_model_data and check_params;

%File I/O
if use_file ~=1;
    comment = sprintf('out_sahira_COCSBPM_pay%dvals%dstudy%d',payoff_scheme,subjective_vals,study);     %The filename will already fill in basic parameters so only use special info for this.
end;
outpath = 'C:\matlab_files\fiance\parameter_recovery\beta_fixed_code\Model_fitting_hybrid_study\outputs';
filename_for_plots = [outpath filesep 'out_SeqLennoIO_ll1pay1vals020230707.mat']; %Unfortunately still needs to be typed in manually

%These correspond to identifiers (not configured implementations like in v2) in the v3_sweep version
model_names = {'CO' 'Cs' 'IO' 'BV' 'BR' 'BPM' 'Opt' 'BPV'}; %IO is a placeholder, don't implement
num_model_identifiers = size(model_names,2);


%Now RUN it!
if check_params == 1;

    %Gets all subs at once and then peels off correct one when time comes.
    disp('Getting subject data ...');

    %now returns (preserving legacy variable names):
    %"mean ratings" which is actually 90*num_subs lists of phase 1 ratings
    %seq_vals, which is 6*8*num_subs lists of sequence values and
    %output, which is now 6*num_subs number of subject draws for each sequence
    [mean_ratings_all seq_vals_all output_all] = get_sub_data(subjective_vals, payoff_scheme, study);

    num_subs_found = 0; %now, a legacy
    subjects = 1:size(mean_ratings_all,2);

    %This loop computes ranks for participants
    %and saves relevant behavioural and paradigm-related data to the main
    %struct to be processed by models later
    for subject = subjects;

        %reset individual subject matrices to be refilled with new sub data
        clear mean_ratings seq_vals output ranks;

        %peel off the data for the current subject in the loop.
        mean_ratings = mean_ratings_all(:,subject);
        seq_vals = seq_vals_all(:,:,subject);
        output = output_all(:,subject);

        %Get ranks
        clear seq_ranks ranks;
        seq_ranks = tiedrank(seq_vals')';
        for i=1:size(seq_ranks,1);
            ranks(i,1) = seq_ranks(i,output(i,1));
        end;    %loop through sequences to get rank for each

        Generate_params.ratings(:,subject) = mean_ratings';
        Generate_params.seq_vals(:,:,subject) = seq_vals;

        Generate_params.num_samples(:,subject) = output;
        Generate_params.ranks(:,subject) = ranks;

        %This iterator continues to be used below but its use is obsolete in this loop
        num_subs_found = num_subs_found + 1;

    end;    %loop through subs

    %Now that you have info on the subs, load up the main struct with all basic info you might need
    Generate_params.subjective_vals = subjective_vals;
    Generate_params.payoff_scheme = payoff_scheme;
    Generate_params.do_io = do_io;
    Generate_params.do_models_identifiers = do_models;
    Generate_params.num_subs =  size(Generate_params.seq_vals,3);
    Generate_params.num_seqs =  size(Generate_params.seq_vals,1);
    Generate_params.seq_length =  size(Generate_params.seq_vals,2);
    Generate_params.num_vals = size(Generate_params.ratings,1);
    Generate_params.rating_bounds = [1 100];    %What is min and max of rating scale?
    Generate_params.BVrange = Generate_params.rating_bounds;


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%SET UP MODELS !!!!!!%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %We build off of (Or rather use part of) the infrastructure I created for configuring models in
    %Param_recover*.m. It involves a template list of default parameters
    %values that is then repmatted into a parameter*model type matrix of
    %default parameters. Then a matching parameter*model type free_parameters
    %matrix marks which parameters to fit down below.These matrices are
    %then used to populate separate model fields in Generate_params, which
    %is then dropped into the estimation function.

    %Make the template parameter list
    opt_rule = ceil(exp(-1)*Generate_params.seq_length);  %37% rule cutoff

    model_template.identifier = 2;    %row 1 in param_config, 1:CO 2:IO 3:Cs 4:BV 5:BR 6:BP 7:optimism 8:BPV
    model_template.kappa = 2;        %row 2 in param_config
    model_template.nu = 1;           %row 3
    model_template.cutoff = opt_rule;     %row 4, initialised to optimal 37%
    model_template.Cs = 0;            %row 5, intiialised to optimal no cost to sample
    model_template.BVslope = 0.2;   %row 6, intialised to 1 (like a threshold)
    model_template.BVmid = 50;      %row 7, initialised to halfway through the rating scale (can't be used with log)
    model_template.BRslope = 1;    %row 8
    model_template.BRmid = 50;      %row 9
    model_template.BP = 0;           %row 10
    model_template.optimism = 0;    %row 11
    model_template.BPV = 0;          %row 12
    model_template.beta = 1;        
    model_template.name = 'template';

    %Repmat the template to create a column for each model. For now, we are
    %doing all possible models, not the ones specified in do_models. We'll
    %reduce this matrix to just those below.
    identifiers = 1:num_model_identifiers;
    num_cols = num_model_identifiers;
    param_config_default = [ ...
        identifiers;    %row 1: identifiers,  1:CO 2:IO 3:Cs 4:BV 5:BR 6:BP 7:optimism 8:BPV
        repmat(model_template.kappa,1,num_cols);   %row 2: kappa
        repmat(model_template.nu,1,num_cols);   %row 3: nu
        repmat(model_template.cutoff,1,num_cols)   %row 4: cutoff
        repmat(model_template.Cs,1,num_cols);   %row 5: Cs
        repmat(model_template.BVslope,1,num_cols);        %row 6: BV slope
        repmat(model_template.BVmid,1,num_cols);       %row 7: BV mid
        repmat(model_template.BRslope,1,num_cols);        %row 8: BR slope
        repmat(model_template.BRmid,1,num_cols);       %row 9: BR mid
        repmat(model_template.BP,1,num_cols);        %row 10: prior mean offset (BP)
        repmat(model_template.optimism,1,num_cols);       %row 11: optimism
        repmat(model_template.BPV,1,num_cols);       %row 12: prior variance offset (BPV)
        repmat(model_template.beta,1,num_cols);   %row 15: beta
        ];

    %Mark which are free/to be estimated
    free_parameters = zeros(size(param_config_default));
    free_parameters(4,1) = 1; %Model indicator 1, parameter 4: Cut off
    free_parameters(5,2) = 1;  %Model indicator 2, parameter 5: Cs
    free_parameters(7,4) = 1;   %Model indicator 4, parameter 7: BV
    free_parameters(9,5) = 1;     %Model indicator 5, parameter 9: BR
    free_parameters(10,6) = 1;  %Model indicator 6, parameter 10: BPM
    free_parameters(11,7) = 1;  %Model indicator 7, parameter 11: Opt
    free_parameters(12,8) = 1;  %Model indicator 8, parameter 12: BPV

    %Now reduce matrices to just those in do_models
    param_config_default = param_config_default(:,do_models);
    free_parameters = free_parameters(:,do_models);

    %Save your work into struct
    Generate_params.num_models = numel(do_models);
    Generate_params.param_config_default = param_config_default;
    Generate_params.free_parameters_matrix = free_parameters;
    Generate_params.comment = comment;
    Generate_params.outpath = outpath;

    Generate_params.analysis_name = Generate_params.comment;
    outname = [Generate_params.analysis_name char(datetime('now','format','yyyyddMM')) '.mat'];
    Generate_params.outname = outname;

    disp( sprintf('Running %s', outname) );

    %Now fill in default parameters to model fields
    for model = 1:Generate_params.num_models;   %How many models are we implementing (do_models)?

        it = 1;
        fields = fieldnames(model_template);
        for field = 1:size(fields,1)-1 %exclude name, the last one
            Generate_params.model(model).(fields{field}) = param_config_default(field,model);
            it=it+1;
        end;
        Generate_params.model(model).name = ...
            model_names{...
            Generate_params.model(model).identifier...
            };  %I think this is the only matrix here that hasn't already been reduced to do_models in the preceding step

        %Fill in this model's free parameters to be estimated later, if you get to the parameter estimation this run
        Generate_params.model(model).this_models_free_parameters = find(free_parameters(:,model)==1);
        Generate_params.model(model).this_models_free_parameter_default_vals = param_config_default(find(free_parameters(:,model)==1),model)';

    end;    %loop through models


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%DO THE MODEL FITTING!!!!!!%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %I'll just specify all the bounds together here and then assign them to
    %Generate_params model structures and pass them to fminsearch in the
    %model loop immediately below
    fitting_bounds.CO = [2 Generate_params.seq_length-1];   %cut off, it's a threshold that must be inside sequence
    fitting_bounds.Cs = [-100 100];   %cost to sample
    fitting_bounds.BV = [1 100]; %biased values (O), it's a threshold that must be inside rating bounds
    fitting_bounds.BR = [1 100]; %biased reward (O), it's a threshold that must be inside rating bounds
    fitting_bounds.BPM = [-100 100];  %biased prior, value can't exit the rating scale
    fitting_bounds.Opt = [-100 100];  %optimism, value can't exit rating scale
    fitting_bounds.BPV = [-100 100];  %biased variances, can't be wider than the whole rating scale
    fitting_bounds.beta = [0 200];

    %In the previous section, we only assigned to the Generate_param
    %struct the models that were in the do_models in that section. So
    %this can only operate on those models (until I start implementing these sections from datafiles later).
    for model = 1:numel(Generate_params.do_models_identifiers);

        for sub = 1:Generate_params.num_subs;


            %You want to fit one model for one subject at a time
            Generate_params.current_model = model;
            Generate_params.num_subs_to_run = sub;

            %Use default params as initial values
            params = [ ...
                Generate_params.model(model).this_models_free_parameter_default_vals ...
                Generate_params.model(model).beta ...
                ];

            %Assign upper and lower bounds
            test_name = Generate_params.model( Generate_params.current_model ).name;
            Generate_params.model(model).lower_bound = eval(sprintf('fitting_bounds.%s(1)',test_name));
            Generate_params.model(model).upper_bound = eval(sprintf('fitting_bounds.%s(2)',test_name));
            Generate_params.model(model).lower_bound_beta = fitting_bounds.beta(1);
            Generate_params.model(model).upper_bound_beta = fitting_bounds.beta(2);


            warning('off');

            disp(...
                sprintf('fitting modeli %d name %s subject %d' ...
                , model ...
                , Generate_params.model( model ).name ...
                , sub ...
                ) );

            %%%%%%%%%%%%%%%%%%%%%%%%
            %%%%Main function call in this section
            %%%%%%%%%%%%%%%%%%%%%%%%%%

            [Generate_params.model(model).estimated_params(sub,:) ...
                ,  Generate_params.model(model).ll(sub,:) ...
                , exitflag, search_out] = ...
                fminsearchbnd(  @(params) f_fitparams( params, Generate_params ), ...
                params,...
                [Generate_params.model(model).lower_bound Generate_params.model(model).lower_bound_beta], ...
                [Generate_params.model(model).upper_bound Generate_params.model(model).upper_bound_beta] ...
                );

            %Let's save as we go so results can be checked for problems
            save([Generate_params.outpath filesep Generate_params.outname], 'Generate_params');

        end;    %Loop through subs


    end;   %loop through models

end;    %estimate parameters of simulated data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%Generate performance from estimated parameters!%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if make_est_model_data == 1;

    if check_params == 0 & use_file == 1;   %use the saved file, if analysis is not succeeding from model fitting

        %should create Generate_params in workspace
        load(filename_for_plots,'Generate_params');
        Generate_params.do_io = do_io;

    end;


    %Run ideal observer if configured to do so
    if isfield(Generate_params,'do_io')
        if Generate_params.do_io == 1;
            Generate_params = run_io(Generate_params);
        end
    end;


    for model = 1:Generate_params.num_models;

        Generate_params.current_model = model;

        %%%%%Here's the main function call in this
        %%%%%section!!!%%%%%%%%%%%%%%%%%%%
        %returns subject*sequences matrices of numbers of draws and ranks
        %I'm using a slightly modified function so I can manipulate params
        %in Generate_params that aren't permanent
        [temp1 temp2] = ...
            generate_a_models_data_est(Generate_params);

        %For
        Generate_params.model(model).num_samples_est = temp1';
        Generate_params.model(model).ranks_est = temp2';

    end;    %Loop through models

    save([Generate_params.outpath filesep Generate_params.outname], 'Generate_params');

end;        %if make_est_model_data?
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


disp('audi5000')

toc
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [num_samples ranks] = generate_a_models_data_est(Generate_params);

%I use this version to get performance data when I need to change from
%configured to estimated parameters first

for sub = 1:Generate_params.num_subs;

    %So, change from configured to estimated parameters first then ...
    %We need to temporarilty change the parameter fields to the current
    %parameter settings if are to use generate_a_models_data to get performance
    it = 1;
    fields = fieldnames(Generate_params.model(Generate_params.current_model));
    for field = Generate_params.model(Generate_params.current_model).this_models_free_parameters';   %loop through all free parameter indices (except beta)

        %theoretical parameter
        Generate_params.model(Generate_params.current_model).(fields{field}) = ...
            Generate_params.model(Generate_params.current_model).estimated_params(sub,it);

        %I guess this is in case I want to use more than one theoretical
        %parameter plus beta. But so far I haven't.
        it=it+1;

    end;

    %...and beta (assume beta is last, after all of the (it = one)
    %theoretical parameters
    Generate_params.model(Generate_params.current_model).beta = ...
        Generate_params.model(Generate_params.current_model).estimated_params(sub,end);

    disp(...
        sprintf('computing performance, fitted modeli %d name %s subject %d' ...
        , Generate_params.current_model ...
        , Generate_params.model( Generate_params.current_model ).name ...
        , sub ...
        ) );

    Generate_params.num_subs_to_run = sub;
    [num_samples(sub,:) ranks(sub,:)] = generate_a_models_data(Generate_params);

end;    %Loop through subs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%








%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function  ll = f_fitparams( params, Generate_params );

%We need to temporarilty change the parameter fields to the current
%parameter settings if are to use generate_a_models_data to get performance
it = 1;
fields = fieldnames(Generate_params.model(Generate_params.current_model));
for field = Generate_params.model(Generate_params.current_model).this_models_free_parameters';   %loop through all free parameter indices (except beta)

    Generate_params.model(Generate_params.current_model).(fields{field}) = params(it);
    it=it+1;

end;
%and now assign beta too
b = params(end);

%(generate_a_models_data can do multiple subjects but here we want to fit
%one subject at a time and the number of subjects to be run is set before f_fitparams function call in a
%field of Generate_params
[num_samples ranks choiceStop_all choiceCont_all] = generate_a_models_data(Generate_params);
%num_samples and ranks are seqs and choice* are both seq*sub

ll = 0;

for seq = 1:Generate_params.num_seqs;

    %Log likelihood for this subject

    %Get action values for this sequence
    %seq*seqpos
    choiceValues = [choiceCont_all(seq,:); choiceStop_all(seq,:)]';

    %Need to limit the sequence by the "subject's" (configured simulation's)
    %number of draws ...

    %How many samples for this sequence and subject
    listDraws = ...
        Generate_params.num_samples(seq,Generate_params.num_subs_to_run);

    %Loop through trials to be modelled to get choice probabilities for
    %each action value
    for drawi = 1 : listDraws
        %cprob seqpos*choice(draw/stay)
        cprob(drawi, :) = exp(b*choiceValues(drawi, :))./sum(exp(b*choiceValues(drawi, :)));
    end;

    %Compute ll
    if listDraws == 1;  %If only one draw
        ll = ll - 0 - log(cprob(listDraws, 2));
    else
            ll = ll - sum(log(cprob((1:listDraws-1), 1))) - log(cprob(listDraws, 2));
%             ll = ll - sum(log(cprob((listDraws-1), 1))) - log(cprob(listDraws, 2));
    end;

end;    %seq loop



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [num_samples ranks choiceStop_all choiceCont_all] = generate_a_models_data(Generate_params);
%returns subject*sequences matrices of numbers of draws and ranks

%note: What is called which_model here is param_to_fit(model) outside this programme,
%at the base level

this_sub = 1;   %Need to assign each sub to output array by how many have been run, rathere than by sub num
for num_subs_found = Generate_params.num_subs_to_run;

    if numel(Generate_params.num_subs_to_run) > 1; %i.e., if model fitting to a single subject is not going on here
        disp(...
            sprintf('generating performance for preconfigured modeli %d name %s subject %d' ...
            , Generate_params.current_model ...
            ,Generate_params.model( Generate_params.current_model ).name ...
            , num_subs_found ...
            ) );
    end;

    for sequence = 1:Generate_params.num_seqs;
        
        %just one sequence
        list.vals = squeeze(Generate_params.seq_vals(sequence,:,num_subs_found));

        %get prior dist moments
        Generate_params.PriorMean = mean(Generate_params.ratings(:,num_subs_found));
        Generate_params.PriorVar = var(Generate_params.ratings(:,num_subs_found));

        %ranks for this sequence
        dataList = tiedrank(squeeze(Generate_params.seq_vals(sequence,:,num_subs_found))');

        %Do cutoff model, if needed
        if Generate_params.model(Generate_params.current_model).identifier == 1;

            %initialise all sequence positions to zero/continue (value of stopping zero)
            choiceStop = zeros(1,Generate_params.seq_length);
            %What's the cutoff?
            estimated_cutoff = round(Generate_params.model(Generate_params.current_model).cutoff);
            if estimated_cutoff < 1; estimated_cutoff = 1; end;
            if estimated_cutoff > Generate_params.seq_length; estimated_cutoff = Generate_params.seq_length; end;
            %find seq vals greater than the max in the period before cutoff and give these candidates a maximal stopping value of 1
            choiceStop(1,find( list.vals > max(list.vals(1:estimated_cutoff)) ) ) = 1;
            %set the last position to 1, whether it's greater than the best in the learning period or not
            choiceStop(1,Generate_params.seq_length) = 1;
            %Reverse 0s and 1's
            choiceCont = double(~choiceStop);

        else;   %Any Bayesian models

            [choiceStop, choiceCont, difVal] = analyzeSecretary_2023_modelFit(Generate_params,list.vals);

        end;    %Cutoff or other model?

        choiceValues = [choiceCont; choiceStop]';

        b = Generate_params.model( Generate_params.current_model ).beta;

        %softmax the action values, using this sub's estimated beta
        for drawi = 1 : Generate_params.seq_length
            %cprob seqpos*choice(draw/stay)
            cprob(drawi, :) = exp(b*choiceValues(drawi, :))./sum(exp(b*choiceValues(drawi, :)));
        end;

        cprob(end,2) = Inf; %ensure stop choice on final sample.

        %Now get samples from uniform distribution
        test = rand(1000,Generate_params.seq_length);
        for iteration = 1:size(test,1);

            samples_this_test(iteration) = find(cprob(:,2)'>test(iteration,:),1,'first');
            ranks_this_test(iteration) = dataList( samples_this_test(iteration) );

        end;    %iterations

        num_samples(sequence,this_sub) = round(mean(samples_this_test));
        ranks(sequence,this_sub) = round(mean(ranks_this_test));

        %Accumulate action values too so you can compute ll outside this function if needed
        choiceStop_all(sequence, :, this_sub) = choiceStop;
        choiceCont_all(sequence, :, this_sub) = choiceCont;

    end;    %loop through sequences

    this_sub = this_sub + 1;

end;    %loop through subs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%








%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%now returns (preserving legacy variable names):
%"mean ratings" which is actually 90(i.e., number of rated stimuli)*num_subs lists of phase 1 ratings
%seq_vals, which is 6(number of sequences)*12(number of options)*num_subs(151) lists of sequence values and
%output, which is now 6(number of sequences)*num_subs(N=151), and holds the number of subject draws for each sequence
function [all_ratings seq_vals all_output] = get_sub_data(subjective_vals, payoff_scheme, study);



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

end;    %ratings phase?



%get sequence data
sequence_data_concatenated = [];
for file=1:num_sequence_files;

    phase2_temp = readtable([data_folder filesep datafiles(file).name]);
    phase2_data = phase2_temp(phase2_temp.Correct==1,sequence_file_headers);

    %standardise the header names
    phase2_data.Properties.VariableNames([5:16]) = header_names{1}(5:16);

    sequence_data_concatenated = [sequence_data_concatenated; phase2_data];

end;    %loop through sequence files


%average the number of draws over sequences per subject (all other important variables are between subs)
% mean_draws = grpstats(sequence_data_concatenated,"ParticipantPrivateID","mean","DataVars",["ScreenNumber"]);
all_output = reshape( ...
    sequence_data_concatenated.ScreenNumber, ...
    numel(unique(sequence_data_concatenated.TrialNumber)), ...
    numel(unique(sequence_data_concatenated.ParticipantPrivateID)) ...
    );
%If it's a condition that uses the Costa & Averbeck two screen format
%(option+response screens) then need to divide screen number by 2 to get
%correct number of drawn options.
if header_format == 1;
    all_output = all_output / 2;
end;

%reformat strings with £ signs in cells to be doubles
if option_chars ~= 2;   %if not full pilot (which already is in doubles for some reason.

    for trial = 1:size(sequence_data_concatenated,1);


        if option_chars == 1;

            sequence_data_concatenated.Option1{trial} = str2double(sequence_data_concatenated.Option1{trial}(2:end));
            sequence_data_concatenated.Option2{trial} = str2double(sequence_data_concatenated.Option2{trial}(2:end));
            sequence_data_concatenated.Option3{trial} = str2double(sequence_data_concatenated.Option3{trial}(2:end));
            sequence_data_concatenated.Option4{trial} = str2double(sequence_data_concatenated.Option4{trial}(2:end));
            sequence_data_concatenated.Option5{trial} = str2double(sequence_data_concatenated.Option5{trial}(2:end));
            sequence_data_concatenated.Option6{trial} = str2double(sequence_data_concatenated.Option6{trial}(2:end));
            sequence_data_concatenated.Option7{trial} = str2double(sequence_data_concatenated.Option7{trial}(2:end));
            sequence_data_concatenated.Option8{trial} = str2double(sequence_data_concatenated.Option8{trial}(2:end));
            sequence_data_concatenated.Option9{trial} = str2double(sequence_data_concatenated.Option9{trial}(2:end));
            sequence_data_concatenated.Option10{trial} = str2double(sequence_data_concatenated.Option10{trial}(2:end));
            sequence_data_concatenated.Option11{trial} = str2double(sequence_data_concatenated.Option11{trial}(2:end));
            sequence_data_concatenated.Option12{trial} = str2double(sequence_data_concatenated.Option12{trial}(2:end));


        else

            %Unless they're from full or full pilot conditions, which only have a £ imported, the option
            %strings have hidden <strong> tags to consider
            sequence_data_concatenated.Option1{trial} = str2double(sequence_data_concatenated.Option1{trial}(9:end-9));
            sequence_data_concatenated.Option2{trial} = str2double(sequence_data_concatenated.Option2{trial}(9:end-9));
            sequence_data_concatenated.Option3{trial} = str2double(sequence_data_concatenated.Option3{trial}(9:end-9));
            sequence_data_concatenated.Option4{trial} = str2double(sequence_data_concatenated.Option4{trial}(9:end-9));
            sequence_data_concatenated.Option5{trial} = str2double(sequence_data_concatenated.Option5{trial}(9:end-9));
            sequence_data_concatenated.Option6{trial} = str2double(sequence_data_concatenated.Option6{trial}(9:end-9));
            sequence_data_concatenated.Option7{trial} = str2double(sequence_data_concatenated.Option7{trial}(9:end-9));
            sequence_data_concatenated.Option8{trial} = str2double(sequence_data_concatenated.Option8{trial}(9:end-9));
            sequence_data_concatenated.Option9{trial} = str2double(sequence_data_concatenated.Option9{trial}(9:end-9));
            sequence_data_concatenated.Option10{trial} = str2double(sequence_data_concatenated.Option10{trial}(9:end-9));
            sequence_data_concatenated.Option11{trial} = str2double(sequence_data_concatenated.Option11{trial}(9:end-9));
            sequence_data_concatenated.Option12{trial} = str2double(sequence_data_concatenated.Option12{trial}(9:end-9));

        end;    %how are option values formatted



    end;    %loop through trials

    sequence_data_concatenated.Option1 = cell2mat(sequence_data_concatenated.Option1);
    sequence_data_concatenated.Option2 = cell2mat(sequence_data_concatenated.Option2);
    sequence_data_concatenated.Option3 = cell2mat(sequence_data_concatenated.Option3);
    sequence_data_concatenated.Option4 = cell2mat(sequence_data_concatenated.Option4);
    sequence_data_concatenated.Option5 = cell2mat(sequence_data_concatenated.Option5);
    sequence_data_concatenated.Option6 = cell2mat(sequence_data_concatenated.Option6);
    sequence_data_concatenated.Option7 = cell2mat(sequence_data_concatenated.Option7);
    sequence_data_concatenated.Option8 = cell2mat(sequence_data_concatenated.Option8);
    sequence_data_concatenated.Option9 = cell2mat(sequence_data_concatenated.Option9);
    sequence_data_concatenated.Option10 = cell2mat(sequence_data_concatenated.Option10);
    sequence_data_concatenated.Option11 = cell2mat(sequence_data_concatenated.Option11);
    sequence_data_concatenated.Option12 = cell2mat(sequence_data_concatenated.Option12);

end;    %if not full pilot and so requires formatting

%Time to loop through and process subs and sequences with models
subs = unique(sequence_data_concatenated.ParticipantPrivateID,'stable');    %make sure sub numbers have same order as in all_output!!!!!!
num_subs = numel(subs);
for subject = 1:num_subs

    disp(sprintf('Participant %d',subs(subject)));


    %Get objective values for this subject
    array_Obj = table2array(sequence_data_concatenated(sequence_data_concatenated.ParticipantPrivateID==subs(subject),5:end));

    %loop through and set up sequence value arrays for model fitting (i.e.,
    %normalise them and transform to subjective values if needed)
    for sequence = 1:size(array_Obj,1);

        if subjective_vals == 1;   %if subjective values

            %Loop through options and replace price values with corresponding ratings for each participant
            clear this_rating_data this_seq_Subj;
            this_rating_data = mean_ratings(mean_ratings.ParticipantPrivateID == subs(subject),:);
            for option=1:size(array_Obj,2);

                this_seq_Subj(1,option) = table2array(this_rating_data(this_rating_data.phone_price==array_Obj(sequence,option),'mean_Response'));

            end;    %loop through options

            all_ratings(:,subject) = this_rating_data.mean_Response; %to be returned by function
            seq_vals(sequence,:,subject) = this_seq_Subj; %to be returned by function

        else;    %if objective values

            %normalise prices vector and accumulate over subs(should be same every subject)
            clear temp_ratings temp_seq_vals
            %all participants have the same raw price distribution
            the_prices = 1.0e+03 *[0.3598    0.3838    0.4318    0.4320    0.4800    0.5040    0.5280    0.5518    0.5520    0.5710    0.5760    0.5910    0.6000 ...
                0.6240    0.6319    0.6320    0.6430    0.6461    0.6670    0.6740    0.6958    0.7150    0.7200    0.7230    0.7260    0.7360 ...
                0.7440    0.7458    0.7460    0.7470    0.7500    0.7680    0.7698    0.7790    0.7870    0.7920    0.7950    0.7960    0.7990 ...
                0.8000    0.8020    0.8110    0.8260    0.8350    0.8397    0.8430    0.8460    0.8640    0.8660    0.8700    0.8720    0.8760 ...
                0.8880    0.8940    0.9100    0.9120    0.9150    0.9180    0.9190    0.9240    0.9350    0.9420    0.9460    0.9900    1.0080 ...
                1.0320    1.0620    1.0660    1.0830    1.1140    1.1160    1.1230    1.1340    1.1400    1.1520    1.1550    1.1560    1.1880 ...
                1.2020    1.2250    1.2300    1.2500    1.2540    1.3200    1.3220    1.3450    1.6830    1.6920    1.7090    1.7640]';

            %         %transform values
            old_min = 1;
            old_max = max(the_prices);
            new_min=1;
            new_max = 100;

            %normalise raw price distribution (need for the models later)
            temp_Obj_ratings = (((new_max-new_min)*(the_prices - old_min))/(old_max-old_min))+new_min;
            temp_Obj_ratings = -(temp_Obj_ratings - 50) + 50;
            all_ratings(:,subject) = temp_Obj_ratings; %to be returned by function

            temp_Obj_vals = (((new_max-new_min)*(array_Obj(sequence,:) - old_min))/(old_max-old_min))+new_min;
            temp_Obj_vals = -(temp_Obj_vals - 50) + 50;

            seq_vals(sequence,:,subject) = temp_Obj_vals;   %to be returned by function

        end;    %objective or subjective values?

    end;    %Loop through sequences
end;    %Loop through subs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%










%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function Generate_params = run_io(Generate_params);

for sub = 1:Generate_params.num_subs;

    disp(...
        sprintf('computing performance, ideal observer subject %d' ...
        , sub ...
        ) );

    for sequence = 1:Generate_params.num_seqs;

        clear sub_data;

        means = log(Generate_params.ratings(:,sub));
        means(find(means == -Inf)) = 0;
        sigs = log(Generate_params.ratings(:,sub));
        sigs(find(sigs == -Inf)) = 0;
        vals = log(Generate_params.seq_vals(sequence,:,sub));
        vals(find(vals == -Inf)) = 0;

        list.mu =  mean(means);
        list.sig = var(sigs);
        list.kappa = 2;
        list.nu = 1;

        list.flip = 0;
        list.vals = vals;
        list.length = size(list.vals,2);
        list.optimize = 0;
        params = 0; %Cs

        %Looks like one participants (22) in full pilot rated every face
        %a 1 in phase 1 and so we end up with everything zeros in
        %subjective value model.

        if list.mu == 0;

            samples(sequence,sub) = NaN;
            ranks(sequence,sub) = NaN;

        else

            %Get action values for sample again and take, and the difference between these action values for each sample
            %This code runs the ideal observer!
            [choiceStop, choiceCont, difVal] = ...
                analyzeSecretaryNick3_io(Generate_params, list);

            %Get number of sanmples with noiseless decision
            samples(sequence,sub) = find(difVal<0,1,'first');

            %rank of chosen option
            dataList = tiedrank(squeeze(Generate_params.seq_vals(sequence,:,sub))');    %ranks of sequence values
            ranks(sequence,sub) = dataList(samples(sequence,sub));

        end;

    end;    %sequence loop

end;    %sub loop

%add new io field to output struct
num_existing_models = size(Generate_params.model,2);
Generate_params.model(num_existing_models+1).name = 'Optimal';
Generate_params.model(num_existing_models+1).num_samples_est = samples;
Generate_params.model(num_existing_models+1).ranks_est = ranks;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
