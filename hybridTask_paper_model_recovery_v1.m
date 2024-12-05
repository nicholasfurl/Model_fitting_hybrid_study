
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [] = hybridTask_paper_model_recovery_v1;

%model recovery v1: based on imageTask_paper_parameter_recovery_v6.m.
%Takes simulation of data used in parameter recovery and uses it for model
%recovery instead.

%v5, 26 June 2024. Cleaned up the code a bit for paper submission

%v4, brought in analyzeSecretaryPR.m, which uses improved BR and BV.
%Cleaned upo a little in prep for GitHub and

%v3 adds a manipulation of preeconfigured betas

%imageTask_paper_parameter_recovery is based on param_recover_v3_sweep_cleaner
%and attempts to incoporate fitting of beta and using bounds when fitting
%and integer values for cutoff instead of linspace. And I've tried to
%simplify code as I went and saw suitable places to do so.

tic

addpath(genpath('C:\matlab_files\fiance\parameter_recovery\beta_fixed_code\Model_fitting_imageTasks\FMINSEARCHBND'))
addpath(genpath('C:\matlab_files\fiance\parameter_recovery\beta_fixed_code\Model_fitting_imageTasks\plotSpread'));
addpath(genpath('C:\matlab_files\fiance\parameter_recovery\beta_fixed_code\Model_fitting_imageTasks\klabhub-bayesFactor-3d1e8a5'));

simulate_stimuli = 1;  %If 1, randomly generates phase 1 distributions and sequences from them and saves info in Generate_params
make_config_model_data = 1;  %If 1, presets model parameters in Generate_params and then uses simulated stimuli to create simulated model performance
check_params = 1;       %fit CS, BP and CO data and output estimated parameters
%make_est_model_data = 0;   %generates behaviour for models fitted to simulated data, to ensure it reproduces the results. Not relevant for model recovery
use_file_for_plots = 0; %In case you want to access just the figure without rerunning the analysis
%make_plots = 1;         %There's only the confusion and inversion confusion matrices so no reason to suppress plots
%all_draws_set = 1;          %legacy
%log_or_not = 0; %legacy
%1 cutoff, 2 Cs, 3 dummy (vestige), 4 BV, 5 BR, 6 BP, 7 Opt, 8 BPV
do_models = [2 1 6];    %These are now what v2 called model identifiers - Applies at the moment to both make_model_data and check_params;
% do_models = [6];    %These are now what v2 called model identifiers - Applies at the moment to both make_model_data and check_params;

comment = 'hybrid_MR_test';    %The filename will already fill in basic parameters so only use special info for this.
outpath = 'C:\matlab_files\fiance\parameter_recovery\beta_fixed_code\Model_fitting_hybrid_study\outputs';

filename_for_plots = [outpath filesep 'out_MR_3models3modelTypes1paramLevels1betaLevels2subs5seqs12opts_hybrid_MR_test_20242710.mat'];  %
num_param_levels = 4;   %Will use start values and increments specified below to run up to this many parameters for each model
fit_betas = [1];
num_fit_betas = numel(fit_betas);
total_params_per_model = num_param_levels*num_fit_betas;
%These correspond to identifiers (not configured implementations like in v2) in the v3_sweep version
model_names = {'CO' 'Cs' 'IO' 'BV' 'BR' 'BPM' 'Opt' 'BPV' }; %IO is a placeholder, don't implement
num_model_identifiers = size(model_names,2);


if simulate_stimuli == 1;

%     Generate_params.log_or_not = log_or_not;
%     Generate_params.all_draws_set = all_draws_set;
    Generate_params.do_models_identifiers = do_models;
    Generate_params.num_param_levels = num_param_levels;
    Generate_params.fit_betas = fit_betas;
    Generate_params.num_fit_betas = num_fit_betas;
    Generate_params.total_params_per_model = total_params_per_model;

    %Configure sequences
    %Careful, the number of items in the sequences (seq_length*num_seqs)
    %should not exceed the number of items rated in phase 1. Ideally the
    %former should be at most 60% of latter.
    Generate_params.num_subs = 4;   %So this will be per parameter value
    Generate_params.num_seqs = 5;
    Generate_params.seq_length = 12; %hybrid might've usually used 12 but imageTask always uses 8!
    Generate_params.num_vals = 426;  %How many items in phase 1 and available as options in sequences? I've used before 426 or 90
    Generate_params.rating_bounds = [1 100];    %What is min and max of rating scale?
    Generate_params.rating_grand_mean = 40;     %Individual subjects' rating means will jitter around this (50 or 39.5. The latter comes from the midpoint between NEW hybrid SV ratings mean (30) and normalised price mean (49.2)
    Generate_params.rating_mean_jitter = 5;     %How much to jitter participant ratings means on average?
    Generate_params.rating_grand_std = 20;       %Individual subjects' rating std devs will jitter around this (5 or 18, the latter is the midpoint b/n NEW hybrid SV and OV)
    Generate_params.rating_var_jitter = 2;     %How much to jitter participant ratings vars on average?

    for sub = 1:Generate_params.num_subs;

        this_sub_rating_mean = Generate_params.rating_grand_mean + normrnd( 0, Generate_params.rating_mean_jitter );
        this_sub_rating_std = Generate_params.rating_grand_std + normrnd( 0, Generate_params.rating_var_jitter );

        %Generate a truncated normal distribution of option values
        pd = truncate(makedist('Normal','mu',this_sub_rating_mean,'sigma',this_sub_rating_std),Generate_params.rating_bounds(1),Generate_params.rating_bounds(2));
        phase1 = random(pd,Generate_params.num_vals,1);

%         if log_or_not == 1;
%             phase1 = log(phase1);
%             Generate_params.BVrange = log( Generate_params.rating_bounds )
%         else
            Generate_params.BVrange = Generate_params.rating_bounds;
%         end;    %transform ratings if log_or_not

        %Save this sub's ratings data
        Generate_params.ratings(:,sub) = phase1;

        %Grab the requisit number of random ratings
        temp_ratings = phase1(randperm(numel(phase1)),1);
        Generate_params.seq_vals(:,:,sub) = reshape(...
            temp_ratings(1:Generate_params.num_seqs*Generate_params.seq_length,1) ...
            ,Generate_params.num_seqs ...
            ,Generate_params.seq_length ...
            );

        Generate_params.PriorMean = mean(Generate_params.ratings(:,sub));
        Generate_params.PriorVar = var(Generate_params.ratings(:,sub));

    end;    %Each subject to create stimuli
end;    %Should I create stimuli for simulation?

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%CONFIGURE MODELS AND GET PERFORMANCE!!!!!!%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if make_config_model_data == 1;

    %First, make a template model structure (looks like IO). This will
    %contain the values for all the parameters that are fixed and the
    %initial values for all the free parameters during model fitting and will later be altered
    %to store the manipulated values of the simulated free parameters.

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
    %     model_template.log_or_not = log_or_not; %1 = log transform (normalise) ratings  %row 13 (This is already legacy - log or not is now controlled by switch at start of programme and simulated data was logged before reaching this point).
%     model_template.all_draws = all_draws_set;  %1 = use all trials when computing ll instead of last two per sequence.   %row 14
    model_template.beta = 1;        %Just for parameter estimation.
    model_template.name = 'template';


    %initialise matrix to all default (template) values (IO is default)
    %Set the model identifier first, the rest of the matrix will adapt its size
    identifiers = [];
    for i=1:num_model_identifiers;
        identifiers = [identifiers i*ones(1,num_param_levels)];
    end;    %Loop through the model types
    num_cols = numel(identifiers);



    param_config = [ ...
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
        %         repmat(model_template.log_or_not,1,num_cols);   %row 13: log or not (at the moment not to be trusted)
%         repmat(model_template.all_draws,1,num_cols);   %row 14: all draws
        repmat(model_template.beta,1,num_cols);   %row 15: beta
        ];

    %Later, when model fitting, I'll need to know which parameters are free
    %and which to leave fixed. Rather than make another redundant and inelegant lookup table,
    %I'm going to operate under the assumption that anything I set to be
    %different from the default is a parameter that I will want to test as
    %free later. So I will save a copy of the default param_config and then
    %mark as free parameters anything parameters whose values I manipulate
    %in the next step.
    param_config_default = param_config;

    %What param levels are needed for each model?
    %param_used is index into param_config (Holds the params to be used in
    %sim), param_config_default (holds the default and startring values) and
    %free_parameters (holds flags for which parameters need estimation)
    %Note 3 is skipped, it's a placeholder for IO, which is implemented in
    %v2 but not here (It's just Cs = 0 so is now an indicator 2 model and
    %doesn't need a separate model identifier so gets a placeholder).

    % %sequence length 12
    if Generate_params.seq_length == 12;
        the_param_levels(1,:) = round(linspace(2,8,num_param_levels)); param_used(1) = 4;   %Model indicator 1: Cut off;
        the_param_levels(2,:) = linspace(-.05,.015,num_param_levels)*100; param_used(2) = 5;              %Model indicator 2: Cs
        the_param_levels(4,:) = linspace(35,100,num_param_levels); param_used(4) = 7;                   %Model indicator 4: BV
        the_param_levels(5,:) = linspace(25,100,num_param_levels); param_used(5) = 9;                   %Model indicator 5: BR
        the_param_levels(6,:) = linspace(-90,100,num_param_levels); param_used(6) = 10;                  %Model indicator 6: BP
        the_param_levels(7,:) = linspace(-12,15,num_param_levels); param_used(7) = 11;                    %Model indicator 7: Opt
        the_param_levels(8,:) = linspace(-100,100,num_param_levels); param_used(8) = 12;                  %Model indicator 8: BVariance (obsolete)

        %sequence length 8
    elseif Generate_params.seq_length == 8 ;
        the_param_levels(1,:) = round(linspace(2,Generate_params.seq_length-1,num_param_levels)); param_used(1) = 4;   %Model indicator 1: Cut off;
        the_param_levels(2,:) = linspace(-.055,.03,num_param_levels)*100; param_used(2) = 5;              %Model indicator 2: Cs
        the_param_levels(4,:) = linspace(25,80,num_param_levels); param_used(4) = 7;                   %Model indicator 4: BV
        the_param_levels(5,:) = linspace(20,80,num_param_levels); param_used(5) = 9;                   %Model indicator 5: BR
        the_param_levels(6,:) = linspace(-50,70,num_param_levels); param_used(6) = 10;                  %Model indicator 6: BP
        the_param_levels(7,:) = linspace(-13,15,num_param_levels); param_used(7) = 11;                    %Model indicator 7: Opt
        the_param_levels(8,:) = linspace(-30,30,num_param_levels); param_used(8) = 12;                  %Model indicator 8: BVariance (obsolete)

    elseif  Generate_params.seq_length == 10;
        the_param_levels(1,:) = linspace(2,Generate_params.seq_length-1,num_param_levels); param_used(1) = 4;   %Model indicator 1: Cut off;
        the_param_levels(2,:) = linspace(-.075,.015,num_param_levels)*100; param_used(2) = 5;              %Model indicator 2: Cs
        the_param_levels(4,:) = linspace(40,80,num_param_levels); param_used(4) = 7;                   %Model indicator 4: BV
        the_param_levels(5,:) = linspace(40,90,num_param_levels); param_used(5) = 9;                   %Model indicator 5: BR
        the_param_levels(6,:) = linspace(-20,20,num_param_levels); param_used(6) = 10;                  %Model indicator 6: BP
        the_param_levels(7,:) = linspace(-10,15,num_param_levels); param_used(7) = 11;                    %Model indicator 7: Opt
        the_param_levels(8,:) = linspace(-30,30,num_param_levels); param_used(8) = 12;                  %Model indicator 8: BVariance (obsolete)
    end;

    %Now fill in configured parameters and flag them for each model indicator in param_config and free_parameters.
    for i=1:numel(param_used);

        if i ~=3;   %Because this is IO placegholder and we don't want to change any parameters for it (Just skip 3 in do_models)

            %Now configure this groups parameters
            param_config(param_used(i), find(param_config(1,:)==i) ) = the_param_levels(i,:);
            free_parameters(param_used(i), find(param_config(1,:)==i) ) = 1;

        end;    %Is this a non-io model?
    end;    %loop through parameterised models (i) and assign their configureations

    %Now do some final manipulations to config_params before finalising it
    %You'll make a temp_config, assign manipuated config_params to it, then assign it back to config_params again
    %You'll reduce matrices to just those in do_models
    %And you'll replicate config_params, but changing beta each time and then concatenenating them, so it'll fit multiple betas
    temp_config = [];
    temp_default = [];
    temp_free = [];
    for beta_val = 1:num_fit_betas;
        for i=1:size(do_models,2);

            identifier = do_models(i);
            indices = find(param_config(1,:)==identifier); %we need to search for all implementations of the same indicator

            %Search for and get configured params for all models with given indicator
            param_config_newBeta = param_config(:,indices);

            %Change configured beta value
            param_config_newBeta(end,:) = fit_betas(beta_val);  %beta param should always be the last one

            temp_config = [temp_config param_config_newBeta]; %this assigns all the models with a certain model type identicator
            temp_default = [temp_default param_config_default(:,indices)];  %defaults don't change, you always keep beta as 1
            temp_free = [temp_free free_parameters(:,indices)];


        end;    %Loop through model types
    end;    %loop through betas

    param_config = temp_config;
    param_config_default = temp_default;
    free_parameters = temp_free;

    %Save your work into struct summarising the configured and estimable parameters over models
    %num_models here are specific implementations, NOT indicators/identifiers
    Generate_params.num_models = size(param_config,2);  %So num_models means implementations, not indicators
    Generate_params.param_config_matrix = param_config;
    Generate_params.free_parameters_matrix = free_parameters;

    %Create filename and send to standard out(It'll be saved at end of loop iteration).
    Generate_params.comment = comment;
    Generate_params.outpath = outpath;

    analysis_name = sprintf(...
        'out_MR_%dmodels%dmodelTypes%dparamLevels%dbetaLevels%dsubs%dseqs%dopts_%s_'...
        , Generate_params.num_models ...
        , numel(Generate_params.do_models_identifiers) ...
        , Generate_params.num_param_levels ...
        , Generate_params.num_fit_betas ...
        , Generate_params.num_subs ...
        , Generate_params.num_seqs ...
        , Generate_params.seq_length ...
        , Generate_params.comment ...
        );

    Generate_params.analysis_name = analysis_name;
    outname = [analysis_name char(datetime('now','format','yyyyddMM')) '.mat'];
    Generate_params.outname = outname;

    disp( sprintf('Running %s', outname) );

    %Now fill parameters into template and attach to Generate_param
    %This model loop is through implementations, NOT identifiers
    for model = 1:Generate_params.num_models;

        Generate_params.current_model = model;  %So now model 1 will be the first model implementation in the param_config array after it has been reduced by do_models

        it = 1;
        fields = fieldnames(model_template);
        for field = 1:size(fields,1)-1 %exclude name, the last one
            Generate_params.model(model).(fields{field}) = param_config(field,model);
            it=it+1;

        end;
        Generate_params.model(model).name = ...
            model_names{...
            Generate_params.model(model).identifier...
            };  %I think this is the only matrix here that hasn't already been reduced to do_models in the preceding step

        %Fill in this model's free parameters to be estimated later, if you
        %get to the parameter estimatioin this run
        Generate_params.model(model).this_models_free_parameters = find(free_parameters(:,model)==1);
        Generate_params.model(model).this_models_free_parameter_default_vals = param_config_default(find(free_parameters(:,model)==1),model)';
        Generate_params.model(model).this_models_free_parameter_configured_vals = param_config(find(free_parameters(:,model)==1),model)';

        %%%%%Here's the main function call in this
        %%%%%section!!!%%%%%%%%%%%%%%%%%%%
        %returns subject*sequences matrices of numbers of draws and ranks
        Generate_params.num_subs_to_run = 1:Generate_params.num_subs;
        [Generate_params.model(model).num_samples Generate_params.model(model).ranks] = ...
            generate_a_models_data(Generate_params);

    end;    %loop through models

    save([Generate_params.outpath filesep Generate_params.outname], 'Generate_params');

end; %Do I want to get performance from the pre-configured models?
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%







%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%DO THE MODEL FITTING!!!!!!%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if check_params == 1;

    %set upper and lower bounds for the different models (A bit
    %disorganised to put it here, I admit, but this is a new section of
    %code grafted onto a much older structure so this bit will look a little more ad hoc

    %I'll just specify all the bounds together here and then assign them to
    %Generate_params model structures and pass them to fminsearch in the
    %model loop immediately below
    fitting_bounds.CO = [2 Generate_params.seq_length-1];   %cut off, it's a threshold that must be inside sequence
    fitting_bounds.Cs = [-100 100];   %cost to sample
    fitting_bounds.BV = [1 100]; %biased values (O), it's a threshold that must be inside sequence
    fitting_bounds.BR = [1 100]; %biased reward (O), it's a threshold that must be inside sequence
    fitting_bounds.BPM = [-100 100];  %biased prior, value can't exit the rating scale
    fitting_bounds.Opt = [-100 100];  %optimism, value can't exit rating scale
    fitting_bounds.BPV = [-100 100];  %biased variances, can't be wider than the whole rating scale
    fitting_bounds.beta = [0 100];   %A bit arbitrary

    %How many models do we want to fit to each simulated model?
    which_models_to_fit = do_models;

    %How many simulated models with different parameter levels each need have data that needs to be fitted in this section?
    do_models = 1:Generate_params.num_models;

    %In the previous section, we only assigned to the Generate_oparam
    %struct the models that were in the do_models in that section. So
    %this can only operate on those models (until I start implementing these sections from datafiles later).
    for simed_model = do_models;    %loop through the (veridical) simulated models

        for sub = 1:Generate_params.num_subs;   %Loop through each participant that was simulated under this veridical simed model

            for fitted_model = which_models_to_fit;   %Now fit every model to every model's simulated data

                %You want to fit one model for one subject at a time
                Generate_params.current_model = fitted_model;
                Generate_params.num_subs_to_run = sub;

                %Use default params as initial values
                params = [ ...
                    Generate_params.model(fitted_model).this_models_free_parameter_default_vals ...
                    Generate_params.model(fitted_model).beta ...
                    ];

                %Assign upper and lower bounds
                test_name = Generate_params.model( Generate_params.current_model ).name;
                Generate_params.model(fitted_model).lower_bound = eval(sprintf('fitting_bounds.%s(1)',test_name));
                Generate_params.model(fitted_model).upper_bound = eval(sprintf('fitting_bounds.%s(2)',test_name));
                Generate_params.model(fitted_model).lower_bound_beta = fitting_bounds.beta(1);
                Generate_params.model(fitted_model).upper_bound_beta = fitting_bounds.beta(2);

                warning('off');

                disp(...
                    sprintf('fitting modeli %d name %s param %3.2f subject %d simulated from model %s' ...
                    , fitted_model ...
                    , Generate_params.model( Generate_params.current_model ).name ...
                    , Generate_params.model( Generate_params.current_model ).this_models_free_parameter_configured_vals ...
                    , sub ...
                    , Generate_params.model( simed_model ).name ...
                    ) );

                %%%%%%%%%%%%%%%%%%%%%%%%
                %%%%Main function call in this section
                %%%%%%%%%%%%%%%%%%%%%%%%%%
                [Generate_params.model(simed_model).estimated_params(sub,:,fitted_model) ...
                    ,  Generate_params.model(simed_model).ll(sub,fitted_model) ...
                    , exitflag, search_out] = ...
                    fminsearchbnd(  @(params) f_fitparams( params, Generate_params ), ...
                    params,...
                    [Generate_params.model(model).lower_bound Generate_params.model(simed_model).lower_bound_beta], ...
                    [Generate_params.model(model).upper_bound Generate_params.model(model).upper_bound_beta] ...
                    );

            end;    %Loop through subs

            %Should save after each model completed
            save([Generate_params.outpath filesep Generate_params.outname], 'Generate_params');

        end;    %loop through fitted models

    end;   %loop through simulated veridical models

end;    %estimate parameters of simulated data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%






%%%%%%DO PLOTS!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if use_file_for_plots == 1; load(filename_for_plots); end;    %use file?

%Let's use subplot to plot accuracy by simulated parameter level for every
%combination of simed and fitted model
h1 = figure('Color',[1 1 1]);

%let's average over parameter levels to make confusion and inverse confusion matrices


%need multilevel loop to extract data again. Remember, model field refers
%to fitted model and inside each model field is an estimated_params field
%that is simulated subject * parameter (2) * simulated model (3)
for fitted_model = 1:size(Generate_params.model,3);

    for subject = 1:size(Generate_params.model(1).estimated_params,2);

        for simed_model = 1:size(Generate_params.model(1).estimated_params,3);



        end;    %loop through simulated models
    end;    %loop through simulated participants
end;    %loop through fitted models


disp('audi5000')

toc
%%%%%%%DO PLOTS!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
















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

    %How many samples for this model for this sequence and subject
    listDraws = ...
        Generate_params.model(Generate_params.current_model).num_samples(seq,Generate_params.num_subs_to_run);

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
%         if  Generate_params.model(Generate_params.current_model).all_draws == 1;
            ll = ll - sum(log(cprob((1:listDraws-1), 1))) - log(cprob(listDraws, 2));
%         else;
%             ll = ll - sum(log(cprob((listDraws-1), 1))) - log(cprob(listDraws, 2));
%         end;
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

        list.vals = squeeze(Generate_params.seq_vals(sequence,:,num_subs_found));

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
            %find first index that is a candidate ....
            num_samples(sequence,this_sub) = find(choiceStop == 1,1,'first');   %assign output num samples for cut off model
            %Reverse 0s and 1's
            choiceCont = double(~choiceStop);

        else;   %Any Bayesian models

            [choiceStop, choiceCont, difVal]  = ...
                analyzeSecretaryPR(Generate_params,list.vals);

            num_samples(sequence,this_sub) = find(difVal<0,1,'first');  %assign output num samples for Bruno model

        end;    %Cutoff or other model?

        %...and its rank
        ranks(sequence,this_sub) = dataList( num_samples(sequence,this_sub) );
        %Accumulate action values too so you can compute ll outside this function if needed
        choiceStop_all(sequence, :, this_sub) = choiceStop;
        choiceCont_all(sequence, :, this_sub) = choiceCont;

    end;    %loop through sequences

    this_sub = this_sub + 1;

end;    %loop through subs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%












