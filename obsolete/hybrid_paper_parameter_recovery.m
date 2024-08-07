
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [] = param_recover_v3_sweep;

%hybrid_paper_parameter_recovery.m adapts Param_recover_v3_sweep_v2.m and
%the output files derived from it and makes the outputs a bit friendler for
%publication.

%v3_sweep_v2 attempts to take v3_sweep and make the plots of original and
%estimated parameters a bit friendlier.

%v3_sweep isn't an improvemnt on v2 but starts a new branch. V2 (and its
%successors if any) will continue to run preconfigured models that simulate
%over versus undersampling. The v3_sweep branch will sweep across parameter
%values for each model and plot configured versus estimated [parameters and
%performance).

%v3 note: I have a bad habit of using identifier and indicator
%interchangeably.

%v2: I changed the biased values and biased rewards models to be single
%parameter (Just threshold with a fixed high slope to resemble a sharp
%threshold). Neither all draws nor last draw seem to vary both parameters
%as configured but mainly varies only slope, with some exception. So number
%of models changes

%v2: Also added to v1 some more visualisatioin tools: scatterplots to compare estimated anbd configured
%performance subject by subject and modifications to parameter visuation figure

%V1: originally implemented slope and threshold versions of BV and BR.

tic

addpath(genpath('C:\matlab_files\fiance\parameter_recovery\beta_fixed_code\Model_fitting_hybrid_study\plotSpread'));
addpath(genpath('C:\matlab_files\fiance\parameter_recovery\beta_fixed_code\Model_fitting_hybrid_study\klabhub-bayesFactor-3d1e8a5'));

%The idea was to save data in files along the way as needed. But as
%implemented now, later sections are dependent on previous ones.
simulate_stimuli = 0;  %If 1, randomly generates phase 1 distributions and sequences from them and saves info in Generate_params
make_config_model_data = 0;  %If 1, presets model parameters in Generate_params and then uses simulated stimuli to create simulated model performance
check_params = 0;       %fit the same model that created the data and output estimated parameters
make_est_model_data = 0;
use_file_for_plots = 1; %Set the above to zero and this to 1 and it'll read in a file you specify (See filename_for_plots variable below) and make plots of whatever analyses are in the Generate_params structure in that file;
make_plots = 1;         %if 1, plots the results
all_draws_set = 1;          %You can toggle how the ll is computed here for all models at once if you want or go bvelow and set different values for different models manually in structure
log_or_not = 0; %I'm changing things so all simulated data is logged at point of simulation (==1) or not
%1: cutoff 2: Cs 3: dummy (formerly IO in v2) 4: BV 5: BR 6: BPM 7: Opt 8: BPV
%(I keep model 3 as a legacy for IO because analyseSecertaryNick_2021
%looks for identifiers 4 and 5 for BV and BR and needs that for v2. Also it keeps the same color scheme as v2)
%do_models = [1 2 4 5 7];    %These are now what v2 called model identifiers - Applies at the moment to both make_model_data and check_params;
comment = 'bigSim28seqs';    %The filename will already fill in basic parameters so only use special info for this.
do_models = [2 1 7 4 5];    %These are now what v2 called model identifiers - Applies at the moment to both make_model_data and check_params;
outpath = 'C:\matlab_files\fiance\parameter_recovery\beta_fixed_code\Model_fitting_hybrid_study\outputs';
%Unfortunately still needs to be typed in manually
% filename_for_plots = 'C:\matlab_files\fiance\parameter_recovery\outputs\out_param_rec_ll1_40models10subs28seqs12opts_bigSim28seqs_20211407.mat';
filename_for_plots = 'C:\matlab_files\fiance\parameter_recovery\beta_fixed_code\Model_fitting_hybrid_study\outputs\out_param_rec_ll1_60models20subs5seqs12opts_bigSim_20212106.mat';
num_param_levels = 8;   %Will use start values and increments specified below to run up to this many parameters for each model
%These correspond to identifiers (not configured implementations like in v2) in the v3_sweep version
model_names = {'Cut off' 'Cs' 'IO' 'BV' 'BR' 'BPM' 'Opt' 'BPV' }; %IO is a placeholder, don't implement
num_model_identifiers = size(model_names,2);


if simulate_stimuli == 1;
    
    Generate_params.log_or_not = log_or_not;
    Generate_params.all_draws_set = all_draws_set;
    Generate_params.do_models_identifiers = do_models;
    Generate_params.num_param_levels = num_param_levels;
    
    %Configure sequences
    %Careful, the number of items in the sequences (seq_length*num_seqs)
    %should not exceed the number of items rated in phase 1. Ideally the
    %former should be at most 60% of latter.
    Generate_params.num_subs = 10;   %So this will be per parameter value
    Generate_params.num_seqs = 28;
    Generate_params.seq_length = 12;
    Generate_params.num_vals = 426;  %How many items in phase 1 and available as options in sequences?
    Generate_params.rating_bounds = [1 100];    %What is min and max of rating scale?
    Generate_params.rating_grand_mean = 50;     %Individual subjects' rating means will jitter around this
    Generate_params.rating_mean_jitter = 5;     %How much to jitter participant ratings means on average?
    Generate_params.rating_grand_std = 5;       %Individual subjects' rating std devs will jitter around this
    Generate_params.rating_var_jitter = 2;     %How much to jitter participant ratings vars on average?
    
    for sub = 1:Generate_params.num_subs;
        
        this_sub_rating_mean = Generate_params.rating_grand_mean + normrnd( 0, Generate_params.rating_mean_jitter );
        this_sub_rating_std = Generate_params.rating_grand_std + normrnd( 0, Generate_params.rating_var_jitter );
        
        %Generate a truncated normal distribution of ratings
        %Keep making vectors until one satisfies contraints
        %         disp('Simulating ratings data');
        phase1 = zeros(Generate_params.num_vals,1);
        while ...
                sum(phase1 < Generate_params.rating_bounds(1)) > 0 ...        %Make sure none are below lower bound
                | sum(phase1 > Generate_params.rating_bounds(2)) > 0;         %Make sure none are above upper bound
            %                       numel(unique(round(phase1)))~=Generate_params.num_vals | ...    %Make sure there are no doubles
            
            phase1 = pearsrnd(   ...
                this_sub_rating_mean ...
                ,this_sub_rating_std ...
                ,0 ...  %skew
                ,3 ...  %kurtosis
                ,Generate_params.num_vals ...  %rows
                ,1 ...                      %cols
                );   %seq length n stuff
            %You'll need to add something here to replace doubles
            %and bound it but that's ok
            
        end;    %while loop for ratings creation
        
        if log_or_not == 1;
            phase1 = log(phase1);
            Generate_params.BVrange = log( Generate_params.rating_bounds )
        else
            Generate_params.BVrange = Generate_params.rating_bounds;
        end;    %transform ratings if log_or_not
        
        %Save this sub's ratings data
        Generate_params.ratings(:,sub) = phase1;
        %         Generate_params.mean_ratings(sub,1) = mean(phase1);
        %         Generate_params.var_ratings(sub,1) = var(phase1);
        
        %Grab the requisit number of random ratings
        temp_ratings = phase1(randperm(numel(phase1)),1);
        Generate_params.seq_vals(:,:,sub) = reshape(...
            temp_ratings(1:Generate_params.num_seqs*Generate_params.seq_length,1) ...
            ,Generate_params.num_seqs ...
            ,Generate_params.seq_length ...
            );
        
        %Compute the things all models need from the simulated stimuli for
        %this subject - reward bins (Should bin 100 values into 100 bins
        %and hence do nothing by default), prior mean, prior variance
        %(already log transformed above, if needed)
        %     Generate_params.nbins_reward = 6;  %Why did I ever need these bins? Is it time to retire them yet?
        Generate_params.nbins_reward = numel(Generate_params.rating_bounds(1):Generate_params.rating_bounds(2));  %This should effectuvely remove the binning
        Generate_params.binEdges_reward = ...
            linspace(...
            Generate_params.BVrange(1) ...
            ,Generate_params.BVrange(2)...
            ,Generate_params.nbins_reward+1 ...
            );   %organise bins by min and max
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
    model_template.BVmid = 55;      %row 7, initialised to halfway through the rating scale (can't be used with log)
    model_template.BRslope = 1;    %row 8
    model_template.BRmid = 55;      %row 9
    model_template.BP = 0;           %row 10
    model_template.optimism = 0;    %row 11
    model_template.BPV = 0;          %row 12
    %     model_template.log_or_not = log_or_not; %1 = log transform (normalise) ratings  %row 13 (This is already legacy - log or not is now controlled by switch at start of programme and simulated data was logged before reaching this point).
    model_template.all_draws = all_draws_set;  %1 = use all trials when computing ll instead of last two per sequence.   %row 14
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
        repmat(model_template.all_draws,1,num_cols);   %row 14: all draws
        repmat(model_template.beta,1,num_cols);   %row 15: beta
        ];
    
    %Later, when model fitting, I'll need to know which paraneters are free
    %and which to leave fixed. Rather than make another redundant and inelegent lookup table,
    %I'm going to operate under the assumption that anything I set to be
    %different from the default is a parameter that I will want to test as
    %free later. So I will save a copy of the default param_config and then
    %mark as free parameters anything parameters whose values I manipulate
    %in the next step.
    param_config_default = param_config;
    %     free_parameters = zeros(size(param_config));
    
    %What param levels are needed for each model? (For now, I'll use as the range the values I was using in v2
    %param_used is index into param_config (Holds the params to be used in
    %sim), param_config_default (holds the default and startring values) and
    %free_parameters (holds flags for which parameters need estimation)
    %Note 3 is skipped, it's a placeholder for IO, which is implemented in
    %v2 but not here (It's just Cs = 0 so is now an indicator 2 model and
    %doesn't need a seprate model identifier so gets a placeholder).
    %     the_param_levels(1,:) = linspace(opt_rule-3,opt_rule+3,num_param_levels); param_used(1) = 4;   %Model indicator 1: Cut off;
    %     the_param_levels(2,:) = linspace(-.005,.005,num_param_levels); param_used(2) = 5;              %Model indicator 2: Cs
    %     the_param_levels(4,:) = linspace(50,60,num_param_levels); param_used(4) = 7;                   %Model indicator 4: BV
    %     the_param_levels(5,:) = linspace(45,65,num_param_levels); param_used(5) = 9;                   %Model indicator 5: BR
    %     the_param_levels(6,:) = linspace(-20,20,num_param_levels); param_used(6) = 10;                  %Model indicator 6: BP
    %     the_param_levels(7,:) = linspace(-2,2,num_param_levels); param_used(7) = 11;                    %Model indicator 7: Opt
    %     the_param_levels(8,:) = linspace(-30,30,num_param_levels); param_used(8) = 12;                  %Model indicator 8: BV
    %Wider params:
    the_param_levels(1,:) = linspace(opt_rule-4,opt_rule+4,num_param_levels); param_used(1) = 4;   %Model indicator 1: Cut off;
    the_param_levels(2,:) = linspace(-.01,.01,num_param_levels); param_used(2) = 5;              %Model indicator 2: Cs
    the_param_levels(4,:) = linspace(40,70,num_param_levels); param_used(4) = 7;                   %Model indicator 4: BV
    the_param_levels(5,:) = linspace(35,75,num_param_levels); param_used(5) = 9;                   %Model indicator 5: BR
    the_param_levels(6,:) = linspace(-20,20,num_param_levels); param_used(6) = 10;                  %Model indicator 6: BP
    the_param_levels(7,:) = linspace(-4,4,num_param_levels); param_used(7) = 11;                    %Model indicator 7: Opt
    the_param_levels(8,:) = linspace(-30,30,num_param_levels); param_used(8) = 12;                  %Model indicator 8: BV
    %     %Even wider params:
    %     the_param_levels(1,:) = linspace(opt_rule-4,opt_rule+5,num_param_levels); param_used(1) = 4;   %Model indicator 1: Cut off;
    %     the_param_levels(2,:) = linspace(-.02,.02,num_param_levels); param_used(2) = 5;              %Model indicator 2: Cs
    %     the_param_levels(4,:) = linspace(30,80,num_param_levels); param_used(4) = 7;                   %Model indicator 4: BV
    %     the_param_levels(5,:) = linspace(25,85,num_param_levels); param_used(5) = 9;                   %Model indicator 5: BR
    %     the_param_levels(6,:) = linspace(-20,20,num_param_levels); param_used(6) = 10;                  %Model indicator 6: BP
    %     the_param_levels(7,:) = linspace(-5,5,num_param_levels); param_used(7) = 11;                    %Model indicator 7: Opt
    %     the_param_levels(8,:) = linspace(-30,30,num_param_levels); param_used(8) = 12;                  %Model indicator 8: BV
    
    %Now fill in configured parameters and flag them for each model indicator in param_config and free_parameters.
    for i=1:numel(param_used);
        
        if i ~=3;   %Because this is IO placegholder and we don't want to change any parameters for it (Just skip 3 in do_models)
            
            %Now configure this groups parameters
            param_config(param_used(i), find(param_config(1,:)==i) ) = the_param_levels(i,:);
            free_parameters(param_used(i), find(param_config(1,:)==i) ) = 1;
            
        end;
    end;
    
    %Now reduce matrices to just those in do_models
    %Do_models in v3_sweep means indicators, not implementations, so its more
    %compolicated, we need to search for all implementations of the same indicator
    temp_config = [];
    temp_default = [];
    temp_free = [];
    for i=1:size(do_models,2);
        
        identifier = do_models(i);
        indices = find(param_config(1,:)==identifier);
        
        temp_config = [temp_config param_config(:,indices)];
        temp_default = [temp_default param_config_default(:,indices)];
        temp_free = [temp_free free_parameters(:,indices)];
        
    end;    %Loop through model types
    
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
        'out_param_rec_ll%d_%dmodels%dsubs%dseqs%dopts_%s_'...
        , Generate_params.all_draws_set ...
        , Generate_params.num_models ...
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
        
        %         disp(['saving C:\matlab_files\fiance\online_domains_01\fitted_datafiles\fitted_domains_' model_strs{param_to_fit(model)} sprintf('_params_exp%d.mat',experiment)]);
        %         save(['C:\matlab_files\fiance\online_domains_01\fitted_datafiles\fitted_domains_' model_strs{param_to_fit(model)} sprintf('_params_exp%d.mat',experiment)]);
        %
    end;    %loop through models
    
    
    
    save([Generate_params.outpath filesep Generate_params.outname], 'Generate_params');
    
end; %Do I want to get performance from the pre-configured models?
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%







%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%DO THE MODEL FITTING!!!!!!%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if check_params == 1;
    
    %         %How many models to do in this section?
    %         do_models = [1 2 3];
    do_models = 1:Generate_params.num_models;
    
    %In the previous section, we only assigned to the Generate_oparam
    %struct the models that were in the do_models in that section. So
    %this can only operate on those models (until I start implementing these sections from datafiles later).
    for model = do_models;
        
        for sub = 1:Generate_params.num_subs;
            
            
            %You want to fit one model for one subject at a time
            Generate_params.current_model = model;
            Generate_params.num_subs_to_run = sub;
            
            %Use default params as initial values
            params = [ ...
                Generate_params.model(model).this_models_free_parameter_default_vals ...
                Generate_params.model(model).beta ...
                ];
            
            %                 options = optimset('Display','iter');
            warning('off');
            
            disp(...
                sprintf('fitting modeli %d name %s subject %d' ...
                , model ...
                , Generate_params.model( Generate_params.current_model ).name ...
                , sub ...
                ) );
            
            %%%%%%%%%%%%%%%%%%%%%%%%
            %%%%Main function call in this section
            %%%%%%%%%%%%%%%%%%%%%%%%%%
            [Generate_params.model(model).estimated_params(sub,:) ...
                ,  Generate_params.model(model).ll(sub,:) ...
                , exitflag, search_out] = ...
                fminsearch(  @(params) f_fitparams( params, Generate_params ), params);
            
            %                     fminsearch(  @(params) f_fitparams( params, Generate_params ), params, options);
            
            
        end;    %Loop through subs
        
        %Should save after each model completed
        save([Generate_params.outpath filesep Generate_params.outname], 'Generate_params');
        
    end;   %loop through models
    
end;    %estimate parameters of simulated data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%






%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%Generate performance from estimated parameters!%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if make_est_model_data == 1;
    
    %     %for temp usage
    %         load(filename_for_plots,'Generate_params');
    
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




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%PLOT!%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if make_plots == 1;
    
    %Use data already in workspace because computed above by or ?
    %Or open a file where these things were computed on previous run?
    if use_file_for_plots == 1;
        
        load(filename_for_plots,'Generate_params');
        
    end;    %Plot data structure from a file?
    
    plot_data(Generate_params);
    
end;    %Do plots or not?


%Just to be safe
% save([Generate_params.outpath filesep Generate_params.outname],'Generate_params');

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
        
        Generate_params.model(Generate_params.current_model).(fields{field}) = ...
            Generate_params.model(Generate_params.current_model).estimated_params(sub,it);
        it=it+1;
        
    end;
    
    disp(...
        sprintf('computing performance, fitted modeli %d name %s subject %d' ...
        , Generate_params.current_model ...
        , Generate_params.model( Generate_params.current_model ).name ...
        , sub ...
        ) );
    
    Generate_params.num_subs_to_run = sub;
    [num_samples(sub,:) ranks(sub,:)] = generate_a_models_data(Generate_params);
    
end;    %Loop through subs

fprintf(' ');








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
        if  Generate_params.model(Generate_params.current_model).all_draws == 1;
            ll = ll - sum(log(cprob((1:listDraws-1), 1))) - log(cprob(listDraws, 2));
        else;
            ll = ll - sum(log(cprob((listDraws-1), 1))) - log(cprob(listDraws, 2));
        end;
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
        
        %         if Generate_params.model(Generate_params.current_model).log_or_not == 1;
        %             Generate_params.binEdges_reward = ...
        %                 linspace(log(Generate_params.rating_bounds(1)),log(Generate_params.rating_bounds(2)),Generate_params.nbins_reward+1);   %organise bins by min and max
        %             Generate_params.PriorMean = mean(log(Generate_params.ratings(:,num_subs_found)));
        %             Generate_params.PriorVar = var(log(Generate_params.ratings(:,num_subs_found)));
        %             Generate_params.BVrange = log(Generate_params.rating_bounds);   %Used for normalising BV
        %             list.allVals = log(squeeze(Generate_params.seq_vals(sequence,:,num_subs_found)));
        %         else
        
        %             Generate_params.BVrange = Generate_params.rating_bounds;    %Used for normalising BV
        list.allVals = squeeze(Generate_params.seq_vals(sequence,:,num_subs_found));
        %         end;
        
        %ranks for this sequence
        dataList = tiedrank(squeeze(Generate_params.seq_vals(sequence,:,num_subs_found))');
        
        %         list.optimize = 0;
        %         list.flip = 1;
        list.vals =  list.allVals;
        %         list.length = Generate_params.seq_length;
        
        %Do cutoff model, if needed
        if Generate_params.model(Generate_params.current_model).identifier == 1;
            
            %get seq vals to process
            this_seq_vals = list.allVals;
            %initialise all sequence positions to zero/continue (value of stopping zero)
            choiceStop = zeros(1,Generate_params.seq_length);
            %What's the cutoff?
            estimated_cutoff = round(Generate_params.model(Generate_params.current_model).cutoff);
            if estimated_cutoff < 1; estimated_cutoff = 1; end;
            if estimated_cutoff > 12; estimated_cutoff = 12; end;
            %find seq vals greater than the max in the period
            %before cutoff and give these candidates a maximal stopping value of 1
            choiceStop(1,find( this_seq_vals > max(this_seq_vals(1:estimated_cutoff)) ) ) = 1;
            %set the last position to 1, whether it's greater than
            %the best in the learning period or not
            choiceStop(1,Generate_params.seq_length) = 1;
            %find first index that is a candidate ....
            num_samples(sequence,this_sub) = find(choiceStop == 1,1,'first');   %assign output num samples for cut off model
            
            %Reverse 0s and 1's for ChoiceCont
            choiceCont = double(~choiceStop);
            
        else;   %Any Bayesian models
            
            [choiceStop, choiceCont, difVal]  = ...
                analyzeSecretaryNick_2021(Generate_params,list);
            
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
function [] = plot_data(Generate_params);

%set up plot appearance
%For now I'll try to match model identifiers to colors. Which means this
%colormap needs to scale to the total possible number of models, not the
%number of models
plot_cmap = hsv(8+1);  %models + subjects
f_a = 0.1; %face alpha
sw = 1;  %ppoint spread width
graph_font = 10;

% %averaged participants scatterplot - Configured versus estimated params per identifier, and configured versus estimated sampling per identifier.
% h10 = figure('NumberTitle', 'off', 'Name',['parameters: averaged participants ' Generate_params.outname]);
% set(gcf,'Color',[1 1 1]);
% %...and for samples
% h11 = figure('NumberTitle', 'off', 'Name',['samples: averaged participants ' Generate_params.outname]);
% set(gcf,'Color',[1 1 1]);

%Scatterplot with ALL participants' fitted parameter values (not just averaged for each configured level) and configured values
h100 = figure('NumberTitle', 'off', 'Name',['parameters: all participants ' Generate_params.outname]);
set(gcf,'Color',[1 1 1]);
%Scatterplot with samples from all participants' fitted parameter values (not just averaged for each configured level) and samples from configured values
h110 = figure('NumberTitle', 'off', 'Name',['samples: all participants ' Generate_params.outname]);
set(gcf,'Color',[1 1 1]);

%For scatterplot subplots
num_rows = floor(sqrt(numel(Generate_params.do_models_identifiers )) );
num_cols = ceil(numel(Generate_params.do_models_identifiers)/num_rows);

% %Let's try it by putting all the parameters and sampling in one column in the same figure
% num_rows = numel(Generate_params.do_models_identifiers);
% num_cols = 2;   %parameter scatters and sampling scatters

plot_it = 1;

% Generate_params.do_models_identifiers = [2 1 7 4 5];

%which parameter value levels do you want to analyse? (The paper uses 12 levels)
fprintf('');
keep_param_vals = [3:7];

identifier_accum = [];
samples_config_accum = [];
samples_fitted_accum = [];
param_config_accum = [];   %configured, for bar plot
param_fitted_accum = [];  %estimated / fitted, for bar plot


model_count = 0;    %all the parameter value/level * indentifier / model type combinations that are slated to be kept and to be plotted in bar plot below.
for identifier = 1:numel(Generate_params.do_models_identifiers);
    %    1 2 4 5 7
    
    parameters_data = [];
    performance_data = [];
    parameters_data_ps = [];
    performance_data_ps = [];
    
    field_it = 1; %is updated for ever model field detected, not for every detection made that's on the keep list (like field_counter)
    field_counter = 0;  %should count up to the number of parameter levels to keep in keep_param_vals, used to index param_accum and parameters_accum
    
    
    %Go searching for this model (here, a model is any configured parameter
    %value for a given identifier (model type), not a model identifier, so
    %might be identifiers*configured values = models or 5*12=60
    for model_field = 1:size(Generate_params.model,2);
        
        %While I happen to be looping through all models, I need to change
        %the model names early on so they appear in graphs matching what
        %they are in other figures generated by other programmes
        if strcmp(Generate_params.model(model_field).name,'Cut off');
            Generate_params.model(model_field).name = 'CO';
        elseif strcmp(Generate_params.model(model_field).name,'Cs');
            Generate_params.model(model_field).name = 'CS';
        elseif strcmp(Generate_params.model(model_field).name,'Opt');
            Generate_params.model(model_field).name = 'O';
        end;
        
        if Generate_params.model(model_field).identifier == ...
                Generate_params.do_models_identifiers(identifier);
            
            %             field_counter = field_counter + 1;
            
            %each time this if/then statement makes a field detection, it
            %can extract the (20 participants estimated parameters / samples
            %for one configured parameter for one model identifier)
            %             if isfield(Generate_params.model(model_field),'estimated_params');
            
            %This pairs each configured value with the mean fitted
            %parameter across simulated participants for that
            %configured parameter value
            
            %                 %configured is 1, estimated is 2
            %                 parameters_data( field_it, 1) = ...
            %                     Generate_params.model(model_field).this_models_free_parameter_configured_vals;
            %                 parameters_data( field_it, 2) = ...
            %                     nanmean(Generate_params.model(model_field).estimated_params(:,1));
            %                 performance_data( field_it, 1) = ...
            %                     nanmean(nanmean(Generate_params.model(model_field).num_samples));
            %                 performance_data( field_it, 2) = ...
            %                     nanmean(nanmean(Generate_params.model(model_field).num_samples_est));
            
            if ismember(field_it,keep_param_vals); %Is this a parameter value level you'd like to plot?

                
                %configured is 1, estimated is 2
                parameters_data = [ parameters_data; ...
                    [Generate_params.model(model_field).this_models_free_parameter_configured_vals nanmean(Generate_params.model(model_field).estimated_params(:,1))] ...
                    ];
                performance_data = [ performance_data; ...
                    [nanmean(nanmean(Generate_params.model(model_field).num_samples)) nanmean(nanmean(Generate_params.model(model_field).num_samples_est))] ...
                    ];
                
                %This accumulation is needed for the bar / point spread plots below
%                 field_counter = field_counter + 1;
                model_count = model_count+1;
                
                %accumulate model identifiers to color bar plot by model later
                identifier_accum(model_count) = Generate_params.do_models_identifiers(identifier);
                name_accum{model_count} = Generate_params.model(model_field).name;
                %accumulate the samples from the kept configured models
                samples_config_accum(:,model_count) = nanmean(Generate_params.model(model_field).num_samples)';
                %accumulate the samples from the kept fitted models
                samples_fitted_accum(:,model_count) = nanmean(Generate_params.model(model_field).num_samples_est)';
                
                %accumulate the parameters from the kept configured models
                param_config_accum(:,model_count) = Generate_params.model(model_field).this_models_free_parameter_configured_vals;
                %accumulate the samples from the kept fitted models
                param_fitted_accum(:,model_count) = Generate_params.model(model_field).estimated_params(:,1);
                
                
%                 %accumulate model identifiers to color bar plot by model later
%                 identifier_accum(field_counter,identifier) = Generate_params.do_models_identifiers(identifier);
%                 %accumulate the samples from the kept configured models
%                 samples_config_accum(:,field_counter,identifier) = nanmean(Generate_params.model(model_field).num_samples)';
%                 %accumulate the samples from the kept fitted models
%                 samples_fitted_accum(:,field_counter,identifier) = nanmean(Generate_params.model(model_field).num_samples_est)';
%                 
%                 %accumulate the parameters from the kept configured models
%                 param_config_accum(field_counter,identifier) = Generate_params.model(model_field).this_models_free_parameter_configured_vals;
%                 %accumulate the samples from the kept fitted models
%                 param_fitted_accum(:,field_counter,identifier) = Generate_params.model(model_field).estimated_params(:,1);
%                 
                
                %This takes every fitted value for this model and pairs its
                %configured value, so all participants' values (and not
                %just averages over participants) can be scatter'ed.
                parameters_data_ps = [ ...
                    parameters_data_ps; ...
                    [ones(size(Generate_params.model(model_field).estimated_params(:,1),1),1)*Generate_params.model(model_field).this_models_free_parameter_configured_vals  Generate_params.model(model_field).estimated_params(:,1)]  ...
                    ];
                performance_data_ps = [ ...
                    performance_data_ps; ...
                    [nanmean(Generate_params.model(model_field).num_samples)' ...
                    nanmean(Generate_params.model(model_field).num_samples_est)' ]  ...
                    ];
                %
                %                 else
                %
                %                     parameters_data( field_it, 1) = NaN;
                %                     parameters_data( field_it, 2) = NaN;
                %                     performance_data( field_it, 1) = NaN;
                %                     performance_data( field_it, 2) = NaN;
                %
                %                 end;    %check that the needed parameters were actually estimated
                
            end; %Is this a parameter value level you'd like to plot?
            
            name = Generate_params.model(model_field).name;
            
            field_it = field_it + 1;
            
        end;  %If this field hold an implementation of the correct model identifier
    end;  %Loop through the model fields (implementations) looking for the needed identifiers
    
    
    %setup general stuff for plotting scatterplots
    clear plot_details;
    plot_details.subplot = [num_rows,num_cols,plot_it];
    plot_details.color = plot_cmap(Generate_params.do_models_identifiers(identifier)+1,:);
    plot_details.graph_font = graph_font;
    plot_details.model_name = name;
    
%     %set up parameter participant-averaged scatterplot
%     plot_details.fig = h10;
%     plot_details.these_data = [parameters_data(:,1), parameters_data(:,2)];
%     plot_details.y_string = 'parameters';
%     %     plot_details.limits = [min(min(plot_details.these_data)) max(max(plot_details.these_data))];
%     plot_details.ticks = [];
%     plot_a_scatter(plot_details,Generate_params);
%     
%     %set up sampling participant-averaged scatterplot
%     plot_details.fig = h11;
%     plot_details.these_data = [performance_data(:,1), performance_data(:,1)];
%     plot_details.y_string = 'samples';
%     plot_details.ticks = [0:2:Generate_params.seq_length];
%     plot_a_scatter(plot_details,Generate_params);
    
    %set up parameter all participants scatterplot
    plot_details.fig = h100;
    plot_details.these_data = [parameters_data_ps(:,1), parameters_data_ps(:,2)];
    plot_details.y_string = 'parameters';
    plot_details.ticks = [];
    %     plot_details.limits = [min(min(plot_details.these_data)) max(max(plot_details.these_data))];
    plot_a_scatter(plot_details,Generate_params);
    
    %set up samples all participants scatterplot
    plot_details.fig = h110;
    plot_details.these_data = [performance_data_ps(:,1), performance_data_ps(:,2)];
    plot_details.y_string = 'samples';
    plot_details.ticks = [0:2:Generate_params.seq_length];
    plot_a_scatter(plot_details,Generate_params);
    
    plot_it = plot_it+1;
    
    
    %     figure(h10);
    %         subplot(num_rows,num_cols,plot_it); hold on;
    %     subplot(num_rows,num_cols,1+(identifier-1)*num_cols); hold on;
    
    %     scatter( ...
    %         parameters_data(:,1)...
    %         , parameters_data(:,2) ...
    %         , 'MarkerEdgeColor', plot_cmap(Generate_params.do_models_identifiers(identifier)+1,:) ...
    %         , 'MarkerFaceColor', plot_cmap(Generate_params.do_models_identifiers(identifier)+1,:) ...
    %         );
    
    %     %regression line
    %     b = regress( ...
    %         parameters_data(:,2) ...
    %         , [ones(Generate_params.num_param_levels,1) parameters_data(:,1)] ...
    %         );
    %     x_vals = [min(parameters_data(:,1)) max(parameters_data(:,1))];
    %     y_hat = b(1) + b(2)*x_vals;
    
    %     plot( x_vals, y_hat ...
    %         , 'Color', plot_cmap(Generate_params.do_models_identifiers(identifier)+1,:) ...
    %         );
    
    %     %I want to regression line
    %     % y_hat = x_vals;
    %     plot( x_vals, y_hat ...
    %         , 'Color', plot_cmap(Generate_params.do_models_identifiers(identifier)+1,:) ...
    %         );
    
    %     min_val = min(min(parameters_data));
    %     max_val = max(max(parameters_data));
    %
    %     %I want to plot the diagonal instead of the regression line
    %     plot([min_val max_val],[min_val max_val], 'Color', [.5 .5 .5]);
    %
    %     set(gca ...
    %         , 'Fontname','Arial' ...
    %         , 'Fontsize',graph_font ...
    %         , 'FontWeight','normal' ...
    %         );
    
    %     if min_val~=max_val & ~isnan(min_val) & ~isnan(max_val);    %If all draws are the same and min=max then set axis chokes on it.
    %         ylim([min_val max_val]);
    %     end;
    %     xlim([min(parameters_data(:,1)) max(parameters_data(:,1))]);
    %     ylim([min(parameters_data(:,2)) max(parameters_data(:,2))]);
    % ylabel('Estimated');
    % xlabel('Configured');
    
    %     title( ...
    %         sprintf('%s',name) ...
    %         , 'Fontname','Arial' ...
    %         , 'Fontsize',graph_font ...
    %         , 'FontWeight','normal' ...
    %         );
    
    %     %Now the samples, configured (derived from) and estimated
    %     figure(h10);
    %     %     subplot(num_rows,num_cols,plot_it); hold on;
    %     subplot(num_rows,num_cols,2+(identifier-1)*num_cols); hold on;
    %
    %
    %     scatter( ...
    %         performance_data(:,1)...
    %         , performance_data(:,2) ...
    %         , 'MarkerEdgeColor', plot_cmap(Generate_params.do_models_identifiers(identifier)+1,:) ...
    %         , 'MarkerFaceColor', plot_cmap(Generate_params.do_models_identifiers(identifier)+1,:) ...
    %         );
    
    %     %regression line
    %     b = regress( ...
    %         performance_data(:,2) ...
    %         , [ones(Generate_params.num_param_levels,1) performance_data(:,1)] ...
    %         );
    %     x_vals = [min(performance_data(:,1)) max(performance_data(:,1))];
    %     y_hat = b(1) + b(2)*x_vals;
    
    %     plot( x_vals, y_hat ...
    %         , 'Color', plot_cmap(Generate_params.do_models_identifiers(identifier)+1,:) ...
    %         );
    
    
    %     y_hat = x_vals;
    %     plot( x_vals, y_hat ...
    %         , 'Color', plot_cmap(Generate_params.do_models_identifiers(identifier)+1,:) ...
    %         );
    
    %     min_val = min(min(performance_data));
    %     max_val = max(max(performance_data));
    
    %     %I want to plot the diagonal instead of the regression line
    %     plot([min_val max_val],[min_val max_val], 'Color', [.5 .5 .5]);
    
    %     set(gca ...
    %         , 'Fontname','Arial' ...
    %         , 'Fontsize',graph_font ...
    %         , 'FontWeight','normal' ...
    %         );
    %     if min_val~=max_val & ~isnan(min_val) & ~isnan(max_val);    %If all draws are the same and min=max then set axis chokes on it.
    %         ylim([0 Generate_params.seq_length]);
    %         xlim([0 Generate_params.seq_length]);
    %     end;
    
    
    %     ylabel('Estimated');
    %     xlabel('Configured');
    
    %     title( ...
    %         sprintf('%s',name) ...
    %         , 'Fontname','Arial' ...
    %         , 'Fontsize',graph_font ...
    %         , 'FontWeight','normal' ...
    %         );
    
    %     plot_it = plot_it+1;
    
end;    %Loop through model types


h1 = figure('Color',[1 1 1],'NumberTitle', 'off', 'Name',['bars samples configured v. estimated ' Generate_params.outname]);
% h3 = figure('Color',[1 1 1],'NumberTitle', 'off', 'Name',['bars parameters configured v. estimated' Generate_params.outname]);

for performance_plot = 1:2;
    
    if performance_plot == 1;
        access_str = 'samples_config_accum(:,model)';
        param_access_str = 'this_models_free_parameter_configured_vals';
    else
        access_str = 'samples_fitted_accum(:,model)';    %Need for correct plotting below on plot 2
        param_access_str = 'estimated_params(:,1)';
    end;
    
    %     plot_two_it = 1;
    %     for model = 1:Generate_params.num_models;
    for model = 1:numel(identifier_accum);   %Loop through kept/accumulated models
        
        
        
        %         %first make plot of draws for configured model simulations. This always runs, since presumably you wouldn't
        %         %call plot_data having done NO analysis. So we won't check first whether
        %         %data are simulated for configured models.
        %         if performance_plot == 1;
        %             if model == 1;
        %                 h1 = figure('NumberTitle', 'off', 'Name',['pre-configured ' Generate_params.outname]);
        %                 set(gcf,'Color',[1 1 1]);
        %             else
        %                 figure(h1);
        %             end;
        %             access_str = [];
        
        %         elseif performance_plot == 2 & isfield(Generate_params.model(1),'num_samples_est');
        %
        %             if model == 1;
        %
        %                 %Average performance bars and spread
        %                 set(gcf,'Color',[1 1 1]);
        %
        % %                 %Scatterplots
        % %                 h4 = figure('NumberTitle', 'off', 'Name',['estimated ' Generate_params.outname]);
        % %                 set(gcf,'Color',[1 1 1]);
        % %
        % %                 %For scatterplot subplots
        % %                 num_rows = floor(sqrt( Generate_params.num_models ) );
        % %                 num_cols = ceil(Generate_params.num_models/num_rows);
        %
        %             end;    %Is it first model or not?
        %
        %             access_str = '_est';    %Need for correct plotting below on plot 2
        %
        %             %Set up scatterplot
        %             figure(h4);
        %             subplot(num_rows,num_cols,plot_two_it); hold on;
        %             plot_two_it = plot_two_it + 1;
        
        %             scatter( ...
        %                 nanmean(Generate_params.model(model).num_samples)...
        %                 , nanmean(Generate_params.model(model).num_samples_est) ...
        %                 , 'MarkerEdgeColor', plot_cmap(Generate_params.model(model).identifier+1,:) ...
        %                 , 'MarkerFaceColor', plot_cmap(Generate_params.model(model).identifier+1,:) ...
        %                 );
        
        %             %regression line
        %             b = regress( ...
        %                 nanmean(Generate_params.model(model).num_samples_est)' ...
        %                 , [ones(Generate_params.num_subs,1) nanmean(Generate_params.model(model).num_samples)'] ...
        %                 );
        %             y_hat = b(1) + b(2)*[1:Generate_params.seq_length];
        
        %             plot( [1:Generate_params.seq_length], y_hat ...
        %                 , 'Color', plot_cmap(Generate_params.model(model).identifier+1,:) ...
        %                 );
        
        %             set(gca ...
        %                 , 'Fontname','Arial' ...
        %                 , 'Fontsize',graph_font ...
        %                 , 'FontWeight','normal' ...
        %                 , 'ylim',[1 Generate_params.seq_length] ...
        %                 , 'xlim',[1 Generate_params.seq_length] ...
        %                 );
        %             ylabel('Estimated samples');
        %             xlabel('Configured samples');
        
        %             title( ...
        %                 sprintf('%s',Generate_params.model(model).name) ...
        %                 , 'Fontname','Arial' ...
        %                 , 'Fontsize',graph_font ...
        %                 , 'FontWeight','normal' ...
        %                 );
        
        %Set up for next plot coming up below
        %             figure(h1);
        
        %         else;   %If not plot 2 or if there is no estimated fields
        %             continue
        %         end;    %Should we plot estimated results (if the second plot iteration and there is estimated data in struct to plot)
        %
        %         subplot(2,1,performance_plot);
        
        %         title( ...
        %             sprintf('%s',Generate_params.comment) ...
        %             ,'Fontname','Arial', ...
        %             'Fontsize',graph_font ...
        %             ,'FontWeight','normal' ...
        %             );
        
        %         this_model_draws = Generate_params.model(model).num_samples;
        %         this_model_draws = eval(['Generate_params.model(model).num_samples' access_str]);
        
        %set up plots
        clear plot_details;
        plot_details.model = model;
        plot_details.model_name = name_accum{model};
        %         plot_details.color = plot_cmap(Generate_params.model(model).identifier+1,:);
        plot_details.color = plot_cmap(identifier_accum(model)+1,:);
        plot_details.sw = sw;
        plot_details.f_a = f_a;
        plot_details.graph_font = graph_font;
        plot_details.keep_param_vals = keep_param_vals;
        plot_details.num_bars = numel(identifier_accum);
        
        %Samples plot
        plot_details.fig = h1;
        plot_details.subplot = [3,1,performance_plot];
%         plot_details.these_data_points = nanmean(eval(['Generate_params.model(model).num_samples' access_str]))';
%         plot_details.these_data_bars = nanmean(nanmean(eval(['Generate_params.model(model).num_samples' access_str]))');
        plot_details.these_data_points = eval(access_str);
        plot_details.these_data_bars = nanmean(eval(access_str));
        plot_details.y_string = 'Samples to decision';
        plot_details.yticks = [0:2:Generate_params.seq_length];
        plot_a_bar(plot_details,Generate_params);
        
        %parameters plot
        if performance_plot == 2;
            fprintf('');
            
            %Need min and max of parameters for a given model to scale plot
            %(parameters_accum is 12 (model param value) * 2 (config or est) * 7 (model identifier)
            
            %get param data to normalise
%             param_config = Generate_params.model(model).this_models_free_parameter_configured_vals;
%             params_fitted = eval(['Generate_params.model(model).' param_access_str]);
            
            param_config = param_config_accum(model);
            params_fitted = param_fitted_accum(:,model);
            
            %get min and max of both config'ed and fitted for all levels of this model
            temp_c = param_config_accum(:,find(identifier_accum==identifier_accum(model)));   %get all data with this identifier
            temp_f = param_fitted_accum(:,find(identifier_accum==identifier_accum(model)));   %get all data with this identifier
            param_max = max(max([temp_c; temp_f]));
            param_min = min(min([temp_c; temp_f]));
            
            
%             param_max = max([param_config max(max( parameters_accum(:,:,Generate_params.model(model).identifier) ) )]);
%             param_min = min([param_config min(min( parameters_accum(:,:,Generate_params.model(model).identifier) ) )]);
            %             param_max = max(param_accum(:,Generate_params.model(model).identifier));
            %             param_min = min(param_accum(:,Generate_params.model(model).identifier));
            
            %normalise both
            param_config_norm = (param_config - param_min)/(param_max - param_min);
            params_fitted_norm = (params_fitted - param_min)/(param_max - param_min);
            
            %plot_details.fig = h3;
            plot_details.subplot = [3,1,3];
            %plot_details.these_data_bars =
            plot_details.these_data_bars = param_config_norm;
            plot_details.these_data_points = params_fitted_norm;
            plot_details.y_string = 'Parameter value';
            plot_details.yticks = [0:.2:1.0];   %nornmalised parameter values
            plot_a_bar(plot_details,Generate_params);
            
        end;    %is it performance 2? (avoids double poltting subplot 3)
        
        %         %average over sequences (rows) but keep sub data (cols) for scatter points
        %         handles = plotSpread(nanmean(this_model_draws,1)' ...
        %             ,'xValues',model ...
        %             ,'distributionColors',plot_cmap(Generate_params.model(model).identifier+1,:) ...
        %             ,'distributionMarkers','.' ...
        %             , 'spreadWidth', sw ...
        %             );
        
        %         bar(model,nanmean(nanmean(this_model_draws,2)) ...
        %             ,'FaceColor',plot_cmap(Generate_params.model(model).identifier+1,:) ...
        %             ,'FaceAlpha',f_a ...
        %             ,'EdgeColor',[0 0 0] ...
        %             );
        
        %         if rem(model, Generate_params.num_param_levels) == round(Generate_params.num_param_levels/2)+2;    %Are we about halfway through the bars for this model?
        %             text( model, -.3 ...
        %                 ,sprintf('%s',Generate_params.model(model).name) ...
        %                 ,'Fontname','Arial' ...
        %                 ,'Fontsize',graph_font ...
        %                 ,'Rotation',0 ...
        %                 ,'HorizontalAlignment','right' ...
        %                 ,'Color',plot_cmap(Generate_params.model(model).identifier+1,:) ...
        %                 );
        %         end;
        
    end;    %loop through models
    
    
    
    
    
    % %     %now, parameters
    %     figure(h3)
    %     subplot(2,1,2);
    %
    %     title( ...
    %         sprintf('%s',Generate_params.comment) ...
    %         ,'Fontname','Arial', ...
    %         'Fontsize',graph_font ...
    %         ,'FontWeight','normal' ...
    %         );
    
    %     for model = 1:Generate_params.num_models;
    %
    %         %         this_model_ranks = Generate_params.model(model).ranks;
    %         this_model_ranks = eval(['Generate_params.model(model).ranks' access_str]);
    %
    %         %average over sequences (rows) but keep sub data (cols) for scatter points
    %         handles = plotSpread(nanmean(this_model_ranks,1)' ...
    %             ,'xValues',model ...
    %             ,'distributionColors',plot_cmap(Generate_params.model(model).identifier+1,:) ...
    %             ,'distributionMarkers','.' ...
    %             , 'spreadWidth', sw ...
    %             );
    %
    %         bar(model,nanmean(nanmean(this_model_ranks,2)) ...
    %             ,'FaceColor',plot_cmap(Generate_params.model(model).identifier+1,:) ...
    %             ,'FaceAlpha',f_a ...
    %             ,'EdgeColor',[0 0 0] ...
    %             );
    %
    %         text( model, -.25 ...
    %             ,sprintf('%s',Generate_params.model(model).name) ...
    %             ,'Fontname','Arial' ...
    %             ,'Fontsize',graph_font ...
    %             ,'Rotation',40 ...
    %             ,'HorizontalAlignment','right' ...
    %             );
    %
    %     end;    %loop through models
    %
    %     set(gca ...
    %         ,'XTick',[] ...
    %         ,'fontSize',graph_font ...
    %         ,'FontName','Arial',...
    %         'XLim',[-1 Generate_params.num_models+2] ...
    %         ,'YLim',[0 Generate_params.seq_length]);
    %     ylabel('Rank of chosen option');
    
end;    %Loop through (potentionally) two performance plots: preconfigured and (optionally) estimated-parameter performance

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

return;

%Now, if parameters of configured models were estimated, make plot of
%configured parameters and the estimated values for the differenbt
%sinmulated participants.

%h2 will plot values in veridical space
%h200 will dplot distributions of differences.
h2 = figure('NumberTitle', 'off', 'Name',['parameters' Generate_params.analysis_name]);
set(gcf,'Color',[1 1 1]);
h200 = figure('NumberTitle', 'off', 'Name',['parameters' Generate_params.analysis_name]);
set(gcf,'Color',[1 1 1]);

%V2 considers only single parameter models
it = 1; %subplot index iterator
% for param = 1:2;    %The (potentially) two parameters per model

for model  = 1:size(Generate_params.model,2);   %Loop through models in structure
    
    %this field does not include the beta parameter, which we aren't plotting
    if isfield(Generate_params.model(model),'estimated_params');
        %             & numel(Generate_params.model(model).this_models_free_parameter_configured_vals) >= param; %Does estimated param exist?
        
        %First plot the points in veridical space
        figure(h2);
        subplot(numel(Generate_params.do_models_identifiers), Generate_params.num_param_levels, model); hold on;
        %         subplot(size(Generate_params.model,2), 1, model); hold on;
        %         subplot(2, size(Generate_params.model,2), model); hold on;
        
        %             %plot the estimated parameter spread across simulated participants
        handles = plotSpread(...
            Generate_params.model(model).estimated_params(:,1) ...
            ,'xValues',1 ...
            ,'distributionColors',plot_cmap(Generate_params.model(model).identifier+1,:) ...
            ,'distributionMarkers','.' ...
            , 'spreadWidth', sw ...
            );
        
        plot(1, ...
            Generate_params.model(model).this_models_free_parameter_configured_vals(1)...
            ,'Marker','o' ...
            ,'MarkerFaceColor',[1 1 1] ...
            ,'MarkerEdgeColor',plot_cmap(Generate_params.model(model).identifier+1,:) ...
            ,'markerSize',8 ...
            ,'LineWidth',2 ...
            );
        
        plot(1, ...
            nanmean(Generate_params.model(model).estimated_params(:,1))...
            ,'Marker','o' ...
            ,'MarkerFaceColor',[1 1 1] ...
            ,'MarkerEdgeColor',plot_cmap(Generate_params.model(model).identifier+1,:) ...
            ,'markerSize',6 ...
            ,'LineWidth',1 ...
            );
        
        set(gca ...
            ,'XTick',1 ...
            ,'fontSize',graph_font - 4 ...
            ,'FontName','Arial'...
            ,'XTickLabel',sprintf('%s',Generate_params.model(model).name) ...
            ,'XTickLabelRotation',20 ...
            );
        
        ylabel( ...
            sprintf('%2.3f | %2.3f',...
            Generate_params.model(model).this_models_free_parameter_configured_vals(1) ...
            , nanmean(Generate_params.model(model).estimated_params(:,1)) ) ...
            ,'Fontname','Arial' ...
            ,'Fontsize',graph_font-4 ...
            ,'FontWeight','normal' ...
            );
        
        %Now add CIs of estimated params to these plots
        x = Generate_params.model(model).estimated_params(:,1);
        SEM = std(x)/sqrt(length(x));               % Standard Error
        ts = tinv([0.025  0.975],length(x)-1);      % T-Score
        CI = mean(x) + ts*SEM;                      % Confidence Intervals
        plot([1 1], CI ...
            , 'Marker','none' ...
            , 'Color', plot_cmap(Generate_params.model(model).identifier+1,:) ...
            );
        
        %Set symmetric axis limits
        
        %         %What's the biggest absolute value that needs to be plotted?
        %         all_data = [...
        %             Generate_params.model(model).estimated_params(:,1) ...
        %             ; CI' ...
        %             ; Generate_params.model(model).this_models_free_parameter_configured_vals(1) ...
        %             ];
        %         ylim([-max(abs(all_data)) max(abs(all_data))]);
        
        %Now, on new plot on second row, plot differences between estimated and configured
        figure(h200);
        subplot(numel(Generate_params.do_models_identifiers) ...
            , Generate_params.num_param_levels ...
            , model ...
            );
        %         subplot(1 ...
        %             , size(Generate_params.model,2)...
        %             , model ...
        %             );
        %         subplot(size(Generate_params.model,2)...
        %             , 1 ...
        %             , model ...
        %             );
        %         subplot(2 ...
        %             , size(Generate_params.model,2)...
        %             , model+size(Generate_params.model,2) ...
        %             );
        hold on;
        
        diffs = Generate_params.model(model).estimated_params(:,1) ...
            - Generate_params.model(model).this_models_free_parameter_configured_vals(1);
        
        h_hist = histogram(diffs,5);
        h_hist.FaceColor = plot_cmap(Generate_params.model(model).identifier+1,:);
        h_hist.EdgeColor = [0 0 0];
        edges = h_hist.BinEdges;
        if abs(edges(1)) > abs(edges(end));
            xrange = [ -abs(edges(1)) abs(edges(1))];
        else
            xrange = [ -abs(edges(end)) abs(edges(end))];
        end
        
        if xrange(2) > 10;
            xlim( [xrange(1) xrange(2)] );
        else
            xlim([-5 5]);
        end;
        %         xlim( [-50 50] );
        ylabel('number subjects');
        xlabel('estimated - configured param');
        
        set(gca ...
            ,'fontSize',graph_font - 4 ...
            ,'FontName','Arial'...
            );
        
    end;    %Test of there is an estimated parameter to plot
    
    %     it = it+1;
    
end;    %Loop through models to plot their estimated parameters

% end;    %Loop through each parameter that potentially could have been estimated
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function plot_a_bar(plot_details,Generate_params);

figure(plot_details.fig);
subplot(plot_details.subplot(1),plot_details.subplot(2),plot_details.subplot(3));

%         if numel(plot_details.these_data) > 1;
%average over sequences (rows) but keep sub data (cols) for scatter points
plotSpread(plot_details.these_data_points ...
    ,'xValues',plot_details.model ...
    ,'distributionColors', plot_details.color ...
    ,'distributionMarkers','.' ...
    , 'spreadWidth', plot_details.sw ...
    );
%         end;

bar(plot_details.model,plot_details.these_data_bars ...
    ,'FaceColor',plot_details.color ...
    ,'FaceAlpha',plot_details.f_a ...
    ,'EdgeColor',[0 0 0] ...
    );

num_vals_per_identifier = numel(plot_details.keep_param_vals);
if rem(plot_details.model, num_vals_per_identifier) == round(num_vals_per_identifier/2);    %Are we about halfway through the bars for this model?
    text( plot_details.model, -(1/Generate_params.seq_length)*plot_details.yticks(end) ...
        ,sprintf('%s',plot_details.model_name) ...
        ,'Fontname','Arial' ...
        ,'Fontsize',plot_details.graph_font ...
        ,'Rotation',0 ...
        ,'HorizontalAlignment','center' ...
        ,'Color',plot_details.color ...
        );
end;

set(gca ...
    ,'XTick',[] ...
    ,'fontSize',plot_details.graph_font ...
    ,'FontName','Arial',...
    'XLim',[-1 plot_details.num_bars+2] ...
    ,'YLim',[plot_details.yticks(1) plot_details.yticks(end)] ...
    , 'YTick', plot_details.yticks ...
    );

ylabel(plot_details.y_string);

box off;
%%%%%%%%plot_a_bar%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function plot_a_scatter(plot_details, Generate_params);

figure(plot_details.fig);
subplot(plot_details.subplot(1), plot_details.subplot(2), plot_details.subplot(3)); hold on;

scatter( ...
    plot_details.these_data(:,1)...
    , plot_details.these_data(:,2) ...
    , 'MarkerEdgeColor',  plot_details.color ...
    , 'MarkerFaceColor', plot_details.color ...
    );

%regression line
b = regress( ...
    plot_details.these_data(:,2) ...
    , [ones(size(plot_details.these_data,1),1) plot_details.these_data(:,1)] ...
    );
x_vals = [min(plot_details.these_data(:,1)) max(plot_details.these_data(:,1))];
y_hat = b(1) + b(2)*x_vals;


%I want regression line
plot( x_vals, y_hat ...
    , 'Color', plot_details.color ...
    );

%I want to plot diagonal
plot([x_vals x_vals],[x_vals x_vals], 'Color', [.5 .5 .5]);

set(gca ...
    , 'Fontname','Arial' ...
    , 'Fontsize',plot_details.graph_font ...
    , 'FontWeight','normal' ...
    );

if ~isempty(plot_details.ticks);
    set(gca,'XTick',plot_details.ticks,'YTick',plot_details.ticks,'xlim',[plot_details.ticks(1) plot_details.ticks(end)],'ylim',[plot_details.ticks(1) plot_details.ticks(end)]);
    % xlim([plot_details.limits(1) plot_details.limits(2)]);
    %     xlim([min(plot_details.these_data(:,1)) max(plot_details.these_data(:,1))] );
    %     ylim([min(plot_details.these_data(:,2)) max(plot_details.these_data(:,2))] );
    % else
    % xlim([plot_details.limits(1) plot_details.limits(2)]);
    % ylim([plot_details.limits(1) plot_details.limits(2)]);
end;
% ylabel(sprintf('Estimated %s', plot_details.y_string));
% xlabel(sprintf('Configured %s', plot_details.y_string));
ylabel(sprintf('Estimated %s', []));
xlabel(sprintf('Configured %s', []));

title( ...
    sprintf('%s',plot_details.model_name) ...
    , 'Fontname','Arial' ...
    , 'Fontsize',plot_details.graph_font ...
    , 'FontWeight','normal' ...
    );

%get correlation to put on plot
fprintf('');
[r pval] = corr(plot_details.these_data);
temp_y = ylim;
temp_x = xlim;
x_inc = (temp_x(2) - temp_x(1))/4;
x_mid = temp_x(1)+(temp_x(2) - temp_x(1))/2;
y_inc = (temp_y(2) - temp_y(1))/10;
text( ...
    x_mid - x_inc ...
    ,temp_y(1) + y_inc ...
    ,sprintf('r = %0.2f, p = %0.2f',r(2,1),pval(2,1)) ...
    ,'Fontname','Arial' ...
    ,'Fontsize',plot_details.graph_font ...
    );
%     min(plot_details.these_data(:,1))+x_inc ...
%%%%%%%%%%%%%plot_a_scatter%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

