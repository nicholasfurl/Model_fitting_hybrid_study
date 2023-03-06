
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [] = model_fitting_hybrid_study;

%model_fitting_hybrid_study updates
%new_fit_code_hybrid_prior_subs_v6_betafix.m so that it reads in and
%processes the Gorilla data directly, instead of using Sahira's
%intermediate *.m files. It's a bit simpler, avoids the confusion with the
%image numbers and is more appropriate for on-line archiving, as it goes
%from raw data to result. I've also removed plotting to simplify. Can use
%new_fit_code_make_plots_2022_v3 to view results.

%Makes some changes to fix small things that came out in comparison with a
%from-scratch implementation of ideal observer.

%v5: I may have ruined v4 by partially changing it to v5. V5 accommodates
%the full and pilotStudy2 studies, which have different file structures and
%so needed a lot of changes to function get_sub_data.

%v4 extrapolates code to work also for hybrid conditions baseline reward
%squares timing. v3 and previous was prior / rating condition only.

%v3: changes how sampling rates predicted from fitted parameters are
%calculated - previously I didn't use beta but now I've implemented some
%room for the degree of response noise (beta) to affect simulated sampling
%rates. I expect this will lower sampling rates to some degree.

%Not sure yet of what all the original version was capable, but I'm
%revisiting after a year and am going to try to

%Should fit individual subs rather than supersubject

%v2: I tried to introduce log_or_not functionality

% new_fit_code_big_trust: I am now modifying my sweep code to fit
% parameters to human participants data with the biggest dataset that we
% have. Big trust.

%v3_sweep isn't an improvemnt on v2 buit starts a new branch. V2 (and its
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

addpath(genpath('C:\matlab_files\fiance\parameter_recovery\beta_fixed_code\Model_fitting_hybrid_study'));
cd('C:\matlab_files\fiance\parameter_recovery\beta_fixed_code\Model_fitting_hybrid_study');

study = 2;  %1: baseline pilot, 2: full pilot, 3: baseline, 4: full, 5: ratings phase, 6: squares 7: timing
subjective_vals = 1;           %Run models using subjective values (ratings) or objective values (prices)?
check_params = 1;       %fit the same model that created the data and output estimated parameters
make_est_model_data = 1;
use_file_for_plots = 0; %use this if you want to use pre-estimated paraneters to generate simulated data (make_est_model_data == 1). References to plotting are obsolete but keeping legacy name
make_plots = 0;         %if 1, plots the results
all_draws_set = 1;          %You can toggle how the ll is computed here for all models at once if you want or go below and set different values for different models manually in structure
log_or_not = 0; %I'm changing things so all simulated data is logged at point of simulation (==1) or not
%1: cutoff 2: Cs 3: dummy (formerly IO in v2) 4: BV 5: BR 6: BPM 7: Opt 8: BPV
%(I keep model 3 as a legacy for IO because analyseSecertaryNick_2021
%looks for identifiers 4 and 5 for BV and BR and needs that for v2. Also it keeps the same color scheme as v2)
do_models = [1 2 4 5 7 ];    %These are now what v2 called model identifiers - Applies at the moment to both make_model_data and check_params;
% do_models = [1 ];    %These are now what v2 called model identifiers - Applies at the moment to both make_model_data and check_params;
if use_file_for_plots ~=1;
    comment = sprintf('out_hybridStudy%d_fromRawData_Log%dvals%d',study,log_or_not,subjective_vals);     %The filename will already fill in basic parameters so only use special info for this.
    %     comment = 'test';
end;
outpath = 'C:\matlab_files\fiance\parameter_recovery\outputs';
payoff_scheme = 1;  %1 if continuous and 3-rank otherwise. This switch is only used for simulations from estimated params, when the payoff scheme wasn't specified before the param estimation stage
filename_for_plots = 'C:\matlab_files\fiance\parameter_recovery\outputs\out_new_ll1_out_hybridFull_fromRawData_Log0vals1_20230303.mat';


%These correspond to identifiers (not configured implementations like in v2) in the v3_sweep version
model_names = {'Cut off' 'Cs' 'IO' 'BV' 'BR' 'BPM' 'Opt' 'BPV' }; %IO is a placeholder, don't implement
num_model_identifiers = size(model_names,2);
% subjects = 1:64;    %big trust sub nums
% subjects = 1;    %big trust sub nums
IC = 2; %1 if AIC, 2 if BIC
do_io = 1;  %If a 1, will add io performance as a final model field when make_est_model_data is switched to 1.



if check_params == 1;
    
    disp('Getting subject data ...');
    num_subs_found = 0;
    
    %This is different from others. Get's all the subs at once and then
    %peels off the correct one when the time comes.
    
    %now returns (preserving legacy variable names):
    %"mean ratings" which is actually 90*num_subs lists of phase 1 ratings
    %seq_vals, which is 6*8*num_subs lists of sequence values and
    %output, which is now 6*num_subs number of subject draws for each sequence
    [mean_ratings_all seq_vals_all output_all, payoff_scheme] = get_sub_data(subjective_vals,study);
    
    subjects = 1:size(mean_ratings_all,2);
    
    for subject = subjects;
        
        %         %sub_fail 1*1, female 1*1, mean_ratings 1*num_stim, seq_vals seqs*opts, output seqs*1
        %         [sub_fail female mean_ratings seq_vals output] = get_sub_data_gorilla(subject);
        
        %instead of getting data like above, peel off the relevant subject.
        
        mean_ratings = mean_ratings_all(:,subject);
        seq_vals = seq_vals_all(:,:,subject);
        output = output_all(:,subject);
        
        num_subs_found = num_subs_found + 1;
        
        %Get ranks
        clear seq_ranks ranks;
        seq_ranks = tiedrank(seq_vals')';
        for i=1:size(seq_ranks,1);
            ranks(i,1) = seq_ranks(i,output(i,1));
        end;    %loop through sequences to get rank for each
        
        if log_or_not == 1;
            Generate_params.ratings(:,num_subs_found) = log(mean_ratings');
            Generate_params.seq_vals(:,:,num_subs_found) = log(seq_vals);
        else;
            Generate_params.ratings(:,num_subs_found) = mean_ratings';
            Generate_params.seq_vals(:,:,num_subs_found) = seq_vals;
        end;
        Generate_params.num_samples(:,num_subs_found) = output;
        Generate_params.ranks(:,num_subs_found) = ranks;
        
    end;
    
    %Now that you have info on the subs, load up the main struct with all
    %the basic info you might need
    %     Generate_params.IC = IC;    %AIC (0) or BIC (1) correction?
    Generate_params.payoff_scheme = payoff_scheme;
    Generate_params.study = study;
    Generate_params.subjective_vals = subjective_vals;
    Generate_params.do_io = do_io;
    Generate_params.log_or_not = log_or_not;
    Generate_params.all_draws_set = all_draws_set;
    Generate_params.do_models_identifiers = do_models;
    Generate_params.num_subs =  size(Generate_params.seq_vals,3);
    Generate_params.num_seqs =  size(Generate_params.seq_vals,1);
    Generate_params.seq_length =  size(Generate_params.seq_vals,2);
    Generate_params.num_vals = size(Generate_params.ratings,1);
    Generate_params.rating_bounds = [1 100];    %What is min and max of rating scale? (Works for big trust anyway)
    if log_or_not == 1;
        Generate_params.rating_bounds = log(Generate_params.rating_bounds);
    end;
    Generate_params.BVrange = Generate_params.rating_bounds;
    Generate_params.nbins_reward = numel(Generate_params.rating_bounds(1):Generate_params.rating_bounds(2));  %This should effectuvely remove the binning
    Generate_params.binEdges_reward = ...
        linspace(...
        Generate_params.BVrange(1) ...
        ,Generate_params.BVrange(2)...
        ,Generate_params.nbins_reward+1 ...
        );   %organise bins by min and max
    
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
    
    %Correct the starting parameters that are in units of ratings
    if log_or_not == 1;
        model_template.BVmid = log(model_template.BVmid);
        model_template.BRmid = log(model_template.BRmid);
        %BP & BVP would be in log units too but can't take negative or
        %0th values so it's best to manually set their starting params
        %and let the estimation find the best value for the context.
        %That means fix them manually if you want them to be logged
    end;
    
    
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
        %         repmat(model_template.log_or_not,1,num_cols);   %row 13: log or not (at the moment not to be trusted)
        repmat(model_template.all_draws,1,num_cols);   %row 14: all draws
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
    %In Param_recover*.m we had distrinctions between model instantiations
    %and
    param_config_default = param_config_default(:,do_models);
    free_parameters = free_parameters(:,do_models);
    
    %Save your work into struct
    Generate_params.num_models = numel(do_models);
    Generate_params.param_config_default = param_config_default;
    Generate_params.free_parameters_matrix = free_parameters;
    Generate_params.comment = comment;
    Generate_params.outpath = outpath;
    analysis_name = sprintf(...
        'out_new_ll%d_%s_'...
        , Generate_params.all_draws_set ...
        , Generate_params.comment ...
        );
    Generate_params.analysis_name = analysis_name;
    outname = [analysis_name char(datetime('now','format','yyyyddMM')) '.mat'];
    Generate_params.outname = outname;
    
    disp( sprintf('Running %s', outname) );
    
    %Now fill in default parameters to model fields
    for model = 1:Generate_params.num_models;   %How many models are we implementing (do_models)?
        
        %         Generate_params.current_model = Generate_params.do_models_identifiers(model);  %So now model 1 will be the first model implementation in the param_config array after it has been reduced by do_models
        
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
        
        %Fill in this model's free parameters to be estimated later, if you
        %get to the parameter estimatioin this run
        Generate_params.model(model).this_models_free_parameters = find(free_parameters(:,model)==1);
        Generate_params.model(model).this_models_free_parameter_default_vals = param_config_default(find(free_parameters(:,model)==1),model)';
        
    end;    %loop through models
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%DO THE MODEL FITTING!!!!!!%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %         %How many models to do in this section?
    %     do_models = Generate_params.do_models_identifiers;
    
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
            
            %                 options = optimset('Display','iter');
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
                fminsearch(  @(params) f_fitparams( params, Generate_params ), params);
            
            %                     fminsearch(  @(params) f_fitparams( params, Generate_params ), params, options);
            
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
    
    if check_params == 0 & use_file_for_plots == 1;   %use the saved file, if analysis is not succeeding from model fitting
        
        %should create Generate_params in workspace\
        load(filename_for_plots,'Generate_params');
        Generate_params.do_io = do_io;
        %Generate_params.subjective_vals = subjective_vals;
        
        if ~isfield(Generate_params,'payoff_scheme');
            Generate_params.payoff_scheme = payoff_scheme;
        end
        
        if ~isfield(Generate_params,'study');
            Generate_params.study = study;
        end
        
    end;    %check if we're starting with simulated performance using a file with estimated params
    
    
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




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%PLOT!%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if make_plots == 1;
    
    
    %Use data already in workspace because computed above by or ?
    %Or open a file where these things were computed on previous run?
    if use_file_for_plots == 1;
        
        load(filename_for_plots,'Generate_params');
        
    end;    %Plot data structure from a file?
    
    Generate_params.IC = IC;    %AIC (0) or BIC (1) correction?
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
        %         Generate_params.PriorMean = mean(log(Generate_params.ratings(:,num_subs_found)));
        %         Generate_params.PriorVar = var(log(Generate_params.ratings(:,num_subs_found)));
        %             Generate_params.BVrange = log(Generate_params.rating_bounds);   %Used for normalising BV
        %             list.allVals = log(squeeze(Generate_params.seq_vals(sequence,:,num_subs_found)));
        %         else
        
        %             Generate_params.BVrange = Generate_params.rating_bounds;    %Used for normalising BV
        list.allVals = squeeze(Generate_params.seq_vals(sequence,:,num_subs_found));
        Generate_params.PriorMean = mean(Generate_params.ratings(:,num_subs_found));
        Generate_params.PriorVar = var(Generate_params.ratings(:,num_subs_found));
        %         end;
        
        %ranks for this sequence
        dataList = tiedrank(squeeze(Generate_params.seq_vals(sequence,:,num_subs_found))');
        
        %         list.optimize = 0;
        %         list.flip = 1;
        list.vals =  list.allVals;
        %         list.length = Generate_params.seq_length;
        
        %So will be 1 if subjective values and zero otherwise
        list.payoff_scheme = Generate_params.subjective_vals;
        
        %Do cutoff model, if needed
        if Generate_params.model(Generate_params.current_model).identifier == 1;
            
            %get seq vals to process
            this_seq_vals = list.allVals;
            %initialise all sequence positions to zero/continue (value of stopping zero)
            choiceStop = zeros(1,Generate_params.seq_length);
            %What's the cutoff?
            estimated_cutoff = round(Generate_params.model(Generate_params.current_model).cutoff);
            if estimated_cutoff < 1; estimated_cutoff = 1; end;
            if estimated_cutoff > Generate_params.seq_length; estimated_cutoff = Generate_params.seq_length; end;
            %find seq vals greater than the max in the period
            %before cutoff and give these candidates a maximal stopping value of 1
            choiceStop(1,find( this_seq_vals > max(this_seq_vals(1:estimated_cutoff)) ) ) = 1;
            %set the last position to 1, whether it's greater than
            %the best in the learning period or not
            choiceStop(1,Generate_params.seq_length) = 1;
            %find first index that is a candidate ....
            %             num_samples(sequence,this_sub) = find(choiceStop == 1,1,'first');   %assign output num samples for cut off model
            
            %Reverse 0s and 1's for ChoiceCont
            choiceCont = double(~choiceStop);
            
        else;   %Any Bayesian models
            
            [choiceStop, choiceCont, difVal]  = model_for_fitting(Generate_params,list);
            
            
            %             if Generate_params.subjective_vals == 0;
            %
            %                 %for prices subjective_val == 0 only!!!
            %                 %implements the roughly 5:3:1 ranked payoff scheme + BR.
            %                 [choiceStop, choiceCont, difVal]  = ...
            %                     analyzeSecretaryNick_2021_prices(Generate_params,list);
            %
            %             else
            %
            %                 %for ratings / subjective values subjective_val == 1 only!!
            %                 %implements payoff proportional to rating. All option values
            %                 %scaled to 100, then 100 bins used. Then normalised to 0 - 1 range.
            %                 [choiceStop, choiceCont, difVal]  = ...
            %                     analyzeSecretaryNick_2021(Generate_params,list);
            %
            %             end;    %prices or subjective vals?
            
            %                 analyzeSecretaryNick_2021_hybrid_prior_cond(Generate_params,list);
            
        end;    %Cutoff or other model?
        
        %Here is the main change associated with v3. Now we let beta
        %introduce some variability into responses.
        
        %             num_samples(sequence,this_sub) = find(difVal<0,1,'first');  %assign output num samples for Bruno model
        
        
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
        
        %...and its rank
        %         ranks(sequence,this_sub) = dataList( round( num_samples(sequence,this_sub) ) );
        %Accumulate action values too so you can compute ll outside this function if needed
        choiceStop_all(sequence, :, this_sub) = choiceStop;
        choiceCont_all(sequence, :, this_sub) = choiceCont;
        
    end;    %loop through sequences
    
    this_sub = this_sub + 1;
    
end;    %loop through subs








%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [all_ratings seq_vals all_output payoff_scheme] = get_sub_data(subjective_vals,study);

%initialise some things that may be changed below depending on study particulars.
ratings_phase = 0;
header_names1 = {'ParticipantPrivateID','Correct','ScreenNumber','TrialNumber','Option1','Option2','Option3','Option4',	'Option5',	'Option6',	'Option7',	'Option8',	'Option9',	'Option10',	'Option11',	'Option12'};
sequence_file_headers = header_names1;  %I want to keep this and the names as separate variables so even if sequence* changes, I can still access the names in header_names1
option_chars = 0;
payoff_scheme = 0;

if study == 1;  %baseline pilot
    data_folder = 'pilot_baseline';
elseif study == 2;  %full pilot
    data_folder = 'pilot_full';
    ratings_phase = 1;
    sequence_file_headers = {'ParticipantPrivateID','Correct','ScreenNumber','TrialNumber','price1a','price2a','price3a','price4a',	'price5a',	'price6a',	'price7a',	'price8a',	'price9a',	'price10a',	'price11a',	'price12a'};
    option_chars = 2;
    payoff_scheme = 1;
elseif study == 3;  %baseline
    data_folder = 'baseline';
elseif study == 4;  %full
    data_folder = 'full';
    ratings_phase = 1;
    sequence_file_headers = {'ParticipantPrivateID','Correct','ScreenNumber','TrialNumber','price1a','price2a','price3a','price4a',	'price5a',	'price6a',	'price7a',	'price8a',	'price9a',	'price10a',	'price11a',	'price12a'};
    option_chars = 1;
    payoff_scheme = 1;
elseif study == 5;  %rating phase
    data_folder = 'rating_phase';
    ratings_phase = 1;
elseif study == 6;  %squares
    data_folder = 'squares';
elseif study == 7;  %timimg
    data_folder = 'timing';
end;

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
    phase2_data.Properties.VariableNames([5:16]) = header_names1(5:16);
    
    sequence_data_concatenated = [sequence_data_concatenated; phase2_data];
    
end;    %loop through sequence files


%average the number of draws over sequences per subject (all other important variables are between subs)
% mean_draws = grpstats(sequence_data_concatenated,"ParticipantPrivateID","mean","DataVars",["ScreenNumber"]);
all_output = reshape( ...
    sequence_data_concatenated.ScreenNumber, ...
    numel(unique(sequence_data_concatenated.TrialNumber)), ...
    numel(unique(sequence_data_concatenated.ParticipantPrivateID)) ...
    );

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
subs = unique(sequence_data_concatenated.ParticipantPrivateID);
num_subs = numel(subs);
for subject = 1:num_subs
    
    disp(sprintf('Participant %d',subs(subject)));
    
    
    %Get objective values for this subject
    array_Obj = table2array(sequence_data_concatenated(sequence_data_concatenated.ParticipantPrivateID==subs(subject),5:end));
    
    %loop through and get io peformance for each sequence
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











%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function Generate_params = run_io(Generate_params);

for sub = 1:Generate_params.num_subs;
    
    disp(...
        sprintf('computing performance, ideal observer subject %d' ...
        , sub ...
        ) );
    
    for sequence = 1:Generate_params.num_seqs;
        
        clear sub_data;
        
        %         Generate_params.ratings( find(Generate_params.ratings == 0) ) = NaN;
        
        
        means = log(Generate_params.ratings(:,sub)+1);
        %         means(find(means == -Inf)) = 0;
        sigs = log(Generate_params.ratings(:,sub)+1);
        %         sigs(find(sigs == -Inf)) = 0;
        vals = log(Generate_params.seq_vals(sequence,:,sub)+1);
        %         vals(find(vals == -Inf)) = 0;
        
        prior.mu =  mean(means);
        prior.var = var(sigs);
        prior.kappa = 2;
        prior.nu = 1;
        
        list.flip = 1;
        list.vals = vals;
        list.length = size(list.vals,2);
        list.optimize = 0;
        %         params = 0; %Cs
        list.Cs = 0;
        list.payoff_scheme = Generate_params.payoff_scheme;
        
        [choiceStop, choiceCont, difVal] = model_io(prior,list);
        
        %         if Generate_params.subjective_vals == 0;
        %
        %             [choiceStop, choiceCont, difVal] = ...
        %                 analyzeSecretaryNick3_io_prices(prior,list,0,0,0);
        %
        %         else;
        %
        %             [choiceStop, choiceCont, difVal] = ...
        %                 analyzeSecretaryNick3_io_subVals(prior,list,0,0,0);
        %
        %         end;    %prices or subjective vals?
        
        samples(sequence,sub) = find(difVal<0,1,'first');
        
        %         %Input to model
        %         sub_data = struct( ...
        %             'sampleSeries',log(squeeze(Generate_params.seq_vals(sequence,:,sub))) ...
        %             ,'prior',prior ...
        %             );
        %
        %         samples(sequence,sub)  = ...
        %             cbm_IO_samplessubjectiveVals(sub_data);
        
        %rank of chosen option
        dataList = tiedrank(squeeze(Generate_params.seq_vals(sequence,:,sub))');    %ranks of sequence values
        ranks(sequence,sub) = dataList(samples(sequence,sub));
        
    end;    %sequence loop
    
end;    %sub loop

% %add new io field to output struct
% num_existing_models = size(Generate_params.model,2);
% Generate_params.model(num_existing_models+1).name = 'Optimal';
% Generate_params.model(num_existing_models+1).num_samples_est = samples;
% Generate_params.model(num_existing_models+1).ranks_est = ranks;

%add new io field to output struct
num_existing_models = Generate_params.num_models;
Generate_params.model(num_existing_models+1).name = 'Optimal';
Generate_params.model(num_existing_models+1).num_samples_est = samples;
Generate_params.model(num_existing_models+1).ranks_est = ranks;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%







