
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [] = all_hybrid_studies_model_space_NEW(all_draws_set,subjective_vals, payoff_scheme, study);

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

%input argument defaults
all_draws_set = 1;          %You can toggle how the ll is computed here for all models at once if you want or go below and set different values for different models manually in structure
subjective_vals = 0;        %Run models using subjective values (ratings) or objective values (prices)?
payoff_scheme = 2;          %1 means continuous reward, 2 means 3-rank (5:3:1), 3 means 3-rank in Sahira's monetary proportion
study = 8;  %1: baseline pilot, 2: full pilot, 3: baseline, 4: full, 5: ratings phase, 6: squares 7: timing 8:payoff

%What model fitting / viz steps should the programme run?
check_params = 1;       %fit the same model that created the data and output estimated parameters
make_est_model_data = 1;
use_file_for_plots = 0; %Set the above to zero and this to 1 and it'll read in a file you specify (See filename_for_plots variable below) and make plots of whatever analyses are in the Generate_params structure in that file;
make_plots = 0;         %if 1, plots the results
analyze_value_positions = 0;    %Create plots with psychometric curves, their thresholds (model fits) and their correlations (nbins_psi hardwired at function call)
do_io = 1;  %If a 1, will add io performance as a final model field when make_est_model_data is switched to 1.

%What kinds of models / fits should be run?
%1: cutoff 2: Cs 3: IO (beta only) 4: BV 5: BR 6: BPM 7: Opt 8: BPV
do_models = [1 2 7 4 5];    %These are now what v2 called model identifiers - Applies at the moment to both make_model_data and check_params;
% do_models = [1 ];    %These are now what v2 called model identifiers - Applies at the moment to both make_model_data and check_params;
IC = 2; %1 if AIC, 2 if BIC
log_or_not = 0; %obsolete

%File I/O
if use_file_for_plots ~=1;
    comment = sprintf('out_sahira_noIO_ll%dpay%dvals%dstudy%d',all_draws_set,payoff_scheme,subjective_vals,study);     %The filename will already fill in basic parameters so only use special info for this.
    %     comment = 'test';
end;
outpath = 'C:\matlab_files\fiance\parameter_recovery\beta_fixed_code\Model_fitting_hybrid_study\outputs';
% filename_for_plots = [outpath filesep 'out_ll1_hybridPilotFull_Pay1studyvals0_20231306.mat']; %Unfortunately still needs to be typed in manually
filename_for_plots = [outpath filesep 'out_sahira_noIO_ll1pay1vals1study220231007.mat']; %Unfortunately still needs to be typed in manually
% filename_for_plots = [outpath filesep 'out_sahira_noIO_ll1pay1vals0study220231007.mat']; %Unfortunately still needs to be typed in manually

%These correspond to identifiers (not configured implementations like in v2) in the v3_sweep version
model_names = {'Cut off' 'Cs' 'IO' 'BV' 'BR' 'BPM' 'Opt' 'BPV' }; %IO is a placeholder, don't implement
num_model_identifiers = size(model_names,2);
% subjects = 1:64;    %big trust sub nums
% subjects = 1;    %big trust sub nums

%Now RUN it!
if check_params == 1;
    
    disp('Getting subject data ...');
    num_subs_found = 0;
    
    %This is different from others. Get's all the subs at once and then
    %peels off the correct one when the time comes.
    
    %now returns (preserving legacy variable names):
    %"mean ratings" which is actually 90*num_subs lists of phase 1 ratings
    %seq_vals, which is 6*8*num_subs lists of sequence values and
    %output, which is now 6*num_subs number of subject draws for each sequence
    [mean_ratings_all seq_vals_all output_all] = get_sub_data(subjective_vals, payoff_scheme, study);
    
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
    Generate_params.analyze_value_positions = analyze_value_positions;  %make psychometric plots if a 1
    %     Generate_params.IC = IC;    %AIC (0) or BIC (1) correction?
    Generate_params.subjective_vals = subjective_vals;
    Generate_params.payoff_scheme = payoff_scheme;
    Generate_params.do_io = do_io;
    Generate_params.log_or_not = log_or_not;
    Generate_params.all_draws_set = all_draws_set;
    Generate_params.do_models_identifiers = do_models;
    Generate_params.num_subs =  size(Generate_params.seq_vals,3);
    Generate_params.num_seqs =  size(Generate_params.seq_vals,1);
    Generate_params.seq_length =  size(Generate_params.seq_vals,2);
    Generate_params.num_vals = size(Generate_params.ratings,1);
    Generate_params.rating_bounds = [1 100];    %What is min and max of rating scale?
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
%     analysis_name = sprintf(...
%         'out_new_ll%d_%s_'...
%         , Generate_params.all_draws_set ...
%         , Generate_params.comment ...
%         );
    Generate_params.analysis_name = Generate_params.comment;
    outname = [Generate_params.analysis_name char(datetime('now','format','yyyyddMM')) '.mat'];
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
            
            
            [choiceStop, choiceCont, difVal] = analyzeSecretary_2023_modelFit(Generate_params,list);
            
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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [choiceStop, choiceCont, difVal] = analyzeSecretary_2023_modelFit(Generate_params, list)

%minValue is repurposed to be 1 if continuous reward and otherwise if
%3-rank payoff
minValue = Generate_params.payoff_scheme;

%Some of this code is based on analyzeSecretaryNick_2021_prices.m
%Assign params. Hopefully everything about model is now pre-specified
%(I would update mean and variance directly from Generate_params)
prior.mu    = Generate_params.PriorMean + Generate_params.model(Generate_params.current_model).BP; %prior mean offset is zero unless biased prior model
prior.sig   = Generate_params.PriorVar + Generate_params.model(Generate_params.current_model).BPV;                        %Would a biased variance model be a distinct model?
if prior.sig < 1; prior.sig = 1; end;   %It can happen randomly that a subject has a low variance and subtracting the bias gives a negative variance. Here, variance is set to minimal possible value.
prior.kappa = Generate_params.model(Generate_params.current_model).kappa;   %prior mean update parameter
prior.nu    = Generate_params.model(Generate_params.current_model).nu;


%Cost to sample
Cs = Generate_params.model(Generate_params.current_model).Cs;      %Will already be set to zero unless Cs model

%If running NEW study, then BVrange should always be 1 to 100, as the
%get_sub_data function should have set subjective and objective values to
%be on the same scales.
%If there are BV parameters specified, then warp the inputs
% if ~isnan(Generate_params.model(Generate_params.current_model).BVslope);
if Generate_params.model(Generate_params.current_model).identifier == 4;   %If identifier is BV
    list.allVals = ...
        (Generate_params.BVrange(2) - Generate_params.BVrange(1)) ./ ...
        (1+exp(-Generate_params.model(Generate_params.current_model).BVslope*(list.allVals-Generate_params.model(Generate_params.current_model).BVmid))); %do logistic transform
end;

% list.vals = list.allVals';



params = [];
distOptions = 0;
% if norm(list.allVals(1:length(list.vals)) - list.vals) > 0
%     dataList = [list.vals; list.allVals(length(list.vals)+1:end)];
%     if dataList(1) == 1380
%         dataList(1) = 900;
%     end
%     fprintf('list mismatch\n');
% else
%     dataList = list.allVals;
% end

% if list.flip == -1
%     sampleSeries = -(dataList - mean(dataList)) + mean(dataList);
% else
%     %     sampleSeries = dataList;
sampleSeries = list.vals;
% end

% N = ceil(list.length - params*list.length/12);
% if N < length(list.vals)
%     N = length(list.vals);
% end

% N = list.length;
N = Generate_params.seq_length;

% prior.mu    = dataPrior.mu;
% prior.kappa = dataPrior.kappa;
%
% if distOptions == 0
% prior.sig   = dataPrior.var;
% prior.nu    = dataPrior.nu;
% else
%     prior.sig   = dataPrior.mean;
%     prior.nu    = 1;
% end

%%% if not using ranks
%%% Cs = params(1)*prior.mu;
%
% Cs = list.Cs;


[choiceStop, choiceCont, difVal, currentRnk] = computeSecretary(params, sampleSeries, prior, N, list, Cs, distOptions,minValue);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [choiceStop, choiceCont, difVal, currentRnk] = computeSecretary(params, sampleSeries, priorProb, N, list, Cs, distOptions,minValue)

sdevs = 8;
dx = 2*sdevs*sqrt(priorProb.sig)/100;
x = ((priorProb.mu - sdevs*sqrt(priorProb.sig)) + dx : dx : ...
    (priorProb.mu + sdevs*sqrt(priorProb.sig)))';

Nchoices = length(list.vals);
%
% if list.optimize == 1
%     Nconsider = length(list.allVals);
% else
Nconsider = length(sampleSeries);
if Nconsider > N
    Nconsider = N;
end
% end

difVal = zeros(1, Nconsider);
choiceCont = zeros(1, Nconsider);
choiceStop = zeros(1, Nconsider);
currentRnk = zeros(1, Nconsider);

for ts = 1 : Nconsider
    
    [expectedStop, expectedCont] = rnkBackWardInduction(sampleSeries, ts, priorProb, N, x, Cs, distOptions,minValue);
    %     [expectedStop, expectedCont] = backWardInduction(sampleSeries, ts, priorProb, x, Cs);
    
    [rnkv, rnki] = sort(sampleSeries(1:ts), 'descend');
    z = find(rnki == ts);
    
    %     fprintf('sample %d rnk %d %.2f %.4f %.2f\n', ts, z, sampleSeries(ts), expectedStop(ts), expectedCont(ts));
    
    difVal(ts) = expectedCont(ts) - expectedStop(ts);
    
    choiceCont(ts) = expectedCont(ts);
    choiceStop(ts) = expectedStop(ts);
    
    currentRnk(ts) = z;
    
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [expectedStop, expectedCont, expectedUtility] = rnkBackWardInduction(sampleSeries, ts, priorProb, ...
    listLength, x, Cs, distOptions,minValue)

if minValue == 1
    
    payoff = sort(sampleSeries,'descend');
    
elseif minValue == 2
    
    payoff = [5 3 1];
    
else;
    
    %maxPayRank = 3;
    %I've taken the actual payments for the top three ranks .12 .08 .04,
    %mapped them from the scale of the full price range to 0 to 100, added 1
    %and taken the log (as we did with the price options)
    %     payoff = log([1.0067    1.0045    1.0022]+1);
    %      payoff = [5 3 1];
    payoff = [.12 .08 .04];
    
    %     payoff = [5 3 1 ];
    % % payoff = [1 0 0 0 0 0];
    
end;


N = listLength;
Nx = length(x);

% payoff = sort(sampleSeries,'descend')';
% payoff = [N:-1:1];
% payoff = (payoff-1)/(N-1);




% % % %bins
% temp = sort(sampleSeries,'descend')';
% % [dummy,payoff] = histc(temp, [minValue(1:end-1) Inf]);
% % nbins = size(minValue,2)-1;
% payoff = (payoff-1)/(nbins-1);

% %normalised rating value
% payoff = sort(sampleSeries,'descend')'; %assign actual values to payoff
% payoff = (payoff-0)/(minValue(end) - 0);    %normalise seq values between zero and 1 relative to maximally rated face


% payoff(find(payoff~=8))=0.0000000000000000000000000000000001;
% payoff(find(payoff==8))=100000000000000000000000000000;
%
% %bound values between zero and 1
% if numel(minValue) > 2;
% payoff = ((payoff-0)/((numel(minValue)-1)-0));
% end;
% payoff = payoff.^40;


%normalise payoff values between zero and 1 relative to maximally-valued option
% payoff = (payoff-log(1))/(log(101) - log(1));
%Should normalise from 1 to 100 to zero to 1
% payoff = (payoff-Generate_params.BVrange(1))/(Generate_params.BVrange(2) - Generate_params.BVrange(1));
% payoff = (payoff-Generate_params.BVrange(1))/(Generate_params.BVrange(2) - Generate_params.BVrange(1));

%normalise payoff values between custom range relative to maximally-valued option
% new_max = 100;
% new_min = 1;
% old_max = log(101);
% old_min = log(1);
% payoff = (((new_max-new_min)*(payoff - old_min))/(old_max-old_min))+new_min;

maxPayRank = numel(payoff);
temp = [payoff zeros(1, 1000)];
payoff = temp;


data.n  = ts;

% if ts > 0

if distOptions == 0
    data.sig = var(sampleSeries(1:ts));
    data.mu = mean(sampleSeries(1:ts));
    
else
    data.mu = mean(sampleSeries(1:ts));
    data.sig = data.mu;
end
% else
%     data.sig = priorProb.sig;
%     data.mu  = priorProb.mu;
% end

utCont  = zeros(length(x), 1);
utility = zeros(length(x), N);

if ts == 0
    ts = 1;
end

[rnkvl, rnki] = sort(sampleSeries(1:ts), 'descend');
z = find(rnki == ts);
rnki = z;

ties = 0;
if length(unique(sampleSeries(1:ts))) < ts
    ties = 1;
end

mxv = ts;
if mxv > maxPayRank
    mxv = maxPayRank;
end

rnkv = [Inf*ones(1,1); rnkvl(1:mxv)'; -Inf*ones(20, 1)];

[postProb] = normInvChi(priorProb, data);
px = posteriorPredictive(x, postProb);
px = px/sum(px);

Fpx = cumsum(px);
cFpx = 1 - Fpx;

for ti = N : -1 : ts
    
    if ti == N
        utCont = -Inf*ones(Nx, 1);
    elseif ti == ts
        utCont = ones(Nx, 1)*sum(px.*utility(:, ti+1));
    else
        utCont = computeContinue(utility(:, ti+1), postProb, x, ti);
    end
    
    %%%% utility when rewarded for best 3, $5, $2, $1
    utStop = NaN*ones(Nx, 1);
    
    rd = N - ti; %%% remaining draws
    id = max([(ti - ts - 1) 0]); %%% intervening draws
    td = rd + id;
    ps = zeros(Nx, maxPayRank);
    
    for rk = 0 : maxPayRank-1
        
        pf = prod(td:-1:(td-(rk-1)))/factorial(rk);
        
        ps(:, rk+1) = pf*(Fpx.^(td-rk)).*((cFpx).^rk);
        
    end
    
    %     psi(:,1) = (Fpx.^(td));
    %     psi(:,2) = td*(Fpx.^(td-1)).*(cFpx);
    %     psi(:,3) = (td*(td-1)/2)*(Fpx.^(td-2)).*(cFpx.^2);
    
    for ri = 1 : maxPayRank+1
        
        z = find(x < rnkv(ri) & x >= rnkv(ri+1));
        utStop(z) = ps(z, 1:maxPayRank)*(payoff(1+(ri-1):maxPayRank+(ri-1))');
        
    end
    
    if sum(isnan(utStop)) > 0
        fprintf('Nan in utStop');
    end
    
    if ti == ts
        [zv, zi] = min(abs(x - sampleSeries(ts)));
        if zi + 1 > length(utStop)
            %             fprintf('accessing utStop at %d value x %.2f\n', zi, x);
            zi = length(utStop) - 1;
        end
        
        %         if rnki > 3 & utStop(zi+1) > 0.0001 & ties == 0
        % %             fprintf('expectedReward %.9f\n', utStop(zi+1));
        %         end
        
        utStop = utStop(zi+1)*ones(Nx, 1);
        
    end
    
    utCont = utCont - Cs;
    
    utility(:, ti)      = max([utStop utCont], [], 2);
    expectedUtility(ti) = px'*utility(:,ti);
    
    expectedStop(ti)    = px'*utStop;
    expectedCont(ti)    = px'*utCont;
    
    %     subplot(2,1,1);
    %     plot(x, utStop, x, utCont, x, utility(:, ti));
    %
    %     subplot(2,1,2);
    %     plot(x, Fpx);
    %
    %     fprintf('');
    
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function utCont = computeContinue(utility, postProb0, x, ti)

postProb0.nu = ti-1;

utCont = zeros(length(x), 1);

% pspx = zeros(length(x), length(x));

expData.n   = 1;
expData.sig = 0;

for xi = 1 : length(x)
    
    expData.mu  = x(xi);
    
    postProb = normInvChi(postProb0, expData);
    spx = posteriorPredictive(x, postProb);
    spx = (spx/sum(spx));
    
    %     pspx(:, xi) = spx;
    
    utCont(xi) = spx'*utility;
    
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%






%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [postProb] = normInvChi(prior, data)

postProb.nu    = prior.nu + data.n;

postProb.kappa = prior.kappa + data.n;

postProb.mu    = (prior.kappa/postProb.kappa)*prior.mu + (data.n/postProb.kappa)*data.mu;

postProb.sig   = (prior.nu*prior.sig + (data.n-1)*data.sig + ...
    ((prior.kappa*data.n)/(postProb.kappa))*(data.mu - prior.mu).^2)/postProb.nu;

if data.n == 0
    postProb.sig = prior.sig;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function prob_y = posteriorPredictive(y, postProb)

tvar = (1 + postProb.kappa)*postProb.sig/postProb.kappa;

sy = (y - postProb.mu)./sqrt(tvar);

prob_y = tpdf(sy, postProb.nu);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%











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










function analyze_value_position_functions(value_data,choice_trial,plot_cmap,binEdges_psi,legend_labels,param_to_fit,two_params);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Look at proportion choice, position and value
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

param_to_fit = [0 param_to_fit];    %the first model in this function is subject so add it to this list as zeroth model so it can be indexed (e.g., in colormaps)

%value_data was log(raw_seq_subs), contains ratings data in sequences and is seq*position*sub
nbins = size(binEdges_psi,2)-1;
num_subs = size(value_data,3);
num_positions = size(value_data,2);
num_seqs = size(value_data,1);
num_models = size(choice_trial,3);
serial_r = value_data;  %only for ratings plots, nothing else

f_a = 0.1; %face alpha
sw = 0.5;  %ppoint spread width
font_size = 12;

%ok. this time, let's be less efficient but more organised. I want to bin
%things right up front before anything so there is a ratings dataset
%(value_data) and a binned dataset (value_bins)
for sub = 1:num_subs;
    binEdges = binEdges_psi(sub,:);
    [dummy,value_bins(:,:,sub)] = histc(value_data(:,:,sub), [binEdges(1:end-1) Inf]);
end;

%nan mask has zeros for view decisions and 1s for take decisions and NaNs
%when no face was seen because of elapsed decision. Can use to mask other
%arrays later ...
nan_mask = NaN(num_seqs,num_positions,num_subs,num_models);
for model=1:num_models;
    for sub=1:num_subs;
        for seq=1:num_seqs;
            nan_mask(seq,1:choice_trial(sub,seq,model),sub,model) = 0;
            nan_mask(seq,round(choice_trial(sub,seq,model)),sub,model) = 1;
        end;
    end;
end;

%now we have two seq*position*subject arrays, one ratings, one bins, now
%make supersubject seq*position versions by concatenating subjects. Effectively, each
%new subject justs adds new sequences so its a long list of sequences
value_data_sub = [];
value_bins_sub = [];
choice_trial_sub = [];

for sub=1:num_subs;
    value_data_sub = [value_data_sub; value_data(:,:,sub)];
    value_bins_sub = [value_bins_sub; value_bins(:,:,sub)]; %bins are still subject specific
    choice_trial_sub = [choice_trial_sub; squeeze(choice_trial(sub,:,:))];
end;

nan_mask_sub = [];
for model=1:num_models;
    temp = [];
    for sub=1:num_subs;
        temp = [temp; squeeze(nan_mask(:,:,sub,model))];
    end;
    nan_mask_sub(:,:,model) = temp;
end;

%yes, it's yet another subject loop. I'm being modular. This one prepares
%the average ratings * serial position data. It also computes the proportion choices *serial position data.
%It also computes proportion subject predicted * serial position data
%This one needed whether using a super subject or fitting all subjects, it's a separate analysis
model_predicted_choices = NaN(num_subs,num_positions,num_models-1); %for proportion correctly predicted subject choices
position_choices = NaN(num_subs,num_positions,num_models);   %for proportion responses serial positions
position_function = zeros(num_positions,num_subs,num_models);   %for average ratings as function of serial position
position_it = zeros(num_positions,num_subs,num_models);         %for average ratings as function of serial position
for sub=1:num_subs;
    
    this_subject_ratings = squeeze(serial_r(:,:,sub));  %only for plotting ratings by serial position
    
    for position=1:num_positions; %loop through the positions
        for model=1:num_models;
            
            sub_choices_this_position = nan_mask(:,position,sub,1);
            model_choices_this_position = nan_mask(:,position,sub,model);
            % %             %computes proportion responses for each position
            position_choices(sub,position,model) = sum( choice_trial(sub,:,model) == position )/size(choice_trial,2);
            
            %find average attractiveness of the choices in each position
            this_subject_choices = squeeze(choice_trial(sub,:,model));
            indices_into_choices = find(this_subject_choices==position);
            if ~isempty(indices_into_choices);
                for i=1:size(indices_into_choices,2);
                    position_function(position,sub,model) = position_function(position,sub,model)+this_subject_ratings(indices_into_choices(i),position);
                    position_it(position,sub,model) = position_it(position,sub,model)+1;
                end;    %loop through values for this position
            end;    %is there a valid position value?
            
        end;    %model
    end;        %position
end;            %subject

%individual subs
position_data_indiv = NaN(nbins,num_positions,num_subs,num_models);            %for value function analyses as function of serial position
ave_rating_per_bin_indiv = NaN(nbins,num_positions,num_subs,num_models);       %use this for curve fitting later

for sub=1:num_subs;
    
    for position=1:num_positions; %loop through the positions
        for model=1:num_models;
            
            this_subject_bins = value_bins(:,:,sub);
            temp2 = squeeze(nan_mask(:,:,sub,model));
            this_subject_bins(isnan(temp2(:)))=NaN;
            
            for val_bin = 1:nbins;
                
                %find bins at this position and,if any, check what are the CHOICES and RATINGS for that bin/position
                trials_with_bins_in_this_position = [];
                trials_with_bins_in_this_position = find( this_subject_bins(:,position) == val_bin );   %on which sequences did a value in this bin occur in this position?
                num_trial_with_vals = numel(trials_with_bins_in_this_position);                     %how many sequences have this value in this position?
                position_data_indiv(val_bin,position,sub,model) = sum(choice_trial(sub,trials_with_bins_in_this_position,model)==position)/ num_trial_with_vals ; %Now I need the number of CHOICES for this positon and bin
                ave_rating_per_bin_indiv(val_bin,position,sub,model) = nanmean(value_data(trials_with_bins_in_this_position,position,sub));   %need this for regression later (no sense of model here)
                
            end;    %value bin
        end;    %model
    end;        %position
end;            %subject


position_data = position_data_indiv;
ave_rating_per_bin = ave_rating_per_bin_indiv;

%loop again, this time through positions and fit averages over subjects
for model=1:num_models;
    for position=1:num_positions;
        
        %computes value slopes for each position, and model
        this_position_no_subs = nanmean(squeeze( position_data(:,position,:,model) ),2);   %returns bin values for a subject in a position
        
        y_this_position = this_position_no_subs(~isnan(this_position_no_subs));
        x_this_position = [1:nbins];
        x_this_position = x_this_position(~isnan(this_position_no_subs))';
        clear f_position;
        if numel(x_this_position)<3 | position==num_positions | sum(this_position_no_subs) == 0;    %if there are too many nans and not enough datapoints, if its that last position with the flat line or all ones, or if no response was ever made
            
            b1(position,model) = NaN;
            b2(position,model) = NaN;
        else
            %             f_position=fit(x_this_position,y_this_position,'1./(1+exp(-p1*(x-p2)))','StartPoint',[1 5],'Lower',[0 1],'Upper',[Inf 8]);
            %             temp_coef = coeffvalues(f_position);
            %             b1(position,model) = temp_coef(1);  %if only slope and mid are free
            %             b2(position,model) = temp_coef(2); %if only slope and mid are free
            if two_params == 1;
                %             Two params free
                f_position=fit(x_this_position,y_this_position,'1./(1+exp(-p1*(x-p2)))','StartPoint',[1 5],'Lower',[0 1],'Upper',[Inf 8]);
                temp_coef = coeffvalues(f_position);
                b1(position,model) = temp_coef(1);  %if only slope and mid are free
                b2(position,model) = temp_coef(2); %if only slope and mid are free
                
            else
                
                % %             %Three params free
                f_position=fit(x_this_position,y_this_position,'p1./(1+exp(-p3*(x-p4)))','StartPoint',[1 1 5],'Lower',[0 0 1],'Upper',[1 Inf 8]);
                temp_coef = coeffvalues(f_position);
                b1(position,model) = temp_coef(2);  %if only slope and mid are free
                b2(position,model) = temp_coef(3); %if only slope and mid are free
            end;
            
        end;    %check is there enough data to do a fit
        
    end;    %loop through positions
end;    %loop through models

b_ci = zeros(size(b2));
b = b2;

%%%%%%%new part: correlation position data for each model with subjects
r_graph = zeros(1,num_models);
r_ci_graph = zeros(1,num_models);
for model = 1:num_models;
    
    if model ~=1;
        
        for sub=1:num_subs;
            
            clear this_subject_data this_model_data this_subject_data_rs this_model_data_rs
            
            %extract_data
            this_subject_data = squeeze(position_data(:,:,sub,1));
            this_model_data = squeeze(position_data(:,:,sub,model));
            
            %reshape data
            this_subject_data_rs = reshape(this_subject_data,prod(size(this_subject_data)),1);
            this_model_data_rs = reshape(this_model_data,prod(size(this_model_data)),1);
            
            %correlate them
            [temp1 temp2] = corrcoef(this_subject_data,this_model_data,'rows','complete');
            r(sub,model-1) = temp1(2,1);
            p(sub,model-1) = temp2(2,1);
            sub_nums(sub,model-1) = sub;
            mod_nums(sub,model-1) = model-1;
            
        end;    %loop through subs
        
    end;    %only consider models other than subjects
    
end;    %models

r_graph = [0 nanmean(r,1)];
r_ci_graph = [0 1.96*(nanstd(r)/sqrt(size(r,1)))];


%average proportion responses, over subjects
mean_position_choices = squeeze(mean(position_choices,1));
ci_position_choices = squeeze(1.96*(std(position_choices,1,1)/sqrt(size(position_choices,1))));
%average ratings as function of serial position
clear ave_ratings ave ci;
ave_ratings = position_function./position_it;

%serial postion plots: average rating, proportion correct and value sensitivity slopes
h3 = figure; set(gcf,'Color',[1 1 1]);  %For serial position/PSE plots/correlation plots
h4 = figure; set(gcf,'Color',[1 1 1]);  %For psychometric function plots
for model = 1:size(choice_trial,3);
    
    markersize = 3;
    
    %average rating as function of serial positon
    legend_locs = [0.5:-0.05:(0.5 - (0.05*5))];
    
    %proportion choices
    figure(h3); subplot( 2,2,3); hold on;
    sph = shadedErrorBar(1:size(mean_position_choices,1),mean_position_choices(:,model),ci_position_choices(:,model),{'MarkerFaceColor',plot_cmap(param_to_fit(model)+1,:),'MarkerEdgeColor',plot_cmap(param_to_fit(model)+1,:),'Marker','o','MarkerSize',markersize,'LineStyle','-'},1); hold on;
    set(sph.mainLine,'Color',plot_cmap(param_to_fit(model)+1,:));
    set(sph.patch,'FaceColor',plot_cmap(param_to_fit(model)+1,:));
    set(sph.edge(1),'Color',plot_cmap(param_to_fit(model)+1,:));
    set(sph.edge(2),'Color',plot_cmap(param_to_fit(model)+1,:));
    %         text(3,legend_locs(model),legend_names{model},'Color',plot_cmap(param_to_fit(model)+1,:),'FontSize',12,'FontName','Arial');
    box off;
    %axis square;
    set(gca,'FontSize',12,'FontName','Arial','xtick',[1:size(b,1)],'ytick',[0.1:0.1:0.8],'Ylim',[0 0.5],'Xlim',[1 size(b,1)],'LineWidth',2);
    xlabel('Position in Sequence'); ylabel('Proportion Choices');
    
    %psychometric function parameters
    figure(h3); subplot( 2,1,1 ); hold on;
    errorbar(1:size(b,1),b(:,model),b_ci(:,model),'Color',plot_cmap(param_to_fit(model)+1,:),'MarkerFaceColor',plot_cmap(param_to_fit(model)+1,:),'MarkerEdgeColor',plot_cmap(param_to_fit(model)+1,:),'Marker','o','MarkerSize',markersize,'LineStyle','-','LineWidth',1); hold on;
    box off;
    set(gca,'FontSize',12,'FontName','Arial','xtick',[1:size(b,1)],'Xlim',[1 size(b,1)],'LineWidth',2);
    xlabel('Position in Sequence'); ylabel('Point of Subjective Equality');
    
    if model ~=1;   %no subjects
        %model correlations with proportion choice
        figure(h3); subplot( 2,2,4 ); hold on;
        legend_positions = [1.1:-.05:0];
        
        handles = plotSpread(r(:,model-1), ...
            'xValues',model,'distributionColors',plot_cmap(param_to_fit(model)+1,:),'distributionMarkers','.', 'spreadWidth', sw);
        
        bar(model,r_graph(model), ...
            'FaceColor',plot_cmap(param_to_fit(model)+1,:),'FaceAlpha',f_a,'EdgeColor',[0 0 0] );
        
        %         text(1.5,legend_positions(model),legend_labels{param_to_fit(model)}, 'Color',plot_cmap(param_to_fit(model)+1,:), 'FontSize',12,'FontName','Arial');
        text(1.5,legend_positions(model),legend_labels{model-1}, 'Color',plot_cmap(param_to_fit(model)+1,:), 'FontSize',12,'FontName','Arial');
        
        set(gca,'FontSize',12,'FontName','Arial', 'xticklabel',{[]},'LineWidth',2);
        ylabel('Model-participant Correlation');
        ylim([0 1.0]);
        xlim([1 numel(r_graph)+0.5]);
        
    end;    %If not a subject
    
    %value psychometric functions (in a different figure with different colormap), with lines for each position
    figure(h4); subplot(1,num_models,model);
    pm_line_colors = cool(size(position_data,2)+1);
    
    for position_line = 1:size(position_data,2)-1;
        
        if numel(size(position_data))==4;
            h = plot( nanmean(squeeze(position_data(:,position_line,:,model)),2) ); hold on;
        else
            h = plot( position_data(:,position_line,model) ); hold on;
        end;
        axis square;
        set(h,'Marker','o','MarkerSize',6,'MarkerEdgeColor',pm_line_colors(position_line,:),'MarkerFaceColor',pm_line_colors(position_line,:),'Color',pm_line_colors(position_line,:),'LineStyle','-','LineWidth',2);
        set(gca,'FontSize',12,'FontName','Arial','xtick',[1:size(position_data,1)],'xlim',[0.5 size(position_data,1)+0.5],'ylim',[0 1.1],'ytick',[0:0.2:1],'LineWidth',2);
        xlabel('Attractiveness Bin'); ylabel('Proportion Choices'); box off;
        
    end;    %position lines
    
    if model == num_models;
        legend('Position 1','Position 2','Position 3','Position 4','Position 5','Position 6','Position 7','Position 8','Position 9','Position 10','Position 11','Position 12');
    end;
    
end;    %loop through models








%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [] = plot_data(Generate_params);

%set up plot appearance
%For now I'll try to match model identifiers to colors. Which means this
%colormap needs to scale to the total possible number of models, not the
%number of models
plot_cmap = hsv(8+1);  %models + subjects
f_a = 0.1; %face alpha
sw = 1;  %ppoint spread width
graph_font = 12;
x_axis_test_offset = .05;   %What percentage of the y axis range should x labels be shifted below the x axis?

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%SAMPLES AND RANKS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Plot samples for participants and different models
h10 = figure('NumberTitle', 'off', 'Name',['parameters ' Generate_params.outname]);
set(gcf,'Color',[1 1 1]);

%Need to accumulate these for analyze_value_position_functions below.
all_choice_trial(:,:,1) = Generate_params.num_samples'; %Should be subs*seqs*models. num_samples here (human draws) is seqs*subs
model_strs = {};
for this_bar = 1:Generate_params.num_models+1;   %+1 is participants
    
    for perf_measure = 1:2;   %samples or ranks
        
        if perf_measure == 1;   %If samples
            subplot(2,2,1); hold on; %Samples plot
            y_string = 'Samples to decision';
            
            if this_bar == 1;   %If participants
                these_data = nanmean(Generate_params.num_samples)';
                plot_color = [1 0 0];
                model_label = 'Participants';
            else;   %if model
                these_data = nanmean(Generate_params.model(this_bar-1).num_samples_est)';
                plot_color = plot_cmap(Generate_params.model(this_bar-1).identifier+1,:);
                model_label = Generate_params.model(this_bar-1).name;
            end;    %partricipants or model?
            
        else;   %If ranks
            subplot(2,2,2); hold on; %Samples plot
            y_string = 'Rank of chosen option';
            if this_bar == 1;   %If participants
                these_data = nanmean(Generate_params.ranks)';
                plot_color = [1 0 0];
                model_label = 'Participants';
            else;   %if model
                these_data = nanmean(Generate_params.model(this_bar-1).ranks_est)';
                plot_color = plot_cmap(Generate_params.model(this_bar-1).identifier+1,:);
                model_label = Generate_params.model(this_bar-1).name;
            end;    %partricipants or model?
            
        end;    %samples or ranks?
        
        %average over sequences (rows) but keep sub data (cols) for scatter points
        handles = plotSpread(these_data ...
            ,'xValues',this_bar ...
            ,'distributionColors',plot_color ...
            ,'distributionMarkers','.' ...
            , 'spreadWidth', sw ...
            );
        
        bar(this_bar,nanmean(these_data) ...
            ,'FaceColor',plot_color ...
            ,'FaceAlpha',f_a ...
            ,'EdgeColor',[0 0 0] ...
            );
        
        set(gca ...
            ,'XTick',[] ...
            ,'fontSize',graph_font ...
            ,'FontName','Arial',...
            'XLim',[0 Generate_params.num_models+2] ...
            ,'YLim',[0 Generate_params.seq_length]);
        ylabel(y_string);
        
        this_offset = -x_axis_test_offset*diff(ylim);
        text( this_bar, this_offset ...
            ,sprintf('%s',model_label) ...
            ,'Fontname','Arial' ...
            ,'Fontsize',graph_font ...
            ,'Rotation',25 ...
            ,'HorizontalAlignment','right' ...
            );
        
    end;    %switch between samples and ranks
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%LOG LIKELIHOOD ANALYSIS
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %Still inside model (this_bar) loop
    if this_bar ~=1;
        
        %%%%%%%%%%%%%%%%%%%%
        %%%%Plot of raw ll's
        subplot(2,3,4);
        
        handles = plotSpread(Generate_params.model(this_bar-1).ll ...
            ,'xValues',this_bar ...
            ,'distributionColors',plot_color ...
            ,'distributionMarkers','.' ...
            , 'spreadWidth', sw ...
            );
        
        bar(this_bar,nanmean(Generate_params.model(this_bar-1).ll) ...
            ,'FaceColor',plot_color ...
            ,'FaceAlpha',f_a ...
            ,'EdgeColor',[0 0 0] ...
            );
        
        set(gca ...
            ,'XTick',[] ...
            ,'fontSize',graph_font ...
            ,'FontName','Arial',...
            'XLim',[1 Generate_params.num_models+2] ...
            );
        %                     ,'YLim',[0 Generate_params.seq_length]...0
        ylabel('Log-likelihood');
        
        this_offset = -x_axis_test_offset*diff(ylim);
        text( this_bar, this_offset ...
            ,sprintf('%s',model_label) ...
            ,'Fontname','Arial' ...
            ,'Fontsize',graph_font ...
            ,'Rotation',25 ...
            ,'HorizontalAlignment','right' ...
            );
        
        %%%%%%%%%%%%%%%%%%%%%%%%
        %Plot of AIC/BIC (Not so relevant if they all have two parameters though)
        subplot(2,3,5);
        
        %Model IC
        no_params = numel( Generate_params.model(this_bar-1).this_models_free_parameters ) + 1; %+1 for beta
        lla = Generate_params.model(this_bar-1).ll;
        if Generate_params.IC == 1; %If AIC (per participant)
            IC_pps = 2*no_params + 2*lla;
            %             IC_sum = nansum(IC_pps);
            %             IC_sum = 2*no_params(param_to_fit(model)) + 2*nansum(lla);
            a_label = 'AIC';
            %             IC_ylims = [800 1350];
        elseif Generate_params.IC == 2; %If BIC (per participant)
            IC_pps = no_params*log(Generate_params.num_seqs) + 2*lla;
            %             IC_sum = nansum(IC_pps);
            %             IC_sum = no_params(param_to_fit(model))*log(numel(lla)*28) + 2*nansum(lla);
            a_label = 'BIC';
            %             IC_ylims = [750 1250];
        end;
        
        handles = plotSpread(IC_pps ...
            , 'xValues',this_bar...
            ,'distributionColors',plot_color ...
            ,'distributionMarkers','.' ...
            , 'spreadWidth', sw ...
            );
        
        bar(this_bar,nanmean(IC_pps), ...
            'FaceColor',plot_color,'FaceAlpha',f_a,'EdgeColor',[0 0 0] );
        
        set(gca ...
            ,'XTick',[] ...
            ,'fontSize',graph_font ...
            ,'FontName','Arial',...
            'XLim',[1 Generate_params.num_models+2] ...
            );
        ylabel(a_label);
        
        this_offset = -x_axis_test_offset*diff(ylim);
        text( this_bar, this_offset ...
            ,sprintf('%s',model_label) ...
            ,'Fontname','Arial' ...
            ,'Fontsize',graph_font ...
            ,'Rotation',25 ...
            ,'HorizontalAlignment','right' ...
            );
        
        %We need to accumulate these data over models in this loop to do the
        %next step more easily
        IC_pps_all_models(:,this_bar-1) = IC_pps';
        %These we need to accumulate so they can be passed into
        %analyze_value_position_functions below
        all_choice_trial(:,:,this_bar) = Generate_params.model(this_bar-1).num_samples_est';
        model_strs{this_bar-1} = Generate_params.model(this_bar-1).name;
        
    end;    %If not participants (this bar ~=1)
    
end;    %loop through models

%the model loop is done but we are still working on the AIC/BIC plot and we
%need to add the sig tests using the accumulated model data
if Generate_params.num_models ~= 1;
    
    %run and plot ttests on IC averages
    pairs = nchoosek(1:Generate_params.num_models,2);
    num_pairs = size(pairs,1);
    [a In] = sort(diff(pairs')','descend');  %lengths of connecting lines
    line_pair_order = pairs(In,:);    %move longest connections to top
    
    %Where to put top line?
    y_inc = 2;
    ystart = max(max(IC_pps_all_models)) + y_inc*num_pairs;
    line_y_values = ystart:-y_inc:0;
    
    for pair = 1:num_pairs;
        
        %run ttest this pair
        [h IC_pp_pvals(pair) ci stats] = ttest(IC_pps_all_models(:,line_pair_order(pair,1)), IC_pps_all_models(:,line_pair_order(pair,2)));
        
        %plot result
        %             subplot(2,4,6); hold on;
        set(gca,'Ylim',[0 ystart]);
        
        if IC_pp_pvals(pair) < 0.05/size(pairs,1);  %multiple comparison corrected
            
            plot([line_pair_order(pair,1)+1 line_pair_order(pair,2)+1],...
                [line_y_values(pair) line_y_values(pair)],'LineWidth',2,'Color',[0 0 0]);
            
        end;    %Do line on plot?;
        
    end;    %loop through ttest pairs
    
end;    %Only compute ttests if there is at least one pair of models

%%%%%%%%%%%%%%%%%%%%%%
%Plot of numbers of winning subs for each model

subplot(2,3,6); hold on; box off;

%winning models
[a, pps_indices] = min(IC_pps_all_models');

for model = 1:Generate_params.num_models;
    
    bar(model,numel(find(pps_indices==model)), ...
        'FaceColor', plot_cmap(Generate_params.model(model).identifier+1,:),'FaceAlpha',f_a,'EdgeColor',[0 0 0] );
    
    set(gca ...
        ,'XTick',[] ...
        ,'fontSize',graph_font ...
        ,'FontName','Arial',...
        'XLim',[0 Generate_params.num_models+1] ...
        );
    ylabel('Frequency');
    
    this_offset = -x_axis_test_offset*diff(ylim);
    text( model, this_offset ...
        ,sprintf('%s',Generate_params.model(model).name) ...
        ,'Fontname','Arial' ...
        ,'Fontsize',graph_font ...
        ,'Rotation',25 ...
        ,'HorizontalAlignment','right' ...
        );
    
end;    %models


%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%INDIVIDUAL PARTICIPANT DATA SCATTERPPLOTS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Scatterplots of participant-model subject by subject relationships (sampling)
h11 = figure('NumberTitle', 'off', 'Name',['parameters ' Generate_params.outname]);
set(gcf,'Color',[1 1 1]);

%For scatterplot subplots
num_rows = floor(sqrt(numel(Generate_params.do_models_identifiers )) );
num_cols = ceil(numel(Generate_params.do_models_identifiers)/num_rows);

for identifier = 1:numel(Generate_params.do_models_identifiers);
    
    subplot(num_rows, num_cols, identifier); hold on; box off;
    
    %Put a diagonal slope = 1 on the plot against which points can be compared
    plot([0 8],[0 8],'Color',[0 0 0]);
    
    scatter( ...
        nanmean(Generate_params.num_samples)' ...
        , nanmean(Generate_params.model(identifier).num_samples_est)' ...
        , 'MarkerEdgeColor', plot_cmap(Generate_params.do_models_identifiers(identifier)+1,:) ...
        , 'MarkerFaceColor', plot_cmap(Generate_params.do_models_identifiers(identifier)+1,:) ...
        );
    
    %regression line
    b = regress( ...
        nanmean(Generate_params.model(identifier).num_samples_est)' ...
        , [ones(Generate_params.num_subs,1) nanmean(Generate_params.num_samples)'] ...
        );
    x_vals = [min(nanmean(Generate_params.num_samples)') max(nanmean(Generate_params.num_samples)')];
    y_hat = b(1) + b(2)*x_vals;
    
    plot( x_vals, y_hat ...
        , 'Color', plot_cmap(Generate_params.do_models_identifiers(identifier)+1,:) ...
        );
    
    set(gca ...
        , 'Fontname','Arial' ...
        , 'Fontsize',graph_font ...
        , 'FontWeight','normal' ...
        );
    
    ylim([0 Generate_params.seq_length]);
    xlim([0 Generate_params.seq_length]);
    ylabel('Predicted sampling');
    xlabel('Participant sampling');
    
    title( ...
        sprintf('%s',Generate_params.model(identifier ).name) ...
        , 'Fontname','Arial' ...
        , 'Fontsize',graph_font ...
        , 'FontWeight','normal' ...
        );
    
end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%Analysis of thresholds!!
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%     if Generate_params.analyze_value_positions == 1;
%             analyze_value_position_functions(raw_seqs_subs,all_choice_trial,plot_cmap,binEdges_psi,model_strs,param_to_fit,two_params);
nbins_psi = 4;
binEdges_psi(1:Generate_params.num_subs,:) = ...
    repmat(...
    linspace(...
    Generate_params.rating_bounds(1) ...
    ,Generate_params.rating_bounds(2) ...
    ,nbins_psi+1 ...
    ), ...
    numel(1:Generate_params.num_subs),1 ...
    );

analyze_value_position_functions(...
    Generate_params.seq_vals ...
    ,all_choice_trial ...
    ,plot_cmap ...
    ,binEdges_psi ...
    ,model_strs ...
    ,Generate_params.do_models_identifiers ...
    ,1 ...
    );

%     end;    %make threshold by serial position plot








function Generate_params = run_io(Generate_params);

for sub = 1:Generate_params.num_subs;
    
    disp(...
        sprintf('computing performance, ideal observer subject %d' ...
        , sub ...
        ) );
    
    for sequence = 1:Generate_params.num_seqs;
        
        clear sub_data;
        
        %         Generate_params.ratings( find(Generate_params.ratings == 0) ) = NaN;
        
        
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
        
        %Looksd like one participants (22) in full pilot rated every face
        %a 1 in phase 1 and so we end up with everything zeros in
        %subjective value model. 
        
        if list.mu == 0;
            
            samples(sequence,sub) = NaN;
            ranks(sequence,sub) = NaN;
            
        else
            
            
        [choiceStop, choiceCont, difVal] = ...
                analyzeSecretaryNick3_io(Generate_params, list);
        
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
        
        end;
        
    end;    %sequence loop
    
end;    %sub loop

%add new io field to output struct
num_existing_models = size(Generate_params.model,2);
Generate_params.model(num_existing_models+1).name = 'Optimal';
Generate_params.model(num_existing_models+1).num_samples_est = samples;
Generate_params.model(num_existing_models+1).ranks_est = ranks;

fprintf('');

