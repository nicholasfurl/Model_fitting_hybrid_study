
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [] = all_hybrid_studies_model_space_seqLen_v2(subjective_vals, seq_cond);

%v2 17th Jan 2024. Made some simplifications.

%This one does the theoretical model fitting for the sequence length study. See
%all_hybrid_studies_model_space_NEW.m or all_hybrid_studies_model_space.m
%for the one that does the NEW study. I'm going to fit one sequence length
%(10 or 14 options) at a time. Seq_cond is a between participants factor where
%column seq_cond = 1 for 10 options and 2 for 14 options.
%Both sequence length conditions are mixed
%together in the files. As the phase 1 rows in the datafile were not
%designated in the sequence length condition column but phase 2 rows are,
%and to facilitate using original data straight off on Javascriot,
%I think I will process all data / participants in get_sub_data and then
%sunset for only one seq_cond as specified in the code.


tic

addpath(genpath('C:\matlab_files\fiance\parameter_recovery\beta_fixed_code\Model_fitting_hybrid_study\FMINSEARCHBND'));

%input argument defaults (comment these out if you like the command line)
if ~exist("subjective_vals","var"); %if arguments have not been specified in a function call
    subjective_vals = 0;        %Run models using subjective values (ratings) or objective values (prices)?
    seq_cond = 2;               %1 means subset participants who had 10 options, 2 means 14 options
end;    %have arguments already been specified?
payoff_scheme = 1;          %1 means continuous reward, 2 means 3-rank (5:3:1), 1 means 3-rank in Sahira's monetary proportion (Implementing only 1 for now, specific to this NEW study


%What model fitting / viz steps should the programme run?
check_params = 1;       %fit the same model that created the data and output estimated parameters
make_est_model_data = 1;
use_file = 0;           %Set the above to zero and this to 1 and it'll read in a file you specify (See filename_for_plots variable below) and make plots of whatever analyses are in the Generate_params structure in that file;
do_io = 1;  %If a 1, will add io performance as a final model field when make_est_model_data is switched to 1.

%What kinds of models / fits should be run?
%1: cutoff 2: Cs 3: IO (beta only) 4: BV 5: BR 6: BPM 7: Opt 8: BPV
do_models =  [2 1 6];    %These are now what v2 called model identifiers - Applies at the moment to both make_model_data and check_params;

%File I/O
outpath = 'C:\matlab_files\fiance\parameter_recovery\beta_fixed_code\Model_fitting_hybrid_study\outputs';
if use_file ~=1;
    comment =sprintf('out_SeqLen_COCSBPM_pay%dvals%dseqcond%d',payoff_scheme,subjective_vals,seq_cond);
else
    filename_for_plots = [outpath filesep 'out_SeqLennoIO_ll1pay1vals020230707.mat']; %Unfortunately still needs to be typed in manually
end;

%These correspond to identifiers (not configured implementations like in v2) in the v3_sweep version
model_names = {'CO' 'Cs' 'IO' 'BV' 'BR' 'BPM' 'Opt' 'BPV' }; %IO is a placeholder, don't implement
num_model_identifiers = size(model_names,2);

%Now RUN it!
if check_params == 1;
    
    disp('Getting subject data ...');
    num_subs_found = 0;
    
    %now returns (preserving legacy variable names):
    %"mean ratings" which is actually 90*num_subs lists of phase 1 ratings
    %seq_vals, which is 6*8*num_subs lists of sequence values and
    %output, which is now 6*num_subs number of subject draws for each sequence
    [mean_ratings_all seq_vals_all output_all] = get_sub_data(subjective_vals, payoff_scheme, seq_cond);
    
    subjects = 1:size(mean_ratings_all,2);
    
    for subject = subjects;

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


        Generate_params.ratings(:,num_subs_found) = mean_ratings';
        Generate_params.seq_vals(:,:,num_subs_found) = seq_vals;

        Generate_params.num_samples(:,num_subs_found) = output;
        Generate_params.ranks(:,num_subs_found) = ranks;

    end;

    %Now that you have info on the subs, load up the main struct with all
    %the basic info you might need
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
    model_template.beta = 1;        %Just for parameter estimation.
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
    Generate_params.analysis_name = Generate_params.comment;;
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
    fitting_bounds.BV = [1 100]; %biased values (O), it's a threshold that must be inside sequence
    fitting_bounds.BR = [1 100]; %biased reward (O), it's a threshold that must be inside sequence
    fitting_bounds.BPM = [-100 100];  %biased prior, value can't exit the rating scale
    fitting_bounds.Opt = [-100 100];  %optimism, value can't exit rating scale
    fitting_bounds.BVar = [-100 100];  %biased variances, can't be wider than the whole rating scale
    fitting_bounds.beta = [0 100];   %A bit arbitrary 
    
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

            %Let's save as we go so results can be checked for problems
            save([Generate_params.outpath filesep Generate_params.outname], 'Generate_params');

            [Generate_params.model(model).estimated_params(sub,:) ...
                ,  Generate_params.model(model).ll(sub,:) ...
                , exitflag, search_out] = ...
                fminsearchbnd(  @(params) f_fitparams( params, Generate_params ), ...
                params,...
                [Generate_params.model(model).lower_bound Generate_params.model(model).lower_bound_beta], ...
                [Generate_params.model(model).upper_bound Generate_params.model(model).upper_bound_beta] ...
                );

        end;    %Loop through subs
    end;   %loop through models
end;    %estimate parameters of simulated data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%Generate performance from estimated parameters!%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if make_est_model_data == 1;
    
    if check_params == 0 & use_file == 1;   %use the saved file, if analysis is not succeeding from model fitting
        
        %should create Generate_params in workspace\
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
    
    %Need to limit the sequence by the "subject's" (configured simulation's) number of draws ...
    
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
    end;

end;    %seq loop
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%





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




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%now returns (preserving legacy variable names):
%"mean ratings" which is actually 90(i.e., number of rated stimuli)*num_subs lists of phase 1 ratings
%seq_vals, which is 6(number of sequences)*12(number of options)*num_subs(151) lists of sequence values and
%output, which is now 6(number of sequences)*num_subs(N=151), and holds the number of subject draws for each sequence
function [all_ratings seq_vals all_output] = get_sub_data(subjective_vals, payoff_scheme, seq_cond);

all_data10_temp = readtable('C:\matlab_files\fiance\parameter_recovery\beta_fixed_code\Model_fitting_hybrid_study\sequence_length\seq-len-study-pt-1-n70-10-v-14-options-5-seqs_oneNoncompletedSubRemoved.csv');
all_data10 = all_data10_temp(:,{'name','run_id','response','price','sequence','option','rank','num_options','seq_cond','num_seqs','reward_cond','array'});
all_data10.run_id = all_data10.run_id + 1000;  %ensure no overlapping sub numbers in different pilots

all_data11_temp = readtable('C:\matlab_files\fiance\parameter_recovery\beta_fixed_code\Model_fitting_hybrid_study\sequence_length\seq-len-study-pt-2-n70-10-v-14-options-5-seqs.csv');
all_data11 = all_data11_temp(:,{'name','run_id','response','price','sequence','option','rank','num_options','seq_cond','num_seqs','reward_cond','array'});
all_data11.run_id = all_data11.run_id + 1100;  %ensure no overlapping sub numbers in different pilots

all_data = vertcat(all_data10, all_data11);

%split off ratings data
ratings_data = all_data(strcmp(all_data.name,'rating_trial'),{'run_id' 'price' 'response'});

%split off phase 2 data where participant said stop
phase2_data = all_data(strcmp(all_data.name,'response_prompt') & all_data.response == 0,{'run_id','response','price','sequence','option','rank','num_options','seq_cond','num_seqs','reward_cond','array'});

%average the two ratingsper subject
group_vars = {'run_id', 'price'};
mean_ratings = grpstats(ratings_data, group_vars, 'mean');

%average the number of draws over sequences per subject (all other important variables are between subs)
group_vars = {'run_id'};
mean_draws = grpstats(phase2_data,"run_id","mean","DataVars",["run_id","price","option","rank","num_options","seq_cond","reward_cond"]);

%extract array data
%get one example of every array
sequenceOne = phase2_data(phase2_data.sequence==1,{'run_id','array','num_seqs','num_options','reward_cond'});

num_subs_found = 0;
for subject = 1:size(sequenceOne,1)
    
    %    disp(sprintf('Participant %d',sequenceOne.run_id(subject)));
    disp(sprintf('subjective vals %d, payoff scheme %d, participant %d',subjective_vals,payoff_scheme, subs(subject)));

    %assemble sequences for this subject
    clear array_Obj;
    array_Obj = reshape(str2double(regexp(sequenceOne.array{subject}, '-?\d+(\.\d+)?', 'match')),[sequenceOne.num_options(subject), sequenceOne.num_seqs(subject)])';
    
    %If number of options in sequence matches programme run parameter, then continue. If not, abort.
     if (seq_cond == 1 & size(array_Obj,2) == 10) |  (seq_cond == 2 & size(array_Obj,2) == 14)   
         num_subs_found = num_subs_found + 1;   %add this subject to index and continue    
     else
         continue;  %bail on this sub and move to next iteration of subject loop
     end;
         
    
    %assemble ratings data for this subject
    this_rating_data = mean_ratings(mean_ratings.run_id == sequenceOne.run_id(subject),:);
    
    
    %Assemble samples for this subject (You'll compute the ranks below in the loop)
    all_output(:,num_subs_found) = table2array(phase2_data( phase2_data.run_id == sequenceOne.run_id(subject),'option' ));
    
    %loop through and get data about each sequence
    for sequence = 1:sequenceOne.num_seqs(subject);
        
        %accumulate in a matrix to be returned by this function.
        %Note that values are sorted by raw price
        if subjective_vals == 1;  %output ratings if modelling using subjective values
            
            %First get distribution of subjective values and save to matrix to be returned by this function
            all_ratings(:,num_subs_found) = this_rating_data.mean_response;
            
            %Then get seq values transformed into subjective vals
            
            %Loop through options and replace price values with corresponding ratings for each participant get sub's ratings
            clear this_seq_Subj;
            for option=1:size(array_Obj(sequence,:),2);
                this_seq_Subj(1,option) = table2array(this_rating_data(this_rating_data.price==array_Obj(sequence,option),'mean_response'));
            end;    %loop through options
            
            %Save subjective sequence values into matrix to be returned by this function
            seq_vals(sequence,:,num_subs_found) = this_seq_Subj;
            
        else;   %output the prices if modelling using objective values
            
            %transform values
            old_min = 1;
            old_max = max(this_rating_data.price);
            new_min=1;
            new_max = 100;
            
            %normalise prices vector and accumulate over subs(should be same every subject)
            clear temp_ratings temp_seq_vals
            temp_Obj_ratings = (((new_max-new_min)*(this_rating_data.price - old_min))/(old_max-old_min))+new_min;
            temp_Obj_ratings = -(temp_Obj_ratings - 50) + 50;
            
            temp_Obj_vals = (((new_max-new_min)*(array_Obj(sequence,:) - old_min))/(old_max-old_min))+new_min;
            temp_Obj_vals = -(temp_Obj_vals - 50) + 50;
            
            %First Get distribution of objective values and save to matrix to
            %be returned by this function
            all_ratings(:,num_subs_found) = temp_Obj_ratings;
            
            %Save objective sequence values into matrix to be returned by this function
            seq_vals(sequence,:,num_subs_found) = temp_Obj_vals;
            
        end;    %subjective or objective values?
        
        %Now get the rank of the chosen option, based on the participant's
        %choice this sequence
        this_seq_ranks = tiedrank( seq_vals(sequence,:,num_subs_found) );
        seq_ranks(sequence,num_subs_found) = this_seq_ranks( all_output(sequence,num_subs_found) );
        
    end;    %Loop through sequences
    
end;    %loop through subjects
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%









%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
        params = 0; 
        
        [choiceStop, choiceCont, difVal] = ...
            analyzeSecretaryNick3_io(Generate_params, list);
        
        samples(sequence,sub) = find(difVal<0,1,'first');
        
        %rank of chosen option
        dataList = tiedrank(squeeze(Generate_params.seq_vals(sequence,:,sub))');    %ranks of sequence values
        ranks(sequence,sub) = dataList(samples(sequence,sub));
        
    end;    %sequence loop
    
end;    %sub loop

%add new io field to output struct
num_existing_models = size(Generate_params.model,2);
Generate_params.model(num_existing_models+1).name = 'Optimal';
Generate_params.model(num_existing_models+1).num_samples_est = samples;
Generate_params.model(num_existing_models+1).ranks_est = ranks;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

