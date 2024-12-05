
function [] = fiance_model_fit_ITM;

%13th Sept 2024, modified fiance_model_fit_LTM.m
%to create fiance_fit_ITM, which outputs only the independent thereshold
%model results in a way amenable to making a paper figure. Works with
%either OV or SV versions of Study 2 / NEW full condition. Hybrid Comms Psych

%C:\matlab_files\fiance\parameter_recovery\beta_fixed_code\Model_fitting_hybrid_study\fiance_model_fit_LTM.m:
%Modified the source code below to compute acceptance thresholds as
%function of sequence position for the NEW study (study 2 in the comm psych
%submission) for both human participants and winning model. This is for
%purposes od responding to reviews asking for more model validation /
%interpretation.

%C:\matlab_files\fiance\fiance_model_fit_LTM.m: Version LTM retains all the stuff from v2 but also adds the linear
%threshold and independent threshold models, the Michael Lee model from the
%Baumann et al. 2020 PNAS paper.

addpath(genpath('C:\matlab_files\fiance\parameter_recovery\beta_fixed_code\Model_fitting_hybrid_study\FMINSEARCHBND'))

IC = 1; %1 if AIC, 2 if BIC
no_params = [8 3]; %Used for IC correction
% model_strs = {'Independent threshold model' 'Linear threshold model'};

outpath = 'C:\matlab_files\fiance\parameter_recovery\beta_fixed_code\Model_fitting_hybrid_study\outputs';  
file_paths = {...
        [outpath filesep 'out_NEW_COCSBPM_pay1vals0study20243110.mat']                  %NEW, OV, payoff 1 (continuous)
    };
%  [outpath filesep 'out_sahira_COCSBPM_pay3vals0study320240303.mat']                  %Study 1 baseline
% [outpath filesep 'out_NEW_COCSBPM_pay1vals1study20240503.mat']    %NEW, SV, payoff 1 (continuous)
%    


data = load(file_paths{1});
data.Generate_params.model(4) = []; %remove optimal

num_models = size(data.Generate_params.model,2);
num_subs_found = size(data.Generate_params.seq_vals,3);

h_threshold = figure('Color',[1 1 1]);
% cmap = lines(num_models+1);
cmap = hsv(9);
% new_order = [3, 2, 7, 1, 4, 5, 6, 8, 9];
cmap = [cmap([3,2,7],:); [0 0 0]];

legendNames = {'Cost to Sample','Cut Off','Biased Prior','Human'};

%participants plus all the models (this ought to mean CS, BV, BP and for now optimal
for model = 1:num_models+1;

    for num_subs_found=1:num_subs_found;

        num_seqs = size(data.Generate_params.seq_vals,1);
        seq_length = size(data.Generate_params.seq_vals,2);

        %get this subs (or corresponding model's number of draws for each sequence
        list.draws = data.Generate_params.num_samples(:,num_subs_found); %should be extracted from num_seqs*num_subsmatrix
        if model ~= num_models + 1; %if not participants
            list.draws = data.Generate_params.model(model).num_samples_est(:,num_subs_found);  %should be extracted from num_seqs*num_subs matrix
        end;

        %get this sub's option data
        list.allVals = squeeze(data.Generate_params.seq_vals(:,:,num_subs_found));

        %initialise parameters
        clear params;
        max_option_value = 100;
        initial_beta = 1;

        clear params

        params = ones(1,seq_length-1)*(max_option_value/2);  %right in the middle for every threshold but last
        params(end+1) = initial_beta;  %set intial beta value

        lower_bounds = [ones(1,seq_length-1)*1 0];
        upper_bounds = [ones(1,seq_length-1)*max_option_value 100];

        [mparams_itm(num_subs_found,:) ll_itm(num_subs_found)] = ...
            fminsearchbnd(@(params) f_fitparams(params, list), params, lower_bounds,upper_bounds);

        % Perform linear regression (1st-degree polynomial)
        p = polyfit(1:seq_length-1, mparams_itm(num_subs_found, 1:seq_length-1), 1);

        % Extract the slope (b) from the result
        slope_b(num_subs_found,model) = p(1);

    end;    %participants

    %plot this model in the appropriate method subplot
    plot(1:seq_length-1, mean(mparams_itm(:,1:seq_length-1)), '-o', 'Color', cmap(model,:), 'MarkerSize', 4, 'Marker', 'o','LineWidth',2);
    hold on;

    %assign this model's output in case I want to use it later
    output(model).name = legendNames{model};
    output(model).itm_thresholds = mparams_itm;
    output(model).itm_ll = ll_itm;

end;    %models (CS, BV, BP, participants)


% Set the x-axis and y-axis ticks and labels
xticks(1:seq_length);
xticklabels(1:seq_length);
yticks(0:20:100);
yticklabels(0:20:100);

% Set x and y axis labels
xlabel('Sequence position');
ylabel('Threshold');

% Optional: Set axis limits if necessary
ylim([0 100]);
xlim([1 seq_length]);

legend(legendNames);

set(gca,'fontSize',12,'FontName','Arial');
box off;



%Correlations between thresholds participant by participant?
[rs ps] = corr(slope_b)



fprintf('');










%%
function ll = f_fitparams(params, list);

%B is always estimated if params is entering this function at all and is always the last one;
b = params(end);
params = params(1:end-1);   %separate beta from others

nTrials = size(list.allVals,1); %number of sequences
nSeq = size(list.allVals,2);    %sequence length

ll = 0;

%loop through sequences
for triali = 1 : nTrials;

    this_seq = list.allVals(triali,:);
    this_draw = list.draws(triali,:);

    thresholds = [params 0];

    choiceValues = [thresholds' this_seq'];    %value of continuing versus value of stopping

    %Compute choice probabilities
    for drawi = 1 : nSeq
        cprob(drawi, :) = exp(b*choiceValues(drawi, :))./sum(exp(b*choiceValues(drawi, :)));
    end

    if this_draw == 1;
        ll = ll - 0 - log(cprob(this_draw, 2));
    else
        ll = ll - sum(log(cprob((this_draw-1), 1))) - log(cprob(this_draw, 2));
    end;

    if ll == Inf;
        fprintf('');
    end;

end;


