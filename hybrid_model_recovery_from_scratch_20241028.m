%%%%%%%%%%%%%Start main body%%%%%%%%%%%%%%%%%%%%%
function [] = hybrid_model_recovery_from_scratch_20241028;

addpath(genpath('C:\matlab_files\fiance\parameter_recovery\beta_fixed_code\Model_fitting_hybrid_study\FMINSEARCHBND'))
addpath(genpath('C:\matlab_files\fiance\parameter_recovery\beta_fixed_code\Model_fitting_hybrid_study\plotSpread'));
addpath(genpath('C:\matlab_files\fiance\parameter_recovery\beta_fixed_code\Model_fitting_hsim_parameter_rangeybrid_study\klabhub-bayesFactor-3d1e8a5'));

%for filename, to be save later
comment = 'earlyTest';

%Simulate stimuli
[input.simed_subs input.N] = make_sequences; %configure within function definition

%prepare parameter levels for simulated models (adapted from param recovery)
%I am trying to cover multiple studies whose estimated parameters might all be 
% %distributed somewhat differently and and I don't suspect the estimated
%parameters will be distributed normally necessarily anyway, so I'm going
%to randomly sample parameter pairs uniformly from the ranges capable of
%modulating the sampling rate for the first parameter and something wide for beta

input.models(1).name = 'CS';
input.models(1).sim_parameter_range(1,:) = [-.05,.015]; %CS
input.models(1).sim_parameter_range(2,:) = [0, 100];  %beta
input.models(1).function = @compute_behaviour_CS;
input.models(1).fit_bounds(1,:) = [-1 1];
input.models(1).fit_bounds(2,:) = [0 100];
input.models(1).fit_initial_params = [0 1];

input.models(2).name = 'CO';
input.models(2).sim_parameter_range(1,:) = [2,8];    %CO
input.models(2).sim_parameter_range(2,:) = [0, 100];  %beta
input.models(2).function = @compute_behaviour_CO;
input.models(2).fit_bounds(1,:) = [2 input.N-1];   %cut off, it's a threshold that must be inside sequence
input.models(2).fit_bounds(2,:) = [0 100];
input.models(2).fit_initial_params = [ceil(exp(-1)*input.N) 1];


input.models(3).name = 'BP';
input.models(3).sim_parameter_range(1,:) = [-90,100];   %BP
input.models(3).sim_parameter_range(2,:) = [0, 100];  %beta
input.models(3).function = @compute_behaviour_BP;
input.models(3).fit_bounds(1,:) = [-100 100];
input.models(3).fit_bounds(2,:) = [0 100];
input.models(3).fit_initial_params = [0 1];


%Create parameter pairs uniformly sampled within these ranges
num_param_levels = 1;
for model = 1:size(input.models,2);

    % Generate N uniform random samples for first parameter and beta
    temp_min = input.models(model).sim_parameter_range(1,1);
    temp_max = input.models(model).sim_parameter_range(1,2);
    param1 = temp_min + (temp_max - temp_min) * rand(num_param_levels, 1);

    temp_min = input.models(model).sim_parameter_range(2,1);
    temp_max = input.models(model).sim_parameter_range(2,2);
    param2 = temp_min + (temp_max - temp_min) * rand(num_param_levels, 1);

    % Combine into an Nx2 matrix for all combinations
    input.models(model).sim_parameter_samples = [param1, param2];

end;    %loop through models

%simulate behaviour for each sequence and fit all three models to each simulated subject
for parameter_level = 1:num_param_levels;
    for subject = 1:size(input.simed_subs,2);
        for simed_model = 1:size(input.models,2);

            clear this_models_draws;

            for sequence = 1:size(input.simed_subs(subject).sequences,1);

                %set up sequence info (changes for every simulation of model)
                input.current_subject = subject;    %needed during model fitting to identify the correct set of sequences
                input.mu = input.simed_subs(subject).mu;
                input.sig = input.simed_subs(subject).sig;
                input.sampleSeries = squeeze(input.simed_subs(subject).sequences(sequence,:));
                input.N = numel(input.sampleSeries);

                %simulate model performance
                params(1) = input.models(simed_model).sim_parameter_samples(parameter_level,1);    %CS
                params(2) = input.models(simed_model).sim_parameter_samples(parameter_level,2);    %beta
                cprob = input.models(simed_model).function(params,input);
                [val index] = max(cprob');
                this_simed_models_draws(sequence,1) = find(index == 2,1);

            end;    %sequences

            %save the draws data for this simulated model for posterity but also keep
            %this_models_draws for now to pass into the next loop to fit
            %these simulated numbers of draws to the three fitted models
            input.models(simed_model).sim_num_draws(parameter_level,subject,:) = this_simed_models_draws;

            %Now fit the three models to the numbers of draws that we just computed above
            for fitted_model = 1:size(input.models,2);

                %get the name of the model to be fitted
                input.models(simed_model).fitted_models(fitted_model).name = input.models(fitted_model).name;
                %get the handle of the model to be fitted (and save to structure for debugging purposes)
                which_model = input.models(fitted_model).function;
                input.models(simed_model).fitted_models(fitted_model).which_model = which_model;
                %get the lower bounds of the two parameters for the model to be fitted
                lower_bounds = [input.models(fitted_model).fit_bounds(1,1) input.models(fitted_model).fit_bounds(2,1)];
                input.models(simed_model).fitted_models(fitted_model).lower_bounds = lower_bounds;
                %get the upper bounds of the two parameters for the model to be fitted
                upper_bounds = [input.models(fitted_model).fit_bounds(1,2) input.models(fitted_model).fit_bounds(2,2)];
                input.models(simed_model).fitted_models(fitted_model).upper_bounds = upper_bounds;
                %get the initial params for the model to be fitted
                params = input.models(fitted_model).fit_initial_params;
                input.models(simed_model).fitted_models(fitted_model).fit_initial_params = params;

                %for debugging purposes, let's just ensure exactly what data were fitted
                input.models(simed_model).fitted_models(fitted_model).fitted_draws(parameter_level,subject,:) = this_simed_models_draws;

                disp(sprintf('fitting parameter combo %d, subject %d, simed model %s fitted model %s',parameter_level,subject,input.models(simed_model).name, input.models(fitted_model).name));

                [input.models(simed_model).fitted_models(fitted_model).estimated_params(subject,:) ...
                    , input.models(simed_model).fitted_models(fitted_model).ll(subject,:) ...
                    , exitflag, search_out] = ...
                    fminsearchbnd(  @(params) f_fitparams(params, input, which_model, this_simed_models_draws), ...
                    params,...
                    lower_bounds, ...
                    upper_bounds ...
                    );

            end;    %fitted models
        end;    %simulated models

        %let's do a save here
        save(['C:\matlab_files\fiance\parameter_recovery\beta_fixed_code\Model_fitting_hybrid_study\outputs\' 'MR_' comment '_' datestr(now, 'yyyymmddHHMM')]);

    end;    %subjects
end;    %parameter levels

disp('audi5000');
%%%%%%%%%%%%%end main body%%%%%%%%%%%%%%%%%%%%%








%%%%%%%%%%%%%Start f_fitparams%%%%%%%%%%%%%%%%%%%%%
function ll = f_fitparams(params, input, which_model, this_simed_models_draws);

%takes the parameters, the input struct from which we need the
%sequence inputs, and which_model, which is the function handle to the
%model we'll use this time

ll = 0;
for sequence = 1:numel(this_simed_models_draws);

    %prepare model for fitting
    input.sampleSeries = squeeze(input.simed_subs(input.current_subject).sequences(sequence,:));    %mu ans sigma should still be loaded into input from preceding simulation code

    %get action probabilities for this sequence
    cprob = which_model(params, input);

    if sum(sum(isnan(cprob))) > 0;
        fprintf('');
    end;

    %get simed draws for this sequence
    listDraws = this_simed_models_draws(sequence);

    %Compute ll
    if listDraws == 1;  %If only one draw
        ll = ll - 0 - log(cprob(listDraws, 2));
    else
        ll = ll - sum(log(cprob((1:listDraws-1), 1))) - log(cprob(listDraws, 2));
    end;

end;    %sequence loop
%%%%%%%%%%%%%end f_fitparams%%%%%%%%%%%%%%%%%%%%%











%%%%%%%%%%%%%Start compute_behaviour_CS%%%%%%%%%%%%%%%%%%%%%
function cprob = compute_behaviour_CS(params,input);

%input.mu: prior mean (from simulation)
%input.sig: prior variance (from simulation)
%input.N: sequence length
%input.sample_series (sequence)

%computes probabilities of take / reject for one option sequence only

%params
input.Cs = params(1);
input.beta = params(2);

sdevs = 8;
dx = 2*sdevs*sqrt(input.sig)/100;
x = ((input.mu - sdevs*sqrt(input.sig)) + dx : dx : ...
    (input.mu + sdevs*sqrt(input.sig)))';

%inialise action values for stop and continue
choiceCont = zeros(1, input.N);
choiceStop = zeros(1, input.N);

for ts = 1:input.N;  %Loop through options

    %get action values for this option using backwards induction
    [expectedStop, expectedCont] = rnkBackWardInduction(input, ts, x);

    choiceCont(ts) = expectedCont(ts);
    choiceStop(ts) = expectedStop(ts);

    %save action probabilities
    choiceValues(ts,:) = [choiceCont(ts) choiceStop(ts)];
    cprob(ts,:) = exp(input.beta*choiceValues(ts,:))./sum(exp(input.beta*choiceValues(ts,:)));

end;


%     cprob(2,end) = 1; %force stopping on last option at least

    fprintf('');
    %%%%%%%%%%%%%End compute_behaviour_CS%%%%%%%%%%%%%%%%%%%%%





%%%%%%%%%%%%%Start compute_behaviour_CO%%%%%%%%%%%%%%%%%%%%%
function cprob = compute_behaviour_CO(params,input);

estimated_cutoff = round(params(1));
if estimated_cutoff < 1; estimated_cutoff = 1; end;
if estimated_cutoff > input.N; estimated_cutoff = input.N; end;

input.beta = params(2);

%initialise all sequence positions to zero/continue (value of stopping zero)
choiceStop = zeros(1,input.N);

%find seq vals greater than the max in the period before cutoff and give these candidates a maximal stopping value of 1
choiceStop(1,find( input.sampleSeries > max(input.sampleSeries(1:estimated_cutoff)) ) ) = 1;

%set the last position to 1, whether it's greater than the best in the learning period or not
choiceStop(1,input.N) = 1;

%find first index that is a candidate ....
num_samples = find(choiceStop == 1,1,'first');   %assign output num samples for cut off model

%Reverse 0s and 1's
choiceCont = double(~choiceStop);

%save action probabilities
choiceValues = [choiceCont; choiceStop];
cprob = exp(input.beta*choiceValues)./sum(exp(input.beta*choiceValues));
cprob(2,end) = 1; %force stopping on last option at least
cprob = cprob';

%%%%%%%%%%%%%End compute_behaviour_CO%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%Start compute_behaviour_BP%%%%%%%%%%%%%%%%%%%%%
function cprob = compute_behaviour_BP(params,input);

%input.mu: prior mean (from simulation)
%input.sig: prior variance (from simulation)
%input.N: sequence length
%input.sample_series (sequence)

%computes probabilities of take / reject for one option sequence only

%params
input.mu = input.mu + params(1);
input.beta = params(2);
input.Cs = 0;


sdevs = 8;
dx = 2*sdevs*sqrt(input.sig)/100;
x = ((input.mu - sdevs*sqrt(input.sig)) + dx : dx : ...
    (input.mu + sdevs*sqrt(input.sig)))';

%inialise action values for stop and continue
choiceCont = zeros(1, input.N);
choiceStop = zeros(1, input.N);

for ts = 1:input.N;  %Loop through options

    %get action values for this option using backwards induction
    [expectedStop, expectedCont] = rnkBackWardInduction(input, ts, x);

    choiceCont(ts) = expectedCont(ts);
    choiceStop(ts) = expectedStop(ts);

    %save action probabilities
    choiceValues(ts,:) = [choiceCont(ts) choiceStop(ts)];
    cprob(ts,:) = exp(input.beta*choiceValues(ts,:))./sum(exp(input.beta*choiceValues(ts,:)));

end;
%%%%%%%%%%%%%End compute_behaviour_BP%%%%%%%%%%%%%%%%%%%%%







%%%%%%%%%%%%%start rnkBackWardInduction%%%%%%%%%%%%%%%%%%%%%
% function [expectedStop, expectedCont, expectedUtility] = rnkBackWardInduction(sampleSeries, ts, priorProb, listLength, x, Cs, Generate_params)
function [expectedStop, expectedCont, expectedUtility] = rnkBackWardInduction(input, ts, x)

N = input.N;
sampleSeries = input.sampleSeries;
priorProb.mu = input.mu;
priorProb.sig = input.sig;
priorProb.kappa = 2;
priorProb.nu = 1;

Nx = length(x);

%set payoffs for different ranks to be the option values themselves
payoff = sort(sampleSeries,'descend');   %sort the sample value
payoff = (payoff - min(payoff))/(max(payoff)-min(payoff));
maxPayRank = numel(payoff);
temp = [payoff zeros(1, 1000)];
payoff = temp;

data.n  = ts;

data.sig = var(sampleSeries(1:ts));
data.mu = mean(sampleSeries(1:ts));

utCont  = zeros(length(x), 1);
utility = zeros(length(x), N);

if ts == 0
    ts = 1;
end

[rnkvl, rnki] = sort(sampleSeries(1:ts), 'descend');
z = find(rnki == ts);
rnki = z;

% ties = 0;
% if length(unique(sampleSeries(1:ts))) < ts
%     ties = 1;
% end

mxv = ts;
if mxv > maxPayRank
    mxv = maxPayRank;
end

rnkv = [Inf*ones(1,1); rnkvl(1:mxv)'; -Inf*ones(20, 1)];

[postProb] = normInvChi(priorProb, data);

postProb.mu ...
    = postProb.mu; %...Then add constant to the posterior mean (will be zero if not optimism model)

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

        utStop = utStop(zi+1)*ones(Nx, 1);

    end

    utCont = utCont - input.Cs;

    utility(:, ti)      = max([utStop utCont], [], 2);
    expectedUtility(ti) = px'*utility(:,ti);

    expectedStop(ti)    = px'*utStop;
    expectedCont(ti)    = px'*utCont;

end
%%%%%%%%%%%%%End rnkBackwardInduction%%%%%%%%%%%%%%%%%%%%%










%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function utCont = computeContinue(utility, postProb0, x, ti)

postProb0.nu = ti-1;

utCont = zeros(length(x), 1);

expData.n   = 1;
expData.sig = 0;

for xi = 1 : length(x)

    expData.mu  = x(xi);

    postProb = normInvChi(postProb0, expData);
    spx = posteriorPredictive(x, postProb);
    spx = (spx/sum(spx));

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
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%






%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function prob_y = posteriorPredictive(y, postProb)

tvar = (1 + postProb.kappa)*postProb.sig/postProb.kappa;

sy = (y - postProb.mu)./sqrt(tvar);

prob_y = tpdf(sy, postProb.nu);
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%






%%%%%%%%%%%%%Start make_sequences%%%%%%%%%%%%%%%%%%%%%
function [simed_subs, seq_length] = make_sequences;

%returns a structure with an element per subject containing field for the
%prior mean, prior var and sequences: num_sequences*seq_length matrix of simulated sequences


num_subs = 1;   %So this will be per parameter value
num_seqs = 2;
seq_length = 12; %hybrid might've usually used 12 but imageTask always uses 8!
num_vals = 426;  %How many items in phase 1 and available as options in sequences? I've used before 426 or 90
rating_bounds = [1 100];    %What is min and max of rating scale?
rating_grand_mean = 40;     %Individual subjects' rating means will jitter around this (50 or 39.5. The latter comes from the midpoint between NEW hybrid SV ratings mean (30) and normalised price mean (49.2)
rating_mean_jitter = 5;     %How much to jitter participant ratings means on average?
rating_grand_std = 20;       %Individual subjects' rating std devs will jitter around this (5 or 18, the latter is the midpoint b/n NEW hybrid SV and OV)
rating_var_jitter = 2;     %How much to jitter participant ratings vars on average?

for sub = 1:num_subs;

    %Make the moments of the prior distribution slightly different for this subject
    this_sub_rating_mean = rating_grand_mean + normrnd( 0, rating_mean_jitter );
    this_sub_rating_std = rating_grand_std + normrnd( 0, rating_var_jitter );

    %Generate a truncated normal distribution of option values
    pd = truncate(makedist('Normal','mu',this_sub_rating_mean,'sigma',this_sub_rating_std),rating_bounds(1),rating_bounds(2));  %creates distribution object for population level of values
    phase1 = random(pd,num_vals,1); %generates values from distribution object to populate the ratings in phase 1 or total prices that could be sampled

    simed_subs(sub).mu = mean(phase1);
    simed_subs(sub).sig = var(phase1);

    simed_subs(sub).sequences = reshape(...
        phase1(1:num_seqs*seq_length,1) ...
        ,num_seqs ...
        ,seq_length ...
        );

end;    %Each subject to create stimuli
%%%%%%%%%%%%%End make_sequences%%%%%%%%%%%%%%%%%%%%%



