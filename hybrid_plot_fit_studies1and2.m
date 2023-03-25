
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [] = hybrid_plot_io_studies1and2;

cd('C:\matlab_files\fiance\parameter_recovery\beta_fixed_code\Model_fitting_hybrid_study');

addpath(genpath('C:\matlab_files\fiance\parameter_recovery\beta_fixed_code\Model_fitting_hybrid_study\plotSpread'));

%hybrid_plot_fit_studies1and2.m Based on hybrid_plot_fit_studies1and2.m, modified to fit beta parameter
%and get log likelihood for the different "io-like" models.

%hybrid_plot_io_studies1and2.m Based on model_fitting_hybrid_study.m, this makes new figures of hybrid
%conditions, using newly debugged code that operates directly on the raw
%data. Hopefully these will be the figure-worthy ones in the end for the
%Communications Psychology revisions.

compute_data = 0;

if compute_data == 1;
    
    %set up a matrix to tell the program what to run
    %put them in order you'd like to see their plots appear
    %col1 -- 1: baseline pilot 1, 2: full pilot, 3: baseline, 4: full, 5: ratings, 6:squares, 7: timing
    %cols 2 and 3 -- 1 run continuous reward scheme for sub, obj
    %cols 4 and 5 -- 1 run 3-rank scheme for sub, obj
    run_studies = [ ...
        2 1 1 1 1; ...
        4 1 1 1 1; ...
        5 1 1 1 1; ...
        1 0 1 0 1; ...
        3 0 1 0 1; ...
        6 0 1 0 1; ...
        7 0 1 0 1; ...
        8 0 1 0 1;
        ];
    
    data.num_studies = numel(unique(run_studies(:,1)));
    data.run_studies = run_studies;
    
    for study = 1:data.num_studies;
        
        study_id = data.run_studies(study,1);
        
        data.study(study).study_id = study_id;
        
        %Now get data for every model requested
        clear models_pick
        models_pick = find(data.run_studies(study,2:end)==1);
        
        for model = 1:numel(models_pick);
            
            %identify model info based pn pick
            subjective_vals = 0;
            if models_pick(model) == 1 | models_pick(model) == 3;
                subjective_vals = 1;
            end;
            
            payoff_scheme = 0;
            if models_pick(model) == 1 | models_pick(model) == 2;
                payoff_scheme = 1;
            end;
            
            %now returns (preserving legacy variable names):
            %"mean ratings" which is actually 90*num_subs lists of phase 1 ratings
            %seq_vals, which is 6*8*num_subs lists of sequence values and
            %output, which is now 6*num_subs number of subject draws for each sequence
            [mean_ratings_all seq_vals_all output_all ranks_all] = ...
                get_sub_data( ...
                study_id, ...
                subjective_vals, ...
                payoff_scheme ...
                );
            
            %participant samples
            data.study(study).samples = output_all;
            data.study(study).ranks = ranks_all;
            
            %model metadata
            data.study(study).model(model).models_pick = models_pick(model);
            data.study(study).model(model).subjective_vals = subjective_vals;
            data.study(study).model(model).payoff_scheme = payoff_scheme;
            data.study(study).model(model).prior_vals = mean_ratings_all;
            data.study(study).model(model).seq_vals = seq_vals_all;
            
            %model samples, ranks, fitted parameter (beta) and log
            %likelihood (in fmodel-specific fields in data)
            disp('RUNNING MODEL NOW');
            data = fit_models(data);
            
        end;    %models loop
        
    end;    %loop through studies
    
    save('hybrid_plot_fit_results_v2.mat','data'); %takes a long time to run and I might want to skip to here and load data later
    
else
    
    load('hybrid_plot_fit_results_v1.mat','data'); %takes a long time to run and I might want to skip to here and load data later
    
end;    %Do I want to run all the models (can take a while or load pre-run file?

plot_results(data);


disp('audi5000')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%







%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function data = fit_models(data)

%Does model for whatever the last study and model saved in data was (so can be evoked
%in a loop through studies/models

fprintf('');

run_study = size(data.study,2); %which study to process?
run_model = size(data.study(run_study).model,2); %which model to process?


num_subs = size(data.study(run_study).model(run_model).seq_vals,3);
num_seqs = size(data.study(run_study).model(run_model).seq_vals,1);

%the only parameter is beta, with starting value equals no noise
%Starting value of beta is zero because exp(0) = 1, or no noise. We use
%exp(params) to ensure the beta enering softmax is always positive.
params = 0;

for subject = 1:num_subs;
    
    fprintf(' subject %d ,',subject);
    
    %get beta that yields best ll for this model and this subject
    [data.study(run_study).model(run_model).estimated_params(subject,:) ...
        ,  data.study(run_study).model(run_model).ll(subject,:) ...
        , exitflag, search_out] = ...
        fminsearch(  @(params) run_models( params, subject, data ), params);
    
    %get action values, samples and ranks for a model using best-fitting parameter
    data = ...
        get_model_performance( ...
        data.study(run_study).model(run_model).estimated_params(subject,:), ...
        subject, ...
        data ...
        );

end;    %loop through subjects

 fprintf('\n');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function data = get_model_performance(params,subject,data);

run_study = size(data.study,2); %which study to process?
run_model = size(data.study(run_study).model,2); %which model to process?

num_subs = size(data.study(run_study).model(run_model).seq_vals,3);
num_seqs = size(data.study(run_study).model(run_model).seq_vals,1);

clear prior;
prior.mu = mean(log(data.study(run_study).model(run_model).prior_vals(:,subject)+1));
prior.var = var(log(data.study(run_study).model(run_model).prior_vals(:,subject)+1));
if prior.var == 0;
    prior.var = eps;
end;
prior.kappa= 2;
prior.nu = 1;

b = exp(params);    %constrain b to be positive

for seq = 1:num_seqs;
    
    clear list choiceCont choiceStop difVal
    list.vals = log(data.study(run_study).model(run_model).seq_vals(seq,:,subject)+1);
    list.length = numel(list.vals);
    list.Cs = 0;
    list.payoff_scheme = data.study(run_study).model(run_model).payoff_scheme;
    list.flip = 1;
    
    [choiceStop, choiceCont, difVal] = analyzeSecretary_nick_2023a(prior,list);
    
    choiceValues = [choiceCont; choiceStop]';
    
    %I need number of draws for this subject and sequence to compare against model
    listDraws = data.study(run_study).samples(seq,subject);
    
    for drawi = 1 : size(list.vals,2);
        %cprob seqpos*choice(draw/stay)
        cprob(drawi, :) = exp(b*choiceValues(drawi, :))./sum(exp(b*choiceValues(drawi, :)));
    end;
    
     cprob(end,2) = Inf; %ensure stop choice on final sample.
     
      %ranks for this sequence
        dataList = tiedrank(list.vals);
        
        %Now get samples from uniform distribution
        test = rand(1000,size(list.vals,2));
        for iteration = 1:size(test,1);
            
            samples_this_test(iteration) = find(cprob(:,2)'>test(iteration,:),1,'first');
            ranks_this_test(iteration) = dataList( samples_this_test(iteration) );
            
        end;    %iterations

        data.study(run_study).model(run_model).samples(seq,subject) = round(mean(samples_this_test));
        data.study(run_study).model(run_model).ranks(seq,subject) = round(mean(ranks_this_test));
        data.study(run_study).model(run_model).ChoiceValues(:,:,seq,subject) = choiceValues;

end; %loop through sequences
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function ll = run_models(params, subject, data);

run_study = size(data.study,2); %which study to process?
run_model = size(data.study(run_study).model,2); %which model to process?


num_subs = size(data.study(run_study).model(run_model).seq_vals,3);
num_seqs = size(data.study(run_study).model(run_model).seq_vals,1);

clear prior;
prior.mu = mean(log(data.study(run_study).model(run_model).prior_vals(:,subject)+1));
prior.var = var(log(data.study(run_study).model(run_model).prior_vals(:,subject)+1));
if prior.var == 0;
    prior.var = eps;
end;
prior.kappa= 2;
prior.nu = 1;

b = exp(params);    %constrain b to be positive

ll=0;
for seq = 1:num_seqs;
    
    clear list choiceCont choiceStop difVal
    list.vals = log(data.study(run_study).model(run_model).seq_vals(seq,:,subject)+1);
    list.length = numel(list.vals);
    list.Cs = 0;
    list.payoff_scheme = data.study(run_study).model(run_model).payoff_scheme;
    list.flip = 1;
    
    [choiceStop, choiceCont, difVal] = analyzeSecretary_nick_2023a(prior,list);
    
    choiceValues = [choiceCont; choiceStop]';
    
    %I need number of draws for this subject and sequence to compare against model
    listDraws = data.study(run_study).samples(seq,subject);
    
    for drawi = 1 : listDraws
        %cprob seqpos*choice(draw/stay)
        cprob(drawi, :) = exp(b*choiceValues(drawi, :))./sum(exp(b*choiceValues(drawi, :)));
    end;
    
    %Compute ll
    if listDraws == 1;  %If only one draw
        ll = ll - 0 - log(cprob(listDraws, 2)); %take the log of just the option decision probability
    else
        ll = ll - sum(log(cprob((1:listDraws-1), 1))) - log(cprob(listDraws, 2));
    end;
    
    if ll == Inf | ll == -Inf;
        fprintf('');
    end;

end;    %sequence loop
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%







%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
function plot_results(data)

h1 = figure; set(gcf,'Color',[1 1 1]);  %samples
plot_colors = lines(6);

rows = ceil(data.num_studies/2);
cols = ceil(data.num_studies/rows);

for study = 1:data.num_studies;
    
    %samples
    figure(h1)
    subplot(rows,cols,study); hold on;
    
    x_axis_it = 1;
    
    %participants
    bar( x_axis_it, mean(mean(data.study(study).samples)), 'FaceAlpha',.1, 'FaceColor', plot_colors(1,:));
    plotSpread( mean(data.study(study).samples)' , 'xValues', x_axis_it, 'distributionColors',plot_colors(1,:) );
    
    %models
    num_models = size(data.study(study).model,2);
    for model = 1:num_models;
        
        x_axis_it = x_axis_it + 1;
        
        bar( x_axis_it, mean(mean(data.study(study).model(model).samples)), 'FaceAlpha',.1, 'FaceColor', plot_colors(data.study(study).model(model).models_pick+1,:));
        plotSpread( mean(data.study(study).model(model).samples)' , 'xValues', x_axis_it, 'distributionColors',plot_colors(data.study(study).model(model).models_pick+1,:) );
        
    end;    %loop through models
    
    
end;    %loop through studies
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%







%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [choiceStop, choiceCont, difVal] = analyzeSecretary_nick_2023a(dataPrior, list)
%minValue is repurposed to be 1 if continuous reward and otherwise if
%3-rank payoff

minValue = list.payoff_scheme;
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

if list.flip == -1
    sampleSeries = -(dataList - mean(dataList)) + mean(dataList);
else
    %     sampleSeries = dataList;
    sampleSeries = list.vals;
end

% N = ceil(list.length - params*list.length/12);
% if N < length(list.vals)
%     N = length(list.vals);
% end

N = list.length;

prior.mu    = dataPrior.mu;
prior.kappa = dataPrior.kappa;
%
% if distOptions == 0
prior.sig   = dataPrior.var;
prior.nu    = dataPrior.nu;
% else
%     prior.sig   = dataPrior.mean;
%     prior.nu    = 1;
% end

%%% if not using ranks
%%% Cs = params(1)*prior.mu;

Cs = list.Cs;


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
    
else
    
    %maxPayRank = 3;
    %payoff = [5 3 1 0 0 0 0 0 0 0 0 0 0 0 0 0 ];
    %I've taken the actual payments for the top three ranks .12 .08 .04,
    %mapped them from the scale of the full price range to 0 to 100, added 1 
    %and taken the log (as we did with the price options)
    payoff = log([1.0067    1.0045    1.0022]+1); 
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

maxPayRank = numel(payoff);
temp = [payoff zeros(1, 20)];
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
%rnkv = [Inf*ones(1,1); rnkvl(1:mxv); -Inf*ones(20, 1)];

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







%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [all_ratings seq_vals all_output seq_ranks] = get_sub_data(study, subjective_vals, payoff_scheme);

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



if study == 1;  %baseline pilot
    data_folder = 'pilot_baseline';
elseif study == 2;  %full pilot
    data_folder = 'pilot_full';
    ratings_phase = 1;
    header_format = 2;     option_chars = 2;
    %payoff_scheme = 1;
elseif study == 3;  %baseline
    data_folder = 'baseline';
elseif study == 4;  %full
    data_folder = 'full';
    ratings_phase = 1;
    header_format = 2;     option_chars = 1;
    %payoff_scheme = 1;
elseif study == 5;  %rating phase
    data_folder = 'rating_phase';
    ratings_phase = 1;
elseif study == 6;  %squares
    data_folder = 'squares';
elseif study == 7;  %timimg
    data_folder = 'timing';
elseif study == 8;  %payoff (stars)
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
    
    disp(sprintf('Study id %d, subjective vals %d, payoff scheme %d, participant %d',study,subjective_vals,payoff_scheme, subs(subject)));
    
    
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
        
        %get ranks
        seq_ranks_temp = tiedrank(seq_vals(sequence,:,subject)')';
        seq_ranks(sequence,subject) = seq_ranks_temp(1,all_output(sequence,subject));
        
        
    end;    %Loop through sequences
end;    %Loop through subs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



