function hybrid_full_matlab_replication_fromScratch

%Hybrid full attempts to use more or less same code for pilot 3 to
%reproduce original results from Gorilla hybrid full study. Will model
%results still be too high?

%get ratings dataset (Note matlab deletes spaces from csv header names during read)
ratings_data_temp = readtable('C:\matlab_files\fiance\parameter_recovery\beta_fixed_code\Model_fitting_hybrid_study\full\rating_data_exp_27661-v5_task-od16.csv');
ratings_data = ratings_data_temp(strcmp(ratings_data_temp.ZoneType,'response_slider_endValue'),{'ParticipantPrivateID','Response','phone_price'});

%get phase 2 dataset
phase2_temp = readtable('C:\matlab_files\fiance\parameter_recovery\beta_fixed_code\Model_fitting_hybrid_study\full\data_exp_27661-v5_task-hlnk.csv');
phase2_data = phase2_temp(phase2_temp.Correct==1,{'ParticipantPrivateID','Correct','ScreenNumber','TrialNumber','price1a','price2a','price3a','price4a',	'price5a',	'price6a',	'price7a',	'price8a',	'price9a',	'price10a',	'price11a',	'price12a'});


%convert weird html strings in price column to proper numbers
ratings_data.phone_price = cell2mat(cellfun(@(x) str2double(x(6:end-5)), ratings_data.phone_price , 'UniformOutput', false));

%average the two ratings per subject
group_vars = {'ParticipantPrivateID', 'phone_price'};
mean_ratings = grpstats(ratings_data, group_vars, 'mean');

%average the number of draws over sequences per subject (all other important variables are between subs)
mean_draws = grpstats(phase2_data,"ParticipantPrivateID","mean","DataVars",["ScreenNumber"]);

%extract array data

%reformat strings with £ signs in cells to be doubles
phase2_data.price1a = cell2mat(cellfun(@(x) str2double(x(2:end)), phase2_data.price1a , 'UniformOutput', false));
phase2_data.price2a = cell2mat(cellfun(@(x) str2double(x(2:end)), phase2_data.price2a , 'UniformOutput', false));
phase2_data.price3a = cell2mat(cellfun(@(x) str2double(x(2:end)), phase2_data.price3a , 'UniformOutput', false));
phase2_data.price4a = cell2mat(cellfun(@(x) str2double(x(2:end)), phase2_data.price4a , 'UniformOutput', false));
phase2_data.price5a = cell2mat(cellfun(@(x) str2double(x(2:end)), phase2_data.price5a , 'UniformOutput', false));
phase2_data.price6a = cell2mat(cellfun(@(x) str2double(x(2:end)), phase2_data.price6a , 'UniformOutput', false));
phase2_data.price7a = cell2mat(cellfun(@(x) str2double(x(2:end)), phase2_data.price7a , 'UniformOutput', false));
phase2_data.price8a = cell2mat(cellfun(@(x) str2double(x(2:end)), phase2_data.price8a , 'UniformOutput', false));
phase2_data.price9a = cell2mat(cellfun(@(x) str2double(x(2:end)), phase2_data.price9a , 'UniformOutput', false));
phase2_data.price10a = cell2mat(cellfun(@(x) str2double(x(2:end)), phase2_data.price10a , 'UniformOutput', false));
phase2_data.price11a = cell2mat(cellfun(@(x) str2double(x(2:end)), phase2_data.price11a , 'UniformOutput', false));
phase2_data.price12a = cell2mat(cellfun(@(x) str2double(x(2:end)), phase2_data.price12a , 'UniformOutput', false));

%Time to loop through and process subs and sequences with models
seq_accumulator = [];
subs = unique(phase2_data.ParticipantPrivateID);
num_subs = numel(subs);
for subject = 1:num_subs
    
    disp(sprintf('Participant %d',subs(subject)));
    

    
    
    %Get objective values for this sequence
    array_Obj = table2array(phase2_data(phase2_data.ParticipantPrivateID==subs(subject),5:end));
    %array_Obj{subject} = reshape(str2double(regexp(sequenceOne.array{subject}, '-?\d+(\.\d+)?', 'match')),[sequenceOne.num_options(subject), sequenceOne.num_seqs(subject)])';
    
    %loop through and get io peformance for each sequence
    for sequence = 1:size(array_Obj,1);
        
        %Loop through options and replace price values with corresponding ratings for each participant
        clear this_rating_data this_seq_Subj;
        this_rating_data = mean_ratings(mean_ratings.ParticipantPrivateID == subs(subject),:);
        for option=1:size(array_Obj,2);
            
            this_seq_Subj(1,option) = table2array(this_rating_data(this_rating_data.phone_price==array_Obj(sequence,option),'mean_Response'));
            
        end;    %loop through options
        
        %Now input this sequence (Obj and Subj versions) to io
        
        %subjective values first
        clear prior list
        prior.mu = mean(log(this_rating_data.mean_Response+1));
        prior.var = var(log(this_rating_data.mean_Response+1));
%         prior.mu = mean(this_rating_data.mean_Response);
%         prior.var = var(this_rating_data.mean_Response);
        prior.kappa= 2;
        prior.nu = 1;
        
        list.vals = log(this_seq_Subj+1);
%         list.vals = this_seq_Subj;

        list.length = numel(list.vals);
        list.Cs = 0;
        list.payoff_scheme = 1;
        list.flip = 1;
        
        disp('Subjective')
        disp(this_seq_Subj)
        
        [choiceCont, choiceStop, difVal] = analyzeSecretary_nick_2023a(prior,list);
        
        num_samples_Subj(sequence,subject) = find(difVal<0,1,'first');
        
        seq_accumulator(sequence,:,subject) = this_seq_Subj;
        
        if subs(subject) == 3128692;
            fprintf('');
        end;
        
        
        %Now objective values
        
        %         %transform values
        old_min = 1;
        old_max = max(this_rating_data.phone_price);
        new_min=1;
        new_max = 100;
        
        %normalise prices vector and accumulate over subs(should be same every subject)
        clear temp_ratings temp_seq_vals
        temp_Obj_ratings = (((new_max-new_min)*(this_rating_data.phone_price - old_min))/(old_max-old_min))+new_min;
        temp_Obj_ratings = -(temp_Obj_ratings - 50) + 50;
        
        
        temp_Obj_vals = (((new_max-new_min)*(array_Obj(sequence,:) - old_min))/(old_max-old_min))+new_min;
        temp_Obj_vals = -(temp_Obj_vals - 50) + 50;
        
        clear prior list
        prior.mu = mean(log(temp_Obj_ratings+1));
        prior.var = var(log(temp_Obj_ratings+1));
        prior.kappa= 2;
        prior.nu = 1;
        
        list.vals = log(temp_Obj_vals+1);
        list.length = numel(list.vals);
        list.Cs = 0;
        list.payoff_scheme = 1;
        list.flip = 1;
        
        disp('Objective')
        disp(temp_Obj_vals)
        
        [choiceCont, choiceStop, difVal] = analyzeSecretary_nick_2023a(prior,list);
        
        num_samples_Obj(sequence,subject) = find(difVal<0,1,'first');
        
    end;    %Loop through sequences
end;    %Loop through subs



%Make plots

%take a look at behaviour
figure;

% seq_lengths = unique(mean_draws.mean_ScreenNumber);
plot_colors = lines(3);
x_axis_it = 1;

%participants
bar( x_axis_it, mean(mean_draws.mean_ScreenNumber), 'FaceAlpha',.1, 'FaceColor', plot_colors(1,:));
plotSpread( mean_draws.mean_ScreenNumber , 'xValues', x_axis_it, 'distributionColors',plot_colors(1,:) );

x_axis_it = x_axis_it + 1;

%subjective values
bar( x_axis_it, mean(mean(num_samples_Subj)), 'FaceAlpha',.1, 'FaceColor', plot_colors(2,:));
plotSpread( mean(num_samples_Subj)' , 'xValues', x_axis_it, 'distributionColors',plot_colors(2,:) );

x_axis_it = x_axis_it + 1;


%objective values
bar( x_axis_it, mean(mean(num_samples_Obj)), 'FaceAlpha',.1, 'FaceColor', plot_colors(3,:));
plotSpread( mean(num_samples_Obj)' , 'xValues', x_axis_it, 'distributionColors',plot_colors(3,:) );

x_axis_it = x_axis_it + 1;


fprintf('')




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

%%% First time probLarger goes under 0.5, subject should stop
% probLarger = forwardInduction(sampleSeries, prior);

% figure;
% [expectedStop, expectedContinue] = rnkBackWardInduction(sampleSeries, 0, prior);

[choiceStop, choiceCont, difVal, currentRnk] = computeSecretary(params, sampleSeries, prior, N, list, Cs, distOptions,minValue);
%
% if list.optimize == 1
%     z = find(difVal < 0);
%     [~, rnki] = sort(sampleSeries, 'descend');
%     rnkValue = find(rnki == z(1));
%
%     winnings = (rnkValue == 1)*5 + (rnkValue == 2)*2 + (rnkValue == 3)*1;
% else
%     winnings = 0;
%     rnkValue = -1*ones(length(list.vals), 1);
% end

% when difVal goes negative, subject should stop
% subplot(2,2,2);
% plot(1:length(choiceStop), choiceStop, 1:length(choiceCont), choiceCont);
% text(length(list.vals), 0, '*');
%
% subplot(2,2,1);
% plot(dataList);


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
function [expectedStop, expectedCont, expectedUtility] = backWardInduction(sampleSeries, ts, priorProb, x, Cs)

N = length(sampleSeries);

data.n  = ts;

% if ts > 0
data.sig = var(sampleSeries(1:ts));
data.mu = mean(sampleSeries(1:ts));
% else
%     data.sig = priorProb.sig;
%     data.mu  = priorProb.mu;
% end

utStop  = x;

utCont = zeros(N, 1);
utility = zeros(length(x), N);

if ts == 0
    ts = 1;
end

for ti = N : -1 : ts
    
    expData = data;
    expData.n = ti;
    
    [postProb] = normInvChi(priorProb, expData);
    
    px = posteriorPredictive(x, postProb);
    
    px = px/sum(px);
    
    if ti == N
        utCont(ti) = -Inf;
    else
        utCont(ti) = sum(px.*utility(:, ti+1)) - Cs;
    end
    
    if ti == ts
        utility(:, ti)   = ones(size(utStop, 1), 1)*max([sampleSeries(ts) utCont(ti)]);
        expectedStop(ti) = sampleSeries(ts);
    else
        utility(:, ti)   = max([utStop ones(size(utStop, 1), 1)*utCont(ti)], [], 2);
        expectedStop(ti) = sum(px.*utStop);
    end
    
    expectedUtility(ti) = sum(px.*utility(:,ti));
    expectedCont(ti)    = utCont(ti);
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [expectedStop, expectedCont, expectedUtility] = rnkBackWardInduction(sampleSeries, ts, priorProb, ...
    listLength, x, Cs, distOptions,minValue)

if minValue == 1
    
    payoff = sort(sampleSeries,'descend');
    
else
    
    %maxPayRank = 3;
    payoff = [5 3 1 0 0 0 0 0 0 0 0 0 0 0 0 0 ];
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

% subplot(2,2,1);
% plot(x, pspx(:, 1:100:end));
%
% subplot(2,2,2);
% plot(x, utility);
% title(ti);
%
% subplot(2,2,3);
% plot(x, utCont);
% axis([min(x) max(x) 0.5 3]);
%
% fprintf('');


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
function prob_y = posteriorPredictive(y, postProb)

tvar = (1 + postProb.kappa)*postProb.sig/postProb.kappa;

sy = (y - postProb.mu)./sqrt(tvar);

prob_y = tpdf(sy, postProb.nu);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [z, fz] = integrateSampling(x, postProb)

sdevs = 8;

maxx = postProb.mu + sdevs*sqrt(postProb.sig);
minx = postProb.mu - sdevs*sqrt(postProb.sig);

dx = (maxx - minx)/10000;

xv = ((minx + dx) : dx : maxx)';

pv = posteriorPredictive(xv, postProb);

pv = pv/sum(pv);

[v, vi] = min(abs(x - xv));

z = sum(pv((vi+1):end));

fz = sum(xv((vi+1):end).*pv(((vi+1):end)))/z;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function probLarger = forwardInduction(sampleSeries, priorProb)

N = length(sampleSeries);

for t = 1 : N
    
    utStop = sampleSeries(t);
    
    data.mu = mean(sampleSeries(1:t));
    data.n  = t;
    
    if t > 1
        data.sig = var(sampleSeries(1:t));
    else
        data.sig = 0;
    end
    
    [postProb] = normInvChi(priorProb, data);
    
    probLarger(t) = 1 - (1 - integrateSampling(sampleSeries(t), postProb)).^(N-t);
    
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function extraCode()

% %%% Need to see if this is a candidate...
% [~, rnki] = sort(sampleSeries(1:ts), 'descend');
% rnk = rnki(end);
%
%
%
% if rnk > 3
%     expectedReward = 0;
% else
%
%     [pgreater, ~] = integrateSampling(sampleSeries(ts), postProb);
%     pless = 1 - pgreater;
%
%     rd = N - ts;
%
%     ps = zeros(length(payoff), 1);
%
%     ps(1+(rnk-1)) = pless^(rd);
%     ps(2+(rnk-1)) = rd*(pless^(rd-1))*(pgreater);
%     ps(3+(rnk-1)) = (rd*(rd-1)/2)*(pless^(rd-2))*(pgreater^2);
%
%     expectedReward = payoff*ps;
%
% end














