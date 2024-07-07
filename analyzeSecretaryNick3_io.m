function [choiceStop, choiceCont, difVal, currentRnk, winnings] = analyzeSecretaryNick3_io(Generate_params, list)

%Based on analyzeSecretaryNick3_io_subVals.m. Modified to take the three
%types of payoff. No longer needs to be subj or obj values because that
%will be accomplished before passing data to function.

%minValue is repurposed to be 1 if continuous reward and otherwise if 3-rank payoff
minValue = Generate_params.payoff_scheme;

%Some of this code is based on analyzeSecretaryNick_2021_prices.m
%Assign params. Hopefully everything about model is now pre-specified
%(I would update mean and variance directly from Generate_params)
prior.mu    = list.mu; %prior mean offset is zero unless biased prior model
prior.sig   = list.sig;                        %Would a biased variance model be a distinct model?
prior.kappa = list.kappa;   %prior mean update parameter
prior.nu    = list.nu;

%Cost to sample
Cs = 0;

distOptions = 0;

if list.flip == -1
    sampleSeries = -(dataList - mean(dataList)) + mean(dataList);
else
    sampleSeries = list.vals;
end

N = list.length;


[choiceStop, choiceCont, difVal, currentRnk] = computeSecretary(sampleSeries, prior, N, list, Cs, distOptions,minValue);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%







%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [choiceStop, choiceCont, difVal, currentRnk] = computeSecretary(sampleSeries, priorProb, N, list, Cs, distOptions,minValue);

sdevs = 8;
dx = 2*sdevs*sqrt(priorProb.sig)/100;
x = ((priorProb.mu - sdevs*sqrt(priorProb.sig)) + dx : dx : ...
    (priorProb.mu + sdevs*sqrt(priorProb.sig)))';

Nchoices = length(list.vals);

if list.optimize == 1
    Nconsider = length(list.allVals);
else
    Nconsider = length(sampleSeries);
    if Nconsider > N
        Nconsider = N;
    end
end

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
    payoff = [.12 .08 .04];

end;


N = listLength;
Nx = length(x);

maxPayRank = numel(payoff);
temp = [payoff zeros(1, 1000)];
payoff = temp;


data.n  = ts;

if distOptions == 0
    data.sig = var(sampleSeries(1:ts));
    data.mu = mean(sampleSeries(1:ts));
    
else
    data.mu = mean(sampleSeries(1:ts));
    data.sig = data.mu;
end

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
% rnkv = [Inf*ones(1,1); rnkvl(1:mxv); -Inf*ones(20, 1)];

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
    
    utCont = utCont - Cs;
    
    utility(:, ti)      = max([utStop utCont], [], 2);
    expectedUtility(ti) = px'*utility(:,ti);
    
    expectedStop(ti)    = px'*utStop;
    expectedCont(ti)    = px'*utCont;
   
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%









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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function prob_y = posteriorPredictive(y, postProb)

tvar = (1 + postProb.kappa)*postProb.sig/postProb.kappa;

sy = (y - postProb.mu)./sqrt(tvar);

prob_y = tpdf(sy, postProb.nu);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

