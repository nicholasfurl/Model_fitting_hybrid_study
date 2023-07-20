
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [] = hybrid_subj_vals_model_comparison;


addpath(genpath('C:\matlab_files\fiance\parameter_recovery\beta_fixed_code\Model_fitting_hybrid_study\plotSpread'));
addpath(genpath('C:\matlab_files\fiance\parameter_recovery\beta_fixed_code\Model_fitting_hybrid_study\klabhub-bayesFactor-3d1e8a5'));


%v2 tries to combine multiple studies / datasets together

%Adapted from code from new_fit_code*.m to take outputs from examples like new_fit_code_hybrid_prior_subs_v2
%and does a big comparison of models made with subjective values versus
%models made with objective values (prices)

%For now, the models that I've run and can use here (rather than all
%possible models) are:
%SV: 1: cutoff 2: Cs 3: BV 4: BR 5: Opt 6: io
%OV: 7: cutoff 8: Cs 9: BV 10: BR 11: Opt 12: io
% do_models = [2 5 7 10];
do_models = [1:5 7:11];    %So SV and OV will be same models each, skip io
%
% %each one creates a structure for which the Generate_params structure is a field
% filename_for_SV = 'C:\matlab_files\fiance\parameter_recovery\outputs\out_new_ll1_HybridPrior2SubsLog0vals1_20223003.mat';       %output from subjective value models
% filename_for_OV = 'C:\matlab_files\fiance\parameter_recovery\outputs\out_new_ll1_HybridPrior2SubsLog0vals0_20223003.mat';    %output from objective value models

%The above ones 20223003 I corrected for some weird BR results and I think
%they're right now. As a sanity check I completely reran after fixing code
%and the results, in the files here. They reproduce the same model wins and
%sampling rates when treated individually - but here the model comparison
%produced different results ....
%

% 
% %%Sequence length 10
% data_for_SV{1} = load('C:\matlab_files\fiance\parameter_recovery\outputs\out_new_ll1_out_test_jscriptSeqLen_Pay1vals1Len1_20231906.mat');       %output from subjective value models
% data_for_OV{1} = load('C:\matlab_files\fiance\parameter_recovery\outputs\out_new_ll1_out_test_jscriptSeqLen_Pay1vals0Len1_20232006.mat');   %output from objective value models

% %%Sequence length 14
% data_for_SV{1} = load('C:\matlab_files\fiance\parameter_recovery\outputs\out_new_ll1_out_test_jscriptSeqLen_Pay1vals1Len2_20231906.mat');       %output from subjective value models
% data_for_OV{1} = load('C:\matlab_files\fiance\parameter_recovery\outputs\out_new_ll1_out_test_jscriptSeqLen_Pay1vals0Len2_20232106.mat');   %output from objective value models
% 
% %full pilot
% data_for_SV{1} = load('C:\matlab_files\fiance\parameter_recovery\outputs\out_new_ll1_out_sahira_pay1vals1study2_20232106.mat');       %output from subjective value models
% data_for_OV{1} = load('C:\matlab_files\fiance\parameter_recovery\outputs\out_new_ll1_out_sahira_pay1vals0study2_20232206.mat');   %output from objective value models


% %full
% data_for_SV{1} = load('C:\matlab_files\fiance\parameter_recovery\outputs\out_new_ll1_out_sahira_pay1vals1study4_20232206.mat');       %output from subjective value models
% data_for_OV{1} = load('C:\matlab_files\fiance\parameter_recovery\outputs\out_sahira_ll1pay1vals0study420232306.mat');   %output from objective value models
% 
% %prior / ratings
% data_for_SV{1} = load('C:\matlab_files\fiance\parameter_recovery\outputs\out_sahira_ll1pay3vals1study520232406.mat');       %output from subjective value models
% data_for_OV{1} = load('C:\matlab_files\fiance\parameter_recovery\outputs\out_sahira_ll1pay3vals0study520232506.mat');   %output from objective value models

%w/o any IO in model comparison
outpath = 'C:\matlab_files\fiance\parameter_recovery\beta_fixed_code\Model_fitting_hybrid_study\outputs';

%NEW no io
% data_for_SV{1} = load([outpath filesep 'out_NEWnoIO_ll1pay1vals020230307.mat']); %Unfortunately still needs to be typed in manually
% data_for_OV{1} = load([outpath filesep 'out_NEWnoIO_ll1pay1vals120230307.mat']); %Unfortunately still needs to be typed in manually

% %seq leng = 1 no io
% data_for_SV{1} = load([outpath filesep 'out_SeqLennoIO_ll1pay1vals020230707.mat']); %Unfortunately still needs to be typed in manually
% data_for_OV{1} = load([outpath filesep 'out_SeqLennoIO_ll1pay1vals120230707.mat']); %Unfortunately still needs to be typed in manually
% 
% %seq length = 2 no io
% data_for_SV{1} = load([outpath filesep 'out_SeqLennoIO_ll1pay1vals120230807.mat']); %Unfortunately still needs to be typed in manually
% data_for_OV{1} = load([outpath filesep 'out_SeqLennoIO_ll1pay1vals020230807.mat']); %Unfortunately still needs to be typed in manually

% %full pilot study 2 no io
% data_for_SV{1} = load([outpath filesep 'out_sahira_noIO_ll1pay1vals1study220231007.mat']); %Unfortunately still needs to be typed in manually
% data_for_OV{1} = load([outpath filesep 'out_sahira_noIO_ll1pay1vals0study220231007.mat']); %Unfortunately still needs to be typed in manually

% %full
% data_for_SV{1} = load([outpath filesep 'out_sahira_noIO_ll1pay1vals1study420231107.mat']); %Unfortunately still needs to be typed in manually
% data_for_OV{1} = load([outpath filesep 'out_sahira_noIO_ll1pay1vals0study420231107.mat']); %Unfortunately still needs to be typed in manually

%full
%data_for_SV{1} = load([outpath filesep 'out_sahira_noIO_ll1pay3vals1study520231207.mat']); %Unfortunately still needs to be typed in manually
% data_for_OV{1} = load([outpath filesep 'out_sahira_noIO_ll1pay3vals0study520231207.mat']); %Unfortunately still needs to be typed in manually

%ratings / prior study 2 no io
data_for_SV{1} = load([outpath filesep 'out_sahira_noIO_ll1pay3vals1study520231207.mat']); %Unfortunately still needs to be typed in manually
data_for_OV{1} = load([outpath filesep 'out_sahira_noIO_ll1pay3vals0study520231207.mat']); %Unfortunately still needs to be typed in manually


num_studies = size(data_for_SV,2);

%Takes list of file paths and then comcatenates the data for the models.
%Different from the below, which treats the same models from different files as
%though they were different.
New_struct_SV = combineStudies(data_for_SV);
New_struct_OV = combineStudies(data_for_OV);

%This bit makes a new structure, but now the models in each input struct
%are treated as separate models in thje output struct, even if they have
%the same names (e.g., so SV Cs and OV Cs are added as separate models)
New_struct = combineModels(New_struct_SV, New_struct_OV);

%plot_data(New_struct);

plot_data(New_struct);


fprintf('')


disp('audi5000');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%






%%%%%%%%%%%%%%%%%%%%%%
%%%%Taken from new_fit_code_hybrid_prior_subs_v2.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [] = plot_data(Generate_params);

%set up plot appearance
%For now I'll try to match model identifiers to colors. Which means this
%colormap needs to scale to the total possible number of models, not the
%number of models
plot_cmap = hsv(Generate_params.num_models+1);  %models + subjects
f_a = 0.1; %face alpha
sw = 1;  %ppoint spread width
graph_font = 12;
x_axis_test_offset = .05;   %What percentage of the y axis range should x labels be shifted below the x axis?

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%SAMPLES AND RANKS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Plot samples for participants and different models
h10 = figure;
set(gcf,'Color',[1 1 1]);

%Need to accumulate these for analyze_value_position_functions below.
all_choice_trial(:,:,1) = Generate_params.num_samples'; %Should be subs*seqs*models. num_samples here (human draws) is seqs*subs
model_strs = {};
for this_bar = 1:Generate_params.num_models+1;
    
    
    %     for perf_measure = 1:2;   %samples or ranks
    %
    %         if perf_measure == 1;   %If samples
    subplot(2,2,1); hold on; %Samples plot
    y_string = 'Samples to decision';
    
    if this_bar == 1;   %If participants
        these_data = Generate_params.num_samples;
        plot_color = [1 0 0];
        model_label = 'Participants';
    else;   %if model
        these_data = Generate_params.model(this_bar-1).num_samples_est;
        plot_color = plot_cmap(Generate_params.model(this_bar-1).identifier+1,:);
        model_label = Generate_params.model(this_bar-1).name;
    end;    %participants or model?
    
    
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
    
    %         end;    %switch between samples and ranks
    
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
[a pps_indices] = min(IC_pps_all_models');

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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

















%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function New_struct = combineModels(New_struct_SV, New_struct_OV);

New_struct.IC = 2; %1=AIC, 2=BIC (They should all be one parameter models anyway)
New_struct.num_samples = New_struct_SV.num_samples; %Both SV and OV should have the same data for participant samples, only models should differ
New_struct.seq_length = New_struct_SV.seq_length; 
New_struct.num_seqs = New_struct_SV.num_seqs; 
New_struct.model = [New_struct_SV.model New_struct_OV.model];
New_struct.num_models = size(New_struct.model,2);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%











%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function New_data_struct = combineStudies(data);

%I've just copied this straight from combine_hybrid_objective_v3 then modified it to fit new context

num_models = size(data{1}.Generate_params.model,2);

New_data_struct.seq_length = data{1}.Generate_params.seq_length;    %assumes all studies use same seq length (num options per seq)
New_data_struct.num_samples = [];
New_data_struct.num_seqs = [];

for j=1:size(data,2);   %datasets
    
    disp(sprintf('dataset 1: %d subjects', size(data{j}.Generate_params.num_samples,2)));
    
    %participants
    New_data_struct.num_samples = ...
        [New_data_struct.num_samples; ...
        nanmean(data{j}.Generate_params.num_samples)' ...
        ];
    
        %participants - I need to get every participant's sequence length
        %so I can compute BIC
    New_data_struct.num_seqs = ...
        [New_data_struct.num_seqs; ...
        sum(data{j}.Generate_params.num_samples>0)' ...
        ];

end;    %datasets

disp(sprintf('TOTAL: %d subjects', numel(New_data_struct.num_samples)));



for model=1:num_models-1; %models, minus 1 if optimal (io) is last one
    
    %copy some stuff over that's common to all studies (assumed)
    New_data_struct.model(model).identifier = data{1}.Generate_params.model(model).identifier;
    New_data_struct.model(model).name = data{1}.Generate_params.model(model).name;
    New_data_struct.model(model).this_models_free_parameters = data{1}.Generate_params.model(model).this_models_free_parameters;
    %initialise some stuff to accumulate over studies
    New_data_struct.model(model).estimated_params = [];
    New_data_struct.model(model).ll = [];
    New_data_struct.model(model).num_samples_est = [];
    
    for j=1:size(data,2);   %datasets
        
        %models
        New_data_struct.model(model).estimated_params = ...
            [New_data_struct.model(model).estimated_params; ...
            data{j}.Generate_params.model(model).estimated_params ...
            ];
        
        New_data_struct.model(model).ll = ...
            [New_data_struct.model(model).ll; ...
            data{j}.Generate_params.model(model).ll ...
            ];
        
        New_data_struct.model(model).num_samples_est = ...
            [New_data_struct.model(model).num_samples_est; ...
            nanmean(data{j}.Generate_params.model(model).num_samples_est)' ...
            ];
        
        %         Generate_params.model(i).ranks_est = ...
        %             [Generate_params.model(i).ranks_est ...
        %             data{j}.Generate_params.model(i).ranks_est ...
        %             ];
        
    end;    %datasets
    
end;    %models
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%








