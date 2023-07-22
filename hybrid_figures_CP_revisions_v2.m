
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [] = hybrid_figures_CP_revisions_v2;

%v2 reorganises things so that participants is treated like a model. It's a
%big structural change so I've saved as v2 to preserve old structure. I
%lazily did it halfway through change so v1 no longer works but at least I
%can see what old form looked like if needed.

%adapted from hybrid_subj_vals_model_comparison_NEW.

addpath(genpath('C:\matlab_files\fiance\parameter_recovery\beta_fixed_code\Model_fitting_hybrid_study\plotSpread'));
addpath(genpath('C:\matlab_files\fiance\parameter_recovery\beta_fixed_code\Model_fitting_hybrid_study\klabhub-bayesFactor-3d1e8a5'));

%Order of bars. These are the indices in the original structs placed in the
%order you want them in the new structs
%1 CO, 2 Cs, 3 Opt, 4 BV, 5 BR, 6 optimal
%Orders will be duplicated across objective and subjective values, if both exist
%Numbers mean where I want a certain model index to appear in new array. So
%putting a 6 first mean the first model in the input struct will be optimal
%zero means participants.
bar_order = [6 0 2 1 3 4 5];

%4 is NEW by itself
figure_num = 5;

% h = figure('Color',[1 1 1]);

%For now, the models that I've run and can use here (rather than all
%possible models) are:
%SV: 1: cutoff 2: Cs 3: BV 4: BR 5: Opt 6: io
%OV: 7: cutoff 8: Cs 9: BV 10: BR 11: Opt 12: io
% do_models = [2 5 7 10];
% do_models = [1:5 7:11];    %So SV and OV will be same models each, skip io


%w/o any IO in model comparison
outpath = 'C:\matlab_files\fiance\parameter_recovery\beta_fixed_code\Model_fitting_hybrid_study\outputs';
if figure_num == 1;
    
    % %full pilot study 2 no io
    data{1}{1} = load([outpath filesep 'out_sahira_noIO_ll1pay1vals1study220231007.mat']); %Unfortunately still needs to be typed in manually
    data{1}{2} = load([outpath filesep 'out_sahira_noIO_ll1pay1vals0study220231007.mat']); %Unfortunately still needs to be typed in manually
    
    
elseif figure_num == 2;
    
    % %full
    data{1}{1} = load([outpath filesep 'out_sahira_noIO_ll1pay1vals1study420231107.mat']); %Unfortunately still needs to be typed in manually
    data{1}{2} = load([outpath filesep 'out_sahira_noIO_ll1pay1vals0study420231107.mat']); %Unfortunately still needs to be typed in manually
    
    
    %ratings / prior study 2 no io
    data{2}{1} = load([outpath filesep 'out_sahira_noIO_ll1pay3vals1study520231207.mat']); %Unfortunately still needs to be typed in manually
    data{2}{2} = load([outpath filesep 'out_sahira_noIO_ll1pay3vals0study520231207.mat']); %Unfortunately still needs to be typed in manually
    
    
elseif figure_num == 3;
    
    %pilot baseline, baseline, squares, timing and payoff
    data{1}{1} = load([outpath filesep 'out_sahira_noIO_ll1pay3vals0study120231707.mat'] );
    data{2}{1} = load( [outpath filesep 'out_sahira_noIO_ll1pay3vals0study320231707.mat']);
    data{3}{1} = load( [outpath filesep 'out_sahira_noIO_ll1pay3vals0study620231807.mat']);
    data{4}{1} = load( [outpath filesep 'out_sahira_noIO_ll1pay3vals0study720231807.mat']);
    data{5}{1} = load([outpath filesep 'out_sahira_noIO_ll1pay2vals0study820231807.mat'] );
    
elseif figure_num == 4;
    
    
    %NEW no io
    temp = load([outpath filesep 'out_NEWnoIO_ll1pay1vals020230307.mat']);
    data{1}{1} =  temp;
    
    temp = load([outpath filesep 'out_NEWnoIO_ll1pay1vals120230307.mat']); %Unfortunately still needs to be typed in manually
    data{1}{2} = temp;
    
elseif figure_num == 5;
    
    
    % %seq leng = 1 no io
    temp = load([outpath filesep 'out_SeqLennoIO_ll1pay1vals020230707.mat']); 
    data{1}{1} = temp;
    temp = load([outpath filesep 'out_SeqLennoIO_ll1pay1vals120230707.mat']);
    data{1}{2} = temp;
    
    %seq length = 2 no io
    temp = load([outpath filesep 'out_SeqLennoIO_ll1pay1vals120230807.mat']);
    data{2}{1} =  temp;
    temp = load([outpath filesep 'out_SeqLennoIO_ll1pay1vals020230807.mat']);
    data{2}{2} = temp;
    
end;


%Some figures involve more than one study in each figure (right now,
%setting each study as a column)
num_studies = size(data,2);

for study = 1:num_studies;
    
    %SV & OV? Or just OV?
    num_datasets = size(data{study},2);
    
    for dataset = 1:num_datasets;
        
        %Takes list of file paths and then concatenates the data for the models.
        %Different from combineModels, which treats the same models from different files as
        %though they were different.
        combined_data{dataset} = reformatData(data{study}{dataset});
        
    end;
    
    %This bit makes a new structure, but now the models in each input struct
    %are treated as separate models in the output struct, even if they have
    %the same names (e.g., so SV Cs and OV Cs are added as separate models)
    New_struct.study(study) = combineModels(combined_data, bar_order);
    
end;    %loop through studies

%update it with handy info that is not study-specific (top-level info)
New_struct.IC = 2; %1=AIC, 2=BIC (They should all be one parameter models anyway)
New_struct.figure = figure_num;
New_struct.bar_order = bar_order;
%New_struct.num_panels = num_panels;
New_struct.num_studies = num_studies;

%Make this figure
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
% num_models_per_study = numel(Generate_params.bar_order);
num_models_per_study = numel(Generate_params.study(1).model);
BayesThresh = 3;

%io first, participants next, theoertical fitted models next
temp = hsv(10);
plot_cmap = [0 0 0; .75 .75 .75; temp(1:end,:)];  %supports up to eight model identifiers (not models per se) plus optimal (identifier 0) plus participants. Probably will use seven identifiers plus optimal plus participants
% plot_cmap = hsv(num_models_per_study+1);  %models + subjects
f_a = 0.1; %face alpha
sw = 1;  %ppoint spread width
graph_font = 9;
x_axis_test_offset = .05;   %What percentage of the y axis range should x labels be shifted below the x axis?
x_rot = 65;
num_panels = 3; %This means samples, parameters, BIC, frequencies

%Should allow easier subplot panel rearrangement by controling these
%studies row-wise
% rows = Generate_params.num_studies;  %subplots
% cols = Generate_params.num_panels; %subplots
% subplot_num = "panel_num+((study-1)*cols)" ;    %Will need to use eval, as study and panel_num don't exist yet
%studies colwise
panel_nums = [1 2 3];   %fixes the subplot locations of the samples, BIC and model win freq (indices 1,2,3 into panel_nums)
rows = num_panels;  %subplots
cols = Generate_params.num_studies; %subplots
subplot_num = "((panel_num-1)*cols)+study" ;    %Will need to use eval, as study and panel_num don't exist yet

%Samples, BIC & frequency
h10 = figure('Color',[1 1 1]);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%SAMPLES
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for study = 1:Generate_params.num_studies;
    
    %Need to accumulate these for analyze_value_position_functions below.
    %     all_choice_trial(:,:,1) = Generate_params.study(1).num_samples'; %Should be subs*seqs*models. num_samples here (human draws) is seqs*subs
    %     model_strs = {};
    
    clear IC_pps_all_models samples_accum;
    human_is_index = [];    %Used for plotting significance connector lines on samples plot, because I need to know which bar is humans for significance comparison
    it_model_comparison = 1;
    
    for this_bar = 1:num_models_per_study;
        
        %subplot(2,2,1); hold on; %Samples plot
        panel_num = panel_nums(1);  %Panels nums can be changed colwise versus rowwise in subplot
        subplot(rows,cols,eval(subplot_num)); hold on;
        
        y_string = 'Samples to decision';
        
        
        %         identifier = 0;             %What's the numeric identifier for this model (used mainly for colouring bars by model identity)
        
        %         if this_bar == 1;   %If participants
        %             these_data = Generate_params.study(study).num_samples;
        %             plot_color = [1 0 0];
        %             model_label = 'Participants';
        %
        %         else;   %if model
        
        these_data = Generate_params.study(study).model(this_bar).num_samples_est;
        %For some reason optimal doesn't come with an identifier
        %             if ~isempty(Generate_params.study(study).model(this_bar).identifier)
        identifier = Generate_params.study(study).model(this_bar).identifier;
        %             end;
        plot_color = plot_cmap(identifier,:); %+2 skips the first color for participants and the second color for optimal
        model_label = Generate_params.study(study).model(this_bar).name;
        %         end;    %participants or model?
        
        %keep data for use later on when plotting significance connector lines
        if Generate_params.study(study).model(this_bar).identifier == 1;    %if participants
            human_is_index = this_bar;
        end;
        samples_accum(:,this_bar) = these_data;
        
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
            'XLim',[0 num_models_per_study+2] ...
            ,'YTick',[0:2:Generate_params.study(study).seq_length]);
        ylabel(y_string);
        
        this_offset = -x_axis_test_offset*diff(ylim);
        text( this_bar, this_offset ...
            ,sprintf('%s',model_label) ...
            ,'Color', [0 0 0] ...
            ,'Fontname','Arial' ...
            ,'Fontsize',graph_font ...
            ,'Rotation',x_rot ...
            ,'HorizontalAlignment','right' ...
            );
        
        %         end;    %switch between samples and ranks
        
        
        
        if Generate_params.study(study).model(this_bar).skip == 0; %If a theoretical model and not participants or IO
            
            %%%%%%%%%%%%%%%%%%%%%%%%
            %Plot of AIC/BIC (Not so relevant if they all have two parameters though)
            %             subplot(2,3,5);
            %subplot(Generate_params.num_panels,Generate_params.num_studies,subplot_indices(3,study));
            panel_num = panel_nums(2);  %Panels nums can be changed colwise versus rowwise in subplot
            subplot(rows,cols,eval(subplot_num));
            
            %Model IC
            no_params = numel( Generate_params.study(study).model(this_bar).this_models_free_parameters ) + 1; %+1 for beta
            lla = Generate_params.study(study).model(this_bar).ll;
            if Generate_params.IC == 1; %If AIC (per participant)
                IC_pps = 2*no_params + 2*lla;
                a_label = 'AIC';
            elseif Generate_params.IC == 2; %If BIC (per participant)
                IC_pps = no_params*log(Generate_params.study(study).num_seqs) + 2*lla;
                a_label = 'Bayesian information criterion';
            end;
            
            handles = plotSpread(IC_pps ...
                , 'xValues',it_model_comparison...
                ,'distributionColors',plot_color ...
                ,'distributionMarkers','.' ...
                , 'spreadWidth', sw ...
                );
            
            bar(it_model_comparison,nanmean(IC_pps), ...
                'FaceColor',plot_color,'FaceAlpha',f_a,'EdgeColor',[0 0 0] );
            
            set(gca ...
                ,'XTick',[] ...
                ,'fontSize',graph_font ...
                ,'FontName','Arial',...
                'XLim',[0 num_models_per_study-(human_is_index-1)] ...
                );
            ylabel(a_label);
            
            %I moved this to after the significance lines have been added,
            %where the y lim has been finalized
%             this_offset = -x_axis_test_offset*5*diff(ylim);
%             text( it_model_comparison, this_offset ...
%                 ,sprintf('%s',model_label) ...
%                 ,'Fontname','Arial' ...
%                 ,'Fontsize',graph_font ...
%                 ,'Rotation',x_rot ...
%                 ,'HorizontalAlignment','right' ...
%                 );
            
            %Some housekeeping
            IC_pps_all_models(:,it_model_comparison) = IC_pps'; %We need to accumulate these data over models in this loop to do the next step more easily
            it_model_comparison = it_model_comparison + 1;
            
            %These we need to accumulate so they can be passed into
            %analyze_value_position_functions below
            %             all_choice_trial(:,:,this_bar) = Generate_params.study(study).model(this_bar-1).num_samples_est';
            %             model_strs{this_bar-1} = Generate_params.study(study).model(this_bar-1).name;
            
        end;    %If a theoretical model (i.e., skip == 0)
        
    end;    %loop through models
    
    
    
    
    
    
    
    %%%%%%%%%This is the finish of the loops that add bars to the samples
    %%%%%%%%%and BIC plots. During this loop, we would have accumulated
    %%%%%%%%%data from those bars into matrixs that, below, we will use in
    %%%%%%%%%new loops to add pairwise test bars to plots and make model
    %%%%%%%%%"win" frequency plots
    
    
    %significance lines
    if num_models_per_study ~= 1;
        
        %%%%%%%%%%%%%%%%%%
        %Samples significance lines
        
        %Connecting just
        %%%participants with all models,
        %%%but no need for models with each other
        
        %return to samples subplot
        panel_num = panel_nums(1);  %Panels nums can be changed colwise versus rowwise in subplot
        subplot(rows,cols,eval(subplot_num));
        
        %get the pairs connecting participants to models and set up their y axis locations
        num_pairs = size(samples_accum,2);  %This means the number of pairs is just the number of cols in the samples matrix (minus the first)
        y_inc = .5;
        
        %OK, seems weird, but I'm fixing the maximum height of the Y axis
        %to 14 samples, which is the largest sequence length across all the
        %studies (Seq study 2) so that all bars are on the same scale and
        %can be compared across studies. But I'll still number the X ticks
        %differently depending on sequence length.
%         ystart = max(max(samples_accum(:,1:human_is_index))) + y_inc*human_is_index +y_inc;
%         ystart = Generate_params.study(study).seq_length + y_inc*human_is_index +y_inc;
        ystart = 14 + y_inc*human_is_index +y_inc;
        %     line_y_values = ystart:-y_inc:0;
        line_y_values = (0:y_inc:ystart) + (max(max(samples_accum)) + 2*y_inc);
        
        %loop through pairs, get "significance" of each, plot them
%         for pair = num_pairs:-1:1;
        for pair = human_is_index:-1:1;

            
            [bf10(pair),samples_pvals(pair),ci,stats] = ...
                bf.ttest( samples_accum(:,human_is_index) - samples_accum(:,pair) );
            
            %             bf.ttest( samples_accum(:,1) - samples_accum(:,num_pairs + 2 - pair) );

            %distance on plot
            %         distance_on_plot = [1 num_pairs + 2 - pair];
            distance_on_plot = [human_is_index pair];
            
            %             if samples_pvals(pair) < 0.05/num_pairs;
            %                 plot(distance_on_plot,...
            %                     [line_y_values(pair) line_y_values(pair)],'LineWidth',.5,'Color',[0 0 0]);
            %             end;
            
            if bf10(pair) < (1/BayesThresh);
                plot(distance_on_plot,...
                    [line_y_values(pair) line_y_values(pair)],'LineWidth',1.5,'Color',[1 0 1]);
            end;
            if bf10(pair) > BayesThresh;
                plot(distance_on_plot,...
                    [line_y_values(pair) line_y_values(pair)],'LineWidth',1,'Color',[0 0 0]);
            end;
            
            
        end;   %pairs: Loop through comparisons between participants and models
        %Set Y lim to 14 + space for sig bars but y ticks to count the real
        %sequene length.
        set(gca,'Ylim',[0 ystart],'YTick',[0:2:Generate_params.study(study).seq_length]);
        
        
        
        
        
        
        
        
        
        %%%%%%%%%%%%%%%%%%
        %BIC significance lines
        %run and plot ttests on IC averages
        
        %return to BIC subplot
        panel_num = panel_nums(2);  %Panels nums can be changed colwise versus rowwise in subplot
        subplot(rows,cols,eval(subplot_num));
        
        pairs = nchoosek(1:size(IC_pps_all_models,2),2);
        pairs= sortrows(pairs(:,[2 1]),'descend');
        num_pairs = size(pairs,1);
%         [a In] = sort(diff(pairs')','descend');  %lengths of connecting lines
        %         line_pair_order = pairs(In,:);    %move longest connections to top
        line_pair_order = pairs;    %move longest connections to top
        
        
        %         %Where to put top line?
        %         y_inc = 2;
        %         ystart = max(max(IC_pps_all_models)) + y_inc*num_pairs;
        %         line_y_values = ystart:-y_inc:0;
        %Where to put top line?
        max_y = max(max(IC_pps_all_models));
        y_inc = .065*max_y;
        ystart = max_y + y_inc*num_pairs;
        line_y_values = ystart:-y_inc:0;
        
        for pair = 1:num_pairs;
            
            
            
            
            %             %%%%%Bayes Factors
            %             [bf10(pair),IC_pp_pvals(pair),ci,stats] = ...
            %                 bf.ttest( IC_pps_all_models(:,line_pair_order(pair,1)) ...
            %                 - IC_pps_all_models(:,line_pair_order(pair,2)) );
            %
            %             yticks = linspace(0, ceil(max_y/20)*20,5);
            %             set(gca,'Ylim',[0 ystart],'YTick',yticks);
            %
            %             if bf10(pair) < (1/BayesThresh);
            %                 plot([line_pair_order(pair,1) line_pair_order(pair,2)],...
            %                     [line_y_values(pair) line_y_values(pair)],'LineWidth',1,'Color',[1 0 1]);
            %             end;
            %             if bf10(pair) > BayesThresh;
            %
            %                 %find identifier of the rightmost model in the pair
            %                 rightmost_num = line_pair_order(pair,2);    %This is an index into IC_pps_all_models which excludes IO and Human
            %                 rightmost_id = Generate_params.study(study).model(line_pair_order(pair,2)+human_is_index).identifier;
            %                 rightmost_color = plot_cmap(rightmost_id,:);    %color associated with identifer of rightmost model in pairwise comparison
            %
            %                 plot([line_pair_order(pair,1) line_pair_order(pair,2)],...
            %                     [line_y_values(pair) line_y_values(pair)],'LineWidth',.5,'Color',rightmost_color);
            % %                     [line_y_values(pair) line_y_values(pair)],'LineWidth',.5,'Color',[0 1 0]); %plot "significant" connector lines in a color like green
            %             end;  %"significant"?
       %end;    %loop through pairs for Bayesian pairwise tests
            

            
            
            
            
%             %%%%%Frequentist t-test (multiple comparison corrected)
            %run ttest this pair
            [h IC_pp_pvals(pair) ci stats] = ttest(IC_pps_all_models(:,line_pair_order(pair,1)), IC_pps_all_models(:,line_pair_order(pair,2)));
            
            ls_inc = 20;
            num_ticks = 4;
            yticks = round(linspace(0, ceil(max_y/ls_inc)*ls_inc,4));
                        set(gca,'Ylim',[0 ystart],'YTick',yticks);
            
            if IC_pp_pvals(pair) < 0.05/size(pairs,1);  %multiple comparison corrected
                
                
                %%find identifier of the rightmost model in the pair
                rightmost_num = line_pair_order(pair,2);    %This is an index into IC_pps_all_models which excludes IO and Human
                rightmost_id = Generate_params.study(study).model(line_pair_order(pair,2)+human_is_index).identifier;
                rightmost_color = plot_cmap(rightmost_id,:);    %color associated with identifer of rightmost model in pairwise comparison
                
                
                plot([line_pair_order(pair,1) line_pair_order(pair,2)],...
                    [line_y_values(pair) line_y_values(pair)],'LineWidth',.5,'Color',rightmost_color);
                %                                 [line_y_values(pair) line_y_values(pair)],'LineWidth',.5,'Color',[0 0 0]);
                
            end;    %Do line on plot?;
            
        end;    %loop through ttest pairs
        
    end;    %Only compute ttests if there is at least one pair of models
    
    


    
    %This works better here, despite the inelegent extra loop, because the
    %y lim has been fixed for certain by this point so I can saftely make
    %the distance of the text from the X axis be dependent on the Y limit
    %(e.g., the x axis labels won't end up with variable diustances as the
    %plot evolves over model iterations).
    for model = 1:size(IC_pps_all_models,2)
        
        this_offset = -x_axis_test_offset*diff(ylim);
        text( model, this_offset ...
            ,sprintf('%s',Generate_params.study(study).model(model+human_is_index).name) ...
            ,'Fontname','Arial' ...
            ,'Fontsize',graph_font ...
            ,'Rotation',x_rot ...
            ,'HorizontalAlignment','right' ...
            );
        
    end;    %loop through models to add x tick labels to BIC plot
    
    
    
    
    %%%%%%%%%%%%%%%%%%%%%%
    %Plot of numbers of winning subs for each model
    
    %     subplot(Generate_params.num_panels,Generate_params.num_studies,subplot_indices(4,study));
    %     subplot(2,3,6);
    panel_num = panel_nums(3);  %Panels nums can be changed colwise versus rowwise in subplot
    subplot(rows,cols,eval(subplot_num));
    hold on; box off;
    
    %winning models
    [a pps_indices] = min(IC_pps_all_models');
    
    model_it = 0;
    for model = 1:num_models_per_study-human_is_index;
        
        %         %skip empty identifiers/ optimal
        %         if isempty(Generate_params.study(study).model(model).identifier);
        %             continue;
        %         else
%         model_it = model_it + 1;
        %         end;
        
        bar(model,numel(find(pps_indices==model)), ...
            'FaceColor', plot_cmap(Generate_params.study(study).model(model+human_is_index).identifier,:),'FaceAlpha',f_a+.2,'EdgeColor',[0 0 0] );
        %             'FaceColor', plot_cmap(Generate_params.study(study).model(model).identifier,:),'FaceAlpha',f_a,'EdgeColor',[0 0 0] );
        
        set(gca ...
            ,'XTick',[] ...
            ,'fontSize',graph_font ...
            ,'FontName','Arial',...
            'XLim',[0 num_models_per_study-(human_is_index-1)] ...
            );
        ylabel('Frequency');
        
    end;    %models
    
    %Like above, it's embarrassing inelegtn to repeat a loop just
    %completed, but I need the max frequency and therefore the ylim to be
    %finalised or I'll end up with different diustances from the x axis here
    for model = 1:num_models_per_study-human_is_index;
        
        this_offset = -x_axis_test_offset*diff(ylim);
        text( model, this_offset ...
            ,sprintf('%s',Generate_params.study(study).model(model+human_is_index).name) ...
            ,'Fontname','Arial' ...
            ,'Fontsize',graph_font ...
            ,'Rotation',x_rot ...
            ,'HorizontalAlignment','right' ...
            );
        
    end;
    
    
end;    %loop through studies
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

















%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function New_struct = combineModels(New_struct_SV, New_struct_OV);
function New_struct = combineModels(data, bar_order);

%I'm making it so identifier 1 is participants, iderntifier 2 is ideal
%observer and the rest of the identifiers (which in original struct were integers 1
%or greater) are now added a two to make room for participants and ideal
%observer. The identifiers in hybrid_figures* are used to access color maps

%This takes the subjective values struct and the objective values struct and
%makes a combined struct from them, with some formastting to prepare it for
%figure-worthy plotting

%bar_order gives the new model type order for the bars. SV and OV model
%types like Cs for example will be placed together in position defined (for both value type) by
%bar_orde. Each index in bar order refers to number in bar_order refers to
%the position in the new struct and the number at that index refers to the
%position in the old array. If a zero appears in an index position, then
%that means put the participant samples data there instead of a model.

%Basically, for the intended procedure, the idea is to move optimal from the
%sixth position to the first and insert participants after optimal and then
%have the theoretical models follow.

%subjective and objective value models? Or only objective values models?
num_datasets = size(data,2);

%initialise based on first dataset
New_struct.num_samples = data{1}.num_samples; %Both SV and OV should have the same data for participant samples, only models should differ
New_struct.seq_length = data{1}.seq_length;
New_struct.num_seqs = data{1}.num_seqs;
New_struct.num_models = size(data{1}.model,2);

%concatenate models
% for model = 1:New_struct.num_models;
if num_datasets == 2;
    model_name_suffix = {'SV','OV'};
else
    model_name_suffix = {'OV'};
end;
it_bar_order = 1;
for model = 1:numel(bar_order);
    
    if bar_order(model) == 0;
        
        %Put participants' samples data in free model location
        temp(it_bar_order).num_samples_est = data{1}.num_samples;
        temp(it_bar_order).identifier = 1;
        temp(it_bar_order).name = "Human";
        temp(it_bar_order).skip = 1;    %1 will tell BIC and frequency plotters to ignore participants
        
        %set up for next model entry in next loop iteration
        it_bar_order = it_bar_order + 1;
        
    else
        
        for dataset = 1:num_datasets;
            
            %Assign sample data for this model and dataset
            %         temp(((model-1)*num_datasets)+dataset) = data{dataset}.model(bar_order(model));
            %             temp(it_bar_order) = data{dataset}.model(bar_order(model));
            temp(it_bar_order).this_models_free_parameters = data{dataset}.model(bar_order(model)).this_models_free_parameters;
            temp(it_bar_order).estimated_params = data{dataset}.model(bar_order(model)).estimated_params;
            temp(it_bar_order).ll = data{dataset}.model(bar_order(model)).ll;
            temp(it_bar_order).num_samples_est = data{dataset}.model(bar_order(model)).num_samples_est;
            
            %Let's do some reformatting of labels to facilitate plotting later
            if bar_order(model) == 6;   %If optimality model
                
                temp(it_bar_order).identifier = 2;  %Humans are 1, ideal observer is 2
                temp(it_bar_order).name = sprintf("IO %s",model_name_suffix{dataset});
                temp(it_bar_order).skip = 1;    %1 will tell BIC and frequency plotters to ignore IO
                
            elseif bar_order(model) == 1;   %If cut-off model
                
                temp(it_bar_order).identifier =  data{dataset}.model(bar_order(model)).identifier + 2; %make two spaces for humans and io
                temp(it_bar_order).name = sprintf("CO %s",model_name_suffix{dataset});    %shorten name and add suffix
                temp(it_bar_order).skip = 0;    %0 will tell BIC and frequency plotters to plot this as a theoretical model
                
           elseif bar_order(model) == 2;   %If Cs model (only want to slightly modify name to conform to others' format)
                
                temp(it_bar_order).identifier =  data{dataset}.model(bar_order(model)).identifier + 2; %make two spaces for humans and io
                temp(it_bar_order).name = sprintf("CS %s",model_name_suffix{dataset});    %shorten name and add suffix
                temp(it_bar_order).skip = 0;    %0 will tell BIC and frequency plotters to plot this as a theoretical model
                
           elseif bar_order(model) == 3;   %If Optimism model (only want to slightly modify name so readers don't confuse Opt with optimal)
                
                temp(it_bar_order).identifier =  data{dataset}.model(bar_order(model)).identifier + 2; %make two spaces for humans and io
                temp(it_bar_order).name = sprintf("O %s",model_name_suffix{dataset});    %shorten name and add suffix
                temp(it_bar_order).skip = 0;    %0 will tell BIC and frequency plotters to plot this as a theoretical model
                
            else;   %If any other model (i.e., BV, BR
                
                temp(it_bar_order).identifier =  data{dataset}.model(bar_order(model)).identifier + 2; %make two spaces for humans and io
                temp(it_bar_order).name = sprintf("%s %s",data{dataset}.model(bar_order(model)).name, model_name_suffix{dataset});    %shorten name and add suffix
                temp(it_bar_order).skip = 0;    %0 will tell BIC and frequency plotters to plot this as a theoretical model
                
            end;
            
            %set up for next model entry in next loop iteration
            it_bar_order = it_bar_order + 1;
            
        end;    %loop sv. ov datasets
        
    end; %check if participants spot in bar order
    
end;    %loop through bar order indices (locations in new struct)

New_struct.model = temp;

fprintf('');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%











%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function New_data_struct = reformatData(data);

%Makes a simplified struct from the data that's easier to combine between
%subjective anbd objective values in the next step

%I've just copied this straight from combine_hybrid_objective_v3 then modified it to fit new context

num_models = size(data.Generate_params.model,2);

New_data_struct.seq_length = data.Generate_params.seq_length;    %assumes all studies use same seq length (num options per seq)
New_data_struct.num_samples = [];
New_data_struct.num_seqs = [];

% for j=1:size(data,2);   %datasets
%
% disp(sprintf('dataset: %d subjects', size(data.Generate_params.num_samples,2)));

%participants
New_data_struct.num_samples = ...
    [New_data_struct.num_samples; ...
    nanmean(data.Generate_params.num_samples)' ...
    ];

%participants - I need to get every participant's sequence length
%so I can compute BIC
New_data_struct.num_seqs = ...
    [New_data_struct.num_seqs; ...
    sum(data.Generate_params.num_samples>0)' ...
    ];

% end;    %datasets

disp(sprintf('TOTAL: %d participants', numel(New_data_struct.num_samples)));



for model=1:num_models; %models, minus 1 if optimal (io) is last one
    
    %copy some stuff over that's common to all studies (assumed)
    New_data_struct.model(model).identifier = data.Generate_params.model(model).identifier;
    New_data_struct.model(model).name = data.Generate_params.model(model).name;
    New_data_struct.model(model).this_models_free_parameters = data.Generate_params.model(model).this_models_free_parameters;
    %initialise some stuff to accumulate over studies
    New_data_struct.model(model).estimated_params = [];
    New_data_struct.model(model).ll = [];
    New_data_struct.model(model).num_samples_est = [];
    
    %     for j=1:size(data,2);   %datasets
    
    %models
    New_data_struct.model(model).estimated_params = ...
        [New_data_struct.model(model).estimated_params; ...
        data.Generate_params.model(model).estimated_params ...
        ];
    
    New_data_struct.model(model).ll = ...
        [New_data_struct.model(model).ll; ...
        data.Generate_params.model(model).ll ...
        ];
    
    New_data_struct.model(model).num_samples_est = ...
        [New_data_struct.model(model).num_samples_est; ...
        nanmean(data.Generate_params.model(model).num_samples_est)' ...
        ];
    
    %         Generate_params.model(i).ranks_est = ...
    %             [Generate_params.model(i).ranks_est ...
    %             data{j}.Generate_params.model(i).ranks_est ...
    %             ];
    
    %     end;    %datasets
    
end;    %models
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%







