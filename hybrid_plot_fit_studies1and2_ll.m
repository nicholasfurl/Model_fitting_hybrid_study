
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [] = hybrid_plot_io_studies1and2_ll;

cd('C:\matlab_files\fiance\parameter_recovery\beta_fixed_code\Model_fitting_hybrid_study');

addpath(genpath('C:\matlab_files\fiance\parameter_recovery\beta_fixed_code\Model_fitting_hybrid_study\plotSpread'));
addpath(genpath('C:\matlab_files\fiance\parameter_recovery\beta_fixed_code\Model_fitting_hybrid_study\klabhub-bayesFactor-3d1e8a5'));

%hybrid_plot_fit_studies1and2_ll.m is a second attempt to compare models
%where beta is fit. This time, I'll work from
%hybrid_plot_io_studies1and2_ll_rebar.m.


%hybrid_plot_io_studies1and2_ll_rebar.m reorganises the order of the model
%bars (and encoding in run_studies).

%hybrid_plot_io_studies1and2_ll.m also paints lines onto plots showing
%results of pairwise tests, Bayesian and frequentist.

%hybrid_plot_io_studies1and2_ll.m also adds in the stars condition, which
%was neglected i the earluier version, and it reorganised the code to allow
%for separate reward structures in the  [.12 .08 .04] monetary reward
%conditions and the 5:3:1 ratio in the stars condition.

%hybrid_plot_io_studies1ans2_ll.m builds in a ll computation that can be
%used to decide which version of the ideal observer comes closest to
%participants' choices. It doesn't fit anything though, not even softmax beta

%Based on model_fitting_hybrid_study.m, this makes new figures of hybrid
%conditions, using newly debugged code that operates directly on the raw
%data. Hopefully these will be the figure-worthy ones in the end for the
%Communications Psychology revisions.

compute_data = 1;

if compute_data == 1;
    
    %set up a matrix to tell the program what to run
    %put them in order you'd like to see their plots appear
    %col1 -- 3: full pilot, 5: full,  6: ratings, 2: baseline pilot 1, 4: baseline,  7:squares, 8: timing 9 stars (Do not use 1 as a study number!)
    %a 1 if run model subjective continuous (col2), 3-rank money (col3), 3-rank 5/3/1 (col4)
    %then  objective continuous (col5, models_pick 4), 3-rank money (col6), 3-rank 5/3/1 (col7)
    
    %a 1 if run model:
    %subjective continuous (col2), objective continulous (col3)
    %subjective 3-rank money (col4), objective 3-rank money (col5)
    %subjective 3-rank 5/3/1 (col6), objective 3-rank 5/3/1 (col7)
%     
%     %%%%%%%%SM Figure, all combinations of values and reward scheme models%%%%%%
%     run_studies = [ ...
%         3 1 1 1 1 1 1; ...
%         5 1 1 1 1 1 1; ...
%         6 1 1 1 1 1 1; ...
%         2 0 1 0 1 0 1; ...
%         4 0 1 0 1 0 1; ...
%         7 0 1 0 1 0 1; ...
%         8 0 1 0 1 0 1; ...
%         9 0 1 0 1 0 1; ...
%         ];
%     %
%     %     %This will create plots that average and analyse rows from run_studies
%     %     %Each row in run averages gives rows to be averaged for a given plot
%     %     %Make sure all run_studies rows for a given average have the same models
%     %     %computed or plot might fail or be hard to interpret
%     run_averages = { ...
%         [1 2 3]; ...  %all the studies with a phase 1 where subj and obj value models can be computed (pilot full, full, prior)
%         [4:8]
%         };
%     out_filename = 'hybrid_plot_allModels_rebar.mat';
%     %%%%%%%%SM Figure, all combinations of values and reward scheme models%%%%%%
%     
    
    
    
    
    
    
%     
%     
%         %%%%%%%%SM Figure, models matched with their subj and obj reward-appropriate counterparts%%%%%%
%     run_studies = [ ...
%         3 1 1 0 0 0 0; ...
%         5 1 1 0 0 0 0; ...
%         6 0 0 1 1 0 0; ...
%         2 0 0 0 1 0 0; ...
%         4 0 0 0 1 0 0; ...
%         7 0 0 0 1 0 0; ...
%         8 0 0 0 1 0 0; ...
%         9 0 0 0 0 0 1; ...
%         ];
%     %
%     %     %This will create plots that average and analyse rows from run_studies
%     %     %Each row in run averages gives rows to be averaged for a given plot
%     %     %Make sure all run_studies rows for a given average have the same models
%     %     %computed or plot might fail or be hard to interpret
%     run_averages = { ...
%         [1 2]; ...  %all the studies with a phase 1 where subj and obj value models can be computed (pilot full, full, prior)
%         [4:8]
%         };
%     out_filename = 'hybrid_plot_appropModels_rebar.mat';
%     %%%%%%%%SM Figure, all combinations of values and reward scheme models%%%%%%
%     
    
    
    
    
    
    
    
    
    
    
    
    
    %     % %%%%%%%For testing purposes, a simple and relatively quick arrangement
        run_studies = [ ...
            3 1 1 0 0 0 0;
            5 1 1 0 0 0 0;
            ];
    
    %     %Make sure all indices on a row here use the same models
        run_averages = { ...
            [1 2];
            };
        out_filename = 'hybrid_plot_testRun_noB.mat';
    %     % %%%%%%%For testing purposes, a simple and relatively quick arrangement
    
    
    
    
    
    
    
    %For posterity
%     data.model_labels = {'Participants' 'Subj-rew1' 'Subj-rew2' 'Subj-rew3' 'Obj-rew1' 'Obj-rew2' 'Obj-rew3'};
    data.model_labels = {'Participants' 'Subj-rew1' 'Obj-rew1' 'Subj-rew2' 'Obj-rew2' 'Subj-rew3' 'Obj-rew3'};
    data.study_labels = {'' 'Baseline pilot' 'Full pilot', 'Baseline' 'Full',  'Ratings'  'Squares' 'Timing' 'Payoff'};
    data.num_studies = numel(unique(run_studies(:,1)));
    data.run_studies = run_studies;
    data.run_averages = run_averages;
    data.out_filename = out_filename;
    
    %initalise averages datastruct
    for this_average = 1:size(data.run_averages,1); %Loop through rows of run averages to figure out how many average and model fields you need
        
        %initialise fields to hold concatenated participant data
        %There will be one field per row of run_averages, however many
        %studies or models there are.
        data.averages(this_average).samples = [];
        data.averages(this_average).ranks = [];
        
        %get number of models, assuming all models for this average run the
        %same models (and so we need only check the first study's number of models
        num_models_this_average = sum( data.run_studies(data.run_averages{this_average}(1),:) == 1 );
        
        %inialise fields to hold concatenated model data
        for model = 1:num_models_this_average;
            
            data.averages(this_average).model(model).samples = [];
            data.averages(this_average).model(model).ll = [];
            data.averages(this_average).model(model).ranks = [];
            
        end;    %model loop
        
    end;    %averaged plot loop
    
    
    
    %Here is the main loop that gets participant data per study and
    %computes all the corresponding models.
    for study = 1:data.num_studies;
        
        study_id = data.run_studies(study,1);
        
        data.study(study).study_id = study_id;
        data.study(study).name = data.study_labels{study_id};
        
        %Now get data for every model requested
        %models_pick just converts the binary / Boolean model specifiers to
        %integer labels according to their index
        clear models_pick
        %Find which indices specifying models are picked out by 1's
        models_pick = find(data.run_studies(study,1:end)==1);
        
        for model = 1:numel(models_pick);
            
            %identify subj/obj values model info based on pick (col num of run_studies, from 2 to 7)
            subjective_vals = 0;    %objective values
            if models_pick(model) == 2 | models_pick(model) == 4 | models_pick(model) == 6; %subjective values
                subjective_vals = 1;
            end;
            
            %identify payoff scheme model info based on pick
            payoff_scheme = 0;  %3-rank expressed in money (GBP), cols 4 and 5
            if models_pick(model) == 2 | models_pick(model) == 3; %continuous valued
                payoff_scheme = 1;
            elseif models_pick(model) == 6 | models_pick(model) == 7; %5/3/1 3-rank (appropriate for stars)
                payoff_scheme = 2;
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
            data.study(study).model(model).name = data.model_labels{models_pick(model)};
            data.study(study).model(model).models_pick = models_pick(model);
            data.study(study).model(model).subjective_vals = subjective_vals;
            data.study(study).model(model).payoff_scheme = payoff_scheme;
            data.study(study).model(model).prior_vals = mean_ratings_all;
            data.study(study).model(model).seq_vals = seq_vals_all;
            
            
            %This is the main change from the io version
            data = fit_models(data);
            
%             %model samples and ranks
%             disp('RUNNING MODEL NOW');
%             data = run_models(data);
            
            fprintf('');
            
            %Now accumulate MODEL data for making plots averaging over studies
            for this_average = 1:size(data.run_averages,1);
                
                %                 if this_average == 2 and model == 2;
                %                     fprintf(' ');
                
                if sum( data.run_averages{this_average} == study ) > 0; %check if this study is in this list of studies to average
                    
                    %concatenate model data
                    data.averages(this_average).model(model).samples = [ data.averages(this_average).model(model).samples; mean(data.study(study).model(model).samples)'];
                    data.averages(this_average).model(model).ll = [ data.averages(this_average).model(model).ll; data.study(study).model(model).ll];
                    data.averages(this_average).model(model).ranks = [ data.averages(this_average).model(model).ranks; mean(data.study(study).model(model).ranks)'];
                    data.averages(this_average).model(model).models_pick = data.study(study).model(model).models_pick;
                    data.averages(this_average).model(model).name = data.study(study).model(model).name;
                end;    %is the study in the average list and so needs to be concateneated?
                
            end;    %loop through rows of averaged plots
            
        end;    %models loop
        
        %Now (outside of model loop but within study loop) accumulate PARTICIPANT data for making plots, which average over studies
        for this_average = 1:size(data.run_averages,1);
            
            if sum( data.run_averages{this_average} == study ) > 0; %check if this study is in this list of studies to average
                
                %concatenate participant data
                data.averages(this_average).samples = [ data.averages(this_average).samples; mean( data.study(study).samples )'];
                data.averages(this_average).ranks = [ data.averages(this_average).ranks; mean( data.study(study).ranks )'];
                
            end;    %is the study in the average list and so needs to be concateneated?
            
        end;    %loop through rows of averaged plots
        
    end;    %loop through studies
    
    %takes a long time to run and I might want to skip to here and load data later
    %     save('data_struct3.mat','data');        %basic result
    %     save('hybrid_plot_io_ll.mat','data');   %basic result, but saves into data struct the ll too
    %     save('test_average_io_ll.mat','data');   %for testing
    %         save('hybrid_payoff2_io_ll_v2).mat','data');   %introducing 5:3:1 payoffs and stars condition analysis
    %save('hybrid_v2_io_ll_v2).mat','data');   %introducing 5:3:1 payoffs and stars condition analysis
    save(out_filename,'data');
    
    
else
    
    %        load('test_noNorm_io_ll.mat','data'); %takes a long time to run and I might want to skip to here and load data later
    %     load('hybrid_plot_io_ll.mat','data'); %takes a long time to run and I might want to skip to here and load data later
    %         load('hybrid_payoff2_io_ll.mat','data');   %introducing 5:3:1 payoffs and stars condition analysis
    %     load('test_average_io_ll.mat','data');   %for twsting
   % load('hybrid_plot_allModels_rebar.mat','data');   %introducing 5:3:1 payoffs and stars condition analysis
         load('hybrid_plot_testRun.mat','data');   %introducing 5:3:1 payoffs and stars condition analysis
    
end;    %studies

plot_results(data);


disp('audi5000')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
function plot_results(data)

%Figures comparing participants and models
h1 = figure; set(gcf,'Color',[1 1 1]);  %samples
h2 = figure; set(gcf,'Color',[1 1 1]);  %ll
h3 = figure; set(gcf,'Color',[1 1 1]);  %ranks
%h4 figure is defined below and shows data collapsing over studies
h5 = figure; set(gcf,'Color',[1 1 1]);  %compares participant performance study by study in one plot (for samples / ranks each)

plot_colors = lines(size(data.run_studies,2));

%For figures h1-h3
cols = ceil(data.num_studies/2);
rows = ceil(data.num_studies/cols);

%misc necessities for below
ll_maxY = 55;   %roughly what you think the maximum Y value containing ll data will be. The sig connector lines will be drawn above this
BayesThresh = 3;    %How big a Bayes factor needed to paint onto plot?
x_rot = 30;
bar_line_width = 2;
bar_alpha = .1;


for study = 1:data.num_studies;
    
    this_study_sampling = [];
    this_study_ll = [];
    this_study_ranks = [];
    
    x_axis_it = 1;
    
    %participants samples
    
    %Set up figure comparing participant sampling /ranks per study
    %Yet another accumulator! This one is to facilitate plots and pairwise tests
    %between participant performance measures in the different
    %conditions. Need this later for pairwise tests
    %Need cell array since studies vary in sample size slightly
    P_accum_samples{study} = mean(data.study(study).samples);
    P_accum_ranks{study} = mean(data.study(study).ranks);
    
    figure(h5)
    
    %samples
    subplot(2,1,1);
    bar( study, mean(P_accum_samples{study}),'LineWidth',bar_line_width, 'FaceAlpha', bar_alpha, 'FaceColor', plot_colors(1,:));
    plotSpread( P_accum_samples{study}' , 'xValues', study, 'distributionColors',plot_colors(1,:) );
    num_options = size(data.study(study).model(1).seq_vals,2);
    ylim([0 num_options]);
    ylabel('Samples to decision');
    set(gca,'FontSize',12,'FontName','Arial','xtick',[],'ytick',[0:2:num_options],'LineWidth',2);
    text( study, -1.5 ...
        ,sprintf('%s',data.study(study).name) ...
        ,'Fontname','Arial' ...
        ,'Fontsize',12 ...
        ,'Rotation',x_rot ...
        ,'HorizontalAlignment','right' ...
        );
    box off;
    
    %ranks
    subplot(2,1,2);
    bar( study, mean(P_accum_ranks{study}),'LineWidth',bar_line_width, 'FaceAlpha', bar_alpha, 'FaceColor', plot_colors(1,:));
    plotSpread( P_accum_ranks{study}' , 'xValues', study, 'distributionColors',plot_colors(1,:) );
    num_options = size(data.study(study).model(1).seq_vals,2);
    ylim([0 num_options]);
    ylabel('Rank of chosen option');
    set(gca,'FontSize',12,'FontName','Arial','xtick',[],'ytick',[0:2:num_options],'LineWidth',2);
    text( study, -1.5 ...
        ,sprintf('%s',data.study(study).name) ...
        ,'Fontname','Arial' ...
        ,'Fontsize',12 ...
        ,'Rotation',x_rot ...
        ,'HorizontalAlignment','right' ...
        );
    box off;
    
    
    
    figure(h1)
    subplot(rows,cols,study); hold on;
    bar( x_axis_it, mean(mean(data.study(study).samples)),'LineWidth',bar_line_width, 'FaceAlpha', bar_alpha, 'FaceColor', plot_colors(1,:));
    plotSpread( mean(data.study(study).samples)' , 'xValues', x_axis_it, 'distributionColors',plot_colors(1,:) );
    num_options = size(data.study(study).model(1).seq_vals,2);
    ylim([0 num_options]);
    text( x_axis_it, -.5 ...
        ,sprintf('%s','Participants') ...
        ,'Fontname','Arial' ...
        ,'Fontsize',12 ...
        ,'Rotation',x_rot ...
        ,'HorizontalAlignment','right' ...
        );
    
    
    %participants ranks
    figure(h3)
    subplot(rows,cols,study); hold on;
    bar( x_axis_it, mean(mean(data.study(study).ranks)),'LineWidth',bar_line_width, 'FaceAlpha', bar_alpha, 'FaceColor', plot_colors(1,:));
    plotSpread( mean(data.study(study).ranks)' , 'xValues', x_axis_it, 'distributionColors',plot_colors(1,:) );
    num_options = size(data.study(study).model(1).seq_vals,2);
    ylim([0 num_options]);
    text( x_axis_it, -.5 ...
        ,sprintf('%s','Participants') ...
        ,'Fontname','Arial' ...
        ,'Fontsize',12 ...
        ,'Rotation', x_rot ...
        ,'HorizontalAlignment','right' ...
        );
    
    %Accumulate participant sampling data and ranks data for pairwise tests below
    this_study_sampling(:,x_axis_it) = mean(data.study(study).samples)';
    this_study_ranks(:,x_axis_it) = mean(data.study(study).ranks)';
    
    %models
    num_models = size(data.study(study).model,2);
    for model = 1:num_models;
        
        x_axis_it = x_axis_it + 1;
        
        %plot samples for this model and study
        figure(h1);
        subplot(rows,cols,study); hold on;
        bar( x_axis_it, mean(mean(data.study(study).model(model).samples)),'LineWidth',bar_line_width, 'FaceAlpha', bar_alpha, 'FaceColor', plot_colors(data.study(study).model(model).models_pick,:));
        plotSpread( mean(data.study(study).model(model).samples)' , 'xValues', x_axis_it, 'distributionColors',plot_colors(data.study(study).model(model).models_pick,:) );
        ylim([0 num_options]);
        ylabel('Samples to decision');
        title(data.study(study).name);
        set(gca,'FontSize',12,'FontName','Arial','xtick',[],'ytick',[0:2:num_options],'LineWidth',2);
        text( x_axis_it, -.5 ...
            ,sprintf('%s',data.study(study).model(model).name) ...
            ,'Fontname','Arial' ...
            ,'Fontsize',12 ...
            ,'Rotation', x_rot ...
            ,'HorizontalAlignment','right' ...
            );
        
        %plot ll for this model and study
        figure(h2);
        subplot(rows,cols,study); hold on;
        bar( x_axis_it-1, mean(data.study(study).model(model).ll),'LineWidth',bar_line_width, 'FaceAlpha', bar_alpha, 'FaceColor', plot_colors(data.study(study).model(model).models_pick,:));
        plotSpread( data.study(study).model(model).ll , 'xValues', x_axis_it-1, 'distributionColors',plot_colors(data.study(study).model(model).models_pick,:) );
        ylabel('Log-likelihood');
        title(data.study(study).name);
        set(gca,'FontSize',12,'FontName','Arial','LineWidth',2,'xtick',[],'ytick',[0:10:ll_maxY]);
        ylim([0 ll_maxY]);
        text( x_axis_it-1, -3 ...
            ,sprintf('%s',data.study(study).model(model).name) ...
            ,'Fontname','Arial' ...
            ,'Fontsize',12 ...
            ,'Rotation', x_rot ...
            ,'HorizontalAlignment','right' ...
            );
        
        %plot ranks for this model and study
        figure(h3);
        subplot(rows,cols,study); hold on;
        bar( x_axis_it, mean(mean(data.study(study).model(model).ranks)),'LineWidth',bar_line_width, 'FaceAlpha',bar_alpha, 'FaceColor', plot_colors(data.study(study).model(model).models_pick,:));
        plotSpread( mean(data.study(study).model(model).ranks)' , 'xValues', x_axis_it, 'distributionColors',plot_colors(data.study(study).model(model).models_pick,:) );
        ylim([0 num_options]);
        ylabel('Rank of chosen option');
        title(data.study(study).name);
        set(gca,'FontSize',12,'FontName','Arial','xtick',[],'ytick',[0:2:num_options],'LineWidth',2);
        text( x_axis_it, -.5 ...
            ,sprintf('%s',data.study(study).model(model).name) ...
            ,'Fontname','Arial' ...
            ,'Fontsize',12 ...
            ,'Rotation', x_rot ...
            ,'HorizontalAlignment','right' ...
            );
        
        %Accumulate model sampling data, ll and ranks data for pairwise tests below
        this_study_sampling(:,x_axis_it) = mean(data.study(study).model(model).samples)';
        this_study_ll(:,x_axis_it-1) = data.study(study).model(model).ll;
        this_study_ranks(:,x_axis_it) = mean(data.study(study).model(model).ranks)';
        
    end;    %loop through models
    
    %I need comparisons for participants against each model for samples and
    %ranks but comparisons for all pairs of models for ll. So loop through
    %all pairs and run ones needed when they come up.
    fprintf('')
    pairs = nchoosek(1:size(this_study_sampling,2),2);
    num_pairs = size(pairs,1);
    [a In] = sort(diff(pairs')','descend');  %lengths of connecting lines
    line_pair_order = pairs(In,:);    %move longest connections to top
    
    %iterators, to standardise y locations of connector lines across graphs
    i_samples = 1;  %for samples
    i_ranks = 1;    %for ranks
    i_ll = 1;   %for log likelihood
    
    for pair = 1:num_pairs;
        
        clear temp;
        
        %Is this a pair involving participants? (needed for samples and ranks)
        if sum(line_pair_order(pair,:) == 1) == 1;
            
            %find difference between vectors for participants and this pair's model
            temp(:,1) = this_study_sampling(:,line_pair_order(pair,1)) - this_study_sampling(:,line_pair_order(pair,2));
            temp(:,2) = this_study_ranks(:,line_pair_order(pair,1)) - this_study_ranks(:,line_pair_order(pair,2));
            
            fig_Hs = [h1 h3];
            
            %Where to put top line (samples and ranks)?
            y_inc = .5;
            %             ystart = num_options + y_inc*num_pairs + 2*y_inc;
            ystart = num_options + 6*y_inc;
            line_y_values = ystart:-y_inc:0;
            
            x_position_1 = line_pair_order(pair,1);
            x_position_2 = line_pair_order(pair,2);
            
            correction = num_models;
            
            iterator = [i_samples i_ranks];
            
        else;   %If the contrast does NOT involve participants then it must be between models, which means we want an ll contrast instead.
            
            temp = this_study_ll(:,line_pair_order(pair,1)-1) - this_study_ll(:,line_pair_order(pair,2)-1);
            
            fig_Hs = [h2];
            
            %Where to put top line (samples and ranks)?
            y_inc = 5;
            ystart = ll_maxY + 15*y_inc;
            line_y_values = ystart:-y_inc:0;
            
            x_position_1 = line_pair_order(pair,1)-1;
            x_position_2 = line_pair_order(pair,2)-1;
            
            if num_models > 1;
                correction = size(nchoosek(1:num_models,2),2);
            else
                correction = 1;
            end;
            
            iterator = [i_ll];
            
            if line_pair_order(pair,1) == 5 & line_pair_order(pair,2) == 6
                fprintf('');
            end;
            
        end;    %check if col 1 (participants) included in this pair
        
        %Now use temp and fig_Hs to compute and draw stats results
        for which_plot = 1:size(temp,2);
            
            %Get stats for this difference
            clear bf10 pval ci stats;
            
            [bf10,pvals,ci,stats] = ...
                bf.ttest( temp(:,which_plot) );
            
            %plot result
            figure(fig_Hs(which_plot));
            ylim([0 ystart]);
            
            switch_i = 0;   %did I use a line at all?
            if pvals < 0.05/correction;
                plot([x_position_1 x_position_2],...
                    [line_y_values(iterator(which_plot)) line_y_values(iterator(which_plot))] ,'LineWidth',4,'Color',[0 0 0]);
                switch_i = 1;
            end;
            
            if bf10 < (1/BayesThresh);
                plot([x_position_1 x_position_2],...
                    [line_y_values(iterator(which_plot)) line_y_values(iterator(which_plot))],'LineWidth',2,'Color',[1 0 1]);
                switch_i = 1;
            end;
            
            if bf10 > BayesThresh;
                plot([x_position_1 x_position_2],...
                    [line_y_values(iterator(which_plot)) line_y_values(iterator(which_plot))],'LineWidth',1,'Color',[0 1 0]);
                switch_i = 1;
            end;
            
            %Not very elegent, but I want to control spaces between
            %connector lines and so want to increment an iterator everytime
            %I draw a line and use it to determine location of next line.
            if switch_i == 1;   %If something was drawn on a plot on this round
                
                %Figure out which plot it was drawn on and increement that plot's iterator
                if fig_Hs(which_plot) == h1; i_samples = i_samples + 1;
                elseif fig_Hs(which_plot) == h3; i_ranks = i_ranks + 1;
                else fig_Hs(which_plot) == h2; i_ll = i_ll + 1;
                end;    %test which y position iterator requires updating on this round
                
            end;    %check if a line was drawn
        end;    %loop through however many differences were computed above
    end;    %loop through condition pairs for pairwise tests
    
    
end;    %loop through runs / studies


%Now that you've accumulated data for all studies, add pairwise tests
%comparing samples and ranks between studies to plot h5
pairs = nchoosek(1:size(P_accum_samples,2),2);
num_pairs = size(pairs,1);
[a In] = sort(diff(pairs')','descend');  %lengths of connecting lines
line_pair_order = pairs(In,:);    %move longest connections to top

%Where to put top line (samples and ranks)?
y_inc = .75;
ystart = num_options + num_pairs*y_inc;
line_y_values = ystart:-y_inc:0;

correction = num_pairs;

for pair = 1:num_pairs;
    
    [bf10(1),pValue(1)] = ttest2(P_accum_samples{line_pair_order(pair,1)}',P_accum_samples{line_pair_order(pair,2)}');
    [bf10(2),pValue(2)] = ttest2(P_accum_ranks{line_pair_order(pair,1)}',P_accum_ranks{line_pair_order(pair,2)}');
    
    x_position_1 = line_pair_order(pair,1);
    x_position_2 = line_pair_order(pair,2);
    
    figure(h5)
    
    for graph = 1:2; %samples and ranks
        
        subplot(2,1,graph);
        if pValue(graph) < 0.05/correction;
            plot([x_position_1 x_position_2],...
                [line_y_values(pair) line_y_values(pair)] ,'LineWidth',4,'Color',[0 0 0]);
        end;
        
        if bf10(graph) < (1/BayesThresh);
            plot([x_position_1 x_position_2],...
                [line_y_values(pair) line_y_values(pair)],'LineWidth',2,'Color',[1 0 1]);
        end;
        
        if bf10(graph) > BayesThresh;
            plot([x_position_1 x_position_2],...
                [line_y_values(pair) line_y_values(pair)],'LineWidth',1,'Color',[0 1 0]);
        end;
        
        ylim([0, ystart])
        set(gca,'ytick',[0:2:num_options]);

        
    end;    %loop through graphs
end;    %loop through pairs
















%Now do figure of averages
h4 = figure; set(gcf,'Color',[1 1 1]);  %averages (samples, ll and ranks in rows and averaged groups of studies in cols)


for ave_plot = 1:size(data.averages,2);
    
    %initialise new stuff this loop
    x_axis_it = 1;
    
    %to hold data for pairwise tests and connector lines on plot later
    ave_accum_samples = [];
    ave_accum_ll = [];
    ave_accum_ranks = [];
    
    %participant samples data
    subplot(size(data.averages,2),3,((ave_plot-1)*3)+1); hold on;
    bar( x_axis_it, mean(data.averages(ave_plot).samples), 'LineWidth',bar_line_width,'FaceAlpha',bar_alpha, 'FaceColor', plot_colors(1,:));
    plotSpread( data.averages(ave_plot).samples , 'xValues', x_axis_it, 'distributionColors',plot_colors(1,:) );
    text( x_axis_it, -.5 ...
        ,sprintf('%s','Participants') ...
        ,'Fontname','Arial' ...
        ,'Fontsize',12 ...
        ,'Rotation', x_rot ...
        ,'HorizontalAlignment','right' ...
        );
    
    %participant ranks data
    subplot(size(data.averages,2),3,((ave_plot-1)*3)+2); hold on;
    bar( x_axis_it, mean(data.averages(ave_plot).ranks),'LineWidth',bar_line_width, 'FaceAlpha',bar_alpha, 'FaceColor', plot_colors(1,:));
    plotSpread( data.averages(ave_plot).ranks , 'xValues', x_axis_it, 'distributionColors',plot_colors(1,:) );
    text( x_axis_it, -.5 ...
        ,sprintf('%s','Participants') ...
        ,'Fontname','Arial' ...
        ,'Fontsize',12 ...
        ,'Rotation', x_rot ...
        ,'HorizontalAlignment','right' ...
        );
    
    %Accumulate participant and model data to facilitate plots and pairwise tests
    %on data averaged over studies later
    ave_accum_samples(:,x_axis_it) = data.averages(ave_plot).samples;
    ave_accum_ranks(:,x_axis_it) = data.averages(ave_plot).ranks;
    
    
    for model = 1:size(data.averages(ave_plot).model,2);
        
        x_axis_it = x_axis_it + 1;
        
        %samples
        subplot(size(data.averages,2),3,((ave_plot-1)*3)+1); hold on;
        bar( x_axis_it, mean(data.averages(ave_plot).model(model).samples),'LineWidth',bar_line_width, 'FaceAlpha',bar_alpha, 'FaceColor', plot_colors(data.averages(ave_plot).model(model).models_pick,:));
        plotSpread( data.averages(ave_plot).model(model).samples , 'xValues', x_axis_it, 'distributionColors',plot_colors(data.averages(ave_plot).model(model).models_pick,:) );
        num_options = 12;
        ylim([0 num_options]);
        set(gca,'FontSize',12,'FontName','Arial','ytick',[0:2:num_options],'LineWidth',2,'xtick',[]);
        ylabel('Samples to decision');
        text( x_axis_it, -.5 ...
            ,sprintf('%s',data.averages(ave_plot).model(model).name) ...
            ,'Fontname','Arial' ...
            ,'Fontsize',12 ...
            ,'Rotation', x_rot ...
            ,'HorizontalAlignment','right' ...
            );
        
        %ranks
        subplot(size(data.averages,2),3,((ave_plot-1)*3)+2); hold on;
        bar( x_axis_it, mean(data.averages(ave_plot).model(model).ranks),'LineWidth',bar_line_width, 'FaceAlpha',bar_alpha, 'FaceColor', plot_colors(data.averages(ave_plot).model(model).models_pick,:));
        plotSpread( data.averages(ave_plot).model(model).ranks , 'xValues', x_axis_it, 'distributionColors',plot_colors(data.averages(ave_plot).model(model).models_pick,:) );
        num_options = 12;
        ylim([0 num_options]);
        set(gca,'FontSize',12,'FontName','Arial','ytick',[0:2:num_options],'LineWidth',2,'xtick',[]);
        ylabel('Rank of chosen option');
        text( x_axis_it, -.5 ...
            ,sprintf('%s',data.averages(ave_plot).model(model).name) ...
            ,'Fontname','Arial' ...
            ,'Fontsize',12 ...
            ,'Rotation', x_rot ...
            ,'HorizontalAlignment','right' ...
            );
        
        %ll
        subplot(size(data.averages,2),3,((ave_plot-1)*3)+3); hold on;
        bar( x_axis_it, mean(data.averages(ave_plot).model(model).ll),'LineWidth',bar_line_width, 'FaceAlpha', bar_alpha, 'FaceColor', plot_colors(data.averages(ave_plot).model(model).models_pick,:));
        plotSpread( data.averages(ave_plot).model(model).ll , 'xValues', x_axis_it, 'distributionColors',plot_colors(data.averages(ave_plot).model(model).models_pick,:) );
        set(gca,'FontSize',12,'FontName','Arial','LineWidth',2,'xtick',[]);
        ylabel('Log-likelihood');
        ylim([0 75]);
        text( x_axis_it, -3 ...
            ,sprintf('%s',data.averages(ave_plot).model(model).name) ...
            ,'Fontname','Arial' ...
            ,'Fontsize',12 ...
            ,'Rotation', x_rot ...
            ,'HorizontalAlignment','right' ...
            );
        
        ave_accum_samples(:,x_axis_it) = data.averages(ave_plot).model(model).samples;
        ave_accum_ll(:,x_axis_it-1) = data.averages(ave_plot).model(model).ll;
        ave_accum_ranks(:,x_axis_it) = data.averages(ave_plot).model(model).ranks;
        
    end;    %loop through models for this average plot
    
    %Now go back and add connector bars to indicate pairwise test results for THIS AVERAGE PLOT.
    %Don't forget, you only need connectors for subjects versus each model if
    %measure is sampling or ranks and you want only connectors between models
    %for ll.
    fprintf('')
    pairs = nchoosek(1:size(ave_accum_samples,2),2);
    num_pairs = size(pairs,1);
    [a In] = sort(diff(pairs')','descend');  %lengths of connecting lines
    line_pair_order = pairs(In,:);    %move longest connections to top
    
    %iterators, to standardise y locations of connector lines across graphs
    i_samples = 1;  %for samples
    i_ranks = 1;    %for ranks
    i_ll = 1;   %for log likelihood
    
    for pair = 1:num_pairs;
        
        clear temp;
        
        %Is this a pair involving participants? (needed for samples and ranks)
        if sum(line_pair_order(pair,:) == 1) == 1;
            
            %find difference between vectors for participants and this pair's model
            temp(:,1) = ave_accum_samples(:,line_pair_order(pair,1)) - ave_accum_samples(:,line_pair_order(pair,2));
            temp(:,2) = ave_accum_ranks(:,line_pair_order(pair,1)) - ave_accum_ranks(:,line_pair_order(pair,2));
            
            fig_Hs = [1 2];   %This time, it refers to the subplot column, which holds samples, ll or ranks
            
            %Where to put top line (samples and ranks)?
            y_inc = .5;
            %             ystart = num_options + y_inc*num_pairs + 2*y_inc;
            ystart = num_options + 6*y_inc;
            line_y_values = ystart:-y_inc:0;
            
            x_position_1 = line_pair_order(pair,1);
            x_position_2 = line_pair_order(pair,2);
            
            correction = num_models;
            
            iterator = [i_samples i_ranks];
            
        else;   %If the contrast does NOT involve participants then it must be between models, which means we want an ll contrast instead.
            
            temp = ave_accum_ll(:,line_pair_order(pair,1)-1) - ave_accum_ll(:,line_pair_order(pair,2)-1);
            
            fig_Hs = [3];  %This time, it refers to the subplot column, which holds samples, ll or ranks
            
            %Where to put top line (samples and ranks)?
            y_inc = 5;
            ystart = ll_maxY + 15*y_inc;
            line_y_values = ystart:-y_inc:0;
            
            x_position_1 = line_pair_order(pair,1);
            x_position_2 = line_pair_order(pair,2);
            
            if num_models > 1;
                correction = size(nchoosek(1:num_models,2),2);
            else
                correction = 1;
            end;
            
            iterator = [i_ll];
            
            if line_pair_order(pair,1) == 5 & line_pair_order(pair,2) == 6
                fprintf('');
            end;
            
        end;    %check if col 1 (participants) included in this pair
        
        %Now use temp and fig_Hs to compute and draw stats results
        for which_plot = 1:size(temp,2);
            
            %Get stats for this difference
            clear bf10 pval ci stats;
            
            [bf10,pvals,ci,stats] = ...
                bf.ttest( temp(:,which_plot) );
            
            %plot result
            subplot(size(data.averages,2),3,((ave_plot-1)*3)+fig_Hs(which_plot)); hold on;
            %             figure(fig_Hs(which_plot));
            ylim([0 ystart]);
            
            switch_i = 0;   %did I use a line at all?
            if pvals < 0.05/correction;
                plot([x_position_1 x_position_2],...
                    [line_y_values(iterator(which_plot)) line_y_values(iterator(which_plot))] ,'LineWidth',4,'Color',[0 0 0]);
                switch_i = 1;
            end;
            
            if bf10 < (1/BayesThresh);
                plot([x_position_1 x_position_2],...
                    [line_y_values(iterator(which_plot)) line_y_values(iterator(which_plot))],'LineWidth',2,'Color',[1 0 1]);
                switch_i = 1;
            end;
            
            if bf10 > BayesThresh;
                plot([x_position_1 x_position_2],...
                    [line_y_values(iterator(which_plot)) line_y_values(iterator(which_plot))],'LineWidth',1,'Color',[0 1 0]);
                switch_i = 1;
            end;
            
            %Not very elegent, but I want to control spaces between
            %connector lines and so want to increment an iterator everytime
            %I draw a line and use it to determine location of next line.
            if switch_i == 1;   %If something was drawn on a plot on this round
                
                %Figure out which plot it was drawn on and increement that plot's iterator
                if fig_Hs(which_plot) == 1; i_samples = i_samples + 1;
                elseif fig_Hs(which_plot) == 2; i_ranks = i_ranks + 1;
                else fig_Hs(which_plot) == 3; i_ll = i_ll + 1;
                end;    %test which y position iterator requires updating on this round
                
            end;    %check if a line was drawn
        end;    %loop through however many differences were computed above
    end;    %loop through condition pairs for pairwise tests
    
    
end;    %loop through average plots
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%





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
        fminsearch(  @(params) run_models_wrapper( params, subject, data ), params);
    
    
    %get action values, samples and ranks for a model using best-fitting parameter
    clear ll samples ranks choiceValues;
    [ll samples ranks choiceValues choiceProbs] = ...
        run_models( ...
        data.study(run_study).model(run_model).estimated_params(subject,:), ...
        subject, ...
        data ...
        );
    
    %Assign results
    data.study(run_study).model(run_model).samples(:,subject) = samples;                %seq*subject
    data.study(run_study).model(run_model).ranks(:,subject) = ranks;                    %seq*subject
    data.study(run_study).model(run_model).choiceValues(:,:,:,subject) = choiceValues;  %draw*response(cont/stop)*seq*subject
    data.study(run_study).model(run_model).choiceProbs(:,:,:,subject) = choiceProbs;    %draw*response(cont/stop)*seq*subject

end;    %loop through subjects

 fprintf('\n');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%Just runs run_models, but drops the other outputs and returns only ll,
%%%for use with fminsearch
function ll = run_models_wrapper( params, subject, data )

[ll samples ranks choiceValues choiceProbs] = run_models(params, subject, data);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%







%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Gets a model's performance and ll for one participant,
%whether model fitting or afterwards

function [ll samples ranks choiceValues_all choiceProbs] = run_models(params, subject, data)

%Does model for whatever the last study and model saved in data was (so can be evoked
%in a loop through studies/models

%configure basic info
run_study = size(data.study,2); %which study to process?
run_model = size(data.study(run_study).model,2); %which model to process?
num_subs = size(data.study(run_study).model(run_model).seq_vals,3);
num_seqs = size(data.study(run_study).model(run_model).seq_vals,1);

%configure model inputs
prior.mu = mean(log(data.study(run_study).model(run_model).prior_vals(:,subject)+1));
prior.var = var(log(data.study(run_study).model(run_model).prior_vals(:,subject)+1));
if prior.var == 0;
    prior.var = eps;    %Once a subject hit 1 for every face. eps is a tiny nonzero number barely resolvable by matlab float precision
end;
prior.kappa= 2;
prior.nu = 1;

ll = 0;
for seq = 1:num_seqs;
    
    clear list choiceCont choiceStop difVal
    list.vals = log(data.study(run_study).model(run_model).seq_vals(seq,:,subject)+1);
    list.length = numel(list.vals);
    list.Cs = 0;
    list.payoff_scheme = data.study(run_study).model(run_model).payoff_scheme;
    list.flip = 1;
    
    [choiceStop, choiceCont, difVal] = analyzeSecretary_nick_2023a(prior,list);

    %update ll for this sequence
    
    %I need number of draws for this subject and sequence to compare against model
    listDraws = data.study(run_study).samples(seq,subject);
    
    choiceValues = [choiceCont; choiceStop]';
    
    %softmax
    b = exp(params);
    for drawi = 1 : size(choiceValues,1);
        %cprob seqpos*choice(draw/stay)
        cprob(drawi, :) = exp(b*choiceValues(drawi, :))./sum(exp(b*choiceValues(drawi, :)));
    end;
    
%     cprob(end,2) = Inf; %ensure stop choice on final sample.
    
    %accumulate ll
    if listDraws == 1;  %If only one draw
        ll = ll - 0 - log(cprob(listDraws, 2)); %take the log of just the option decision probability
    else
        ll = ll - sum(log(cprob((1:listDraws-1), 1))) - log(cprob(listDraws, 2));
    end;
    
%     %ranks for this sequence
    seq_ranks = tiedrank(list.vals')';
%     
%     %Now get samples from uniform distribution
%     test = rand(1000,size(list.vals,2));
%     for iteration = 1:size(test,1);
%         
%         samples_this_test(iteration) = find(cprob(:,2)'>test(iteration,:),1,'first');
%         ranks_this_test(iteration) = seq_ranks( samples_this_test(iteration) );
%         
%     end;    %iterations
    
    %samples(seq,1) = round(mean(samples_this_test));
    %ranks(seq,1) = round(mean(ranks_this_test));
    samples(seq,1) = find(difVal<0,1,'first');
    ranks(seq,1) = seq_ranks( samples(seq,1) );
    choiceValues_all(:,:,seq) = choiceValues;
    choiceProbs(:,:,seq) = cprob;
    
end;    %loop through seqs

%data.study(run_study).model(run_model).ll(subject,1) = ll;


%Now get performance: samples and ranks
%data.study(run_study).model(run_model).samples(seq,subject) = find(difVal<0,1,'first');
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
payoff = (payoff-log(1))/(log(101) - log(1));

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



if study == 2;  %baseline pilot
    data_folder = 'pilot_baseline';
elseif study == 3;  %full pilot
    data_folder = 'pilot_full';
    ratings_phase = 1;
    header_format = 2;     option_chars = 2;
    %payoff_scheme = 1;
elseif study == 4;  %baseline
    data_folder = 'baseline';
elseif study == 5;  %full
    data_folder = 'full';
    ratings_phase = 1;
    header_format = 2;     option_chars = 1;
    %payoff_scheme = 1;
elseif study == 6;  %rating phase
    data_folder = 'rating_phase';
    ratings_phase = 1;
elseif study == 7;  %squares
    data_folder = 'squares';
elseif study == 8;  %timimg
    data_folder = 'timing';
elseif study == 9;  %payoff (aka stars)
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
subs = unique(sequence_data_concatenated.ParticipantPrivateID,'stable');    %Get sub numbers but MAKE SURE TO MAINTAIN THEIR CORRECT ORDER (i.e., stable) so they line up correctly with number of draws or ranks and model performance will be miscomputed.
num_subs = numel(subs);
for subject = 1:num_subs
    
    disp(sprintf('Study id %d, subjective vals %d, payoff scheme %d, participant %d',study,subjective_vals,payoff_scheme, subs(subject)));
    
    
    %Get objective values for this subject
    array_Obj = table2array(sequence_data_concatenated(sequence_data_concatenated.ParticipantPrivateID==subs(subject),5:end));
    
    %loop through and get io peformance for each sequence
    for sequence = 1:size(array_Obj,1);
        
        if subjective_vals == 1;   %if subjective values
            
            %Subject who rated almost every face a 1 in study 4 (full)
            if subject == 34;
                fprintf('');
            end;
            
            %Loop through options and replace price values with corresponding ratings for each participant
            clear this_rating_data this_seq_Subj;
            this_rating_data = mean_ratings(mean_ratings.ParticipantPrivateID == subs(subject),:);
            for option=1:size(array_Obj,2);
                
                this_seq_Subj(1,option) = table2array(this_rating_data(this_rating_data.phone_price==array_Obj(sequence,option),'mean_Response'));
                
            end;    %loop through options
            
            %I've subtracted a 1 so that both prices and ratings will then
            %be on a 0 to 100 scale. And then I can take log(value+1) of
            %both and the smallest log transformed value of both will be zero.
            all_ratings(:,subject) = this_rating_data.mean_Response - 1; %to be returned by function
            seq_vals(sequence,:,subject) = this_seq_Subj - 1; %to be returned by function
            
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
        seq_ranks_temp = tiedrank(seq_vals(sequence,:,subject));
        seq_ranks(sequence,subject) = seq_ranks_temp(all_output(sequence,subject));
        
        
    end;    %Loop through sequences
end;    %Loop through subs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



