#!/home/lunpin/conda/bin/python3

import argparse, csv, datetime, gc, gzip, json, math, matplotlib, numpy, os, random, sys, time
matplotlib.use ('Agg')
import matplotlib.pyplot
import util.util as util


###############################################################
##                           Main                            ##
###############################################################
def main ():
    timestamp = datetime.datetime.now ()
    operation = {
        'behsa': [BehavioralHeatmapResultsAnalysis, ],
        'acobe': [BehavioralHeatmapAnomalyDetection, BehavioralHeatmapResultsAnalysis, ],
        'behra': [BehavioralHeatmapReportAnomaly, ],
        'genbh': [GenerateFeaturesForBehavioralHeatmaps, ], 
        'label': [GenerateLabelsFromOriginalData, ], 
        'parse': [ParseFromOriginalData, ], 
        'testf': [TestFunctions, ], }
    parser = argparse.ArgumentParser (description='Experiment based on CERT 2016 dataset')
    parser.add_argument ('operation', help='Specify an operation (' + ', '.join (operation.keys ()) + ')')
    # Anomaly Detection
    parser.add_argument ('-i', '--input', help='Specify a input file or directory that contains data (e.g., mydata/)', default=['data'], nargs='+')
    parser.add_argument ('-o', '--output', help='Specify an output file to write to', default=timestamp.strftime ('%Y_%m_%d_%H_%M_%S'))
    parser.add_argument ('-l', '--label', help='Specify a labeled ldap file', default='labeledldap.txt')
    parser.add_argument ('-m', '--model', help='Specify a deep learning model', default='ae2')
    parser.add_argument ('-g', '--group', help='Speicfy a group granularity (0: bu, 1: fu, 2: dept, 3: team)', type=int, default=0)
    parser.add_argument ('-v', '--verbose', help='Toggle Verbose', action='store_true')
    parser.add_argument ('--version', help='Specify the feature file for anomaly detection', default='v1')
    parser.add_argument ('--use-gpu', help='Use GPU id (default: all GPUs), put -1 for not using GPU', nargs='+', type=int)
    parser.add_argument ('--train-start', help='Specify the starting date of building heatmaps (e.g., 11/01/2010)', default='11/01/2010')
    parser.add_argument ('--train-end', help='Specify the ending date of building heatmaps (e.g., 12/31/2010)', default='12/31/2010')
    parser.add_argument ('--test-start', help='Specify the starting date of building heatmaps (e.g., 01/01/2011)', default='01/01/2011')
    parser.add_argument ('--test-end', help='Specify the ending date of building heatmaps (e.g., 01/31/2011)', default='01/31/2011')
    parser.add_argument ('--sample', help='Sample only N negatives (for FPR)', type=int, default=9999)
    # Behavioral Heatmap 
    parser.add_argument ('--beh-window', help='Specify the window size in terms of days (default=# feature)', type=int, default=None)
    parser.add_argument ('--beh-slice', help='Specify the number of slices per each day', type=int, default=2)
    parser.add_argument ('--beh-sigma', help='Specify the sigma value', type=float, default=3.0)
    parser.add_argument ('--beh-feature', help='Specify feature keywords', nargs='+')
    parser.add_argument ('--beh-weight', help='Toggle applying weights to features (0: no-weight, 1: log-occurance, 2: mean-abs-sigma, 3+: std', type=int, default=0)
    parser.add_argument ('--beh-staticmeanstd', help='Toggle to apply static mean and std from trainset to testset', action='store_true')
    parser.add_argument ('--beh-reportthres', help='Reporting threshold for anomalies', type=float, default=5.0)
    parser.add_argument ('--beh-nogroupbeh', help='Toggle to exclude group behavior in heatmaps', action='store_true')
    # Parser
    args = parser.parse_args ()
    if len (args.input) == 1: args.input = args.input [0]
    for op in operation [args.operation]: 
        gc.collect ()
        op (args)



###############################################################
##             Behavioral Heatmap Report Anomaly             ##
###############################################################
def BehavioralHeatmapReportAnomaly (args):
    # preprocessing date information
    anomalies = {} # anomalies [group][user]
    labels = {}
    # for each input directory
    for din in args.input:
        # for each group 
        for group in os.listdir (din):
            evaluation = {} # evaluation [model][uid]
            groupdir = os.path.join (din, group)
            if not os.path.isdir (os.path.join (groupdir, 'results')): continue
            # read labels (csv)
            for line in csv.reader (open (os.path.join (groupdir, args.label), 'r')):
                user = GeneratedLabel (line)
                labels [user.uid] = user
            # read results (json)
            for filename in os.listdir (os.path.join (groupdir, 'results')):
                if '.result.log' not in filename [-11: ]: continue
                for line in open (os.path.join (groupdir, 'results', filename), 'r'):
                    uid, score, date, model = json.loads (line)
                    if model not in evaluation: evaluation [model] = {}
                    if uid not in evaluation [model]: evaluation [model][uid] = []
                    evaluation [model][uid].append (score)
            # compare to mean and std
            group_anom = {}
            for model in evaluation:
                # build mean and std from all the data points
                means, stds, scores = {}, {}, []
                for uid in evaluation [model]: 
                    scores.extend (evaluation [model][uid])
                    means [uid] = numpy.mean (evaluation [model][uid])
                    stds [uid] = numpy.std (evaluation [model][uid])
                mean = numpy.mean (scores)
                std = numpy.std (scores)
                # compare to mean and std
                for uid in evaluation [model]:
                    for score in evaluation [model][uid]:
                        if uid not in group_anom: group_anom [uid] = 0
                        if score - mean > args.beh_reportthres * std: 
                            # if score - means [uid] > args.beh_reportthres * stds [uid]:
                            group_anom [uid] += 1
            # write to anomalies if more than N models reported anomaly
            for uid in group_anom:
                if group_anom [uid] >= len (evaluation):
                    if group not in anomalies: anomalies [group] = {}
                    if uid not in anomalies [group]: anomalies [group][uid] = 0
                    anomalies [group][uid] += 1
    # deduce anomalies based on votings from inputs
    for group in anomalies:
        for uid in anomalies [group]:
            if anomalies [group][uid] > 1:
                print (uid)
                            


###############################################################
##            Behavioral Heatmap Results Analysis            ##
###############################################################
def BehavioralHeatmapResultsAnalysis (args):
    # preprocessing date information
    index_to_date = enumdates ('01/02/2010', '05/31/2011')
    date_to_index = {d: i for i, d in enumerate (index_to_date)}
    test_dates = enumdates (args.test_start, args.test_end)
    # for each group 
    for group in os.listdir (args.input):
        evaluation = {} # evaluation [date][model]
        groupdir = os.path.join (args.input, group)
        if not os.path.isdir (os.path.join (groupdir, 'results')): continue
        # read labels (csv)
        labels, normals, abnormals = {}, set (), set ()
        for line in csv.reader (open (os.path.join (groupdir, args.label), 'r')):
            user = GeneratedLabel (line)
            labels [user.uid] = user
            if user.isAbnormal (): abnormals.add (user.uid)
            else: normals.add (user.uid)
        # read results (json)
        for filename in os.listdir (os.path.join (groupdir, 'results')):
            if '.result.log' not in filename [-11: ]: continue
            for line in open (os.path.join (groupdir, 'results', filename), 'r'):
                uid, score, date, model = json.loads (line)
                obj = {'uid': uid, 'score': score, 'date': date, 'model': model}
                if date not in evaluation: evaluation [date] = {}
                if model not in evaluation [date]: evaluation [date][model] = [] 
                evaluation [date][model].append (obj)
        # sort evaluation
        for date in evaluation:
            for model in evaluation [date]:
                evaluation [date][model] = sorted (evaluation [date][model], key=lambda obj: obj ['score'], reverse=True)
        models = sorted (evaluation [date].keys ())
        date_index = [date_to_index [date] for date in evaluation]
        start_index, end_index = min (date_index), max (date_index) + 1
        dates = index_to_date [start_index: end_index]
        if True: # derive trends over dates
            fontsize, linewidth, markersize = 6, 0.5, 0.5
            trends = {} # rank trends [uid][model][date]
            for date in evaluation:
                for model in evaluation [date]:
                    for rank, user in enumerate (evaluation [date][model]): 
                        uid = user ['uid']
                        if uid not in trends: trends [uid] = {}
                        if model not in trends [uid]: trends [uid][model] = {}
                        trends [uid][model][date] = {
                            'rank': rank + 1,
                            'score': user ['score']}
            # select users for later plots
            userlist = set () 
            for m in models: 
                userlist |= set ([evaluation [dates [-1]][m][0]['uid']]) # the most abnormal user on the last day
                userlist |= set ([evaluation [dates [-1]][m][-1]['uid']]) # the least abnormal user on the last day
                if args.sample > 0:
                    allusers = set ([evaluation [dates [-1]][m][i]['uid'] for i in range (0, len (evaluation [dates [-1]][m]))])
                    allusers = list (allusers - userlist)
                    random.shuffle (allusers)
                    allusers = allusers [: args.sample - 1]
                    userlist |= set (allusers)
            userlist = list (userlist - abnormals) + list (abnormals)
            # plot trends (for selective users + anomalous users)
            axisX = numpy.arange (len (dates))
            models = sorted (evaluation [dates [0]].keys ())
            rows, columns = len (models), 2
            output = os.path.join (groupdir, '_'.join  (['user', 'trends']))
            fig, axs = matplotlib.pyplot.subplots (nrows=rows, ncols=columns, sharex=True)
            for index, model in enumerate (models):
                # ranks
                ax = axs [index][0] if len (models) > 1 else axs [index]
                for uid in userlist: 
                    if uid not in trends: continue
                    uid_label = uid if uid in abnormals else None
                    lw = linewidth * 3 if uid in abnormals else linewidth
                    ax.plot (axisX, [trends [uid][model][index_to_date [d]]['rank'] for d in range (start_index, end_index)], 
                        '-o', lw=lw, markersize=markersize, label=uid_label)
                    if uid in abnormals: # emphasize on compromise dates
                        compromises = labels [uid].compromises
                        for d in compromises: 
                            if d not in dates: continue
                            ax.scatter (date_to_index [d] - start_index, trends [uid][model][d]['rank'], color='red')
                ax.set_title ('_'.join ([model, 'ranks']) + ' (' + str (len (labels)) +' users)', fontsize=fontsize)
                ax.set_xticks (axisX); ax.set_xticklabels (dates, fontsize=fontsize, rotation=45, ha='right', rotation_mode='anchor')
                ax.yaxis.set_tick_params (labelsize=fontsize)
                bottom, top = ax.get_ylim (); ax.set_ylim (top, bottom)
                # scores
                ax = axs [index][1] if len (models) > 1 else axs [index + 1]
                for uid in userlist:
                    if uid not in userlist: continue
                    uid_label = uid if uid in abnormals else None
                    lw = linewidth * 3 if uid in abnormals else linewidth
                    ax.plot (axisX, [trends [uid][model][index_to_date [d]]['score'] for d in range (start_index, end_index)],
                        '-o', lw=lw, markersize=markersize, label=uid_label)
                    if uid in abnormals: # emphasize on compromise dates
                        compromises = labels [uid].compromises
                        for d in compromises: 
                            if d not in dates: continue
                            ax.scatter (date_to_index [d] - start_index, trends [uid][model][d]['score'], color='red')
                ax.set_title ('_'.join ([model, 'scores']), fontsize=fontsize)
                ax.set_xticks (axisX); ax.set_xticklabels (dates, fontsize=fontsize, rotation=45, ha='right', rotation_mode='anchor')
                ax.yaxis.set_tick_params (labelsize=fontsize)
            ax.legend (fontsize=fontsize)
            fig.tight_layout ()
            matplotlib.pyplot.savefig (output + '.png', dpi=600)
            matplotlib.pyplot.close (fig)
        if True: # plot heatmaps (anomalous users)
            import acobe.util
            bhinputdir = os.path.join (groupdir, 'heatmaps.raw')
            bhoutputdir = os.path.join (groupdir, 'heatmaps')
            os.makedirs (bhoutputdir, exist_ok=True)
            for filename in os.listdir (bhinputdir):
                _, uid, date = filename.split ('_')
                date = date.replace ('-', '/')
                if True: # print heatmaps on the targeted dates
                    if date not in dates: continue
                    if dates.index (date) % 30 != 0: continue
                if uid in abnormals and date in dates:
                    heatmap = acobe.util.Heatmap.load (os.path.join (bhinputdir, filename))
                    heatmap.plot (bhoutputdir) 
        if False: # plot correlations upon different models per day
            fontsize, linewidth, markersize = 4, 0.5, 0.2
            outputdir = os.path.join (groupdir, 'models')
            os.makedirs (outputdir, exist_ok=True)
            def getPlotData (evalX):
                userX = [obj ['uid'] for obj in evalX]
                scoreX = [obj ['score'] for obj in evalX]
                maxscore, minscore = max (scoreX), min (scoreX)
                scoreX = [len (userX) * (1.0 - (score - minscore) / (maxscore - minscore)) for score in scoreX]
                return userX, scoreX
            for date in evaluation:
                if len (evaluation [date]) < 2: continue
                title = '_'.join ([group, 'models', date.replace ('/', '-')])
                output = os.path.join (outputdir, date.replace ('/', '-'))
                fig, axs = matplotlib.pyplot.subplots (nrows=len (models), ncols=len (models), sharey=True)
                for i, modelX in enumerate (models):
                    userX, scoreX = getPlotData (evaluation [date][modelX])
                    axisX = numpy.arange (len (userX))
                    for j, modelY in enumerate (models):
                        userY, scoreY = getPlotData (evaluation [date][modelY])
                        axisY = numpy.arange (len (userY))
                        # plot
                        ax = axs [i][j]
                        ax.set_title ('X: ' + modelX + '\nY: ' + modelY, fontsize=fontsize)
                        ax.set_xticks ([axisX [0], axisX [-1]])
                        ax.set_xticklabels (['anom', 'norm'], fontsize=fontsize)
                        ax.yaxis.set_tick_params (labelsize=fontsize)
                        # plot rank correlation
                        ax.plot (axisX, [userY.index (userX [x]) for x in axisX], 'ro', markersize=markersize)
                        # plot score correlation
                        ax.plot (scoreX, [scoreY [userY.index (userX [x])] for x in axisX], 'bo', markersize=markersize) 
                        # plot score X
                        ax.plot (axisX, scoreX, 'y-', lw=linewidth)
                        # plot score Y
                        ax.plot (scoreY, axisY, 'c-', lw=linewidth)
                        # plot rank difference
                        ax.plot (axisX, [2 * len (set (userY [: x]) - set (userX [: x])) for x in axisX], 'k-', lw=linewidth)
                fig.tight_layout ()
                matplotlib.pyplot.savefig (output + '.png', dpi=600)
                matplotlib.pyplot.close (fig)  
        

        
###############################################################
##           Behaviral Heatmap Anomaly Detection             ##
###############################################################
def BehavioralHeatmapAnomalyDetection (args):
    import tensorflow 
    from acobe.util import Heatmap
    from acobe.model import Autoencoder1, Autoencoder2, AnoGAN
    gc.collect () 
    gpus = tensorflow.config.experimental.list_physical_devices ('GPU') 
    if args.use_gpu is None: lgpus = gpus
    elif len (args.use_gpu) == 1 and args.use_gpu [0] == -1: lgpus = []
    else: lgpus = [gpus [i] for i in args.use_gpu]
    tensorflow.config.experimental.set_visible_devices (devices=lgpus, device_type='GPU')
    for i in range (0, len (gpus)): tensorflow.config.experimental.set_memory_growth (gpus [i], True)
    print ('\033[91mUsing GPU\n', '\n'.join ([str (gpu) for gpu in lgpus]), '\033[0m')
    # preprocessing date information
    date_to_index = {d: i for i, d in enumerate (enumdates ('01/02/2010', '05/31/2011'))}
    train_dates = enumdates (args.train_start, args.train_end)
    test_dates = enumdates (args.test_start, args.test_end)
    slices = args.beh_slice
    # selected features
    if args.beh_feature is not None and len (args.beh_feature) > 0:
        print ('[ \033[32mGiven keywords for features:\033[0m ]')
        for f in args.beh_feature: print ('    ' + f)
    # read labels 
    progress, labels, bar = 0, {}, util.ProgressBar ('Read Anomaly Labels', 4000)
    for line in csv.reader (open (args.label, 'r')):
        progress += 1; bar.update (progress)
        user = GeneratedLabel (line)
        assigned_group = ' / '.join (user.groups [: args.group + 1])
        if assigned_group not in labels: labels [assigned_group] = {}
        labels [assigned_group][user.uid] = user
    bar.finish ()
    # sample only N normal users 
    if args.sample > 0:
        for group in labels:
            normals, abnormals = set (), set ()
            for uid in labels [group]:
                if labels [group][uid].isNormal (): normals.add (uid)
                else: abnormals.add (uid)
            normals = list (normals)
            random.shuffle (normals)
            normals = list (normals) [: args.sample]
            for uid in list (labels [group].keys ()):
                if uid not in normals and uid not in abnormals: del (labels [group][uid])     
    # read labels and build heatmap groups
    features, fset, progress, bar = {}, set (), 0, util.ProgressBar ('Preprocess Features', sum (len (labels [group]) for group in labels))
    for group in labels:
        for uid in labels [group]:
            progress += 1
            bar.update (progress)
            # feature fin = [fname][date][hrs]
            fin, fout = json.loads (open (os.path.join (args.input, uid, args.version + '.features'), 'r').read ()), {}
            for f in fin:
                # filter features and add to global yaxis
                if args.beh_feature is not None and len (args.beh_feature) > 0:
                    if not any (fkey in f for fkey in args.beh_feature): continue
                fset.add (f)
                # preprocess daily slicing
                for date in fin [f]: 
                    hrs = fin [f][date]
                    hrs = hrs [6: 18] + hrs [18: ] + hrs [: 6] # shuffle working/off hours (from 0-24 to 6-18 and 18-6)
                    hrs = [sum (hrs [i * 24 // slices: (i + 1) * 24 // slices]) for i in range (0, slices)] # slicing
                    fin [f][date] = hrs
                # redefine feature fout = [fname][hrs][date]
                fout [f] = [] 
                for hr in range (0, slices):
                    hrs = [0] * len (date_to_index)
                    for date in fin [f]:
                        if date not in date_to_index: continue # exclude starting/ending date of the CERT2016 dataset
                        hrs [date_to_index [date]] = fin [f][date][hr]
                    fout [f].append (hrs)
            if group not in features: features [group] = {}
            features [group][uid] = fout
    bar.finish ()
    fset = sorted (fset)
    print ('\033[91m[ Features for Behavioral Heatmaps ]')
    print (json.dumps (fset, indent=4))
    print ('Totally', len (fset), 'features\033[0m')    
    # build heatmaps 
    window = len (fset) if args.beh_window is None else args.beh_window
    heatmaps = {}
    def getWeights (group, dates, base=10):
        dates = enumdates (dateback (dates [0], window), dates [0]) + dates [1: ]
        while dates [0] not in date_to_index: dates = dates [1: ]
        while dates [len (dates) - 1] not in date_to_index: dates = dates [: len (dates) - 1]
        idx_start = date_to_index [dates [0]]
        idx_end = date_to_index [dates [len (dates) - 1]] + 1
        weights, bar = {}, util.ProgressBar ('Build Weights', len (features [group]))
        for progress, uid in enumerate (features [group]):
            weightfactor = {}
            weights [uid] = {}
            fu = features [group][uid] # feature [group][fname][hrs][date]
            for f in fset: 
                weightfactor [f] = [] 
                for hr in range (0, slices):
                    occurance = fu [f][hr][idx_start: idx_end] if f in fu else [0] * (len (dates))
                    factor = sum (occurance) + 1
                    weightfactor [f].append (factor)
            # scale = math.log (sumfactor / minfactor, base) # scale = largest weight
            sumfactor = sum (weightfactor [f])
            scale = math.log (sumfactor / min (weightfactor [f]), base)
            for f in fset:
                weights [uid][f] = []
                for hr in range (0, slices):
                    f_weight = math.log (sumfactor / weightfactor [f][hr], base)
                    weights [uid][f].append (f_weight / scale)
        return weights
    def getHeatmapsFromGroup (group, dates, groupdir=None, weight_mode=None, meanstd_mode=None, groupbeh=True): 
        dates = enumdates (dateback (dates [0], window), dates [0]) + dates [1: ]
        while dates [0] not in date_to_index: dates = dates [1: ]
        while dates [len (dates) - 1] not in date_to_index: dates = dates [: len (dates) - 1]
        idx_start = date_to_index [dates [0]]
        idx_end = date_to_index [dates [len (dates) - 1]] + 1
        # build deviation [uid][fname][hrs][date]
        deviation, bar = {}, util.ProgressBar ('Build Behavioral Deviation Matrix', len (features [group]))
        weight = {}
        meanstd = {}
        for progress, uid in enumerate (features [group]):
            bar.update (progress)
            fu = features [group][uid] # feature [group][fname][hrs][date]
            deviation [uid] = {}
            weight [uid] = {}
            meanstd [uid] = {}
            # transform occ to (0, 1] if window == 1 (no deviation and no weights)
            if window <= 1: # liuliu 
                # copy occurance to deviation
                for f in fset:
                    deviation [uid][f] = []
                    for hr in range (0, slices):
                        occurance = fu [f][hr][idx_start: idx_end] if f in fu else [0] * (len (dates))
                        deviation [uid][f].append (occurance)
                # normalize to [0, 1] on dates
                for date in range (0, len (dates)):
                    maxocc = 0.0000001
                    for f in fset:
                        for hr in range (0, slices):
                            maxocc = max (maxocc, deviation [uid][f][hr][date])
                    for f in fset:
                        for hr in range (0, slices):
                            deviation [uid][f][hr][date] /= maxocc
            # calculate approaximate deviation if window > 1
            else:                 
                # derive behavioral deviations and apply weights
                for f in fset:
                    deviation [uid][f] = []
                    weight [uid][f] = []
                    meanstd [uid][f] = [] 
                    for hr in range (0, slices):
                        ## approaximate deviation for efficiency
                        occurance = fu [f][hr][idx_start: idx_end] if f in fu else [0] * (len (dates))
                        devs = []
                        # table for sum
                        summation = [sum ([occ for occ in occurance [0: window - 1]])] * (window - 1) # exclude the last day already
                        for i in range (window - 1, len (dates)): summation.append (summation [-1] - occurance [i - window + 1] + occurance [i])
                        # table for mean
                        mean = [s / (window - 1) for s in summation]
                        # table for factor = (x - m) ** 2
                        factor = [(occurance [i] - mean [i]) ** 2 for i in range (0, len (occurance))]
                        # table for summation of factors
                        sfactor = [sum (factor [0: window - 1])] * (window - 1) # exclude the last day already
                        for i in range (window - 1, len (factor)): 
                            sfactor.append (sfactor [-1] - factor [i - window + 1] + factor [i])
                            if sfactor [-1] < 0.0: sfactor [-1] = 0.0 # floating error
                        # calculate std iteratively
                        std = []
                        for i in range (0, len (sfactor)): 
                            s = numpy.sqrt (sfactor [max (i - 1, 0)] / (window - 1))
                            s = 0.001 if s < 0.001 else s # avoid divide by zero
                            std.append (s)
                            if meanstd_mode is not None and isinstance (meanstd_mode, dict):
                                dev = (occurance [i] - meanstd_mode [uid][f][hr]['mean']) / meanstd_mode [uid][f][hr]['std'] # sigma (static meanstd)
                            else: dev = (occurance [i] - mean [max (i - 1, 0)]) / std [i] # sigma (sliding meanstd)
                            dev = max (-1.0, min (dev / args.beh_sigma, 1.0)) # normalize-and-bound sigma (-1, 1)
                            devs.append (dev)
                        # calculate weights
                        if weight_mode is None or weight_mode == 0: w = 1
                        elif isinstance (weight_mode, dict): w = weight_mode [uid][f][hr]
                        elif weight_mode == 2: w = 1 / (4 * numpy.mean (numpy.abs (devs)))
                        elif weight_mode == 3: w = 1 / math.log (max (2, numpy.mean (std)), 2)
                        elif weight_mode == 4: w = 1 / max (1, numpy.mean (std))
                        weight [uid][f].append (min (w, 1.0))
                        # static mean-and-std
                        meanstd [uid][f].append ({'mean': numpy.mean (mean), 'std': numpy.mean (std), })
                        # apply weights and rescale to (0, 1)
                        for i in range (0, len (devs)):
                            dev = devs [i]
                            dev = dev * weight [uid][f][hr] # apply weight
                            dev = (dev + 1) / 2 # scale sigma to (0, 1)
                            devs [i] = dev
                        deviation [uid][f].append (devs)
        bar.finish ()
        # build heatmaps [date] 
        heatmaps, bar = {}, util.ProgressBar ('Build Behavioral Heatmap', len (deviation))
        for progress, uid in enumerate (deviation):
            bar.update (progress)
            du = deviation [uid]
            for end in range (window - 1, len (dates)):
                # build bitmap 
                bitmap, xaxis, yaxis = [], dates [end - window + 1: end + 1] * slices, fset
                for f in du: 
                    row = []
                    for hr in range (0, slices): row.extend (du [f][hr][end - window + 1: end + 1])
                    bitmap.append (row)
                index = dates [end]
                if index not in heatmaps: heatmaps [index] = {}
                heatmaps [index][uid] = Heatmap (
                    bitmap=bitmap, 
                    name=uid, 
                    index=dates [end], 
                    group=group, 
                    xaxis=xaxis,
                    yaxis=yaxis)
        bar.finish ()
        # build average behavior and redefine-and-plot heatmaps 
        if groupbeh:
            bar = util.ProgressBar ('Refine Heatmaps with Group Behaviors (' + str (group) + ')', len (heatmaps))
            for progress, index in enumerate (heatmaps):
                bar.update (progress)
                gbh = sum ([heatmaps [index][uid] for uid in heatmaps [index]]) / len (heatmaps [index])
                for uid in heatmaps [index]: 
                    heatmaps [index][uid] = Heatmap.concat (heatmaps [index][uid], gbh)
            bar.finish ()
        # save heatmaps to group dir
        if groupdir is not None:
            heatmap_dir = os.path.join (groupdir, 'heatmaps.raw'); os.makedirs (heatmap_dir, exist_ok=True)
            bar = util.ProgressBar ('Refine Heatmaps with Group Behaviors (' + str (group) + ')', len (heatmaps))
            for progress, index in enumerate (heatmaps):
                bar.update (progress)
                for uid in heatmaps [index]: 
                    heatmaps [index][uid].save (heatmap_dir)
            bar.finish ()
        # save heatmaps to group dir
        if meanstd_mode is None or meanstd_mode == False: meanstd = None
        return heatmaps, weight, meanstd
    # build training arguments and functions
    class ModelArguments (object):
        def __init__ (self, verbose):
            self.verbose = verbose
    modelargs, model = ModelArguments (verbose=args.verbose), None
    def train_and_test (trainX, testX):
        height = len (fset) * 2 if not args.beh_nogroupbeh else len (fset)
        width = window * slices
        # autoencoder1
        if model_name in ['ae1', 'autoencoder1']:
            model = Autoencoder1 (height=height, width=width, args=modelargs)
            model.train (X=list (trainX), epochs=64, batchsize=32)
            scores, syns = model.test (X=list (testX))
        # autoencoder2
        if model_name in ['ae2', 'autoencoder2']:
            if True: # color-channel methods
                print ('Shape before color-transformation', trainX [0].shape)
                for i in range (0, len (trainX)): trainX [i] = trainX [i].toColormap () 
                for i in range (0, len (testX)): testX [i] = testX [i].toColormap ()
                print ('Shape after color-transformation', trainX [0].shape)
                print ('# of Channels', trainX [0].shape [2])                               
                # trainX [0].plot ('./', 'colormap', raw=True)
                shape = trainX [0].shape
                model = Autoencoder2 (height=shape [0], width=shape [1], args=modelargs, channels=shape [2])
            else: model = Autoencoder2 (height=height, width=width, args=modelargs, channels=1)
            model.train (X=list (trainX), epochs=16, batchsize=32)
            scores, syns = model.test (X=list (testX))
        # anogan
        if model_name in ['gan', 'anogan']:
            os.environ ['TF_CPP_MIN_LOG_LEVEL'] = '2'
            os.environ ['KMP_WARNINGS'] = 'off'
            tensorflow.compat.v1.logging.set_verbosity (tensorflow.compat.v1.logging.ERROR)
            model = AnoGAN (height=height, width=width, args=modelargs)
            model.train (X=list (trainX), epochs=4, batchsize=32)
            scores, syns = model.test (X=list (testX))
        # evaluate
        evaluation = sorted (enumerate (scores), key=lambda tup: tup [1], reverse=True)
        for i in range (0, len (evaluation)):
            index, score = evaluation [i]
            fig = testX [index]
            name = fig.name
            date = fig.index
            evaluation [i] = [score, name, date]
        # cleanup
        if model is not None: 
            import tensorflow.keras as keras
            keras.backend.clear_session ()
            del (model); gc.collect ()
        return evaluation
    # build train/test set and train/test the model
    for group in labels:
        # make group directory
        groupdir = os.path.join (args.output, 'beh_group_' + group.replace (' ', '').replace ('_','').replace ('/', '_'))
        os.makedirs (groupdir, exist_ok=True)
        # continue if completed (do not overwrite)
        files = os.listdir (groupdir)
        if 'user_trends.png' in files: continue
        # Training/Testing set
        with open (os.path.join (groupdir, args.label), 'w') as fout:
            for uid in labels [group]: fout.write (labels [group][uid].csvstr ()) 
        if args.beh_weight == 0: weights = None
        elif args.beh_weight == 1: weights = getWeights (group, train_dates)
        else: weights = args.beh_weight
        print ('[\033[32m Build Training Set', group, '\033[0m]')
        trainset, weights, meanstd = getHeatmapsFromGroup (group, train_dates, groupdir=groupdir, weight_mode=weights, meanstd_mode=args.beh_staticmeanstd, groupbeh=not args.beh_nogroupbeh)
        print ('[\033[32m Build Testing Set', group, '\033[0m]')
        testset, _, _ = getHeatmapsFromGroup (group, test_dates, groupdir=groupdir, weight_mode=weights, meanstd_mode=meanstd, groupbeh=not args.beh_nogroupbeh)
        if args.model == 'all': model_list = ['ae1', 'ae1', 'gan']
        elif args.model == 'aes': model_list = ['ae1', 'ae2']
        else: model_list = [args.model]
        for model_name in model_list: 
            print ('\033[32m[ Train/Test on Group ' + group + ' (' + model_name + ') ]\033[0m')
            evaluation, bar = [], util.ProgressBar ('Train and Test (' + group + ')', len (labels [group]))
            for progress, uid in enumerate (labels [group]):
                bar.update (progress)
                trainX, testX = [], []
                for index in trainset: trainX.append (trainset [index][uid])
                for index in testset: testX.append (testset [index][uid])
                evaluation.extend (train_and_test (trainX, testX))
            bar.finish ()
            evaluation = sorted (evaluation, key=lambda tup: tup [0], reverse=True)
            index_output = {}
            for score, name, index in evaluation:
                if index not in index_output: index_output [index] = []
                index_output [index].append ([name, score, index, model_name])
            for date in test_dates:
                if date not in index_output: continue
                os.makedirs (os.path.join (groupdir, 'results'), exist_ok=True)
                with open (os.path.join (groupdir, 'results', '.'.join ([date.replace ('/', '-'), model_name, 'result.log'])), 'w') as fout:
                    for obj in index_output [date]: fout.write (json.dumps (obj) + '\n')
        # clean
        del (trainX)
        del (testX)
        gc.collect ()
        with open (os.path.join (groupdir, 'user_trends.png'), 'w') as fout:
            fout.write ('BEH completed but not yet rendered\n')
    # configure args for behavioral heatmap analysis
    args.input = args.output
        


###############################################################
##          Generate Features For Behavior Heatmaps          ##
###############################################################
def GenerateFeaturesForBehavioralHeatmaps (args):
    features = {}
    version = args.version
    if version == 'v1': version = 'lunpin'
    for filename in os.listdir (args.input):
        if '.csv' != filename [-4: ]: continue
        ftype = filename [: filename.find ('.')] if '.' in filename else filename
        today, today_events, history_events = None, {}, {} # event statistics for each filetype
        def update_events (history_events, today_events):
            for key in list (today_events.keys ()): 
                if key not in history_events: history_events [key] = 0
                history_events [key] += today_events [key]
                del (today_events [key])
        if ftype not in ['device', 'email', 'file', 'http', 'logon']: continue
        for line in csv.reader (open (os.path.join (args.input, filename), 'r')):
            # parse each line
            line = ParseLineFromOriginalData (line, args, ftype=ftype)
            date, hr = line ['date'].split (' ')
            activity = line ['activity'].lower () if 'activity' in line else None
            feature, host = {}, line ['pc']
            hr = int (hr.split (':') [0]) # get hour as index
            if date != today: update_events (history_events, today_events)
            # device
            if ftype == 'device':
                if version == 'liuliu': feature [activity] = 1
                if version == 'lunpin':
                    feature [activity] = 1
                    event = '-'.join ([host, activity])
                    if event not in today_events: today_events [event] = 0
                    if event not in history_events: feature ['newevent'] = 1
                    today_events [event] += 1
            # email
            if ftype == 'email':        
                if 'view' in activity: continue
                if version == 'liuliu': feature [activity] = 1      
                if version == 'lunpin':
                    if '@dtaa' not in line ['from']: continue
                    internals, externals = [], []
                    attachments = line ['attachments'].split (';') if line ['attachments'] != '' else []
                    for recipient in line ['to'].split (';') + line ['cc'].split (';') + line ['bcc'].split (';'):
                        if '' == recipient: continue
                        if '@dtaa.com' in line: internals.append (recipient)
                        else: externals.append (recipient)
                    for tag, recipients in [['internal', internals], ['external', externals]]:
                        feature ['-'.join ([tag, 'send'])] = 1
                        feature ['-'.join ([tag, 'recipient'])] = len (recipients)
                        feature ['-'.join ([tag, 'newrecpt'])] = 0 # new recipient
                        feature ['-'.join ([tag, 'attach'])] = len (attachments)
                        for recipient in recipients:
                            if recipient not in today_events: today_events [recipient] = 0
                            if recipient not in history_events: feature ['-'.join ([tag, 'newrecpt'])] += 1
                            today_events [recipient] += 1
            # file
            if ftype == 'file':
                if version == 'liuliu': feature [activity] = 1
                if version == 'lunpin':
                    if 'copy' in activity:
                        activity += ' fr' if eval (line ['from_removable_media']) else ' fl'
                        activity += ' tr' if eval (line ['to_removable_media']) else ' tl'
                    if 'delete' in activity:
                        activity += ' fr' if eval (line ['from_removable_media']) else ' fl'
                    if 'open' in activity:
                        activity += ' fr' if eval (line ['from_removable_media']) else ' fl'
                    if 'write' in activity:
                        activity += ' tr' if eval (line ['to_removable_media']) else ' tl'
                    filepath = line ['filename']
                    extension = filepath [filepath.rfind ('.') + 1: ]
                    dirpath = filepath [: filepath.rfind ('\\') + 1]
                    feature ['-'.join (['activity', activity])] = 1
                    feature ['-'.join (['extension', extension])] = 1 # doc, exe, jpg, pdf, txt, zip
                    event = '-'.join ([activity, dirpath, extension])
                    if event not in today_events: today_events [event] = 0
                    if event not in history_events: feature ['newevent'] = 1
                    today_events [event] += 1
            # http
            if ftype == 'http':
                if version == 'liuliu': feature [activity] = 1
                if version == 'lunpin':
                    # if 'visit' in activity: continue
                    url = line ['url']
                    domain = url [url.find ('//') + 2: ]
                    domain = domain [: domain.find ('/')]
                    resource = url [url.rfind ('/') + 1: ]
                    extension = resource [resource.rfind ('.') + 1: ]
                    if 'visit' not in activity: activity += ' ' + extension
                    feature ['-'.join (['activity', activity])] = 1
                    event = '-'.join ([activity, domain])
                    if event not in today_events: today_events [event] = 0
                    if event not in history_events: feature ['newevent'] = 1
                    today_events [event] += 1 
            # logon
            if ftype == 'logon': 
                if version == 'liuliu': feature [activity] = 1
                if version == 'lunpin':
                    if 'logoff' in activity: continue
                    feature ['count'] = 1
                    host = line ['pc']
                    if host not in today_events: today_events [event] = 0
                    if host not in history_events: feature ['newhost'] = 1
                    today_events [event] += 1
            # add to features
            for f in feature:
                fname = '-'.join ([ftype, f.lower ()])
                if fname not in features: features [fname] = {}
                if date not in features [fname]: features [fname][date] = [0] * 24 # default ts_gram = 24
                features [fname][date][hr] += feature [f]
        # save event statistics
        update_events (history_events, today_events)
        with open (os.path.join (args.input, ftype + '.events'), 'w') as fout:
            total = sum (history_events [e] for e in history_events)
            events = sorted (history_events.items (), reverse=True, key=lambda item: item [1])
            for e in events: fout.write (json.dumps (list (e) + [e [1] / total]) + '\n')
    # write features
    with open (os.path.join (args.input, args.output + '.features'), 'w') as fout:
        fout.write (json.dumps (features) + '\n')



###############################################################
##                   Read Data from files                    ##
###############################################################
def ParseLineFromOriginalData (line, args, ftype=None):
    dataformat = {
        'decoy_file': ['filename', 'pc'],
        'device': ['date', 'pc', 'activity'],
        'email': ['date', 'pc' ,'to' ,'cc', 'bcc', 'from', 'activity', 'size', 'attachments'],
        'file': ['date', 'pc', 'filename', 'activity', 'to_removable_media', 'from_removable_media'],
        'http': ['date', 'pc', 'url', 'activity'],
        'logon': ['date', 'pc', 'activity'],
        'psychometric': ['employee_name', 'user_id', 'O', 'C', 'E', 'A', 'N'],}
    if isinstance (line, str): line = list (csv.reader ([line])) [0]
    if ftype is None: 
        ftype = args.input [args.input.rfind ('/') + 1: ] 
        ftype = ftype [: ftype.find ('.')] if '.' in ftype else ftype
    ret = {key: line [i] for i, key in enumerate (dataformat [ftype])}
    return ret



###############################################################
##                 Parse From Original Data                  ##
###############################################################
def ParseFromOriginalData (args):
    ftype = args.input [args.input.rfind ('/') + 1: ]
    ftype = ftype [: ftype.find ('.')] if '.' in ftype else ftype
    writer = util.DirectoryWriter (args.output, ftype + '.csv')
    linecount = -1 
    for line in csv.reader (open (args.input, 'r'), delimiter=','):
        linecount += 1
        if linecount == 0: continue # first line is just header
        user = None
        if ftype == 'decoy_file':    
            # decoy_filename, pc
            pass
        if ftype == 'device':
            # id,date,user,pc,file_tree,activity ----> date,pc,activity
            user = line [2]
            line = [line [1], line [3], line [5]]
        if ftype == 'email':
            # id,date,user,pc,to,cc,bcc,from,activity,size,attachments,content 
            # ----> date,pc,to,cc,bcc,from,activity,size,attachments
            user = line [2]
            line = [line [1]] + line [3: -1]
        if ftype == 'file':
            # file.csv: id,date,user,pc,filename,activity,to_removable_media,from_removable_media,content
            # ----> date,pc,filename,activity,to_removable_media,from_removable_media
            user = line [2]
            line = [line [1]] + line [3: -1]
        if ftype == 'http':
            # http.csv: id,date,user,pc,url,activity,content
            # ----> date,pc,url,activity
            user = line [2]
            line = [line [1]] + line [3: -1]
        if ftype == 'logon':
            # id,date,user,pc,activity ----> date,pc,activity
            user = line [2]
            line = [line [1]] + line [3: ]
        if ftype == 'psychometric':
            # employee_name,user_id,O,C,E,A,N
            pass
        if user is not None: writer.write (user, ','.join (line) + '\n')



###############################################################
##                     Generated Labels                      ##
###############################################################
class GeneratedLabel (object):
    ABNORMAL_TAGS = ['Abnormal', 'abnormal']
    def __init__ (self, line):
        if isinstance (line, str): line = list (csv.reader ([line])) [0]
        self.uid  = line [0]
        self.groups = line [1: 5] # business_unit,functional_unit,department,team
        self.label = line [5]
        self.compromises = line [6: ]
    def isAbnormal (self, date=None):
        if date is None: return self.label in self.ABNORMAL_TAGS
        return self.label in self.ABNORMAL_TAGS and date in self.compromises
    def isNormal (self, date=None):
        return not self.isAbnormal (date)
    def csv (self):
        line = [self.uid] + self.groups + [self.label]
        if self.isAbnormal (): line += self.compromises
        return line
    def csvstr (self):
        return ','.join (self.csv ()) + '\n'
 


###############################################################
##           Generate Labels From Orignal Data               ##
###############################################################
def GenerateLabelsFromOriginalData (args):
    users = {}
    # ldap data
    for ldap in os.listdir (args.input):
        for line in csv.reader (open (os.path.join (args.input, ldap), 'r')):
            # employee_name,user_id,email,role,projects,business_unit,functional_unit,department,team,supervisor
            uid = line [1] 
            if 'user_id' in uid: continue
            groups = line [5: -1]
            if uid not in users: users [uid] = {'label': 'Normal', 'groups': groups}
            elif users [uid]['groups'] != groups: 
                print ('LDAP not consistent! It should be consistent for CERT2016 dataset.')
                print (uid)
                print (groups)
                print (users [uid]['groups'])
    # answers
    for line in csv.reader (open (args.label, 'r')): # should concat the answers first
        # example: logon,"{N0F6-R4VY47YV-3468ADPN}","01/06/2011 04:51:28","CSF2712","PC-3343","Logon"
        uid = line [3]
        if 'user_id' in uid: continue
        date = line [2].split (' ') [0]
        if uid not in users: print ('User not in LDAP! Users should be in LDAP for CERT2016 dataset.'); continue
        user = users [uid]
        user ['label'] = 'Abnormal'
        if 'date' not in user: user ['date'] = set ([date])
        else: user ['date'].add (date)
        users [uid] = user
    # output
    with open (args.output, 'w') as fout:
        for uid in users: 
            user = users [uid]
            label = user ['label']
            line = [uid] + user ['groups'] + [label]
            if label != 'Normal': line += user ['date']
            fout.write (','.join (line) + '\n')



###############################################################
##                           MISC                            ##
###############################################################
def enumdates (start, end):
    dinMonth = [[-1, 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]]
    dinMonth += [[-1, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]] * 3
    MM, DD, YYYY = [int (split) for split in start.split ('/')]
    ret = []
    while True:
        ret.append ('/'.join ([str (MM).zfill (2), str (DD).zfill (2), str (YYYY)]))
        if ret [-1] == end: break
        DD += 1
        if DD > dinMonth [YYYY % 4][MM]: DD = 1; MM += 1
        if MM > 12: MM = 1; YYYY += 1
    return ret

def dateback (date, n=0):
    dinMonth = [[-1, 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]]
    dinMonth += [[-1, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]] * 3
    MM, DD, YYYY = [int (split) for split in date.split ('/')]
    while n > 0:
        DD -= 1 
        if DD == 0:
            MM -= 1
            if MM == 0: MM = 12; YYYY -= 1
            DD = dinMonth [YYYY % 4][MM]
        n -= 1
    return '/'.join ([str (MM).zfill (2), str (DD).zfill (2), str (YYYY)])

def getTimestamp (ts_str, f='%m/%d/%Y %H:%M:%S'):
    return time.mktime (datetime.datetime.strptime (ts_str, f).timetuple ())

def TestFunctions (args):
    pass

if __name__ == '__main__':
    main ()

