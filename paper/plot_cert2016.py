#!/usr/bin/python3
import csv, gzip, json, matplotlib, numpy, os, random
matplotlib.use ('Agg')
import matplotlib.pyplot

exp = 'expbeh'
votes = 3
r1 = '/home/lunpin/anom/cert2016/r6.1/' + exp
r2 = '/home/lunpin/anom/cert2016/r6.2/' + exp

r1 = '/media/lunpin/ext-drive/bizon/anom/cert2016/r6.1/' + exp
r2 = '/media/lunpin/ext-drive/bizon/anom/cert2016/r6.2/' + exp

def main ():

    print ('heatmap for Group (r1s2)')
    http = Heatmap ('JPH1910', r1, '2020-03-07_s1beh_3', 'anom1_labeledldap.txt', group=True)
    httpscale = Heatmap ('JPH1910', r1, '2020-03-07_s1beh_3', 'anom1_labeledldap.txt', group=True, scaling=True)
    plotHeatmaps ('acobe-r1s2-group_' + exp, 'JPH1910', [http, httpscale],
        ['HTTP (working hours)', 'HTTP (off hours)',
         'Scaled (working hours)', 'Scaled (off hours)'])

    print ('heatmap for JPH1910 (r1s2)')
    device = Heatmap ('JPH1910', r1, '2020-03-07_s1beh_1', 'anom1_labeledldap.txt')
    http = Heatmap ('JPH1910', r1, '2020-03-07_s1beh_3', 'anom1_labeledldap.txt')
    plotHeatmaps ('acobe-r1s2-JPH1910_' + exp, 'JPH1910', [device, http], 
        ['Device (working hours)', 'Device (off hours)', 
         'HTTP (working hours)', 'HTTP (off hours)'])

    print ('Plot Metrics')

    acobe = readExperiments (r1, '2020-03-07_s1beh_1', 'anom1_labeledldap.txt')
    acobe.extend (readExperiments (r1, '2020-03-07_s1beh_2', 'anom1_labeledldap.txt'))
    acobe.extend (readExperiments (r1, '2020-03-07_s1beh_3', 'anom1_labeledldap.txt'))
    acobe.extend (readExperiments (r2, '2020-03-07_s1beh_1', 'anom1_labeledldap.txt'))
    acobe.extend (readExperiments (r2, '2020-03-07_s1beh_2', 'anom1_labeledldap.txt'))
    acobe.extend (readExperiments (r2, '2020-03-07_s1beh_3', 'anom1_labeledldap.txt'))
    acobe.extend (readExperiments (r2, '2020-03-07_s2beh_1', 'anom2_labeledldap.txt'))
    acobe.extend (readExperiments (r2, '2020-03-07_s2beh_2', 'anom2_labeledldap.txt'))
    acobe.extend (readExperiments (r2, '2020-03-07_s2beh_3', 'anom2_labeledldap.txt'))
    
    plotMetric ('acobe_metric_' + exp, [
        ['N=3', getMetric (acobe, days=1, votes=3)], 
        ['N=2', getMetric (acobe, days=1, votes=2)], 
        ['N=1', getMetric (acobe, days=1, votes=1)]])
    
    ain1 = readExperiments (r1, '2020-02-15_s1beh_16', 'anom1_labeledldap.txt')
    ain1.extend (readExperiments (r2, '2020-02-15_s1beh_16', 'anom1_labeledldap.txt'))
    ain1.extend (readExperiments (r2, '2020-02-15_s2beh_16', 'anom2_labeledldap.txt'))
    
    egh = readExperiments (r1, '2020-04-04_s1beh_1', 'anom1_labeledldap.txt')
    egh.extend (readExperiments (r1, '2020-04-04_s1beh_2', 'anom1_labeledldap.txt'))
    egh.extend (readExperiments (r1, '2020-04-04_s1beh_3', 'anom1_labeledldap.txt'))
    egh.extend (readExperiments (r2, '2020-04-04_s1beh_1', 'anom1_labeledldap.txt'))
    egh.extend (readExperiments (r2, '2020-04-04_s1beh_2', 'anom1_labeledldap.txt'))
    egh.extend (readExperiments (r2, '2020-04-04_s1beh_3', 'anom1_labeledldap.txt'))
    egh.extend (readExperiments (r2, '2020-04-04_s2beh_1', 'anom2_labeledldap.txt'))
    egh.extend (readExperiments (r2, '2020-04-04_s2beh_2', 'anom2_labeledldap.txt'))
    egh.extend (readExperiments (r2, '2020-04-04_s2beh_3', 'anom2_labeledldap.txt'))

    t4f = readExperiments (r1, '2020-04-05_s1beh_1', 'anom1_labeledldap.txt')
    t4f.extend (readExperiments (r1, '2020-04-05_s1beh_2', 'anom1_labeledldap.txt'))
    t4f.extend (readExperiments (r1, '2020-04-05_s1beh_3', 'anom1_labeledldap.txt'))
    t4f.extend (readExperiments (r2, '2020-04-05_s1beh_1', 'anom1_labeledldap.txt'))
    t4f.extend (readExperiments (r2, '2020-04-05_s1beh_2', 'anom1_labeledldap.txt'))
    t4f.extend (readExperiments (r2, '2020-04-05_s1beh_3', 'anom1_labeledldap.txt'))
    t4f.extend (readExperiments (r2, '2020-04-05_s2beh_1', 'anom2_labeledldap.txt'))
    t4f.extend (readExperiments (r2, '2020-04-05_s2beh_2', 'anom2_labeledldap.txt'))
    t4f.extend (readExperiments (r2, '2020-04-05_s2beh_3', 'anom2_labeledldap.txt'))
    
    d1 = readExperiments (r1, '2020-05-27_s1beh_1', 'anom1_labeledldap.txt')
    d1.extend (readExperiments (r1, '2020-05-27_s1beh_2', 'anom1_labeledldap.txt'))
    d1.extend (readExperiments (r1, '2020-05-27_s1beh_3', 'anom1_labeledldap.txt'))
    d1.extend (readExperiments (r2, '2020-05-27_s1beh_1', 'anom1_labeledldap.txt'))
    d1.extend (readExperiments (r2, '2020-05-27_s1beh_2', 'anom1_labeledldap.txt'))
    d1.extend (readExperiments (r2, '2020-05-27_s1beh_3', 'anom1_labeledldap.txt'))
    d1.extend (readExperiments (r2, '2020-05-27_s2beh_1', 'anom2_labeledldap.txt'))
    d1.extend (readExperiments (r2, '2020-05-27_s2beh_2', 'anom2_labeledldap.txt'))
    d1.extend (readExperiments (r2, '2020-05-27_s2beh_3', 'anom2_labeledldap.txt'))
 
    liu = readExperiments (r1, '2020-06-07_s1beh_1', 'anom1_labeledldap.txt')
    liu.extend (readExperiments (r1, '2020-06-07_s1beh_2', 'anom1_labeledldap.txt'))
    liu.extend (readExperiments (r1, '2020-06-07_s1beh_3', 'anom1_labeledldap.txt'))
    liu.extend (readExperiments (r1, '2020-06-07_s1beh_4', 'anom1_labeledldap.txt'))
    liu.extend (readExperiments (r2, '2020-06-07_s1beh_1', 'anom1_labeledldap.txt'))
    liu.extend (readExperiments (r2, '2020-06-07_s1beh_2', 'anom1_labeledldap.txt'))
    liu.extend (readExperiments (r2, '2020-06-07_s1beh_3', 'anom1_labeledldap.txt'))
    liu.extend (readExperiments (r2, '2020-06-07_s1beh_4', 'anom1_labeledldap.txt'))
    liu.extend (readExperiments (r2, '2020-06-07_s2beh_1', 'anom2_labeledldap.txt'))
    liu.extend (readExperiments (r2, '2020-06-07_s2beh_2', 'anom2_labeledldap.txt'))
    liu.extend (readExperiments (r2, '2020-06-07_s2beh_3', 'anom2_labeledldap.txt'))
    liu.extend (readExperiments (r2, '2020-06-07_s2beh_4', 'anom2_labeledldap.txt'))

    liuf = readExperiments (r1, '2020-06-08_s1beh_1', 'anom1_labeledldap.txt')
    liuf.extend (readExperiments (r1, '2020-06-08_s1beh_2', 'anom1_labeledldap.txt'))
    liuf.extend (readExperiments (r1, '2020-06-08_s1beh_3', 'anom1_labeledldap.txt'))
    liuf.extend (readExperiments (r2, '2020-06-08_s1beh_1', 'anom1_labeledldap.txt'))
    liuf.extend (readExperiments (r2, '2020-06-08_s1beh_2', 'anom1_labeledldap.txt'))
    liuf.extend (readExperiments (r2, '2020-06-08_s1beh_3', 'anom1_labeledldap.txt'))
    liuf.extend (readExperiments (r2, '2020-06-08_s2beh_1', 'anom2_labeledldap.txt'))
    liuf.extend (readExperiments (r2, '2020-06-08_s2beh_2', 'anom2_labeledldap.txt'))
    liuf.extend (readExperiments (r2, '2020-06-08_s2beh_3', 'anom2_labeledldap.txt'))


    plotMetric ('metric_' + exp, [
        ['Acobe', getMetric (acobe, days=1, votes=votes)],
        ['Base-FF', getMetric (liuf, days=1, votes=votes)],
        ['Baseline', getMetric (liu, days=1, votes=4)],
        ['1-Day', getMetric (d1, days=1, votes=votes)],
        ['No-Group', getMetric (egh, days=1, votes=votes)],
        ['All-in-1', getMetric (ain1, days=1, votes=1)],
        ])


    print ('exlude group behavior')
    plotExperiment ('egh-r1s1-device_' + exp, r1, '2020-04-04_s1beh_1', 'anom1_labeledldap.txt')
    plotExperiment ('egh-r1s1-http_' + exp, r1, '2020-04-04_s1beh_3', 'anom1_labeledldap.txt')

    print ('acobe')
    plotExperiment ('acobe-r1s1-device_' + exp, r1, '2020-03-07_s1beh_1', 'anom1_labeledldap.txt')
    plotExperiment ('acobe-r1s1-file_' + exp, r1, '2020-03-07_s1beh_2', 'anom1_labeledldap.txt')
    plotExperiment ('acobe-r1s1-http_' + exp, r1, '2020-03-07_s1beh_3', 'anom1_labeledldap.txt')

    print ('all features together in one autoencoder')
    plotExperiment ('ain1-r1s1_' + exp, r1, '2020-02-15_s1beh_16', 'anom1_labeledldap.txt')

    print ('slice a day by 24 time frames')
    plotExperiment ('4tf-r1s1-device_' + exp, r1, '2020-04-05_s1beh_1', 'anom1_labeledldap.txt')
    plotExperiment ('4tf-r1s1-http_' + exp, r1, '2020-04-05_s1beh_3', 'anom1_labeledldap.txt')

    print ('single day reconstruction')
    plotExperiment ('1dr-r1s1-device_' + exp, r1, '2020-05-27_s1beh_1', 'anom1_labeledldap.txt')
    plotExperiment ('1dr-r1s1-http_' + exp, r1, '2020-05-27_s1beh_3', 'anom1_labeledldap.txt')

    print ('liuliu')
    plotExperiment ('liu-r1s1_' + exp, r1, '2020-06-07_s1beh_3', 'anom1_labeledldap.txt')
    plotExperiment ('liuf-r1s1_' + exp, r1, '2020-06-08_s1beh_3', 'anom1_labeledldap.txt')
    
    

def plotMetric (filepath, metrics):
    # preprocessing metrics
    for label, metric in metrics:
        TP = metric ['TP']
        FP = metric ['FP']
        TN = [FP [-1] - fp for fp in FP]
        FN = [TP [-1] - tp for tp in TP]
        for m in ['Precision', 'Recall', 'TP Rate', 'FP Rate']: metric [m] = []
        for i in range (0, len (TP)):
            tp, fp, tn, fn = TP [i], FP [i], TN [i], FN [i]
            metric ['Precision'].append ( 0.0 if tp + fp == 0 else tp / (tp + fp) )
            metric ['Recall'].append ( 0.0 if tp + fn == 0 else tp / (tp + fn) )
            metric ['FP Rate'].append ( 0.0 if fp +tn == 0 else fp / (fp + tn) )
            metric ['TP Rate'] = metric ['Recall']
    # plot ROC and F1
    styles = ['k-o', 'k-s', 'k-^', 'k:v', 'k:D', 'k:X', 'k:P']
    fontsize, lw, ms = 14, 2.5, 10
    for figname, xmetric, ymetric, in [
        ['roc', 'FP Rate', 'TP Rate'],
        ['f1', 'Recall', 'Precision']]:
        fig = matplotlib.pyplot.figure ()
        ax = fig.add_subplot (1, 1, 1)
        for i, obj in enumerate (metrics):
            label, metric = obj 
            auc = 0
            for j in range (1, len (metric [xmetric])):
                a_b = metric [ymetric][j] + metric [ymetric][j - 1]
                a_h = metric [xmetric][j] - metric [xmetric][j - 1]
                auc += a_b * a_h / 2
            if figname == 'roc': label = label + ' (' + '{:5.2f}'.format (auc * 100) + '%)'
            print (label, metric ['TP'], metric ['FP'])
            if figname == 'roc': ax.plot (metric [xmetric], metric [ymetric], styles [i], lw=lw, ms=ms, label=label)
            if figname == 'f1': ax.plot (metric [xmetric][1:], metric [ymetric][1:], styles [i], lw=lw, ms=ms, label=label)
        ax.legend (fontsize=fontsize)
        fig.tight_layout ()
        matplotlib.pyplot.savefig ('_'.join ([filepath, figname]) + '.png', dpi=128)
        matplotlib.pyplot.close (fig)
        



def getMetric (experiments, days=1, votes=1):
    # get lists of normal/abnormal users
    normals, abnormals = [], []
    for experiment in experiments: 
        normals.extend (experiment.normals)
        abnormals.extend (experiment.abnormals)
    normals = set (normals)
    abnormals = set (abnormals)
    # get user ranks
    ranks = {}
    for experiment in experiments:
        r = experiment.getAbnormalUserRank (days)
        for user in r:
            if user not in ranks: ranks [user] = []
            ranks [user].append (r [user])
    # derive TPs and FPs
    Ps, TPs, FPs = set (), [0], [0]
    for rank in range (0, len (normals) + len (abnormals)):
        # if len (TPs) - 1 == len (abnormals): break
        for user in ranks:
            if user in Ps: continue # already reported
            if sum ([r >= rank for r in ranks [user]]) >= votes:
                Ps.add (user) # add to reported list
                if user in normals: FPs [-1] += 1
                else: # identify TP: extend arrays
                    TPs.append (TPs [-1] + 1)
                    FPs.append (FPs [-1])
    # append the rest of normal users
    TPs.append (TPs [-1])
    FPs.append (len (normals))
    return {'TP': TPs, 'FP': FPs, 'abnormals': {user: ranks [user] for user in abnormals}}
    



def plotExperiment (filename, dataset, expid=None, label=None):
    experiments = readExperiments (dataset, expid, label)
    for index, experiment in enumerate (experiments):
        output = filename + 'g' + str (index + 1) if len (experiments) > 1 else filename
        plotTrends (output, experiment)
    return experiments [0] if len (experiments) == 1 else experiments



def plotTrends (filename, experiment, head=0, tail=9999, sample=9999):
    fontsize, normallw, abnormallw, markersize = 14, 0.5, 3, 10
    normals = list (experiment.normals)
    abnormals = list (experiment.abnormals)
    dates = experiment.dates
    xaxis = numpy.arange (len (dates))
    fig = matplotlib.pyplot.figure ()
    ax = fig.add_subplot (1, 1, 1)
    scores = []
    # plot scores of normal users
    random.shuffle (normals)
    normals = normals [: sample]
    discarded = 0
    for uid in normals:
        trend = experiment.users [uid].trends ['score']
        scores.extend (trend)
        discard = False
        for i in range (0, len (trend) - 15):
            subtrend = trend [i: i+15]
            if min (subtrend) == max (subtrend): discard = True; discarded += 1; break
        # if discard: continue # disccard straight lines for EGH
        ax.plot (xaxis, trend, '-', color='grey', lw=normallw)
    # plot scores of abnormal users
    for uid in abnormals:
        trend = experiment.users [uid].trends ['score']
        scores.extend (trend)
        ax.plot (xaxis, trend, 'k-', lw=abnormallw, markersize=markersize, label=uid)
        # emphasize on compromise dates
        x, y = [], []
        for date in experiment.users [uid].compromises:
            if date not in dates: continue
            index = dates.index (date)
            x.append (index)
            y.append (trend [index])
        minscore = min (scores)
        ax.plot (x, [minscore] * len (x), 'k*', markersize=markersize)
    # xticks and labels
    xticks = [xaxis [0]]
    for index, date in enumerate (dates):
        if '/01/' in date: xticks.append (index)
    xticks.append (xaxis [-1])
    xticklabels = [dates [i] for i in xticks]
    # title
    mean = numpy.mean (scores)
    std = numpy.std (scores)
    title = 'mean: ' + '{:.4f}'.format (mean) + ', std: ' + '{:.4f}'.format (std)
    # finalize
    ax.set_xticks (xticks)
    ax.set_xticklabels (xticklabels, fontsize=fontsize, rotation=45, ha='right', rotation_mode='anchor')
    ax.yaxis.set_tick_params (labelsize=fontsize)
    ax.legend (fontsize=fontsize)
    ax.set_title (title, fontsize=fontsize)
    fig.tight_layout ()
    matplotlib.pyplot.savefig ('_'.join ([filename, 'trends']) + '.png', dpi=128)
    matplotlib.pyplot.close (fig)
    if discarded > 0: print (filename, 'trends discarded', str (discarded))



def plotHeatmaps (filepath, user, heatmaps, titles=None):            
    fontsize, markersize = 9, 5
    n_bitmaps, bitmapIndex = sum (heatmap.timeframes for heatmap in heatmaps), 0
    fig, axs = matplotlib.pyplot.subplots (nrows=n_bitmaps, ncols=1, sharex=True, sharey=False)
    # plot heatmaps
    for heatmap in heatmaps:
        for timeframe in range (0, heatmap.timeframes):
            ax = axs [bitmapIndex]
            title = None if titles is None else titles [bitmapIndex]
            bitmapIndex += 1
            # enlarge features (rows) by scale and then plot
            imap, scale = [], 10
            for row in heatmap.bitmap [timeframe]:
                for i in range (0, scale): imap.append (row)
            # xticks and xticklabels
            dates, xticks = heatmap.experiment.dates, [0]
            for index, date in enumerate (dates):
                if '/01/' in date: xticks.append (index)
            xticks.append (len (dates) -1)
            xticklabels = [dates [i] for i in xticks]   
            # yticks and yticklabels
            features = heatmap.features
            yticks = [i * scale for i in range (0, len (features) + 1)]
            yticklabels = ['f' + str (i + 1) for i in range (0, len (yticks) - 1)] + [None]
            # plot compromises on the last figures
            if bitmapIndex == n_bitmaps: 
                compromises = []
                for i in range (0, scale): imap.append ([0] * len (dates)) # empty line for compromise marks 
                yticks = yticks + [(len (features) + 1) * scale]
                for date in heatmap.experiment.users [user].compromises:
                    if date not in dates: continue
                    compromises.append (dates.index (date))
                    positions = [yticks [-1] - (scale / 2)] * len (compromises)
                ax.plot (compromises, positions, 'k*', markersize=markersize)
            # de-normalize imap
            vmin, vmax = 0.0, 1.0
            if True:
                vmax = 3.0
                vmin = 0.0 - vmax
                imap = numpy.array (imap) * 2 * vmax - vmax
            # finalize subfigure
            if heatmap.scaling: im = ax.imshow (imap, cmap='gray_r', aspect='auto')
            else: im = ax.imshow (imap, cmap='gray_r', aspect='auto', vmin=vmin, vmax=vmax)
            ax.set_xticks (xticks)
            ax.set_xticklabels (xticklabels, fontsize=fontsize)
            matplotlib.pyplot.setp (ax.get_xticklabels (), rotation=45, ha='right', rotation_mode='anchor')
            ax.set_yticks (yticks)
            ax.set_yticklabels (yticklabels, fontsize=fontsize)
            ax.set_title (title, fontsize=fontsize)
            # matplotlib.pyplot.setp (ax.get_yticklabels (), rotation=45, ha='right', rotation_mode='anchor')
            cb = fig.colorbar (im, ax=ax) 
            cb.minorticks_on ()
    # finalize whole figure
    fig.tight_layout ()
    matplotlib.pyplot.savefig (filepath + '.png', dpi=128)
    matplotlib.pyplot.close (fig)



class Experiment (object):
    def __init__ (self, directory, label='label.txt'):
        # read labels (csv)
        users, normals, abnormals = {}, set (), set ()
        for line in csv.reader (open (os.path.join (directory, label), 'r')):
            user = User (line)
            users [user.uid] = user
            if user.isAbnormal (): abnormals.add (user.uid)
            else: normals.add (user.uid)
        # read results (json)
        evaluation = {} # evaluation [date]
        for filename in os.listdir (os.path.join (directory, 'results')):
            if '.result.log' not in filename [-11: ]: continue
            for line in open (os.path.join (directory, 'results', filename), 'r'):
                uid, score, date, model = json.loads (line)
                obj = {'uid': uid, 'score': score, 'date': date, 'model': model}
                if date not in evaluation: evaluation [date] = []
                evaluation [date].append (obj)
        # sort evaluation
        for date in evaluation:
            evaluation [date] = sorted (evaluation [date], key=lambda obj: obj ['score'], reverse=True)
        # get sorted dates
        def dtoi (date):
            MM, DD, YYYY = date.split ('/')
            YYYY = int (YYYY) * 10000
            MM = int (MM) * 100
            DD = int (DD) * 1
            return YYYY + MM + DD
        dates = sorted (list (evaluation.keys ()), key=lambda date: dtoi (date))
        # get trends
        for date in dates:
            currentrank, currentscore = 0, 0
            for rank, user in enumerate (evaluation [date]): 
                if user ['score'] != currentscore: 
                    currentrank, currentscore = rank, user ['score']
                uid = user ['uid']
                users [uid].trends ['rank'].append (currentrank + 1)
                users [uid].trends ['score'].append (currentscore)
        # finalize
        self.directory = directory
        self.users = users
        self.normals = normals
        self.abnormals = abnormals
        self.dates = dates
    def getAbnormalUserRank (self, days=1):
        # sort all abnormal data points
        scores = [] 
        for user in self.abnormals: scores.extend (self.users [user].trends ['score'])
        scores = sorted (list (set (scores)), reverse=True)
        ret_rank = {}
        rank_count = 0
        TPs = 0
        # score as moving threshold
        for score in scores:
            for user in self.users: # order of normal users is irrelevant to results
                if self.users [user].countAbnormalDays (score) >= days: 
                    if user not in ret_rank:
                        rank_count += 1
                        ret_rank [user] = rank_count
                        if user in self.abnormals: TPs += 1
            if TPs == len (self.abnormals): break
        return ret_rank



class User (object):
    ABNORMAL_TAGS = ['Abnormal', 'abnormal']
    def __init__ (self, line):
        if isinstance (line, str): line = list (csv.reader ([line])) [0]
        self.uid  = line [0]
        self.groups = line [1: 5] # business_unit,functional_unit,department,team
        self.label = line [5]
        self.compromises = line [6: ]
        self.trends = {'rank': [], 'score': []}
        self.sortedtrends = None
        self.hasAbnormalRaise = None
        self.mean = None
        self.std = None
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
    def sortedTrends (self):
        if self.sortedtrends is None: self.sortedtrends = sorted (self.trends ['score'], reverse=True)
        return self.sortedtrends
    def mean (self): 
        if self.mean is None: self.mean = numpy.mean (self.trends ['score'])
        return self.mean
    def std (self):
        if self.std is None: self.std = numpy.std (self.trends ['score'])
        return self.std
    def hasAbnormalRaise (self, scale=3.0):
        if self.hasAbnormalRaise is not None: return self.hasAbnormalRaise
        trends = self.trends ['score']
        for index in range (1, len (trends)):
            past = trends [: index]
            mean, std = numpy.mean (past), numpy.std (past)
            std = 0.0001 if std == 0.0 else std
            if (trend [index] - mean) / std > scale: 
                self.hasAbnormalRaise = True
                return hasAbnormalRaise    
    def countAbnormalDays (self, threshold):
        ret = 0
        for score in self.sortedTrends ():
            if score <= threshold: return ret
            ret += 1
        return ret



class Heatmap (object):    
    def __init__ (self, user, dataset, expid=None, label=None, timeframes=2, group=False, scaling=False):
        experiments = readExperiments (dataset, expid, label)
        # find experiment
        for experiment in experiments:
            if user in experiment.normals or user in experiment.abnormals: break
        # read data and save it to a heatmap
        heatmap = {}
        directory = os.path.join (experiment.directory, 'heatmaps.raw')
        for filename in os.listdir (directory):
            if user not in filename: continue
            obj = json.loads (gzip.open (os.path.join (directory, filename), 'r').read ())
            dates, features, bitmap = obj ['xaxis'], obj ['yaxis'], obj ['bitmap']
            features = features [: int (len (features) / 2)] # exclude group behavior
            for i, feature in enumerate (features):
                if feature not in heatmap: heatmap [feature] = {}
                for j, date in enumerate (dates):
                    if date not in heatmap [feature]: heatmap [feature][date] = {}
                    timeframe = int (j / (int (len (dates) / 2)))
                    if group: heatmap [feature][date][timeframe] = bitmap [i + len (features)][j]
                    else: heatmap [feature][date][timeframe] = bitmap [i][j]
        # transform heatmap to bitmaps
        bitmap = []
        for timeframe in range (0, timeframes):
            bitmap.append ([])
            for feature in features:
                bitmap [timeframe].append ([])
                for date in experiment.dates:
                    bitmap [timeframe][-1].append (heatmap [feature][date][timeframe])            
        # finalize
        self.experiment = experiment
        self.timeframes = timeframes
        self.features = features
        self.heatmap = heatmap
        self.bitmap = bitmap
        self.scaling = scaling
 


def readExperiments (dataset, expid=None, label=None):
    if expid is not None:
        experiments = []
        for group in os.listdir (os.path.join (dataset, expid)):
            experiments.append (Experiment (os.path.join (dataset, expid, group), label))
            # except: print (dataset, expid, label, 'not ready')
    else: experiments = dataset if isinstance (dataset, list) else [dataset]
    return experiments
 

if __name__ == '__main__': main ()

