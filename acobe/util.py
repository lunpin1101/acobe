#!/usr/bin/python3
import gzip, json, matplotlib, numpy, os
matplotlib.use ('Agg')
import matplotlib.pyplot

class Heatmap (numpy.ndarray):
    def __new__ (cls, bitmap, name, index, group, xaxis=None, yaxis=None):
        ret = numpy.asarray (bitmap).view (cls)
        return ret
    def __init__ (self, bitmap, name, index, group, xaxis=None, yaxis=None):
        self.name = name
        self.index = index
        self.group = group
        self.xaxis = xaxis
        self.yaxis = yaxis
    def save (self, dirname, filename=None):
        if filename is None: filename = '_'.join (['heatmap', self.name, self.index.replace ('/', '-')])
        with gzip.open (os.path.join (dirname, filename), 'wt') as fout:
            fout.write (json.dumps ({
                'name': self.name,
                'index': self.index,
                'group': self.group,
                'xaxis': self.xaxis,
                'yaxis': self.yaxis,
                'bitmap': self.tolist (),}) + '\n')
    @classmethod
    def load (cls, filename):
        obj = json.loads (gzip.open (filename, 'r').read ())
        return cls (
            bitmap = obj ['bitmap'],
            name = obj ['name'],
            index = obj ['index'],
            group = obj ['group'],
            xaxis = obj ['xaxis'],
            yaxis = obj ['yaxis'])
    @classmethod
    def concat (cls, h1, h2):
        ret = numpy.concatenate ((h1, h2))
        ret = cls (ret,
            name = h1.name,
            index = h1.index,
            group = h1.group,
            xaxis = h1.xaxis if h1.xaxis is not None else None,
            yaxis = h1.yaxis * 2 if h1.yaxis is not None else None)
        return ret
    def plot (self, dirpath, filename=None, raw=False):
        # color transformation
        if not raw:
            bitmap = [list () for _ in range (0, self.shape [0])]
            for i in range (0, self.shape [0]):
                for j in range (0, self.shape [1]):
                    observation = self [i][j] * 2 - 1 # rescale from (0, 1) to (-1, 1)
                    if observation >= 0.0: observation = [observation, 0.0, 0.0] # red gradient
                    else: observation = [-observation, -observation, -observation] # gray gradient
                    bitmap [i].append (observation)
        else: bitmap = self.tolist ()
        # plot
        fig, ax = matplotlib.pyplot.subplots (nrows=1, ncols=1)
        ax.set_title (self.name)
        ax.imshow (numpy.array (bitmap))
        if self.xaxis is not None:
            ax.set_xticks (numpy.arange (len (self.xaxis)))
            ax.set_xticklabels (self.xaxis, fontsize=4)
            matplotlib.pyplot.setp (ax.get_xticklabels (), rotation=45, ha='right', rotation_mode='anchor')
        if self.yaxis is not None:
            ax.set_yticks (numpy.arange (len (self.yaxis)))
            ax.set_yticklabels (self.yaxis, fontsize=4)
            matplotlib.pyplot.setp (ax.get_yticklabels (), rotation=45, ha='right', rotation_mode='anchor')
        fig.tight_layout ()
        if filename is None: filename = '_'.join (['heatmap', self.name, self.index.replace ('/', '-')])
        matplotlib.pyplot.savefig (os.path.join (dirpath, filename + '.png'), dpi=600)
        matplotlib.pyplot.close (fig)
    def toColormap (self):
        bitmap, newX, newY = [], [], []
        for x in self.xaxis: # dates
            if x not in newX: newX.append (x)
        for y in self.yaxis: # features
            if y not in newY: newY.append (y)
        for j in range (0, len (newY)): # features
            bitmap.append (list ())
            for i in range (0, len (newX)): # dates
                bitmap [j].append (list ())
        for i in range (0, self.shape [0]):
            for j in range (0, self.shape [1]):
                bitmap [i % len (newY)][j % len (newX)].append (self [i][j])
        return self.__class__ (bitmap,
            name = self.name,
            index = self.index,
            group = self.group,
            xaxis = newX,
            yaxis = newY)
        
 

