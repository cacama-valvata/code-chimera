import os
import pandas as pd
import nibabel as nib
import numpy as np
from nilearn.plotting import plot_roi,plot_stat_map
from nilearn import datasets, image, plotting
from nimare.dataset import Dataset
from nimare.decode import discrete
from nimare.utils import get_resource_path
from nimare.decode import continuous
from nimare.utils import get_template
from nilearn.image import resample_to_img,load_img,get_data
import matplotlib as mpl
import matplotlib.pyplot as plt
# %matplotlib inline
from wordcloud import WordCloud
import glob
import datetime
import pathlib
from nimare.results import MetaResult
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from statistics import mean
import statistics
from sklearn_extra.cluster import KMedoids
from nimare.meta.kernel import ALEKernel
import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler
import glob
# from https://stackoverflow.com/questions/41416498/dendrogram-or-other-plot-from-distance-matrix
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform
from statistics import mean
import seaborn as sns

from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
from factor_analyzer.factor_analyzer import calculate_kmo
import scipy.cluster.hierarchy as sch
import networkx as nx
from nilearn.image import binarize_img
import matplotlib.table as mtable
from matplotlib.font_manager import FontProperties
from scipy.stats import spearmanr
from math import pi


def make_ns_table(C1_NSallLDA,OutFile="File.png", save=False):


    VarTab=C1_NSallLDA.iloc[:,0:4]
    VarTab.iloc[:,1:]=round(C1_NSallLDA.iloc[:,1:],3)

    CelDat=VarTab.values.tolist()

    rLabs=['LDA-50','','','','',
          'LDA-100','','','','',
          'LDA-200','','','','',
          'LDA-400','','','','',]

    CLabs=VarTab.columns.tolist()
    CLabs=[i.replace('_zRvrs','') for i in CLabs]
    CLabs= ['PCC' if 'Precuneus' in s else s for s in CLabs]
    CLabs= ['rDLPFC' if 'rIFG' in s else s for s in CLabs]
    CLabs= ['rDLPFC' if 'rdLPFC' in s else s for s in CLabs]
    CLabs= ['lDLPFC' if 'ldLPFC' in s else s for s in CLabs]

    fig = plt.figure()
    # Table plot Add an axes rect [left, bottom, width, height]
    TabAx = fig.add_axes([0.1, 0.1, 0.8, 0.8])

    TabPlt=mtable.table(ax=TabAx,cellText=CelDat,rowLabels=rLabs,colWidths=[.5,.15,.15,.15],
             cellLoc='center',rowLoc='right',colLoc='center',colLabels=CLabs,loc='upper center')

    for (row, col), cell in TabPlt.get_celld().items():
        cell.set_linewidth(0)
        if (row == 0) or (col == -1):
            cell.set_text_props(fontproperties=FontProperties(weight='bold'))          
        if col == -1:
            cell.set_text_props(horizontalalignment='right')
        if (col == 0) & (row>0):
            cell.set_text_props(horizontalalignment='left')
        if (row == 1) or (row == 6) or (row == 11) or (row == 16):
            cell.set_text_props(fontproperties=FontProperties(weight='semibold'))

    # remove default tickmarks 
    TabAx.set_xticks([])
    TabAx.set_yticks([])

    # remove frame from bottom and right of correlation matrix
    TabAx.spines['right'].set_visible(False)
    TabAx.spines['left'].set_visible(False)

    if save:
        fig.savefig(OutFile,format='png', bbox_inches='tight',dpi=300)


def make_heatmapdend(DistMat,OutFile='HeatMapDend.png',Title='Heat-map with dendrogram',Cthrsh=.5,save=True,optimal_ordering=True):

    '''
    this creates a heatmap dendrogram from distance matrix, it requires: 
    distance matrix

    has defaults set for 
    OutFile='HeatMapDend.png'
    Title='Heat-map with dendrogram'
    Cthrsh=.5 
    save=True
    optimal_ordering=True
    '''
    
    Columns=DistMat.columns.tolist()
    Columns= ['PCC' if 'Precuneus' in s else s for s in Columns]
    Columns= ['rDLPFC' if 'rIFG' in s else s for s in Columns]
    Columns= ['lDLPFC' if 'ldLPFC' in s else s for s in Columns]
    Columns= ['rDLPFC' if 'rdLPFC' in s else s for s in Columns]
    DistMat.columns=Columns
    DistMat.index=Columns


    # create Dice (correlation) map
    DiceMat=1-DistMat

    # create 8x8 figure
    fig = plt.figure(figsize=(10,10))

    # add title

    # top side dendogram
    ax1 = fig.add_axes([0.3, 0.7, 0.6, 0.2])
    Y = sch.linkage(DiceMat, method='ward',optimal_ordering=optimal_ordering)
    Z = sch.dendrogram(Y,color_threshold=Cthrsh)
    ax1.set_xticks([])
    ax1.set_yticks([])
    plt.title(Title,fontdict={'fontweight': 'bold'})

    # add axes to dendrogram
    ax1.spines['top'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_visible(False)

    # get index of the ROIs based on optimal ordering for dendrogram
    idx1 = Z['leaves']

    # reorder matrices based on index
    DistMat = DistMat.iloc[idx1, :]
    DistMat = DistMat.iloc[:, idx1]

    DiceMat = DiceMat.iloc[idx1, :]
    DiceMat = DiceMat.iloc[:, idx1]

    # main heat-map
    axmatrix = fig.add_axes([0.3, 0.1, 0.6, 0.6])

    # use seaborn heatmap function to make heat map 
    sns_plot=sns.heatmap(round(DiceMat,2), cmap='RdYlBu_r', annot=True, ax=fig.add_axes([0.3, 0.1, 0.6, 0.6]),
                annot_kws={"size": 7}, vmin=0, vmax=1,cbar_ax=fig.add_axes([0.94, 0.1, 0.02, 0.6]));

    # get x and y labels
    XLabs=sns_plot.get_xticklabels()
    YLabs=sns_plot.get_yticklabels()

    # Add x/y labels with new font format
    sns_plot.set_xticklabels(XLabs,fontdict={'fontsize': 9,
     'fontweight': 'semibold',
     'rotation': 90})

    sns_plot.set_yticklabels(YLabs,fontdict={'fontsize': 9,
     'fontweight': 'semibold',
     'rotation': 0})

    # remove default tickmarks 
    axmatrix.set_xticks([])
    axmatrix.set_yticks([])

    # remove frame from bottom and right of correlation matrix
    axmatrix.spines['right'].set_visible(False)
    axmatrix.spines['bottom'].set_visible(False)

    # save and show plot
    if save:
        plt.savefig(OutFile, format='png', bbox_inches='tight',dpi=300)

    plt.show()



def make_scree(DiceMat,OutFile='ScreePlot.png',Title='EFA Scree Plot of MACM Associations',save=True):


    # set up fa parameters
    fa = FactorAnalyzer(rotation = 'varimax',n_factors=DiceMat.shape[1],is_corr_matrix=True)
    # fit
    fa.fit(DiceMat)
    # get eigenvalues
    ev,_ = fa.get_eigenvalues()

    fig = plt.figure(figsize=(10,10))

    # Scree plot Add an axes at position rect [left, bottom, width, height]
    ScreeAx=fig.add_axes([0.14, 0.34, 0.65, 0.65])
    plt.scatter(range(1,DiceMat.shape[1]+1),ev)
    plt.plot(range(1,DiceMat.shape[1]+1),ev)
    #kwargs = {'labelsize': 0}
    kwargs = {'labelbottom': False}
    plt.tick_params(axis='x', which='both',**kwargs)

    plt.title(Title, fontdict={'fontweight': 'semibold'})
    plt.xlabel('Factors', fontdict={'fontweight': 'semibold'})

    plt.ylabel('Eigen Value', fontdict={'fontweight': 'semibold'})
    #ScreePlt.set_xticks([])
    plt.grid()

    VarTab=pd.DataFrame(fa.get_factor_variance(),index=['Variance','Proportional Var','Cumulative Var']).iloc[:,0:9]
    VarTab.columns=range(1,10)
    VarTab=round(VarTab,3)
    CelDat=VarTab.values.tolist()

    # Table plot Add an axes rect [left, bottom, width, height]
    TabAx = fig.add_axes([0.02, 0.1, 0.9, 0.2])

    TabPlt=mtable.table(ax=TabAx,cellText=CelDat,rowLabels=VarTab.index.tolist(),colWidths=[.083]*9,
             cellLoc='center',rowLoc='right',colLoc='center',colLabels=VarTab.columns.tolist(),loc='upper center')

    for (row, col), cell in TabPlt.get_celld().items():
        cell.set_linewidth(0)
        if (row == 0) or (col == -1):
            cell.set_text_props(fontproperties=FontProperties(weight='semibold'))

    # remove default tickmarks 
    TabAx.set_xticks([])
    TabAx.set_yticks([])

    # remove frame from bottom and right of correlation matrix
    TabAx.spines['right'].set_visible(False)
    TabAx.spines['bottom'].set_visible(False)
    TabAx.spines['left'].set_visible(False)
    TabAx.spines['top'].set_visible(False)
    # save and show plot
    if save:
        plt.savefig(OutFile, format='png', bbox_inches='tight',dpi=300)
    
    plt.show()




def silhouette_graph(BDdist,OutFile="SilhouetteGraph.png",Sse=True,Clusters=True,save=False,colors=['yellow','lightblue','red','orange', 'green','yellow','blue','purple','darkred','darkgreen','lightred','lightgreen','lightyellow']):

    # A list holds the silhouette coefficients for each k
    silhouette_coefficients = []
    sse = []
    labels = []
    
    # clean ROI names if needed
    Columns=BDdist.index.tolist()
    Columns= ['PCC' if 'Precuneus' in s else s for s in Columns]
    Columns= ['rDLPFC' if 'rIFG' in s else s for s in Columns]
    Columns= ['lDLPFC' if 'ldLPFC' in s else s for s in Columns]
    Columns= ['rDLPFC' if 'rdLPFC' in s else s for s in Columns]
    BDdist.index=Columns
    BDdist.columns=Columns
    
    # get number of ROIs to know number of iterations
    numRois=len(BDdist)-1
    
    
    # Notice you start at 2 clusters for silhouette coefficient
    for k in range(2, numRois):     
        kmeans = KMedoids(n_clusters=k,metric='precomputed',method='pam').fit(BDdist)
        labels.append(kmeans.labels_.tolist())
        sse.append(kmeans.inertia_)
        score = silhouette_score(BDdist, kmeans.labels_,metric ="precomputed")
        silhouette_coefficients.append(score)


    # if making Sse
    if Sse:
        plt.plot(range(2, numRois), sse)
        plt.xticks(range(2, numRois))
        plt.xlabel("Number of Clusters")
        plt.ylabel("SSE")

        if save:
            OutFile2=OutFile.split('.')[0:-1][0] + "_sse." + str(OutFile.split('.')[-1])
            plt.savefig(OutFile2, format='png', bbox_inches='tight',dpi=300)

        plt.show()

    # Plot the Sillhouete plot
    plt.plot(range(2, numRois), silhouette_coefficients)
    plt.xticks(range(2, numRois))
    plt.xlabel("Number of Clusters",fontdict={'fontweight': 'semibold'})
    plt.ylabel("Average silhouette coefficient",fontdict={'fontweight': 'semibold'})
    if save:
            plt.savefig(OutFile, format='png', bbox_inches='tight',dpi=300)      
    plt.show()
    
    # start working on cluster color plot 
    if Clusters:
        
        # created the kmeans cluster label DF
        KmeanDf=pd.DataFrame(np.array(labels).T.tolist(),index=BDdist.index.to_list(),columns=list(range(2,len(labels)+2)))
        
        # set up colors to use, and color bounds
        # colors=['lightblue', 'green','yellow','orange','blue','red','purple','darkred','darkgreen','lightred','lightgreen','lightyellow']
        
        bounds=list(range(len(labels)+1))
        colors=colors[:len(bounds)]
        
        # sort values
        KmeanDf.sort_values(by=KmeanDf.columns.tolist(),axis=0,ascending=True, inplace=True,ignore_index=False)
        
        # create discrete colormap
        cmap = mpl.colors.ListedColormap(colors)
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

        # enable or disable frame
        plt.figure(frameon=True)

        plt.xticks(np.arange(0, KmeanDf.shape[1], 1),KmeanDf.columns.tolist(),fontdict={'fontsize': 8,
         'fontweight': 'semibold',
         'rotation': 0})
        plt.yticks(np.arange(0, KmeanDf.shape[0], 1),KmeanDf.index.tolist(),fontdict={'fontsize': 8,
         'fontweight': 'semibold',
         'rotation': 0})
        plt.ylabel('ROI',fontdict={'fontsize': 10,'fontweight': 'semibold'})
        plt.xlabel('Number of clusters',fontdict={'fontsize': 10,'fontweight': 'semibold'})
        
        plt.imshow(KmeanDf, cmap=cmap, norm=norm)
        if save:
            OutFile3=OutFile.split('.')[0:-1][0] + "_Clusters." + str(OutFile.split('.')[-1])
            plt.savefig(OutFile3, format='png', bbox_inches='tight',dpi=300)
            




        

def make_heatmapdend_raw(Data,OutFile,Title='Heat-map with dendrogram',Raw=True,Cthrsh=.5,save=True,method='ward',metric='euclidean',optimal_ordering=True):
    
    # if raw data, create correlation (dice) and distance matrices
    if Raw:
        # get rois from index
        rois=Data.index.tolist()
        rois= ['PCC' if 'Precuneus' in s else s for s in rois]
        rois= ['rDLPFC' if 'rIFG' in s else s for s in rois]
        rois= ['lDLPFC' if 'ldLPFC' in s else s for s in rois]
        rois= ['rDLPFC' if 'rdLPFC' in s else s for s in rois]
        Data.index=rois
        # correlate
        Mat=np.corrcoef(Data)
        # make dataframe
        DiceMat=pd.DataFrame(Mat,columns = rois,index=rois)
        # create distance mat
        DistMat=1-DiceMat
        for r in range(len(DiceMat.iloc[:,0])):
            DistMat.iloc[r,r]=0
    
    # if not raw data, determine if it is distance or corr/dice mat      
    else:
        rois=Data.index.tolist()
        rois= ['PCC' if 'Precuneus' in s else s for s in rois]
        rois= ['rDLPFC' if 'rIFG' in s else s for s in rois]
        rois= ['lDLPFC' if 'ldLPFC' in s else s for s in rois]
        rois= ['rDLPFC' if 'rdLPFC' in s else s for s in rois]
        Data.index=rois
        Data.columns=rois
        # if the identity cells are 0, it's distance
        if Data.iloc[0,0]==0:
            DistMat=Data.copy()
            # create Dice (correlation) map
            DiceMat=1-DistMat
        else:
            DiceMat=Data.copy()
            DistMat=1-DiceMat
            
 
    # create 8x8 figure
    fig = plt.figure(figsize=(10,10))
    
    # add title
#     plt.title(Title,fontdict={'fontweight': 'bold'})

    # top side dendogram
    ax1 = fig.add_axes([0.3, 0.7, 0.6, 0.2])
    Y = sch.linkage(DiceMat, method=method,metric=metric,optimal_ordering=optimal_ordering)
    Z = sch.dendrogram(Y,color_threshold=Cthrsh)
    ax1.set_xticks([])
    ax1.set_yticks([])
    plt.title(Title,fontdict={'fontweight': 'bold'})
    
    # add axes to dendrogram
    ax1.spines['top'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_visible(False)

    # get index of the ROIs based on optimal ordering for dendrogram
    idx1 = Z['leaves']
    
    # reorder matrices based on index
    DistMat = DistMat.iloc[idx1, :]
    DistMat = DistMat.iloc[:, idx1]

    DiceMat = DiceMat.iloc[idx1, :]
    DiceMat = DiceMat.iloc[:, idx1]

    # main heat-map
    axmatrix = fig.add_axes([0.3, 0.1, 0.6, 0.6])

    # use seaborn heatmap function to make heat map 
    sns_plot=sns.heatmap(round(DiceMat,2), cmap='RdYlBu_r', annot=True, ax=fig.add_axes([0.3, 0.1, 0.6, 0.6]),
                annot_kws={"size": 7}, vmin=0, vmax=1,cbar_ax=fig.add_axes([0.94, 0.1, 0.02, 0.6]));
    
    # get x and y labels
    XLabs=sns_plot.get_xticklabels()
    YLabs=sns_plot.get_yticklabels()

    # Add x/y labels with new font format
    sns_plot.set_xticklabels(XLabs,fontdict={'fontsize': 9,
     'fontweight': 'semibold',
     'rotation': 90})
    
    sns_plot.set_yticklabels(YLabs,fontdict={'fontsize': 9,
     'fontweight': 'semibold',
     'rotation': 0})
    
    # remove default tickmarks 
    axmatrix.set_xticks([])
    axmatrix.set_yticks([])
    
    # remove frame from bottom and right of correlation matrix
    axmatrix.spines['right'].set_visible(False)
    axmatrix.spines['bottom'].set_visible(False)
    
    # save and show plot
    if save:
        plt.savefig(OutFile, format='png', bbox_inches='tight',dpi=300)
    
    plt.show()



def annotate(decodefile):
    
    # read in file
    Decoded=pd.read_csv(decodefile)
    cols = Decoded.columns.tolist()
    First=cols[0]
    
    # split bmap classes
    Decoded[['Class','Feature']]=Decoded[First].str.split('.',n=1, expand=True)
    cols2=Decoded.columns.tolist()
    
    # reorder columns
    cols=[[cols[0]],cols2[-2:],cols[1:]]
    cols = [item for sublist in cols for item in sublist]
    AllTerms=Decoded[cols]
    
    # get index for good classes 
    # Indx=(AllTerms['Class']=='BD') | (AllTerms['Class']=='PC') | (AllTerms['Class']=='ST')
    IndxBdPc=(AllTerms['Class']=='BD') | (AllTerms['Class']=='PC')
    IndxS=(AllTerms['Class']=='ST') | (AllTerms['Class']=='SM')
    IndxR=(AllTerms['Class']=='RT') | (AllTerms['Class']=='RM')
    # GoodClasses=AllTerms.loc[Indx]
    
    BdPc=AllTerms.loc[IndxBdPc]
    Stim=AllTerms.loc[IndxS]
    Resp=AllTerms.loc[IndxR]
    
    BD=AllTerms.loc[AllTerms['Class']=='BD']
    PC=AllTerms.loc[AllTerms['Class']=='PC']

    

#     return AllTerms,GoodClasses,BD,PC,ST
    return AllTerms,BdPc,BD,PC,Stim,Resp;



def make_radar(WideDF,CurDom,OutFile,save=True,clean=True,legend=False,Title='',Colors=['r','y','b','g','o','p']):
    
    CurDom2=CurDom + '.'
    
    if clean:
        Cog=WideDF[list(WideDF.columns[WideDF.columns.str.contains(CurDom)])]
        Cog.columns=Cog.columns.str.replace(CurDom2,"")
        Cog.columns=Cog.columns.str.replace("."," ")
        Cog.columns=Cog.columns.str.replace("Language ","Lang-")
        Cog.columns=Cog.columns.str.replace("Memory ","Mem-")
#         Cog.columns=Cog.columns.str.replace("Memory ","Mem-")
#         Cog.columns=Cog.columns.str.replace("Memory Explicit","Explicit Mem")
        
    else:
        
        Cog=WideDF
        Cog.columns=Cog.columns.str.replace("."," ")
    
    CogVals=Cog.to_numpy().flatten()

    # ------- PART 1: Create background

    # number of variable

    categories=list(Cog.columns)
    
    # categories=list(BD_ProbRev['Feature'])
    N = len(categories)

    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    # Initialise the spider plot
    ax = plt.subplot(111, polar=True)

    # If you want the first axis to be on top:
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    # Draw one axe per variable + add labels
    plt.xticks(angles[:-1], categories)
#     plt.xticks(angles[:-1],categories,rotation=45,rotation_mode='anchor',horizontalalignment=Rots)


    ax.set_rlabel_position(0)
#     plt.ylim(min(CogVals),max(CogVals))
    plt.ylim(0,.9)


    # ------- PART 2: Add plots

    # Plot each individual = each line of the data
    # I don't make a loop, because plotting more than 3 groups makes the chart unreadable
    colors=Colors
    
    for i in range(len(Cog.iloc[:,0])):
        values=Cog.iloc[i,:].tolist()
        values += values[:1]
        if i==1:
            ax.plot(angles, values, linewidth=1.4, linestyle='solid', label=Cog.index[i],color=colors[i],alpha=1)
            ax.fill(angles, values, color=colors[i], alpha=0.2)
        else:
            ax.plot(angles, values, linewidth=1.4, linestyle='solid', label=Cog.index[i],color=colors[i],alpha=0.9)
            ax.fill(angles, values, color=colors[i], alpha=0.15)

        


     # Add legend
    if legend: 
        plt.legend(loc='upper right', bbox_to_anchor=(-.1, 0.9))

    plt.title(Title,fontdict={'fontsize': 11,'fontweight' : 'bold'})

    if save:
        plt.savefig(OutFile,format='png', bbox_inches='tight',dpi=300)
    # Show the graph
    plt.show()



# define function for reading in decoding file, and annotating it
def annotate_ns(decodefile):

    
    # read in file
    Decoded=pd.read_csv(decodefile)
    
    Cols=Decoded.columns
    
    if sum(Cols=='Term') == 1:
        XLab='Term'           
    else:
        XLab='feature'
        
            
    Decoded[XLab]=Decoded[XLab].str.replace('LDA50_','')
    Decoded[XLab]=Decoded[XLab].str.replace('LDA100_','')
    Decoded[XLab]=Decoded[XLab].str.replace('LDA200_','')
    Decoded[XLab]=Decoded[XLab].str.replace('LDA400_','')
    Decoded[XLab]=Decoded[XLab].str.replace('abstract_weight__','')
    Decoded[XLab]=Decoded[XLab].str.replace('abstract_weight_','')
    Decoded[XLab]=Decoded[XLab].str.replace('_',' ')
    Decoded[XLab] = Decoded[XLab].str.replace('\d+', '')
    Decoded
    return Decoded;


def make_macm_fig(CurImg,RoiImg,OutFile,Title="Meta-Analytic Coactivation",save=True):
    
   
    fig = plt.figure(figsize=(16, 13),facecolor='white')
    ax = plt.axes()
    
    display=plot_stat_map(CurImg,threshold=1.645,display_mode='mosaic',cut_coords=5,draw_cross=False,colorbar=False,symmetric_cbar=False,vmax=5,figure=fig,axes=ax)
    
    display.add_overlay(RoiImg)
    ax.set_title(Title, fontsize = 20,fontweight="bold")

    # fig.suptitle(RoiName, fontsize=26,fontweight="bold")
    if save:
        fig.savefig(OutFile,facecolor="w", edgecolor="w",format='png', bbox_inches='tight',dpi=300)

    plt.show()