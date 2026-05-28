





































#===============|
# [[INITIATION]]|
#===============|

#--------------------------------------------------------------------------------------------------------------------------------------------

# [setup]
gather=False # (data)->{data collection or data analysis}
extract=20000 # (data)->{number of records to be extracted from dataset}
lim=0 # (data)->{limit for density of mental classified texts (one user)}
veclen=50 # (embedding)->{length of embedding vector}
alpha=.1 # (embedding)->{learning rate of vectorizer (word2vec,fasttext)}
ngram=1 # (embedding)->{word range}
testsize=.2 # (split)->{train-test split ratio}
strtfy=True # (split)->{stratify or not}
cnnlyrs=[.5] # (cnn)->{architecture of sequential convolution layers}
hdnlyrs=[] # (cnn)->{architecture of hidden fully connected layer}
kernel=1 # (cnn)->{kernel size of convolutions}
drptrglz=.1 # (cnn)->{dropout layer ratio}
lstmlyrs=[.5] # (lstm)->{architecture of recurrent layers}
nestbag=5 # (bagging)->{number of base estimators}
smpl=.2 # (bagging)->{input samples percentage drawn for each estimator}
ftrs=.8 # (bagging)->{input features percentage drawn for each estimator}
nptnc=1 # (training)->{max allowable number of early stopping counter}
patience=.0001 # (training)->{early stopping sensitivity}
clipgrad=.01 # (training)->{weight clipping}
lr_adam=.01 # (optimizers)->{learning rate for adam}
lr_adamax=.01 # (optimizers)->{learning rate for adamax}
lr_adadelta=2 # (optimizers)->{learning rate for adadelta}
lr_adagrad=.01 # (optimizers)->{learning rate for adagrad}
lr_rmsprop=.001 # (optimizers)->{learning rate for rmsprop}
l2wghtdcy=.01 # (optimizers)->{weight regularization}
tst_szs=list(range(98,19,-20)) # (validation)-> {train sizes for validation curve}

# [libraries]
import time,torch,nltk,warnings,pickle,random,os,math
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils as nn_utils
import matplotlib.pyplot as plt
from tabulate import tabulate
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer,PorterStemmer
from stop_words import get_stop_words
from gensim.models import Word2Vec,FastText,Phrases
from sklearn.preprocessing import normalize,MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score,precision_score,recall_score,roc_curve,roc_auc_score,ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from scipy.optimize import curve_fit

# [definitions]
colist=['Model','Type','Embedding','Activation Function','Solver Algorithm','Base Model','Final Estimator','Accuracy','F1 Score','Precision','Recall','AUC','Confusion Matrix','Fitting Time (s)','Validation Time (s)'] # dataframe column names
filepath=r'D:\Python\aridis\pyprj\prj3\datas\newdata1.csv' # input dataset path
dfpath=r'D:\Python\aridis\pyprj\prj3\outputs\df.csv' # saving output dataframe path
sdfpath=r'D:\Python\aridis\pyprj\prj3\outputs\sdf.csv' # saving output dataframe path
ofpath=r'D:\Python\aridis\pyprj\prj3\outputs\of.csv' # saving output calculacions path
rfpath=r'D:\Python\aridis\pyprj\prj3\outputs\rf.csv' # saving method rankings path
tfpath=r'D:\Python\aridis\pyprj\prj3\outputs\tf.csv' # saving method run time path
trfpath=r'D:\Python\aridis\pyprj\prj3\outputs\trf.csv' # saving total run time path
svpath=r'D:\Python\aridis\pyprj\prj3\outputs\models\.'[:-1] # saving models path
cnpath=r'D:\Python\aridis\pyprj\prj3\outputs\cnfsn\.'[:-1] # saving confusion matrices path
cvpath=r'D:\Python\aridis\pyprj\prj3\outputs\plots\.'[:-1] # saving plots path
numblist='''0123456789''' # list of numbers
punclist='''!()-+[]{};:'"\,<>./?#$%^&*_~=|`''' # list of punctuations
emblist=['Count','TF-IDF'][-1:] # embedding methods
actlist=['RRelu','Tanh','Mish','Elu'][:1] # activation functions for cnn-lstm model
slvlist=['Adam','Adamax','Adadelta','Adagrad','RMSprop'][:1] # solver algorithms for cnn-lstm model
ens1list=['Bagging'] # list of single input ensembles
fin1list=['Majority Voting'] # solvers of single input ensembles
ens2list=['Stacking'] # list of ensembles with multiple inputs
fin2list=['Logistic Regression'] # solvers of ensembles with multiple inputs
lenopt=len(actlist)*len(slvlist) # combinations for cnn-lstm activation and optimization (each embedding)
lenlim=(len(ens1list)+1)*lenopt # state combinitions number (single input ensembles for each embedding)
lendor=lenlim+len(ens2list) # state combinitions number (cnn-lstm & bagging & all ensembles for each embedding)
lenmix=lendor*len(emblist) # total number of models (cnn-lstm & bagging & stacking for all embeddings)
tst_szs=[x /100 for x in tst_szs] # convert percent integer to float %
paths=[svpath,cnpath,cvpath]
file_container=[os.listdir(svpath),os.listdir(cnpath),os.listdir(cvpath)] # delete last files in folder
fcnt=0 # initiating file path counter
for file_list in file_container: # getting the files in folder
    pth=paths[fcnt] # getting the folder path
    for file_name in file_list: # checking a file in folder
        file_path=os.path.join(pth,file_name) # merging path with file name for full address
        if os.path.isfile(file_path): # checks if the file still exists or already deleted
           os.remove(file_path) # remove the file from path
    fcnt=fcnt+1 # go to next folder path
e=math.e # euler number
rtimes=time.time() # total run timer

# [functions]
warnings.filterwarnings('ignore') # bypass warnings
wnl=WordNetLemmatizer() # lemmatizing function
wps=PorterStemmer() # stemming function
stoplist=get_stop_words('en') # english stop words function
lossfunc=nn.CrossEntropyLoss() # loss function
class CNNLSTMClassifier(nn.Module): # cnn-lstm
      def __init__(self,vcln): # basic definitions
          super(CNNLSTMClassifier,self).__init__() # initiate the parent class
          self.conv=nn.ModuleList() # convolution layer initiation
          self.normc=nn.ModuleList() # normalization layer initiation (cnn)
          self.actc=nn.ModuleList() # activation layer initiation (cnn)
          self.pool=nn.ModuleList() # pooling layer initiation
          insize=int(vcln) # input size of nodes
          for lyr in range(len(cnnlyrs)): # convolution layers
              if 0<lyr: # cheking the first layer
                 insize=outsize # updtate node size (input)
              outsize=int(cnnlyrs[lyr]*vcln) # update node size (output)
              self.conv.append(nn.Conv1d(insize,outsize,kernel_size=kernel)) # adding convolution layer
              self.actc.append(actfnc) # adding activation layer
              self.pool.append(nn.MaxPool1d(kernel_size=kernel)) # adding pooling layer
              self.normc.append(nn.BatchNorm1d(outsize)) # adding normalization layer
          self.dropout=nn.Dropout(drptrglz) # dropout layer
          self.lstm=nn.ModuleList() # recurrent layer initiation
          insize=outsize # update node size (input)
          for lyr in range(len(lstmlyrs)): # recurrent layers
              if 0<lyr: # cheking the first layer
                 insize=outsize # updtate node size (input)
              outsize=int(lstmlyrs[lyr]*vcln) # update node size (output)
              self.lstm.append(nn.LSTM(insize,outsize,batch_first=True,num_layers=1)) # adding recurrent layer
          self.actr=actfnc # activation for lstm
          self.fc=nn.Linear(outsize,2) # fully connected layer for output
          self.sm=nn.Softmax(dim=1) # softmax
      def forward(self,Xtrn): # forward pass
          for lyr in range(len(cnnlyrs)): # going through convolution layers
              Xtrn=self.normc[lyr](self.pool[lyr](self.actc[lyr](self.conv[lyr](Xtrn)))) # getting output of convolution layers
          Xtrn=Xtrn.permute(0,2,1) # reshape the output
          Xtrn=self.dropout(Xtrn) # applying the dropout layer
          for lyr in range(len(lstmlyrs)):
              Xtrn,_=self.lstm[lyr](Xtrn)
          Xtrn=self.actr(Xtrn) # applying activation function on lstm
          Xtrn=self.fc(Xtrn) # getting the output with a fully connected layer
          Xtrn=Xtrn.squeeze(0) # reshape the output for returning
          Xtrn=self.sm(Xtrn) # applying softmax on final probabilities
          return Xtrn
      def fit(self,Xtrn,Ytrn,Xtst,Ytst): # training
          best_loss=1 # initiation of best loss
          best_acc=0 # initiation of best accucary
          best_model=None # initiation of best model
          early_stopping_counter=0 # initiation of early stopping counter
          epoch=0 # initiation of epochs
          while True: # main training loop
                optimizer.zero_grad() # disable gradient calculation for predicting and scoring
                probatrn=self.forward(Xtrn) # probability outputs of train set
                predict_train=torch.tensor(probatrn.argmax(dim=1).tolist()) # predict outputs of train set
                probatst=self.forward(Xtst) # probability outputs of test set
                predict_test=torch.tensor(probatst.argmax(dim=1).tolist()) # predict outputs of test set
                train_score=round(accuracy_score(Ytrn,predict_train),2) # score of train data
                train_f1scr=round(f1_score(Ytrn,predict_train),2) # f1 score of train
                train_prcsn=round(precision_score(Ytrn,predict_train),2) # precision score of train
                train_recall=round(recall_score(Ytrn,predict_train),2) # recall score of train
                train_auc=round(roc_auc_score(Ytrn,predict_train),2) # auc score of train
                trnscrslrn.append(train_score) # collecting list of train scores for learning curve
                trnf1lrn.append(train_f1scr) # collecting list of train f1 for learning curve
                trnprclrn.append(train_prcsn) # collecting list of train precision for learning curve
                trnrcllrn.append(train_recall) # collecting list of train recall for learning curve
                trnauclrn.append(train_auc) # collecting list of train auc for learning curve
                probatst=self.forward(Xtst) # probability outputs of test set
                test_score=round(accuracy_score(Ytst,predict_test),2) # test accuracy score
                test_f1scr=round(f1_score(Ytst,predict_test),2) # f1 score of test
                test_prcsn=round(precision_score(Ytst,predict_test),2) # precision of test
                test_recall=round(recall_score(Ytst,predict_test),2) # recall of test
                test_auc=round(roc_auc_score(Ytst,predict_test),2) # auc score of test
                tstscrslrn.append(test_score) # collecting list of test scores for learning curve
                tstf1lrn.append(test_f1scr) # collecting list of test f1 for learning curve
                tstprclrn.append(test_prcsn) # collecting list of test precision for learning curve
                tstrcllrn.append(test_recall) # collecting list of test recall for learning curve
                tstauclrn.append(test_auc) # collecting list of test auc for learning curve
                loss=lossfunc(probatrn,Ytrn) # calculating overall loss
                lossval=round(loss.item(),4) # getting loss value
                # (((loss curve for paper)))
                losstrn=lossval # calculating train loss
                trnlosslrn.append(lossval) # collecting list of train scores for learning curve
                losstst=lossfunc(probatst,Ytst) # calculating test loss
                lossval2=round(losstst.item(),4) # getting loss value
                tstlosslrn.append(lossval2) # collecting list of test scores for learning curve
                # (((loss curve for paper)))
                iterlist.append(epoch+1) # collecting iterations for x axis of learning curves
                print(epoch+1,'|',round(lossval,4),',',round(train_score,2),round(test_score,2),'          ',end='\r')
                if best_acc<test_score: # checking improvement of test score
                   best_acc=test_score # updating best test score
                   best_model=self.state_dict() # updating model 
                if lossval<(best_loss-patience): # early stopping patience threshold check
                   best_loss=lossval # updating best loss value
                else: # if model didnt improve
                   early_stopping_counter+=1 # increase early stop counter by 1
                   if nptnc<=early_stopping_counter: # early stopping patience number check
                      break # break if early stop condition is met
                loss.backward() # back propagation
                if clipgrad==True:
                   nn.utils.clip_grad_norm_(self.parameters(),1) # clipping the parameters of network
                optimizer.step() # optimize one step
                epoch+=1 # next step
          self.load_state_dict(best_model) # load the best model parameters
          return # return self
def baggpredict(nestbag): # predicting for bagging model
    listprobatrn=[] # list of probas of base estimators (train)
    listprdctrn=[] # list of predictions of base estimators (train)
    listprobatst=[] # list of probas of base estimators (test)
    listprdctst=[] # list of predictions of base estimators (test)
    for bgcnt in range(nestbag): # base estimators of bagging
        listprobatrn.append(mdl[bgcnt].forward(X_train[:,ftrsind[bgcnt]])) # update list of probas (train)
        listprdctrn.append(listprobatrn[-1].argmax(dim=1).tolist()) # update list of predictions (train)
        listprobatst.append(mdl[bgcnt].forward(X_test[:,ftrsind[bgcnt]])) # update list of probas (test)
        listprdctst.append(listprobatst[-1].argmax(dim=1).tolist()) # update list of predictions (test)
    listprobatrn=torch.stack(listprobatrn).numpy()
    probatrn=np.mean(listprobatrn,axis=0)
    listprdctrn=np.array(listprdctrn).transpose() # reshaped array
    listprdctrn=np.sum(listprdctrn,axis=1) # gathering votes
    listprobatst=torch.stack(listprobatst).numpy()
    probatst=np.mean(listprobatst,axis=0)
    listprdctst=np.array(listprdctst).transpose() # reshaped array
    listprdctst=np.sum(listprdctst,axis=1) # gathering votes
    predict_train=[1 if x>int(nestbag/2) else 0 for x in listprdctrn] # more than half of votes is 1
    predict_test=[1 if x>int(nestbag/2) else 0 for x in listprdctst] # more than half of votes is 1
    return predict_train,probatrn,predict_test,probatst # return predictions of train and test
def learncurve(trnscrlst,tstscrlst,fnm):
    # (((loss curve for paper)))
    curve_func1=lambda x,a: 1-a*(e**(-1/x)) # a*(e**(-1/x)) # regression function  
    # (((loss curve for paper)))
    popt1,_=curve_fit(curve_func1,iterlist,trnscrlst) # fit train scores to curve
    popt2,_=curve_fit(curve_func1,iterlist,tstscrlst) # fit test scores to curve
    x_fit=np.linspace(iterlist[0],iterlist[-1],50) # devide x axis
    y1_fit=curve_func1(x_fit,popt1) # calculate y1
    y2_fit=curve_func1(x_fit,popt2) # calculate y2
    plt.plot(x_fit,y1_fit,'-',color="m",label="Train Score") # plot y1 curve for train
    plt.plot(x_fit,y2_fit,'-',color="c",label="Test Score") # plot y2 curve for test
    plt.title("Learning Curve") # table title
    plt.xlabel("Training Iterations") # x axis label
    plt.ylabel(fnm) # y axis label
    # (((loss curve for paper)))
    plt.legend(loc="upper right") # "lower right" # legend location
    # (((loss curve for paper)))
    plt.grid(True,which='both',linestyle='--',linewidth=0.5) # grid enabled
    plt.minorticks_on() # axis sections enabled
    plt.gca().set_xticks(np.arange(min(iterlist),max(iterlist),5)) # applying grid
    plt.gca().set_yticks(np.arange(min(min(trnscrlst),min(tstscrlst)),max(max(trnscrlst),max(tstscrlst)),0.05)) # applying grid
    imagename1=cvpath+modelname+'-'+embedding+' '+active+' '+solve+' '+fnm+'.jpg' # name of file to be saved
    plt.savefig(imagename1) # save the file
    plt.close() # close the plot
    plt.clf() # reset memory
    return
def ranker(rnkmethod,rnklist):
    sctn=[]
    accrnk=[]
    tavgrnk=[]
    for crnk in rnklist:
        sctn.append(rnkmethod)
        acclist=[]
        rnktime=[]
        for csrch in range(len(df)):
            if crnk==list(df[rnkmethod])[csrch]:
               acclist.append(list(df['Accuracy'])[csrch][1])
               rnktime.append(list(df['Fitting Time (s)'])[csrch])
        accrnk.append(round(sum(acclist)/len(acclist),2))
        tavgrnk.append(round(sum(rnktime)/len(rnktime),2))
    sctn,rnklist,accrnk,tavgrnk=zip(*sorted(zip(sctn,rnklist,accrnk,tavgrnk),key=lambda x: x[2],reverse=True))
    for section,method,accuracy,elapsed in zip(sctn,rnklist,accrnk,tavgrnk):
        new_row=[section,method,accuracy,elapsed]
        rf.loc[len(rf)]=new_row
    return rf

#--------------------------------------------------------------------------------------------------------------------------------------------







































#============|
# [[DATASET]]|
#============|

#--------------------------------------------------------------------------------------------------------------------------------------------

# [collecting]
if gather==True:
   allusers_allsentences=[] # initiating empty list of texts (all users)
   oneuser_allsentences=[] # initiating empty list of texts (one user)
   taglist=[] # initiating empty list of output labels of text classification (one user)
   usertags=[] # initiating empty list of output labels of text classification (all users)
   datasrc=pd.read_csv(filepath,encoding='latin-1') # input dataset in filepath is presorted by users
   dataset=datasrc.sample(extract) # take sample from data (1)
   dataset=dataset.reset_index(drop=True) # reset index
   print(dataset) # report the raw datas to cmd
   print()
   print('target class ratio:',round(list(dataset['label']).count(1)/len(dataset),2)) # report the class ratios to cmd
   print()
   cnt=0 # initiating counter
   for mkr in range(len(dataset)): # convert data to needed shape
       cnt=cnt+1 # step counter plus one
       print('reading dataset...',int((cnt/len(dataset))*100),'%',end='\r') # report progress to cmd
       if mkr==0: # first data
          ol='' # doesnt have last user id in list
       if 0<mkr: # not first data
          ol=list(dataset['user'])[mkr-1] # last user id in list
       nl=list(dataset['user'])[mkr] # new user id in list
       oneuser_allsentences.append(list(dataset['text'])[mkr]) # concatenate one user texts
       taglist.append(list(dataset['label'])[mkr]) # convert labels from dataset to list (for one user texts)
       if nl!=ol or mkr==len(dataset)-1: # True,collect one user texts (if last user is different from the new user in the list->next user)
          allusers_allsentences.append(oneuser_allsentences) # add one user text to list of all user texts
          oneuser_allsentences=[] # clear the list for next user texts collection
          usertags.append(sum(taglist)/len(taglist)) # density of mental illness classified texts (used to classify users for output of model)
          if lim<usertags[-1]: # limit for density of mental illness classified texts (one user)
             usertags[-1]=1 # mental illness accepted
          if usertags[-1]<=lim: # less than limit
             usertags[-1]=0 # mental illness rejected
          taglist=[] # clear labels list for one user texts
   print('reading dataset... done!') # report to cmd

   # [preprocess]
   qntuser=len(allusers_allsentences) # number of all users
   qntuserposts=[] # initiating empty list of posts number (all users)
   allusers_concat=[] # initiating empty list of processed texts (all users)->sentences
   allusers_tokenized=[] # initiating empty list of processed texts (all users)->words
   cnt=0 # initiating counter
   savelist1=allusers_allsentences[:20]
   df = pd.DataFrame(savelist1, columns=['raw']) 
   df.to_excel('savelist1.xlsx', index=False) 
   for oneuser_allsentences in allusers_allsentences: # processing all user texts (user by user)
       cnt=cnt+1 # step counter plus one
       print('preprocessing data...',int((cnt/qntuser)*100),'%',end='\r') # report progress to cmd
       oneuser_concat_pre1='' # initiating empty list for processed words
       qntpost=len(oneuser_allsentences) # number of posts (one user)
       ngram_trans=Phrases(oneuser_allsentences) # auto detect multi words (New York City)
       for oneuser_onesentence_raw in oneuser_allsentences: # {{preprocess: numbers & punctuations}}
           oneuser_onesentence_raw=ngram_trans[oneuser_onesentence_raw] # {{perform auto ngram recognization}}
           processed='' # initiating empty string for words
           for letters in oneuser_onesentence_raw: # check letters
               if letters in (numblist+punclist): # eliminate (numbers & punctuations)
                  letters='' # initiating empty string for words
               processed=processed+letters # remaking words with clean letters
           oneuser_concat_pre1=oneuser_concat_pre1+processed.lower()+' ' # add word to text for one user (lower caseing & adding space between words) # {{preprocess: lowercase}}
       oneuser_concat_split=oneuser_concat_pre1.split() # tokenizing user text to words
       oneuser_concat_pre2='' # initiating empty list for processed words
       for words in oneuser_concat_split: # processing one user texts (word by word)
           if (words not in stoplist) and (1<len(words)) and ('http' not in words) and ('@' not in words): # single letters # {{preprocess: stopwords,urls,mentions}}
              postag=list(nltk.pos_tag([words])[0])[1][0] # recognizing word position in sentence # {{preprocess: stemming and lemmatizing}}
              wnpstg=wordnet.NOUN # initiate position (noun)
              if postag=='J': # recognize position
                 wnpstg=wordnet.ADJ # adjective
              if postag=='V': # recognize position
                 wnpstg=wordnet.VERB # verb
              if postag=='R':  # recognize position
                 wnpstg=wordnet.ADV # adverb
              words=wnl.lemmatize(words,wnpstg) # lemmatizing words
              words=wps.stem(words) # stemming words
              oneuser_concat_pre2=oneuser_concat_pre2+words+' ' # add processed words to texts of one user
       allusers_concat.append(oneuser_concat_pre2) # list of processed sentences by users
       qntuserposts.append(qntpost) # user texts quantity
   nodelist=[] # initiating empty list for deleted users id
   ndel=0 # initiating counter for deleted users
   for lister in range(len(allusers_concat)): # tokenizing user sententces
       if len(allusers_concat[lister])==0: # user text is fully removed after process
          ndel=ndel+1 # deleted users counter plus one
       if len(allusers_concat[lister])!=0: # user text has remained after process
          nodelist.append(lister) # list of users that their texts hasnt been fully removed after process
          allusers_tokenized.append(allusers_concat[lister].split()) # adding tokenized sentences to list of all user texts
   tagtemp=[] # initiating empty list for updating output label values
   alltemp=[] # initiating empty list for updating untokenized input texts
   for udel in range(len(usertags)): # syncing output labels with final input
       if udel in nodelist: # user texts remained after process
          tagtemp.append(usertags[udel]) # temporary output label list
          alltemp.append(allusers_concat[udel]) # temporary untokenized input texts list
   usertags=tagtemp # updating output labels
   allusers_concat=alltemp  # updating untokenized input texts list
   adlistof=[len(allusers_tokenized),len(dataset)-ndel,veclen] # data values for dataframe
   trshld=usertags.count(1)/adlistof[0] # setting threshold of activation function based on sample frequencies
   print('preprocessing data... done!') # report to cmd
   print('========================================================')
   print()
   print('users quantity:  ',adlistof[0]) # report number of users to cmd
   print('texts posted:    ',adlistof[1]) # report number of texts (all users) to cmd
   print()
   savelist2=allusers_concat[:20]
   df = pd.DataFrame(savelist2, columns=['process']) 
   df.to_excel('savelist2.xlsx', index=False) 
   pickle.dump(allusers_tokenized,open('allusers_tokenized.pkl','wb')) # save tokenized sentences to file
   pickle.dump(allusers_concat,open('allusers_concat.pkl','wb')) # save concatenated senteces to file
   pickle.dump(usertags,open('usertags.pkl','wb')) # save class labels list to file
   pickle.dump(adlistof,open('adlistof.pkl','wb')) # save data properties to file
   pickle.dump(trshld,open('trshld.pkl','wb')) # save activation threshold amount to file
   exit() # preprocessing data mode

# [split train-test]
allusers_tokenized=pickle.load(open('allusers_tokenized.pkl','rb')) # load tokenized sentences from file
allusers_concat=pickle.load(open('allusers_concat.pkl','rb')) # load concatenated sentences from file
usertags=pickle.load(open('usertags.pkl','rb')) # load class labels from file
adlistof=pickle.load(open('adlistof.pkl','rb')) # load data properties from file
trshld=pickle.load(open('trshld.pkl','rb')) # load threshold from file
random_indices=random.sample(list(range(adlistof[1])),extract) # take sample from data (2)
allusers_tokenized=[allusers_tokenized[i] for i in random_indices] # tokenized samples
allusers_concat=[allusers_concat[i] for i in random_indices] # concatenated samples
usertags=[usertags[i] for i in random_indices] # class label samples
tf=pd.DataFrame(columns=['Embedding','Run Time (s)']) # create dataframe for models
df=pd.DataFrame(columns=colist) # create dataframe for models
of=pd.DataFrame(columns=['users quantity','texts posted','embedding vector length']) # create dataframe for calculations
of.loc[len(of)]=adlistof # ad calculations values to dataframe
of.to_csv(ofpath,index=False) # save calculations dataframe as csv file
stratify=None
if strtfy==True:
   stratify=usertags
xsplit_train,xsplit_test,yembedtrn,yembedtst=train_test_split(range(len(allusers_concat)),usertags,test_size=testsize,stratify=stratify) # spliting indices and labels
xembedtrn=[allusers_concat[i] for i in xsplit_train] # spliting features data for train
xembedtst=[allusers_concat[i] for i in xsplit_test] # spliting features data for test

# [embedding]
print('embedding vector length:',veclen)
print()
cnt=0 # initiating counter
for embedding in emblist: # applying all embedding methods
    tembs=time.time() # start timer for embedding vectorizer
    lodmod=[] # initiating empty list for model files of stacking ensemble
    lodnmd=[] # initiating empty list for model names of stacking ensemble
    if embedding=='Count': # choosing embedding method
       vectorizer=CountVectorizer(max_features=veclen,ngram_range=(ngram,ngram)) # embedding function definition
    if embedding=='TF-IDF': # choosing embedding method
       vectorizer=TfidfVectorizer(max_features=veclen,ngram_range=(ngram,ngram)) # embedding function definition
    if embedding in ['Count','TF-IDF']: # recognizing embedding method for special calculations
       #fitting=vectorizer.fit(allusers_concat) # learn vocabulary !!!!AAAA!!!!
       fitting=vectorizer.fit(xembedtrn) # learn vocabulary
       vectors=vectorizer.transform(allusers_concat) # vectorize texts
       w2v=vectors.toarray() # convert vectorize to useable shape
       #wordcol=vectorizer.vocabulary_ # get words list
    w2v_normalized=normalize(w2v,norm='l2') # l2 normalization of embedding vector
    w2v_scaled=MinMaxScaler(feature_range=(0,1)).fit_transform(w2v_normalized) # feature scaling of input
    w2v_zero_scaled=(w2v_scaled-w2v_scaled.mean(axis=0))/w2v_scaled.std(axis=0) # zero-centering and scaling
    w2v_preprocessed=np.clip(w2v_zero_scaled,-1,1) # clipping or capping (optional)
    w2v=w2v_preprocessed # rewrite values
    tembf=time.time() # end timer for embedding vectorizer
    tembdelta=round(tembf-tembs,2) # time calculation
    tf.loc[len(tf)]=[embedding,tembdelta]
    print()
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('<',embedding,'> embedding time for vectorizing the texts: ',tembdelta,'s') # report embedding time to cmd
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print()
    X_train,X_test,Y_train,Y_test=train_test_split(w2v,usertags,test_size=testsize,stratify=usertags) # spliting data !!!EDIT!!!
    X_train=[w2v[i] for i in xsplit_train] # spliting features data for train
    X_test=[w2v[i] for i in xsplit_test] # spliting features data for test
    Y_train=yembedtrn
    Y_test=yembedtst

    #--------------------------------------------------------------------------------------------------------------------------------------------


    


















    



















    #==========|
    # [[MODEL]]|
    #==========|

    #--------------------------------------------------------------------------------------------------------------------------------------------

    # (create & fit)
    X_train=torch.tensor(np.array(X_train).transpose(),dtype=torch.float32).unsqueeze(0) # reshaping and format correction
    Y_train=torch.tensor(np.array(Y_train),dtype=torch.long) # reshaping and format correction
    X_test=torch.tensor(np.array(X_test).transpose(),dtype=torch.float32).unsqueeze(0) # reshaping and format correction
    Y_test=torch.tensor(np.array(Y_test),dtype=torch.long) # reshaping and format correction
    for actslv in range(lendor): # model options states
        iterlist=[0] # fitting iterations
        trnscrslrn=[0] # train accuracy for learning curve
        trnf1lrn=[0] # train f1 for learning curve
        trnprclrn=[0] # train precision for learning curve
        trnrcllrn=[0] # train recall for learning curve
        trnauclrn=[0] # train auc for learning curve
        tstscrslrn=[0] # test accuracy for learning curve
        tstf1lrn=[0] # test f1 for learning curve
        tstprclrn=[0] # test precision for learning curve
        tstrcllrn=[0] # test recall for learning curve
        tstauclrn=[0] # test auc for learning curve
        numsize=[] # train sizes for validation
        trnscrsvld=[] # train scores for validation curve
        tstscrsvld=[] # test scores for validation curve
        # (((loss curve for paper)))
        trnlosslrn=[1] # train loss for learning curve
        tstlosslrn=[1] # test loss for learning curve
        trnlossvld=[] # train loss for validation curve
        tstlossvld=[] # test loss for validation curve
        # (((loss curve for paper)))
        cnt=cnt+1 # step counter plus one
        if actslv<lenlim: # model is cnn-lstm or single input ensemble
           lenact=actslv%len(actlist) # address of activation function in list
           lenslv=int(actslv/len(actlist))%len(slvlist) # address of solver algorithm in list
           active=actlist[lenact] # selecting activation function
           actmlp=active.lower() # lower case function name for model input
           solve=slvlist[lenslv] # selecting solver algorithm
           slvmlp=solve.lower() # lower case algorithm name for model input

           # (activation function selection)
           if active=='RRelu': # checking activation function
              actfnc=nn.RReLU() # choosing the function
           if active=='Tanh': # checking activation function
              actfnc=nn.Tanh() # choosing the function
           if active=='Mish': # checking activation function
              actfnc=nn.Mish() # choosing the function
           if active=='Elu': # checking activation function
              actfnc=nn.ELU() # choosing the function

           # (solver algorithm selection)
           if solve=='Adam': # checking optimizer
              slvalg=optim.Adam # choosing the optimizer
              d_lr=lr_adam # tuning the learning rate
           if solve=='Adamax': # checking optimizer
              slvalg=optim.Adamax # choosing the optimizer
              d_lr=lr_adamax # tuning the learning rate
           if solve=='Adadelta': # checking optimizer
              slvalg=optim.Adadelta # choosing the optimizer
              d_lr=lr_adadelta # tuning the learning rate
           if solve=='Adagrad': # checking optimizer
              slvalg=optim.Adagrad # choosing the optimizer
              d_lr=lr_adagrad # tuning the learning rate
           if solve=='RMSprop': # checking optimizer
              slvalg=optim.RMSprop # choosing the optimizer
              d_lr=lr_rmsprop # tuning the learning rate

        # (cnn-lstm)
        if actslv<lenopt: # model is not an ensemble model
           modelname='CNN-LSTM' # loading model name
           modeltype='Neural Network' # type of classifier
           basemodel='..............' # not an ensemble model
           exclsv='..............' # not an ensemble model
           mdl=CNNLSTMClassifier(veclen) # creating model
           optimizer=slvalg(mdl.parameters(),lr=d_lr,weight_decay=l2wghtdcy) # select method for optimizing weights
           t1=time.time() # solver time start
           mdl.fit(X_train,Y_train,X_test,Y_test) # fitting the model
           t2=time.time() # solver time end

        # (bagging)
        if lenopt<=actslv<lenlim: # ensemble model (single input)
           poslist=int(actslv/lenopt)-1 # position of method in list
           modelname=ens1list[poslist] # loading ensemble method name
           modeltype='Ensemble' # type of classifier
           exclsv=fin1list[poslist] # final optimizer of ensemble
           basemodel='CNN-LSTM' # base model name
           if modelname=='Bagging': # bagging ensemble model
              idmod=str(cnt-1-(1*lenopt)) # id of model file
              filename=svpath+str(idmod) # name of file
              baseestim=pickle.load(open(filename,"rb")) # load base model
              estims=[] #  # initiating list of base estimators (for using now)
              mdl=[] # initiating list of base estimators (for using later)
              Y_smplist=[] # class label of bagging data splits
              ftrsind=[] # features of bagging data splits
              trnbaglist=[] # list of base estimators train scores
              tstbaglist=[] # list of base estimators test scores
              t1=time.time() # start bagging timer
              for bgcnt in range(nestbag): # fit all base estimators
                  X_smpl,X_remn,Y_smpl,Y_remn=train_test_split(X_train.squeeze(0).numpy().transpose(),Y_train.numpy(),test_size=1-smpl,stratify=Y_train) # selecting random samples
                  ftrsind.append(random.sample(range(veclen),int(ftrs*veclen))) # selecting random features
                  X_smpl=X_smpl[:,ftrsind[-1]] # selecting features for train samples in estimator
                  X_remn=X_remn[:,ftrsind[-1]] # selecting features for test samples in estimator
                  X_smpl=torch.tensor(np.array(X_smpl).transpose(),dtype=torch.float32).unsqueeze(0) # reshape and format data
                  Y_smpl=torch.tensor(np.array(Y_smpl),dtype=torch.long) # reshape and format data
                  X_remn=torch.tensor(np.array(X_remn).transpose(),dtype=torch.float32).unsqueeze(0) # reshape and format data
                  Y_remn=torch.tensor(np.array(Y_remn),dtype=torch.long) # reshape and format data
                  estims.append(CNNLSTMClassifier(int(ftrs*veclen))) # update the list of base estimators
                  baggestim=estims[-1] # select estimator for fitting
                  optimizer=slvalg(baggestim.parameters(),lr=d_lr,weight_decay=l2wghtdcy) # selecting solver for bagging
                  baggestim.fit(X_smpl,Y_smpl,X_remn,Y_remn) # fit the base estimator.
                  estims[-1]=baggestim # update the last estimator with fitted parameters
              mdl=estims # list of base estimators to be used later
              t2=time.time() # end bagging timer

        # (stacking)
        if lenlim<=actslv: # ensemble model (multiple input)
           active='Multiple' # models with different activation functions are used
           solve='Multiple' # models with different solver algorithms are used
           poslist=actslv-lenlim # position of multi input ensemble in list of methods
           modelname=ens2list[poslist] # loading ensemble method name
           modeltype='Ensemble' # type of classifier
           basemodel='Hybrid ('+str(len(lodmod))+')' # base model name
           exclsv=fin2list[poslist] # final optimizer of ensemble
           stckprdctrn=[] # list of base models predictions (train)
           stckprdctst=[] # list of base models predictions (train)
           stckprobatrn=[] # list of base models predict probas (train)
           stckprobatst=[] # list of base models predict probas (test)
           with torch.no_grad(): # disable gradients for predictions
                for cmdl in range(len(lodmod)): # base model selection
                    mdl=lodmod[cmdl] # loading model
                    mdnm=lodnmd[cmdl] # loading name
                    if mdnm=='CNN-LSTM': # base estimator: cnn model
                       probatrn=mdl.forward(X_train) # forward pass of train
                       probatst=mdl.forward(X_test) # forward pass of test
                       predict_train=probatrn.argmax(dim=1).tolist() # calculating train predictions
                       predict_test=probatst.argmax(dim=1).tolist() # calculating test predictions
                       probatrn=probatrn.tolist() # convert to numpy array format
                       probatst=probatst.tolist() # convert to numpy array format
                    if mdnm=='Bagging': # base estimator: bagging ensemble
                       [predict_train,probatrn,predict_test,probatst]=baggpredict(nestbag)
                    stckprdctrn.append(predict_train) # updating inputs of stacking model with predictions (train)
                    stckprobatrn.append(probatrn) # updating inputs of stacking model with probabilities (train)
                    stckprdctst.append(predict_test) # updating inputs of stacking model with predictions (test)
                    stckprobatst.append(probatst) # updating inputs of stacking model with probabilities (test)
           if modelname=='Stacking': # stacking ensemble
              stckprobatrn=np.array(stckprobatrn) # convert the train array for stacking to numpy array
              [s0,s1,s2]=stckprobatrn.shape # get the shape of train array
              stckprobatrn=stckprobatrn.reshape(s1,s0*s2).tolist() # reshape and concatenate probabilities as features
              stckprdctrn=torch.transpose(torch.tensor(stckprdctrn),0,1).tolist() # reshaping input features of predictions
              stckprobatst=np.array(stckprobatst) # convert the test array for stacking to numpy array
              [s0,s1,s2]=stckprobatst.shape # get the shape of test array
              stckprobatst=stckprobatst.reshape(s1,s0*s2).tolist() # reshape and concatenate probabilities as features
              stckprdctst=torch.transpose(torch.tensor(stckprdctst),0,1).tolist() # reshaping input features of predictions
              mdl=LogisticRegression()
              t1=time.time() # start timer for stacking model
              mdl.fit(stckprdctrn,Y_train.tolist()) # fitting the stacking model
              t2=time.time() # end timer for stacking model

        # (timer)
        tdelta=round(t2-t1,2) # calculating elapsed time








        

        





























        #===============|
        # [[Evaluation]]|
        #===============|

        #--------------------------------------------------------------------------------------------------------------------------------------------

        # [scores]
        with torch.no_grad():
             if modelname=='CNN-LSTM': # cnn-lstm model
                probatrn=mdl.forward(X_train) # forward pass (predict probas train)
                probatst=mdl.forward(X_test) # forward pass (predict probas test)
                predict_train=probatrn.argmax(dim=1).tolist() # predict train
                predict_test=probatst.argmax(dim=1).tolist() # predict test
             if modelname=='Bagging': # bagging ensemble
                [predict_train,probatrn,predict_test,probatst]=baggpredict(nestbag)
             if modelname=='Stacking': # stacking ensemble
                predict_train=mdl.predict(stckprdctrn) # predict train
                predict_test=mdl.predict(stckprdctst) # predict test
        train_score=round(accuracy_score(Y_train,predict_train),2) # accuracy of train
        train_f1scr=round(f1_score(Y_train,predict_train),2) # f1 score of train
        train_prcsn=round(precision_score(Y_train,predict_train),2) # precision score of train
        train_recall=round(recall_score(Y_train,predict_train),2) # recall score of train
        train_auc=round(roc_auc_score(Y_train,predict_train),2) # auc score of train
        train_conf=confusion_matrix(Y_train,predict_train) # confusion matrice of train
        test_score=round(accuracy_score(Y_test,predict_test),2) # accuracy of test
        test_f1scr=round(f1_score(Y_test,predict_test),2) # f1 score of test
        test_prcsn=round(precision_score(Y_test,predict_test),2) # precision of test
        test_recall=round(recall_score(Y_test,predict_test),2) # recall of test
        test_auc=round(roc_auc_score(Y_test,predict_test),2) # auc score of test
        # (confusion matrix)
        test_conf=confusion_matrix(Y_test,predict_test) # confusion matrice of test
        mtrxname1=cnpath+embedding+' '+active+' '+solve+' (train).jpg' # name of file to be saved
        cm=confusion_matrix(Y_train,predict_train) # calculate confusion matrix
        disp=ConfusionMatrixDisplay(confusion_matrix=cm) # ready for plot
        disp.plot() # generate the matrix picture
        plt.savefig(mtrxname1) # save the file
        plt.close() # close plot
        plt.clf() # reset memory
        mtrxname2=cnpath+embedding+' '+active+' '+solve+' (test).jpg' # name of file to be saved
        cm=confusion_matrix(Y_test,predict_test) # calculate confusion matrix
        disp=ConfusionMatrixDisplay(confusion_matrix=cm) # ready for plot
        disp.plot() # generate the matrix picture
        plt.savefig(mtrxname2) # save the file
        plt.close() # close plot
        plt.clf() # reset memory

        # [curves]
        tval1=time.time() # start timer for validation
        if modelname=='CNN-LSTM': # cnn-lstm model

           # (roc)
           fprtrn,tprtrn,_=roc_curve(Y_train,predict_train,drop_intermediate=False) # roc curve calculation for train
           fprtst,tprtst,_=roc_curve(Y_test,predict_test,drop_intermediate=False) # roc curve calculation for test
           smooth_x=np.linspace(0,1,20) # smooth x axis
           smooth_ytrn=np.interp(smooth_x,fprtrn,tprtrn) # smooth train roc
           smooth_ytst=np.interp(smooth_x,fprtst,tprtst) # smooth test roc
           curve_func2=lambda x,a,b: (x**a)+b # regression function
           popt1,_=curve_fit(curve_func2,smooth_x,smooth_ytrn) # fit train scores to curve
           popt2,_=curve_fit(curve_func2,smooth_x,smooth_ytst) # fit test scores to curve
           y1_fit=curve_func2(smooth_x,popt1[0],popt1[1]) # calculate y1
           y2_fit=curve_func2(smooth_x,popt2[0],popt2[1]) # calculate y2
           plt.plot(smooth_x,smooth_x,'--',color="g",linewidth=.5)
           plt.plot(smooth_x,y1_fit,'-',color="m",label="Train ROC") # plot y1 curve for train
           plt.plot(smooth_x,y2_fit,'-',color="c",label="Test ROC") # plot y2 curve for test
           plt.title('ROC Curve') # plot title
           plt.xlabel('False Positives Rate') # x axis label
           plt.ylabel('True Positive Rate')  # y axis label
           plt.legend(loc="lower right") # legend location
           plt.grid(True,which='both',linestyle='--',linewidth=0.5) # grid enabled
           plt.minorticks_on() # axis sections enabled
           imagename2=cvpath+modelname+'-'+embedding+' '+active+' '+solve+' (ROC Curve).jpg' # name of file to be saved
           plt.savefig(imagename2) # save the file
           plt.close() # close the plot
           plt.clf() # reset memory
           # (iterations learning)
           learncurve(trnscrslrn,tstscrslrn,'Performence Score (Accuracy)')
           learncurve(trnf1lrn,tstf1lrn,'Performence Score (F1)')
           learncurve(trnprclrn,tstprclrn,'Performence Score (Precision)')
           learncurve(trnrcllrn,tstrcllrn,'Performence Score (Recall)')
           learncurve(trnauclrn,tstauclrn,'Performence Score (AUC)')
           # (((loss curve for paper)))
           learncurve(trnlosslrn,tstlosslrn,'Cross-Entropy Loss')
           # (((loss curve for paper)))
           """
           # (train size validation)
           clv=0 # counter of validation train size
           for ctsz in tst_szs: # spliting data (different sample numbers)
               clv=clv+1 # counter plus one
               nsz=int((1-ctsz)*len(w2v)) # number of train samples 
               numsize.append(nsz-nsz%10) # list of train size
               X_used,X_dont,Y_used,Y_dont=train_test_split(w2v,usertags,test_size=ctsz) # splitting for train size validation
               X_used=torch.tensor(np.array(X_used).transpose(),dtype=torch.float32).unsqueeze(0) # shape and formats
               Y_used=torch.tensor(np.array(Y_used),dtype=torch.long) # shape and formats
               X_dont=torch.tensor(np.array(X_dont).transpose(),dtype=torch.float32).unsqueeze(0) # shape and formats
               Y_dont=torch.tensor(np.array(Y_dont),dtype=torch.long) # shape and formats
               vld=CNNLSTMClassifier(veclen) # defining model for validating
               optimizer=slvalg(vld.parameters(),lr=d_lr,weight_decay=l2wghtdcy) # selecting solver algorithm
               t1=time.time() # starting validation time
               vld.fit(X_used,Y_used,X_dont,Y_dont) # fit the model on validation dataset
               t2=time.time() # ending validation time
               trnscrsvld.append(round(accuracy_score(Y_used,vld.forward(X_used).argmax(dim=1).tolist()),2)) # kfolds train scores list
               tstscrsvld.append(round(accuracy_score(Y_dont,vld.forward(X_dont).argmax(dim=1).tolist()),2)) # kfolds test scores list
               probaused=vld.forward(X_used) # probability outputs of used set
               probadont=vld.forward(X_dont) # probability outputs of unused set
               # (((loss curve for paper)))
               lossused=lossfunc(probaused,Y_used) # calculating train loss
               trnlossvld.append(round(lossused.item(),4)) # collecting list of train loss for learning curve
               lossdont=lossfunc(probadont,Y_dont) # calculating test loss
               tstlossvld.append(round(lossdont.item(),4)) # collecting list of test loss for learning curve
               # (((loss curve for paper)))
           nsz=int((1-testsize)*extract)
           numsize.append(nsz-nsz%10)
           trnscrsvld.append(train_score)
           tstscrsvld.append(test_score)
           curve_func3=lambda x,a,b: a/(-x**.5)+b # a/(-x**.5)+b # function for regression
           # (((loss curve for paper)))
           losstrn=round(lossfunc(probatrn,Y_train).item(),4) # calculating overall loss
           losstst=round(lossfunc(probatst,Y_test).item(),4) # calculating test loss
           trnlossvld.append(losstrn)
           tstlossvld.append(losstst)
           popt1,_=curve_fit(curve_func3,numsize,trnlossvld) #trnscrsvld) # fitting train scores to curve
           popt2,_=curve_fit(curve_func3,numsize,tstlossvld) #tstscrsvld) # fitting test scores to curve
           # (((loss curve for paper)))
           a_fit1,b_fit1=popt1 # regression parameters for train curve
           a_fit2,b_fit2=popt2 # regression parameters for test curve
           x_fit=np.linspace(numsize[0],numsize[-1],80) # deviding x axis
           y1_fit=curve_func3(x_fit,a_fit1,b_fit1) # calculating y1 (train curve)
           y2_fit=curve_func3(x_fit,a_fit2,b_fit2) # calculating y2 (test curve)
           plt.plot(x_fit,y1_fit,'-',color='m',label='Train Score') # plotting train curve
           plt.plot(x_fit,y2_fit,'-',color='c',label='Test Score') # plotting test curve
           plt.xlabel('Training Set Size') # x axis label
           plt.ylabel('Performence Score (Accuracy)') # y axis label
           plt.title('Validation Curve') # plot name
           plt.legend(loc='lower right') # place of legend
           plt.grid(True,which='both',linestyle='dotted',linewidth=0.5) # grid enabled
           plt.minorticks_on() # axis sections enabled
           plt.gca().set_xticks(np.arange(0,max(numsize),2500)) # applying grid
           # (((loss curve for paper)))
           # plt.gca().set_yticks(np.arange(min((min(trnscrsvld),min(tstscrsvld))*100)%10,max(max(trnscrsvld),max(tstscrsvld)),0.05)) # applying grid
           plt.gca().set_yticks(np.arange(min((min(trnlossvld),min(tstlossvld))*100)%10,max(max(trnlossvld),max(tstlossvld)),0.05)) # applying grid
           # (((loss curve for paper)))
           imagename3=cvpath+modelname+'-'+embedding+' '+active+' '+solve+' (Validation Curve).jpg' # name of file to be saved
           plt.savefig(imagename3) # save the file
           plt.close() # close the plot
           plt.clf() # reset memory
           """
        tval2=time.time() # end validating timer
        tvald=round(tval2-tval1,2) # calculate validating time
        print('                                                                             ') # printing space for removing last epochs report in cmd
        print()
        print()
        print()

        # (save)
        savename=svpath+str(cnt-1) # file name id
        pickle.dump(mdl,open(savename,"wb")) # save model to file for single input bagging ensembles
        lodmod.append(mdl) # save model to list for multi input ensembles
        lodnmd.append(modelname)# save name to list for multi input ensembles
        adlistdf=[modelname,modeltype,embedding,active,solve,basemodel,exclsv,[train_score,test_score],[train_f1scr,test_f1scr],[train_prcsn,test_prcsn],[train_recall,test_recall],[train_auc,test_auc],[train_conf.tolist(),test_conf.tolist()],tdelta,tvald] # row to be added to the dataframe
        df.loc[len(df)]=adlistdf # add row to dataframe

        #--------------------------------------------------------------------------------------------------------------------------------------------





 


































        #============|
        # ((reports))|
        #============|

        #--------------------------------------------------------------------------------------------------------------------------------------------

        print('-----------------------------------------------------------',cnt,'/',lenmix) # states counter
        print()
        print('    Model:',modelname)
        print('    Type:',modeltype)
        if lenopt<=actslv:
           print('    Base Model:',basemodel)
           print('    Optimization Algorithm:',exclsv)
        print('    Embedding:',embedding)
        if actslv<lenlim:
           print('    Activation Function:',active)
           print('    Solver Algorithm:',solve)
        print()
        print('     Scores                 Train          Test')
        print('     Accuracy              ',train_score,'         ',test_score)
        print('     F1score               ',train_f1scr,'         ',test_f1scr)
        print('     Precision             ',train_prcsn,'         ',test_prcsn)
        print('     Recall                ',train_recall,'         ',test_recall)
        print('     AUC                ',train_auc,'         ',test_auc)
        print()
        print('     Confusion Matrix:')
        print('                         ',train_conf[0],'   ',test_conf[0])
        print('                         ',train_conf[1],'   ',test_conf[1])
        print()
        print('     Elapsed Time:')
        print('                   Fitting:',tdelta,'s')
        print('                   Validation:',tvald,'s')
        print()
        print()
        print()
df.index=df.index+1 # start dataframe index from 1
df.to_csv(dfpath,index=False) # save dataframe to csv file
sdf=df.sort_values(by='Accuracy',ascending=False) # sort the dataframe
sdf.to_csv(sdfpath,index=False) # save dataframe to csv file
print('=========================================================================================================================================================================================================================================================================================')
print(tabulate(df,headers=colist)) # report outputs to cmd (models)
print('=========================================================================================================================================================================================================================================================================================')
print()
print('sorting...')
print()
print('=========================================================================================================================================================================================================================================================================================')
print(tabulate(sdf,headers=colist)) # report outputs to cmd (sorted)
print('=========================================================================================================================================================================================================================================================================================')
print()
print()
print()

# (ranks) 
# calculating average time for each classifier and method of embeddings,activation functions and optimizations
ralist=['Section','Method','Accuracy','Average Fitting Time (s)']
rf=pd.DataFrame(columns=ralist)
rf=ranker('Embedding',emblist)
rf=ranker('Activation Function',actlist)
rf=ranker('Solver Algorithm',slvlist)
rf.index=rf.index+1 # start dataframe index from 1
rf.to_csv(rfpath,index=False) # save calculations dataframe as csv file
print('========================================================')
print(tabulate(rf,headers=ralist)) # report outputs to cmd (methods rank)
print('========================================================')
print()
print()
print()
print('========================================================')
print(tabulate(tf,headers=tf.columns)) # report outputs to cmd (times)
print('========================================================')
print()
print()
print()
tf.to_csv(tfpath,index=False) # save method times dataframe as csv file
rtimef=time.time() # calculate total run time
rtimedelta=round((rtimef-rtimes)/60,1) # convert second to minutes
trf=pd.DataFrame([rtimedelta],columns=['Total Run Time (mins)'])
trf.to_csv(trfpath,index=False) # save total run time dataframe as csv file
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print('Total running time (mins):',rtimedelta,'min') # report total run time o cmd
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print()
print()
print()

#--------------------------------------------------------------------------------------------------------------------------------------------