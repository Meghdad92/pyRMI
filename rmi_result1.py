# house_price.py
# predict price from AC, sq ft, style, nearest school
# PyTorch 1.7.0-CPU Anaconda3-2020.02  Python 3.7.6
# Windows 10 


import numpy as np
import time
import os
import pickle
from itertools import cycle
import torch as T
import pandas as pd
from torch.optim.lr_scheduler import StepLR
device = T.device("cpu")  # apply to Tensor or Module
T.set_num_threads(20)

# Hyper-parameters 
input_size = 1
hidden_size = 32
num_classes = 1
num_epochs = 100
batch_size = 16
learning_rate = 0.001
branching_factor = 10



class AttributeData(T.utils.data.Dataset):
  def __init__(self, src_file=None, att=None, m_rows=None):
    assert src_file is not None
    assert att is not None
    self.src_file = src_file
    self.att=att 
    self.m_rows = m_rows

    df = pd.read_csv(self.src_file, sep=',',usecols=[self.att])
    stats_df = df.groupby('age')['age'].agg('count').pipe(pd.DataFrame).rename(columns = {att: 'frequency'})
    stats_df['pdf'] = stats_df['frequency'] / sum(stats_df['frequency'])
    stats_df['cdf'] = stats_df['pdf'].cumsum()
    stats_df = stats_df.reset_index()
    print ("base file stats:")
    print (stats_df)

    x = stats_df[att].to_numpy().reshape(-1,1)
    y = stats_df["cdf"].to_numpy().reshape(-1,1)


    self.x_tensor = T.as_tensor(x).type(T.FloatTensor).to(device)
    self.y_tensor = T.as_tensor(y).type(T.FloatTensor).to(device)      

  def newdata(self,loop="all",condition=60,updated_file=None):
    if updated_file == None:
      df_base = pd.read_csv(self.src_file, sep=',',usecols=[self.att])
      con = df_base[self.att] > condition
      df_new = df_base[con == True].sample(frac=5,replace=True)


      df = pd.concat([df_base,df_new])

      df.to_csv("data/census_updated.csv",sep=',',index=None,header=True)
    else:
      df = pd.read_csv(updated_file, sep=',',usecols=[self.att])



    stats_df = df.groupby('age')['age'].agg('count').pipe(pd.DataFrame).rename(columns = {self.att: 'frequency'})
    stats_df['pdf'] = stats_df['frequency'] / sum(stats_df['frequency'])
    stats_df['cdf'] = stats_df['pdf'].cumsum()
    stats_df = stats_df.reset_index()
    print ("new file stats:")
    print (stats_df)


    df_transfer = stats_df[stats_df[self.att] <= condition].sample(frac=0.2)
    df_update = stats_df[stats_df[self.att] > condition].sample(frac=1)


    if loop=="all":
      x = stats_df[self.att].to_numpy().reshape(-1,1)
      y = stats_df["cdf"].to_numpy().reshape(-1,1)
    elif loop=="transferset":
      x = df_transfer[self.att].to_numpy().reshape(-1,1)
      y = df_transfer["cdf"].to_numpy().reshape(-1,1)
      print ("transferset shape:", x.shape)
    elif loop=="updatebatch":
      x = df_update[self.att].to_numpy().reshape(-1,1)
      y = df_update["cdf"].to_numpy().reshape(-1,1)
      print ("updatebatch shape:",x.shape)

    self.x_tensor = None
    self.y_tensor = None
    self.x_tensor = T.as_tensor(x).type(T.FloatTensor).to(device)
    self.y_tensor = T.as_tensor(y).type(T.FloatTensor).to(device)


    
  def __len__(self):
    return len(self.x_tensor)

  def __getitem__(self, idx):
    preds = self.x_tensor[idx,:]  # or just [idx]
    price = self.y_tensor[idx,:] 
    return (preds, price)       # tuple of two matrices

class SyntheticIntegers(T.utils.data.Dataset):
  def __init__(self, src_file=None, m_rows=None):
    if src_file==None:
      # Synthetic data
      #data = np.random.randint(0, 1000000, size=100000, dtype="int32")
      data = np.random.normal(loc=1000000,scale=5,size=10000)
      data = [int(i) for i in data]
      _min = np.min(data)
      _max = np.max(data)
      #data = [[(i-_min)/(_max-_min)] for i in data]
      data = [[i] for i in data]
      idxs = sorted(range(len(data)), key=lambda k: data[k])
      _min = np.min(idxs)
      _max = np.max(idxs)
      idxs = [[(i-_min)/(_max-_min)] for i in idxs]
      #idxs = [[i] for i in idxs]
      

      dataset = np.concatenate((data, idxs),axis=1)
      fmt = '%d', '%1.4f'
      #np.savetxt("SyntheticIntegers.txt",dataset,delimiter='\t',fmt=fmt)
    else:
      all_xy = np.loadtxt(src_file, max_rows=m_rows,usecols=[0,1], delimiter="\t",comments="#", skiprows=0, dtype=np.float32)

      data = all_xy[:,0]
      idxs = all_xy[:,1]

      data = [[i] for i in data]
      idxs = [[i] for i in idxs]



    self.x_tensor = T.as_tensor(data).type(T.FloatTensor).to(device)
    self.y_tensor = T.as_tensor(idxs).type(T.FloatTensor).to(device)
    
    
    # Data loader
    #train_loader = T.utils.data.DataLoader(dataset=(x_tensor,y_tensor), batch_size=batch_size, shuffle=True)
    
    #test_loader = T.utils.data.DataLoader(dataset=(x_tensor,y_tensor), batch_size=10, shuffle=False)

    #self.x_data = T.tensor(tmp_x, \ dtype=T.float32).to(device)
    #self.y_data = T.tensor(tmp_y, \ dtype=T.float32).to(device)

  def __len__(self):
    return len(self.x_tensor)

  def __getitem__(self, idx):
    preds = self.x_tensor[idx,:]  # or just [idx]
    price = self.y_tensor[idx,:] 
    return (preds, price)       # tuple of two matrices 

class HouseDataset(T.utils.data.Dataset):
  # AC  sq ft   style  price   school
  # -1  0.2500  0 1 0  0.5650  0 1 0
  #  1  0.1275  1 0 0  0.3710  0 0 1
  # air condition: -1 = no, +1 = yes
  # style: art_deco, bungalow, colonial
  # school: johnson, kennedy, lincoln

  def __init__(self, src_file, m_rows=None, frac=None):
    all_xy = np.loadtxt(src_file, max_rows=m_rows,
      usecols=[0,1,2,3,4,5,6,7,8], delimiter="\t",
      # usecols=range(0,9), delimiter="\t",
      comments="#", skiprows=0, dtype=np.float32)

    if frac is not None:
      idx = np.random.randint(all_xy.shape[0], size=int(all_xy.shape[0]*frac))
      all_xy = all_xy[idx,:]

    tmp_x = all_xy[:,[0,1,2,3,4,6,7,8]]
    tmp_y = all_xy[:,5].reshape(-1,1)    # 2-D required

    self.x_data = T.tensor(tmp_x, \
      dtype=T.float32).to(device)
    self.y_data = T.tensor(tmp_y, \
      dtype=T.float32).to(device)

  def __len__(self):
    return len(self.x_data)

  def __getitem__(self, idx):
    preds = self.x_data[idx,:]  # or just [idx]
    price = self.y_data[idx,:] 
    return (preds, price)       # tuple of two matrices 

class HouseDatasetPermuted(T.utils.data.Dataset):

  def __init__(self, src_file, m_rows=None):
    columns = [0,1,2,3,4,5,6,7,8]  
    all_xy = np.loadtxt(src_file, max_rows=m_rows,
      usecols=columns, delimiter="\t",
      # usecols=range(0,9), delimiter="\t",
      comments="#", skiprows=0, dtype=np.float32)


    all_xy_permuted = None
    for i in columns:
        clm = all_xy[:,i]
        clm = np.sort(clm).reshape(-1,1)
        if i == 0:
            all_xy_permuted = clm
        else:
            all_xy_permuted = np.concatenate((all_xy_permuted,clm),axis=1)

    idx = np.random.randint(all_xy_permuted.shape[0], size=int(all_xy_permuted.shape[0]*0.2))
    all_xy_permuted = all_xy_permuted[idx,:]
    
    new_dataset = np.concatenate((all_xy, all_xy_permuted))
    fmt = '%d', '%1.4f', '%d', '%d', '%d', '%1.4f', '%d', '%d', '%d'
    np.savetxt("new_dataset.txt",new_dataset,delimiter='\t',fmt=fmt)

    
    
    tmp_x = all_xy_permuted[:,[0,1,2,3,4,6,7,8]]
    tmp_y = all_xy_permuted[:,5].reshape(-1,1)    # 2-D required

    self.x_data = T.tensor(tmp_x, \
      dtype=T.float32).to(device)
    self.y_data = T.tensor(tmp_y, \
      dtype=T.float32).to(device)

  def __len__(self):
    return len(self.x_data)

  def __getitem__(self, idx):
    preds = self.x_data[idx,:]  # or just [idx]
    price = self.y_data[idx,:] 
    return (preds, price)       # tuple of two matrices 

class Net(T.nn.Module):
  def __init__(self, inputSize, outputSize):
    super(Net, self).__init__()
    self.hid1 = T.nn.Linear(inputSize, 32)  # 8-(10-10)-1
    self.hid2 = T.nn.Linear(32, 32)
    self.oupt = T.nn.Linear(32, outputSize)

    T.nn.init.xavier_uniform_(self.hid1.weight)
    T.nn.init.zeros_(self.hid1.bias)
    T.nn.init.xavier_uniform_(self.hid2.weight)
    T.nn.init.zeros_(self.hid2.bias)
    T.nn.init.xavier_uniform_(self.oupt.weight)
    T.nn.init.zeros_(self.oupt.bias)

  def forward(self, x):
    z = T.relu(self.hid1(x))
    z = T.relu(self.hid2(z))
    z = self.oupt(z)  # no activation
    return z

class linearRegression(T.nn.Module):
  def __init__(self, inputSize, outputSize):
    super(linearRegression, self).__init__()
    self.linear = T.nn.Linear(inputSize, outputSize)

  def forward(self, x):
    out = self.linear(x)
    return out

def create_model(_type, inputSize, outputSize):
  if _type == "linear":
    return linearRegression(inputSize, outputSize).to(device)
  elif _type == "neural_net":
    return Net(inputSize, outputSize).to(device)

        
class RMI(Net):
  def __init__(self, num_stages=2, branching_factor=5, model_types=["linear","neural_net"], input_size=8, output_size=1):
    super(RMI, self).__init__(input_size, output_size)
    self.B = branching_factor
    self.S = num_stages
    self.types = model_types
    assert len(self.types)==self.S or len(self.types)==1
    self.inputSize = input_size
    self.outputSize = output_size
    self.rmi_list = []
    
    self.build_RMI()

    
  def build_RMI(self):
    if self.S<3:
      if len(self.types)==1:
        self.types = self.types*self.S
      
      self.rmi_list.append(create_model(self.types[0], self.inputSize, self.outputSize))
      for s in range(1,self.S):
        for b in range(self.B):
          self.rmi_list.append(create_model(self.types[s], self.inputSize, self.outputSize))

    else:
      raise ValueError('RMI with more than two stages is not implemented yet!')
      

  def train_single_model(self,model_idx,data_ldr):
      bat_size = 128          
      max_epochs = 500
      ep_log_interval = 50
      lrn_rate = 0.01
    
      loss_func = T.nn.MSELoss()
      # optimizer = T.optim.SGD(net.parameters(), lr=lrn_rate)
      optimizer = T.optim.Adam(self.rmi_list[model_idx].parameters(), lr=lrn_rate)
      scheduler = StepLR(optimizer, step_size=1, gamma=0.96)

      self.rmi_list[model_idx].train()  # set mode
      for epoch in range(0, max_epochs):
        T.manual_seed(1+epoch)  # recovery reproducibility
        epoch_loss = 0  # for one full epoch
        #scheduler.step()
        for batch_idx, (X,Y) in enumerate(data_ldr):
          X = X.to(device)
          Y = Y.to(device)       
          optimizer.zero_grad()          # prepare gradients
          oupt = self.rmi_list[model_idx](X).to(device)                  # predicted prices
          loss_val = loss_func(oupt, Y).to(device)  # avg per item in batch
          epoch_loss += loss_val.item()  # accumulate avgs
          loss_val.backward()            # compute gradients
          optimizer.step()               # update wts
    
        if epoch % ep_log_interval == 0:
          print("epoch = %4d   loss = %0.4f" % \
           (epoch, epoch_loss))

    
      self.rmi_list[model_idx].eval()
      print("Done ")

  def finetune_single_model(self,model_idx,data_ldr):

      max_epochs = 500
      ep_log_interval = 50
      lrn_rate = 0.0002
    
      loss_func = T.nn.MSELoss()
      # optimizer = T.optim.SGD(net.parameters(), lr=lrn_rate)
      optimizer = T.optim.Adam(self.rmi_list[model_idx].parameters(), lr=lrn_rate)
      scheduler = StepLR(optimizer, step_size=1, gamma=0.96)
    
      print("\nStart update by finetuning node {}".format(model_idx))

      self.rmi_list[model_idx].train()  # set mode
      for epoch in range(0, max_epochs):
        T.manual_seed(1+epoch)  # recovery reproducibility
        epoch_loss = 0  # for one full epoch
        #scheduler.step()
        for  X, Y in data_ldr:
                 
          optimizer.zero_grad()       
       
          oupt3 = self.rmi_list[model_idx](X)
          

          loss_val = loss_func(oupt3, Y)
          
          
          epoch_loss += loss_val.item()  
          loss_val.backward()     
          optimizer.step()          
    
        if epoch % ep_log_interval == 0:
          print("epoch = %4d   loss = %0.4f" % (epoch, epoch_loss))

    
      self.rmi_list[model_idx].eval()
      print("Done ")

  def update_single_model(self,model_idx,data_ldr,transfer_set_ldr, lamda=0.99, alpha=1/2):
      
      max_epochs = 500
      ep_log_interval = 50
      lrn_rate = 0.01
    
      loss_func = T.nn.MSELoss()
      # optimizer = T.optim.SGD(net.parameters(), lr=lrn_rate)
      optimizer = T.optim.Adam(self.rmi_list[model_idx].parameters(), lr=lrn_rate)
      scheduler = StepLR(optimizer, step_size=1, gamma=0.96)

    
      print("\nStart update by distilling node {}".format(model_idx))
      pre_model = self.rmi_list[model_idx]
      pre_model.eval()
      self.rmi_list[model_idx].train()  # set mode


      for epoch in range(0, max_epochs):
        T.manual_seed(1+epoch)  # recovery reproducibility
        epoch_loss = 0  # for one full epoch
        #scheduler.step()
        if len(transfer_set_ldr) > len(data_ldr):
          for (X_tr, Y_tr), (X, Y) in (zip(transfer_set_ldr, cycle(data_ldr))):                   
            optimizer.zero_grad()       
            oupt1 = pre_model(X_tr)
            oupt2 = self.rmi_list[model_idx](X_tr)          
            oupt3 = self.rmi_list[model_idx](X)
            
            loss1 = kd_loss(oupt1, oupt2)
            loss2 = loss_func(oupt2, Y_tr)
            loss3 = loss_func(oupt3, Y)
            
            loss_val = (1-lamda)*((1-alpha)*loss1 + alpha*loss2) + lamda*loss3
            
            epoch_loss += loss_val.item()  
            loss_val.backward()     
            optimizer.step()          
      
          if epoch % ep_log_interval == 0:
            print("epoch = %4d   loss = %0.4f" % (epoch, epoch_loss))

        else:
          for (X, Y), (X_tr, Y_tr) in (zip(data_ldr, cycle(transfer_set_ldr))):          
            optimizer.zero_grad()       
            oupt1 = pre_model(X_tr)
            oupt2 = self.rmi_list[model_idx](X_tr)          
            oupt3 = self.rmi_list[model_idx](X)
            
            loss1 = kd_loss(oupt1, oupt2)
            loss2 = loss_func(oupt2, Y_tr)
            loss3 = loss_func(oupt3, Y)
            
            loss_val = (1-lamda)*((1-alpha)*loss1 + alpha*loss2) + lamda*loss3
            
            epoch_loss += loss_val.item()  
            loss_val.backward()     
            optimizer.step()          
      
          if epoch % ep_log_interval == 0:
            print("epoch = %4d   loss = %0.4f" % (epoch, epoch_loss))


    
      self.rmi_list[model_idx].eval()
      print("Done ")

  def train(self, dataset, src_file, att):
      if dataset == "synthetic":
        train_ds = SyntheticIntegers()
        bat_size = 5
        data_ldr = T.utils.data.DataLoader(train_ds,
          batch_size=bat_size, shuffle=True)          
      elif dataset == "attributes":
        train_ds = AttributeData(src_file=src_file, att=att)
        bat_size = 5
        data_ldr = T.utils.data.DataLoader(train_ds,batch_size=bat_size, shuffle=True)
      elif dataset == "HouseDataset":
        train_ds = HouseDataset(src_file=src_file)
        bat_size = 5
        data_ldr = T.utils.data.DataLoader(train_ds,
          batch_size=bat_size, shuffle=True)       
      else:
        raise("Dataset not found!")

      print ("training root of RMI ...")
      self.train_single_model(0, data_ldr)
      
      self.rmi_list[0].eval()


      self._max = -np.inf
      self._min = np.inf
      for (batch_idx, batch) in enumerate(data_ldr):
         (X, Y) = batch
         oupt = self.rmi_list[0](X)
         _max = oupt.max().item()
         _min = oupt.min().item()

         if _max > self._max:
             self._max = _max
         if _min < self._min:
             self._min = _min
      print ("min prediction = {} and max prediction = {}".format(self._min, self._max))
      print ("splitting data to the branches ...")
      cat_X = None
      cat_Y = None
      cat_out = None
      for (batch_idx, batch) in enumerate(data_ldr):
         (X, Y) = batch
         oupt = self.rmi_list[0](X)
  
         idx = T.floor((oupt-self._min)/(self._max-self._min) * (self.B-1)).int() + 1
         if batch_idx==0:
             cat_X = X
             cat_Y = Y
             cat_out = idx
         else:
            cat_X = T.cat((cat_X,X))
            cat_Y = T.cat((cat_Y,Y))      
            cat_out = T.cat((cat_out,idx))
 

      sub_X = {}
      sub_Y = {}
      for i in range(len(cat_out)):
          if cat_out[i].item() not in sub_X.keys():
              sub_X[cat_out[i].item()] = cat_X[i].reshape(1,-1)
              sub_Y[cat_out[i].item()] = cat_Y[i].reshape(1,-1)
          else:
              sub_X[cat_out[i].item()] = T.cat((sub_X[cat_out[i].item()], cat_X[i].reshape(1,-1)))
              sub_Y[cat_out[i].item()] = T.cat((sub_Y[cat_out[i].item()], cat_Y[i].reshape(1,-1)))

      print ("training each node ...\n\n")
      bat_size = 5#int(bat_size/self.B) 
      for sub in sub_X.keys():
          print ("training node {}".format(sub))

          sub_dataset = T.utils.data.TensorDataset(sub_X[sub],sub_Y[sub])
          sub_data_ldr = T.utils.data.DataLoader(sub_dataset,batch_size=bat_size, shuffle=True)
      

          
          self.train_single_model(sub, sub_data_ldr)
          
  def incremental_update(self,dataset, src_file, att):
      if dataset == "synthetic":
        train_ds = SyntheticIntegers()
        bat_size = 5
        data_ldr = T.utils.data.DataLoader(train_ds,batch_size=bat_size, shuffle=True)          
      elif dataset == "attributes":
        train_ds = AttributeData(src_file=src_file, att=att)
        train_ds.newdata(loop="updatebatch")
        bat_size = 5
        update_batch_ldr = T.utils.data.DataLoader(train_ds,batch_size=bat_size, shuffle=True)

        train_ds.newdata(loop="transferset",updated_file="data/census_updated.csv")
        transfer_ldr = T.utils.data.DataLoader(train_ds,batch_size=bat_size, shuffle=True)

      elif dataset == "HouseDataset":
        train_ds = HouseDataset(src_file=src_file)
        bat_size = 5
        data_ldr = T.utils.data.DataLoader(train_ds,batch_size=bat_size, shuffle=True)       
      else:
        raise("Dataset not found!")
          
    
      loss_func = T.nn.MSELoss()

      self.update_single_model(0, update_batch_ldr, transfer_ldr)
      #self.finetune_single_model(0, update_batch_ldr)
      
      self.rmi_list[0].eval()


      for (batch_idx, batch) in enumerate(update_batch_ldr):
         (X, Y) = batch
         oupt = self.rmi_list[0](X)
         _max = oupt.max().item()
         _min = oupt.min().item()

         if _max > self._max:
             self._max = _max
         if _min < self._min:
             self._min = _min

      for (batch_idx, batch) in enumerate(transfer_ldr):
         (X, Y) = batch
         oupt = self.rmi_list[0](X)
         _max = oupt.max().item()
         _min = oupt.min().item()

         if _max > self._max:
             self._max = _max
         if _min < self._min:
             self._min = _min

      print ("\nsplitting data to the branches ...")



      cat_X = None
      cat_Y = None
      cat_out = None
      for (batch_idx, batch) in enumerate(update_batch_ldr):
         (X, Y) = batch
         oupt = self.rmi_list[0](X)
         idx = T.floor((oupt-self._min)/(self._max-self._min) * (self.B-1)+1).int()
         if batch_idx==0:
             cat_X = X
             cat_Y = Y
             cat_out = idx
         else:
            cat_X = T.cat((cat_X,X))
            cat_Y = T.cat((cat_Y,Y))      
            cat_out = T.cat((cat_out,idx))
 

      sub_X = {}
      sub_Y = {}
      for i in range(len(cat_out)):
          if cat_out[i].item() not in sub_X.keys():
              sub_X[cat_out[i].item()] = cat_X[i].reshape(1,-1)
              sub_Y[cat_out[i].item()] = cat_Y[i].reshape(1,-1)
          else:
              sub_X[cat_out[i].item()] = T.cat((sub_X[cat_out[i].item()], cat_X[i].reshape(1,-1)))
              sub_Y[cat_out[i].item()] = T.cat((sub_Y[cat_out[i].item()], cat_Y[i].reshape(1,-1)))



      cat_X_tr = None
      cat_Y_tr = None
      cat_out_tr = None
      for (batch_idx, batch) in enumerate(transfer_ldr):
         (X, Y) = batch
         oupt = self.rmi_list[0](X)
         idx = T.floor((oupt-self._min)/(self._max-self._min) * (self.B-1)+1).int()
         if batch_idx==0:
             cat_X_tr = X
             cat_Y_tr = Y
             cat_out_tr = idx
         else:
            cat_X_tr = T.cat((cat_X_tr,X))
            cat_Y_tr = T.cat((cat_Y_tr,Y))      
            cat_out_tr = T.cat((cat_out_tr,idx))
 

      sub_X_tr = {}
      sub_Y_tr = {}
      for i in range(len(cat_out_tr)):
          if cat_out_tr[i].item() not in sub_X_tr.keys():
              sub_X_tr[cat_out_tr[i].item()] = cat_X_tr[i].reshape(1,-1)
              sub_Y_tr[cat_out_tr[i].item()] = cat_Y_tr[i].reshape(1,-1)
          else:
              sub_X_tr[cat_out_tr[i].item()] = T.cat((sub_X_tr[cat_out_tr[i].item()], cat_X_tr[i].reshape(1,-1)))
              sub_Y_tr[cat_out_tr[i].item()] = T.cat((sub_Y_tr[cat_out_tr[i].item()], cat_Y_tr[i].reshape(1,-1)))


      bat_size = 5#int(bat_size/self.B)

      for sub in sub_X_tr.keys():
          
          sub_dataset = T.utils.data.TensorDataset(sub_X[sub],sub_Y[sub])
          sub_transfer = T.utils.data.TensorDataset(sub_X_tr[sub],sub_Y_tr[sub])
          sub_data_ldr = T.utils.data.DataLoader(sub_dataset,batch_size=bat_size, shuffle=True)
          sub_transfer_ldr = T.utils.data.DataLoader(sub_transfer,batch_size=bat_size, shuffle=True)
          
          self.update_single_model(sub, sub_data_ldr, sub_transfer_ldr)
          #self.finetune_single_model(sub, sub_data_ldr)

  def predict_all(self,dataset, src_file, att):
      if dataset == "synthetic":
        train_ds = SyntheticIntegers()
        bat_size = 5
        data_ldr = T.utils.data.DataLoader(train_ds,batch_size=bat_size, shuffle=True)          
      elif dataset == "attributes":
        train_ds = AttributeData(src_file=src_file, att=att)
        bat_size = 5
        data_ldr = T.utils.data.DataLoader(train_ds,batch_size=bat_size, shuffle=True)
      elif dataset == "HouseDataset":
        train_ds = HouseDataset(src_file=src_file)
        bat_size = 5
        data_ldr = T.utils.data.DataLoader(train_ds,
          batch_size=bat_size, shuffle=True)       
      else:
        raise("Dataset not found!")
          


      cat_X = None
      cat_Y = None
      cat_out = None
      for (batch_idx, batch) in enumerate(train_ds):
         (X, Y) = batch
         oupt = self.rmi_list[0](X)
         if oupt.item() > self._max:
             self._max = oupt.item()
         if oupt.item() < self._min:
             self._min = oupt.item()
         idx = T.floor((oupt-self._min)/(self._max-self._min) * (self.B-1)+1).int()
         if batch_idx==0:
             cat_X = X
             cat_Y = Y
             cat_out = idx
         else:
            cat_X = T.cat((cat_X,X))
            cat_Y = T.cat((cat_Y,Y))      
            cat_out = T.cat((cat_out,idx))
 

      sub_X = {}
      sub_Y = {}
      for i in range(len(cat_out)):
          if cat_out[i].item() not in sub_X.keys():
              sub_X[cat_out[i].item()] = cat_X[i].reshape(1,-1)
              sub_Y[cat_out[i].item()] = cat_Y[i].reshape(1,-1)
          else:
              sub_X[cat_out[i].item()] = T.cat((sub_X[cat_out[i].item()], cat_X[i].reshape(1,-1)))
              sub_Y[cat_out[i].item()] = T.cat((sub_Y[cat_out[i].item()], cat_Y[i].reshape(1,-1)))


      print (np.sort(list(sub_X.keys())))
      total_err = []
      means = []
      medians= []
      percentiles = []
      for sub in sub_X.keys():
          try:
            sub_dataset = T.utils.data.TensorDataset(sub_X[sub],sub_Y[sub])          
            err,_mean,_median,_percentile = rel_err(self.rmi_list[sub], sub_dataset)
            means.append(_mean)
            medians.append(_median)
            percentiles.append(_percentile)
            total_err = total_err + err
          except:
            print ("exit with errors in predict_all!")
            return


      print ("sub_node\tsize\tmean_err\tmedian_err\t95th_err")
      for i,sub in enumerate(np.sort(list(sub_X.keys()))):
          print ("{}\t{}\t{:.2f}\t{:.2f}\t{:.2f}".format(sub, sub_X[sub].shape[0],means[i], medians[i],percentiles[i]))
      print ("\nrelative errors :")
      print ("total mean= {:0.2}\t total median= {:0.2}\t total 95th= {:0.2}".format(np.mean(total_err),np.median(total_err),np.percentile(total_err,95)))         

        
def kd_loss(out_old, out_new,device='cpu'):
    distance = T.square(out_old - out_new)
    return T.mean(distance)

def rel_err(model, ds):

  err = []
  i = 0
  for X, Y in ds:
    #(X, Y) = x_tensor[i],y_tensor[i]            # (predictors, target)
    with T.no_grad():
      oupt = model(X)         # computed price

    y = Y.item()
    if Y.item()==0:
      y = 0.00000000009
    #print (X,Y,oupt)
    i  = i +1
    if i==1000000:
      os.sys.exit()
    rel_err = np.abs(oupt.item()-y)/y * 100
    err.append(rel_err)


  return err, np.mean(err), np.median(err), np.percentile(err,95)


def update():
  T.manual_seed(4)  # representative results 
  np.random.seed(4)
  
  
  rmi = None
  with open('base.pckl', 'rb') as f:
      rmi = pickle.load(f)
      
  rmi.incremental_update(dataset="attributes", src_file="data/census.csv",att="age")

  with open('updated.pckl', 'wb') as f:
      pickle.dump(rmi,f)

def train():
  T.manual_seed(4)  # representative results 
  np.random.seed(4)


  rmi = RMI(input_size=1)
  rmi.train(dataset="attributes", src_file="data/census.csv",att="age")

  with open('base.pckl', 'wb') as f:
      pickle.dump(rmi,f)


def test():

  rmi = None
  with open('updated.pckl', 'rb') as f:
      rmi = pickle.load(f)

  rmi.predict_all(dataset="attributes", src_file="data/census_updated.csv",att="age") 

if __name__ == "__main__":
  #train()
  update()
  test() 



