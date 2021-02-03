import numpy as np
from scipy.linalg import eigh

class model:
    
    def __init__(self,data,index):        
        self.data = data
        self.index = index
        
        self.ensembleLen = 100
        self.numTimePred = 3
        self.tauEmb = 1 # number of lead time embedded
    
    def standardize_in_sample(self, is_validation = False):
        if(is_validation):
            self.inSampleEmb_len = self.index.validate_start - self.m *self.tauEmb
        else:
            self.inSampleEmb_len = self.index.test_start - self.m *self.tauEmb
            
        #### X
        self.inSampleX = np.repeat(np.nan,self.inSampleEmb_len * self.m * self.numLocs).reshape(self.inSampleEmb_len, self.m, -1)
        for i in range(self.inSampleEmb_len):
            self.inSampleX[i,] = self.data.ts[range(i,(self.m * self.tauEmb + i), self.tauEmb)]

        self.inSampleX_mean = self.inSampleX.mean(axis=0)
        self.inSampleX_std = self.inSampleX.std(axis=0)
            
        self.inSampleX = (self.inSampleX -self.inSampleX_mean)/self.inSampleX_std
        self.inSampleDesignMatrix = np.column_stack([np.repeat(1,self.inSampleEmb_len),self.inSampleX.reshape(self.inSampleEmb_len,-1)])
        
        #### Y
        self.inSampleY = self.data.ts[range(self.m * self.tauEmb,self.inSampleEmb_len + (self.m * self.tauEmb))]

        self.inSampleY_mean=self.inSampleY.mean(axis=0)
        self.inSampleY_std=self.inSampleY.std(axis=0)

        self.inSampleY = (self.inSampleY-self.inSampleY_mean)/self.inSampleY_std
        
        
    def standardize_out_sample(self, is_validation = False):
        if(is_validation):
            self.outSampleEmb_index = np.arange(self.index.validate_start, self.index.validate_end+1)
        else:
            self.outSampleEmb_index = np.arange(self.index.test_start, self.index.test_end+1)
            
        self.outSampleEmb_len = len(self.outSampleEmb_index)
        
        #### X
        self.outSampleX = np.zeros((self.outSampleEmb_len, self.m, self.numLocs)) * np.nan
        for i,ind in enumerate(self.outSampleEmb_index):
            self.outSampleX[i,] = self.data.ts[range(ind - self.tauEmb * self.m,ind, self.tauEmb)]
        self.outSampleX = (self.outSampleX - self.inSampleX_mean)/self.inSampleX_std
        self.outSampleDesignMatrix=np.column_stack([np.repeat(1,self.outSampleEmb_len),self.outSampleX .reshape(self.outSampleEmb_len,-1)])

        #### Y
        self.outSampleY = (self.data.ts[self.outSampleEmb_index] - self.inSampleY_mean)/self.inSampleY_std
        
    def get_w_and_u(self):
        wMat = np.random.uniform(-self.wWidth,self.wWidth,self.nh*self.nh).reshape(self.nh,-1)
        uMat = np.random.uniform(-self.uWidth,self.uWidth,self.nh*self.nColsU).reshape(self.nColsU,-1)

        #Make W Matrix Sparse 
        for i in range(self.nh):
            numReset=self.nh-np.random.binomial(self.nh,self.wSparsity)
            resetIndex = np.random.choice(self.nh, numReset, replace = False)
            wMat[resetIndex,i]=0

        #Make U Matrix Sparse 
        for i in range(self.nColsU):
            numReset=self.nh-np.random.binomial(self.nh,self.uSparsity)
            resetIndex = np.random.choice(self.nh, numReset, replace = False)
            uMat[i,resetIndex]=0

        #Scale W Matrix
        v = eigh(wMat,eigvals_only=True)
        spectralRadius = max(abs(v))
        wMatScaled=wMat*self.delta/spectralRadius
        
        return wMatScaled, uMat
    
    def get_hMat(self,wMat,uMat):
        #Create H Matrix in-sample
        hMatDim = 2*self.nh
        uProdMat=self.inSampleDesignMatrix.dot(uMat);

        hMat = np.zeros((hMatDim,self.inSampleEmb_len))

        xTemp = uProdMat[0,:]
        xTemp = np.tanh(xTemp)

        hMat[0:self.nh,0] = xTemp
        hMat[self.nh:,0] = xTemp*xTemp

        for t in range(1,self.inSampleEmb_len):
            xTemp = wMat.dot(xTemp)+uProdMat[t,:]
            xTemp = np.tanh(xTemp)

            hMat[0:self.nh,t] = xTemp*self.alpha + hMat[0:self.nh,t-1]*(1-self.alpha)
            hMat[self.nh:,t] = hMat[0:self.nh,t]*hMat[0:self.nh,t]

        #Create H Matrix out-sample
        uProdMatOutSample = self.outSampleDesignMatrix.dot(uMat)
        hMatOutSample = np.zeros((self.outSampleEmb_len,hMatDim))

        xTemp = wMat.dot(xTemp)+uProdMatOutSample[0,:]
        xTemp = np.tanh(xTemp)

        hMatOutSample[0,0:self.nh] = xTemp
        hMatOutSample[0,self.nh:] = xTemp*xTemp

        for t in range(1,self.outSampleEmb_len):
            xTemp = wMat.dot(xTemp)+uProdMatOutSample[t,:]
            xTemp = np.tanh(xTemp)

            hMatOutSample[t,0:self.nh] = xTemp*self.alpha + hMatOutSample[t-1,0:self.nh]*(1-self.alpha)
            hMatOutSample[t,self.nh:] = hMatOutSample[t,0:self.nh]*hMatOutSample[t,0:self.nh]

        return hMat, hMatOutSample
        
    def train(self,hyper_para):
        
        self.m, self.nh, self.ridge, self.delta, self.alpha, self.wWidth, self.uWidth, self.wSparsity, self.uSparsity = hyper_para; 
        
        self.m = int(self.m)
        self.nh = int(self.nh)
        
        self.numLocs = self.data.ts.shape[1]
        
        self.standardize_in_sample()
            
    def forecast(self):
        print("The forecasting (a total of {} ensemble replicates are required) takes about 2 hours to complete".format(self.ensembleLen))
        print("Forecasting ensemble replicate ", end = "")

        self.standardize_out_sample()
        self.forMat = np.ones((self.ensembleLen,self.outSampleEmb_len,self.numLocs,self.numTimePred)) * np.nan
        self.nColsU = self.numLocs * self.m + 1

        for iEnsem in range(self.ensembleLen):
            print(iEnsem+1, end=" ")

            wMat, uMat = self.get_w_and_u();

            hMat, hMatOutSample = self.get_hMat(wMat,uMat);
            
            #Ridge Regression to get out-sample forecast
            tmp = hMat.dot(hMat.transpose())
            np.fill_diagonal(tmp,tmp.diagonal()+self.ridge)

            self.forMat[iEnsem,:,:,0] = hMatOutSample.dot(np.linalg.solve(tmp,hMat.dot(self.inSampleY)))

            #Transform to the original scale
            self.forMat[iEnsem,:,:,0] = self.forMat[iEnsem,:,:,0] * self.inSampleY_std + self.inSampleY_mean
                
            #### Prediction at t + pred_lag + 1 where t is the current time
            hMatDim = 2*self.nh
            for pred_lag in range(1,self.numTimePred):

                #Create H Matrix out-sample for prediction more than one lead time
                self.outSampleX_mixed = self.outSampleX.copy()

                for i in range(min(pred_lag,self.m)):
                    ii = i+1
                    self.outSampleX_mixed[pred_lag:,-ii,:] = (self.forMat[iEnsem,(pred_lag-ii):(-ii),:,pred_lag-ii] - 
                                                              self.inSampleX_mean[-ii])/self.inSampleX_std[-ii]

                self.outSampleX_mixed[0:pred_lag,] = np.nan
                self.outSampleDesignMatrix_mixed = np.column_stack([np.repeat(1,self.outSampleEmb_len),
                                                    self.outSampleX_mixed.reshape(self.outSampleEmb_len,-1)])
    
                uProdMatOutSample = self.outSampleDesignMatrix_mixed.dot(uMat)

                hMatOutSample_new = np.zeros((self.outSampleEmb_len,hMatDim)) * np.nan

                xTemp = hMatOutSample[pred_lag-1,0:self.nh]
                xTemp = wMat.dot(xTemp)+uProdMatOutSample[pred_lag,:]
                xTemp = np.tanh(xTemp)

                hMatOutSample_new[pred_lag,0:self.nh] = xTemp
                hMatOutSample_new[pred_lag,self.nh:] = xTemp*xTemp

                for t in range(pred_lag+1,self.outSampleEmb_len):
                    xTemp = hMatOutSample[t-1,0:self.nh]
                    xTemp = wMat.dot(xTemp)+uProdMatOutSample[t,:]
                    xTemp = np.tanh(xTemp)

                    hMatOutSample_new[t,0:self.nh] = xTemp*self.alpha + hMatOutSample_new[t-1,0:self.nh]*(1-self.alpha)
                    hMatOutSample_new[t,self.nh:] = hMatOutSample_new[t,0:self.nh] * hMatOutSample_new[t,0:self.nh]

                hMatOutSample = hMatOutSample_new.copy()
                
                self.forMat[iEnsem,:,:,pred_lag] = hMatOutSample.dot(np.linalg.solve(tmp,hMat.dot(self.inSampleY)))
                
                #Transform to the original scale
                self.forMat[iEnsem,:,:,pred_lag] = self.forMat[iEnsem,:,:,pred_lag] * self.inSampleY_std + self.inSampleY_mean
    
    def process_results(self):
        self.forMat.sort(axis=0)

        self.quantile5 = self.forMat[int(0.05*self.ensembleLen)]
        self.quantile50 = self.forMat[int(0.5*self.ensembleLen)]
        self.quantile95 = self.forMat[int(0.95*self.ensembleLen)]

    def calculate_MSPE(self):
        diff = self.forMat.mean(axis=0) - self.data.ts[self.outSampleEmb_index]
        self.MSPE = np.mean(diff**2)

    def cross_validation_multiple(self,cv_para):

        self.numLocs = self.data.ts.shape[1]
        
        self.m, self.nh, self.ridge, self.delta, self.alpha, self.wWidth, self.uWidth, self.wSparsity, self.uSparsity = cv_para; 
    
        self.m = int(self.m)
        self.nh = int(self.nh)
        
        self.nColsU = self.numLocs * self.m + 1
        
        self.standardize_in_sample(True)
        
        forMatCV = np.zeros((self.ensembleLen,self.outSampleEmb_len,self.numLocs,self.numTimePred))

        for iEnsem in range(self.ensembleLen):
            wMat, uMat = self.get_w_and_u();

            hMat, hMatOutSample = self.get_hMat(wMat,uMat);

            #Ridge Regression to get out-sample forecast
            tmp = hMat.dot(hMat.transpose())
            np.fill_diagonal(tmp,tmp.diagonal()+self.ridge)

            forMatCV[iEnsem,:,:,0] += hMatOutSample.dot(np.linalg.solve(tmp,hMat.dot(self.inSampleY)))
            
            #### Prediction at t + pred_lag + 1 where t is the current time
            hMatDim = 2*self.nh
            for pred_lag in range(1,self.numTimePred):

                #Create H Matrix out-sample for prediction more than one lead time
                outSampleX_mixed = self.outSampleX.copy()

                for i in range(min(pred_lag,self.m)):
                    ii = i+1
                    forMatCV_scaled_back  = forMatCV[iEnsem,(pred_lag-ii):(-ii),:,pred_lag-ii] * self.inSampleY_std + self.inSampleY_mean
                    outSampleX_mixed[pred_lag:,-ii,:] = (forMatCV_scaled_back - self.inSampleX_mean[-ii])/self.inSampleX_std[-ii]

                outSampleX_mixed[0:pred_lag,] = np.nan
                outSampleDesignMatrix_mixed = np.column_stack([np.repeat(1,self.outSampleEmb_len),
                                                    outSampleX_mixed.reshape(self.outSampleEmb_len,-1)])
    
                uProdMatOutSample = outSampleDesignMatrix_mixed.dot(uMat)

                hMatOutSample_new = np.zeros((self.outSampleEmb_len,hMatDim)) * np.nan

                xTemp = hMatOutSample[pred_lag-1,0:self.nh]
                xTemp = wMat.dot(xTemp)+uProdMatOutSample[pred_lag,:]
                xTemp = np.tanh(xTemp)

                hMatOutSample_new[pred_lag,0:self.nh] = xTemp
                hMatOutSample_new[pred_lag,self.nh:] = xTemp*xTemp

                for t in range(pred_lag+1,self.outSampleEmb_len):
                    xTemp = hMatOutSample[t-1,0:self.nh]
                    xTemp = wMat.dot(xTemp)+uProdMatOutSample[t,:]
                    xTemp = np.tanh(xTemp)

                    hMatOutSample_new[t,0:self.nh] = xTemp*self.alpha + hMatOutSample_new[t-1,0:self.nh]*(1-self.alpha)
                    hMatOutSample_new[t,self.nh:] = hMatOutSample_new[t,0:self.nh] * hMatOutSample_new[t,0:self.nh]

                hMatOutSample = hMatOutSample_new.copy()
                
                forMatCV[iEnsem,:,:,pred_lag] = hMatOutSample.dot(np.linalg.solve(tmp,hMat.dot(self.inSampleY)))
        
        
        forMatCVmean = forMatCV.mean(axis = 0)

        diff = np.ndarray(shape = forMatCVmean.shape) * np.nan

        for i in range(self.numTimePred):
            diff[:,:,i] = forMatCVmean[:,:,i] - self.outSampleY

        MSPE = np.nanmean(diff**2,axis=(0,1))
        
        return MSPE