
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from tkinter import *
from tkinter import messagebox
from PIL import ImageTk, Image
from sklearn import svm

df = pd.read_excel('Dry_Bean_Dataset.xlsx')
Info = {'FeaturesIndex': -1,'CombIndex':-1,'Learning_rate':0,'numEphocs':0,'MSE_Threshold':0.5,'boolBias':False, 'selectedAlgorithm': ' '}
selectedFeatures = []
SelectedClasses = []
data = pd.DataFrame()
master = Tk() 
master.title("Dry Beans Classifier")
master.geometry('800x600')




# background_image=PhotoImage('background.png')
# background_label = Label(master, image=background_image)
# background_label.place(x=0, y=0, relwidth=1, relheight=1)

def getInfo():
        
        #Get status of checkboxes
        
        F = {'Area':feature1.get(),'Perimeter':feature2.get(),'MajorAxisLength':feature3.get(),'MinorAxisLength':feature4.get(),'Roundness':feature5.get()}
        if(F['Area'] == 1):
            selectedFeatures.append('Area')
        if(F['Perimeter']==1):
            selectedFeatures.append('Perimeter')
        if(F['MajorAxisLength']==1):
            selectedFeatures.append('MajorAxisLength')
        if(F['MinorAxisLength']==1):
            selectedFeatures.append('MinorAxisLength')
        if(F['Roundness']==1):
            selectedFeatures.append('Roundness')

        if(len(selectedFeatures) != 2):
            messagebox.showinfo('Error!', "Please Select Exactly two features!")

        #for i in range (len(selectedFeatures)):
        if(selectedFeatures[0] == 'Area'): 
            if(selectedFeatures[1] == 'Perimeter'):
                Info['FeaturesIndex'] = 0
            elif(selectedFeatures[1] == 'MajorAxisLength'):
                Info['FeaturesIndex'] = 1
            elif(selectedFeatures[1] == 'MinorAxisLength'):
                Info['FeaturesIndex'] = 2
            elif(selectedFeatures[1] == 'Roundness'):
                Info['FeaturesIndex'] = 3
        elif(selectedFeatures[0] == 'Perimieter'):
            if(selectedFeatures[1] == 'MajorAxisLength'):
                Info['FeaturesIndex']= 4
            elif(selectedFeatures[1] == 'MinorAxisLength'):
                Info['FeaturesIndex']= 5
            elif(selectedFeatures[1] == 'Roundness'):
                Info['FeaturesIndex'] = 6
        elif(selectedFeatures[0] == 'MajorAxisLength'):
            if(selectedFeatures[1] == 'MinorAxisLength'):
                Info['FeaturesIndex'] = 7
            elif(selectedFeatures[1] == 'Roundness'): 
                Info['FeaturesIndex']= 8
        else:
            Info['FeaturesIndex'] = 9
                
        #print(selectedFeatures)
    #------------------------------
        C = {'comb1':comb1.get(),'comb2':comb2.get(),'comb3':comb3.get()}
        
    
        if(C['comb1']==1):
            SelectedClasses.append('Bomay')
            SelectedClasses.append('Cali')
            Info['CombIndex'] = 1
        if(C['comb2']==2):
            SelectedClasses.append('Bomay')
            SelectedClasses.append('Sira')
            Info['CombIndex'] = 2
        if(C['comb3']==3):
            Info['CombIndex'] = 3
            SelectedClasses.append('Sira')
            SelectedClasses.append('Cali')

        if(len(SelectedClasses) != 2):
            messagebox.showinfo('Error!', 'Please select only 1 combination of classes')
        #print(SelectedClasses)
    #---------------------------------
        Info['Learning_rate'] = float(e.get())
        #print(Learning_rate)
    #----------------------------------
        if(int(a.get()) < 0):
            messagebox.showinfo('Error!', 'Please enter appropriate value of learning rate!')

        Info['numEphocs'] = int(a.get())
        #print(Info['numEphocs'])
    #------------------------------------
        Info['MSE_Threshold'] = float(MSE.get())
        #print(MSE_Threshold)
    #------------------------------------
        Info['boolBias'] = bias.get()
        #print(boolBias)
    #------------------------------------
        A = {'Adaline':Algorithm1.get(),'Perceptron':Algorithm2.get()}
        if((A['Adaline'] == 1)&(A['Perceptron']==2)):
            messagebox.showinfo('Error!', 'Please choose 1 Algorithm!')
        elif(A['Adaline'] == 1):
            Info['selectedAlgorithm'] = 'Adaline'
        elif(A['Perceptron']==2):
            Info['selectedAlgorithm']  = 'Perceptron'
        else:
            messagebox.showinfo('Error!', 'Please choose 1 Algorithm!')
        #print(selectedAlgorithm)

        print('==========================info======================================')
        print(Info)
        df.isnull().sum()

        df.duplicated().sum()

        df['MinorAxisLength'].fillna(df['MinorAxisLength'].median(), inplace=True)

        df.isnull().sum()


        x=df.drop(columns=['Class'])



        y=df['Class']

        # Define the features
        features = x.columns

        # Define the classes
        classes = y.unique()
        # print(classes)

        # Generate all combinations of features and classes
        feature_combinations = []
        for i in range(len(features)):
            for j in range(i+1, len(features)):
                feature_combinations.append([features[i], features[j]])


        def getfeatures(index ,x):
            z = feature_combinations[index]
            return x[z]


        scaler = StandardScaler()
        x = scaler.fit_transform(x)

        x = pd.DataFrame(x, columns=features)
        data = pd.concat([x, y], axis=1)


        c1=data[0:50] #bombai
        c2=data[50:100] #cali
        c3=data[100:]  #sira



        x0=c1.drop(columns=['Class'])
        y0=c1['Class']


        x1=c2.drop(columns=['Class'])
        y1=c2['Class']


        x2=c3.drop(columns=['Class'])
        y2=c3['Class']



        #[['BOMBAY', 'CALI'], ['BOMBAY', 'SIRA'], ['CALI', 'SIRA']]
        from sklearn.model_selection import train_test_split
        X_train0, X_test0, y_train0, y_test0 = train_test_split(x0, y0, test_size =0.4, random_state = 0) #bomai

        from sklearn.model_selection import train_test_split
        X_train1, X_test1, y_train1, y_test1 = train_test_split(x1, y1, test_size =0.4, random_state = 0) #cali

        from sklearn.model_selection import train_test_split
        X_train2, X_test2, y_train2, y_test2 = train_test_split(x2, y2, test_size =0.4, random_state = 0) #sira


        def convertclasses(index):
            if(index == 1) :
                combined_X_train = pd.concat([X_train0, X_train1], axis=0)
                combined_y_train = pd.concat([y_train0, y_train1], axis=0)

                combined_X_test = pd.concat([X_test0, X_test1], axis=0)
                combined_y_test = pd.concat([y_test0, y_test1], axis=0)
            elif(index == 2) :
                combined_X_train = pd.concat([X_train0, X_train2], axis=0)
                combined_y_train = pd.concat([y_train0, y_train2], axis=0)

                combined_X_test = pd.concat([X_test0, X_test2], axis=0)
                combined_y_test = pd.concat([y_test0, y_test2], axis=0)
            elif(index == 3) :
                combined_X_train = pd.concat([X_train1, X_train2], axis=0)
                combined_y_train = pd.concat([y_train1, y_train2], axis=0)

                combined_X_test = pd.concat([X_test1, X_test2], axis=0)
                combined_y_test = pd.concat([y_test1, y_test2], axis=0)

            return combined_X_train ,combined_y_train ,combined_X_test ,combined_y_test

        def ReformatY(y_, index):
            target = []
            for val in y_:
                if index == 1:
                    if val == 'CALI':
                        target.append(1)
                    elif val == 'BOMBAY':
                        target.append(-1)
                elif index == 2:
                    if val == 'SIRA':
                        target.append(1)
                    elif val == 'BOMBAY':
                        target.append(-1)
                else:
                    if val == 'SIRA':
                        target.append(1)
                    elif val == 'CALI':
                        target.append(-1)
            return target


        xtrain , ytrain , xtest , ytest = convertclasses(Info['CombIndex'])
        


        xtrain = getfeatures(Info['FeaturesIndex']  ,xtrain )
        xtest = getfeatures(Info['FeaturesIndex']  ,xtest )

        # print(xtrain)
        # print(ytrain)

        ytrain = ReformatY(ytrain , Info['CombIndex'])
        ytest = ReformatY(ytest , Info['CombIndex'])
        # print(ytest)

        def confusion_matrix(true_labels, predicted_scores):
            TP = FP = TN = FN = 0

            for true, predicted in zip(true_labels, predicted_scores):
                if true == 1 and predicted == 1:
                    TP += 1
                elif true == -1 and predicted == 1:
                    FP += 1
                elif true == -1 and predicted == -1:
                    TN += 1
                elif true == 1 and predicted == -1:
                    FN += 1

                confusion_mat = [[TP, FN],
                                [FP, TN]]
            return confusion_mat
        

        if(Info['selectedAlgorithm'] == 'Adaline'):

            def Adaline_fit(x, y, threshold, lr, bias):

                y = np.array(y).reshape(len(y), 1)
                weights = np.zeros((x.shape[1], 1))
                counter = 0
                mse = 1000
                bias = 0
                while (mse > threshold):
                    w = weights
                    for i in range(len(x)):
                        xi = x.iloc[i]
                        xi = np.array(xi).reshape(1, 2)

                        y_prediction = np.dot(xi, w) + bias
                        # print(y_prediction)
                        # print(y)

                        e = y[i] - y_prediction
                        xi = xi.reshape(2, 1)

                        if (bool == True):
                            bias = bias + lr * e

                        w = w + lr * e * xi

                    counter = counter + 1
                    if (counter > 1):
                        # mse = (1 / len(x)) * sum((e) ^ 2)
                        y_pred = np.dot(x, w) + bias
                        error = y_pred - y
                        mse = np.sum(np.square(error)) * (1 / (len(x)))
                        # print(y_pred)
                        # print(mse)

                    return w, bias


            w, b = Adaline_fit(xtrain, ytrain, Info['MSE_Threshold'], Info['Learning_rate'], Info['boolBias'])


            def Adaline_predict(X_test, w, b):
                predictions = []
                for i in range(len(X_test)):
                    # xi_test = X_test[i]
                    xi_test = X_test.iloc[i]
                    xi_test = np.array(xi_test).reshape(1, 2)
                    y_prediction = np.dot(xi_test, w) + b
                    if y_prediction >= 0:
                        predictions.append(1)
                    else:
                        predictions.append(-1)
                return predictions

                
            #bias mtshaf l kolo???????????????????????????????????????????
            pred = Adaline_predict(xtest, w, b)
            accuracy = accuracy_score(ytest, pred)
            print("Accuracy:", accuracy)


        else:
            def fit(x_train, y_train, lr, bool, n_iters):
                y_train = np.array(y_train).reshape(len(y_train), 1)
                weights = np.zeros((x_train.shape[1], 1))
                bias = 0

                # gradient descent
                for _ in range(n_iters):
                    for i in range(len(x_train)):
                        xi = x_train.iloc[i]
                        # print(xi)
                        xi = np.array(xi).reshape(1, 2)

                        linear_output = np.dot(xi, weights) + bias
                        y_predicted = signum(linear_output)
                        # print(y_prediction)
                        # print(y)



                        # update weights and bias
                        update = lr * (y_train[i] - y_predicted)
                        xi = xi.reshape(2, 1)
                        weights += update * xi
                        if (bool == True):
                            bias += update

                return weights,bias
            

            def predict(X_test,w,b):
                predictions = []
                for i in range(len(X_test)):
                    # xi_test = X_test[i]
                    xi_test = X_test.iloc[i]
                    xi_test = np.array(xi_test).reshape(1, 2)
                    y_prediction = np.dot(xi_test, w) + b
                    predictions.append(signum(y_prediction))
                return predictions
                #linear_output = np.dot(X_test, w) + b
                #return signum(linear_output)

            def signum(x):
                if x > 0: return 1
                else: return -1


            w,b = fit(xtrain, ytrain,Info['Learning_rate'],Info['boolBias'],Info['numEphocs'])
            pred = predict(xtest,w,b)
            print(pred)
            accuracy = accuracy_score(ytest, pred)
            print("Accuracy:", accuracy)
        print(confusion_matrix(ytest , pred ))

        def visualizeData(scaled_df):  # 10 combs
            import matplotlib.pyplot as plt


            # fL1, fL2 = selectedFeatures[0], selectedFeatures[1]
            # f1, f2 = scaled_df[fL1], scaled_df[fL2]

            plt.xlabel(selectedFeatures[0])
            plt.ylabel(selectedFeatures[1])


            CF =[ (xtrain[xtrain.columns[0]][:30] , xtrain[xtrain.columns[1]][:30]) ,(xtrain[xtrain.columns[0]][30:] , xtrain[xtrain.columns[1]][30:])]
            for plot in range(2):
                plt.scatter(CF[plot][0], CF[plot][1])


            plt.legend([SelectedClasses[0],SelectedClasses[1]])

            x_values = np.linspace(xtrain.min(), xtrain.max())

            yValues = (-w[0] * x_values - b) / w[1]

            plt.plot(x_values, yValues, c='green', label='Decision Boundary')

            plt.show()
        visualizeData(data)
      



        
        
#----------------------------------------------------------------------------------------------------------------------     
Label(master, text='Please select two features').grid(row=0, sticky=W)
feature1 = IntVar()
feature2 = IntVar()
feature3 = IntVar()
feature4 = IntVar()
feature5 = IntVar()
Checkbutton(master, text='Area', variable=feature1).grid(row=1, sticky=W) 
Checkbutton(master, text='Perimeter', variable=feature2).grid(row=2, sticky=W)
Checkbutton(master, text='MajorAxisLength', variable=feature3).grid(row=3, sticky=W)
Checkbutton(master, text='MinorAxisLength', variable=feature4).grid(row=4, sticky=W)
Checkbutton(master, text='Roundness', variable=feature5).grid(row=5, sticky=W)

Label(master, text='Please select One combination of classes').grid(row=6, sticky=W)
comb1 = IntVar() 
comb2 = IntVar() 
comb3 = IntVar()
Radiobutton(master, text='Bombay & Cali', variable=comb1, value=1).grid(row=7, sticky=W)
Radiobutton(master, text='Bombay & Sira', variable=comb2, value=2).grid(row=8, sticky=W)
Radiobutton(master, text='Sira & Cali', variable=comb3, value=3).grid(row=9, sticky=W)

Label(master, text='Please enter Learning rate (eta)').grid(row=10, sticky=W)
e = Entry(master)
e.place(x=170,y=243)
e.focus_set()

Label(master, text='Please enter Number of Epochs').grid(row=11, sticky=W)
a = Entry(master)
a.place(x=170,y=265)
a.focus_set()

Label(master, text='Please enter MSE Threshold').grid(row=12, sticky=W)
MSE = Entry(master)
MSE.place(x=170,y=287)


Label(master, text='').grid(row=12, sticky=W)
bias = BooleanVar()
Checkbutton(master, text='Add Bias', variable=bias).grid(row=13, sticky=W)

Label(master, text='Choose the algorithm').grid(row=14, sticky=W)
Algorithm1 = IntVar()
Algorithm2 = IntVar()
Radiobutton(master, text='Adaline', variable=Algorithm1, value=1).grid(row=50, sticky=W)
Radiobutton(master, text='Perceptron', variable=Algorithm2, value=2).grid(row=60, sticky=W)


GetInfoButton = Button(master, text='Generate', width=10, command=getInfo)
GetInfoButton.place(x=450 , y=300)
# PlotButton = Button(master, text='Plot graph', width=10, command = visualizeData(data))
# GetInfoButton.place(x=650 , y=400)

master.mainloop() 








#visualizeData(data)

# plt.style.use("seaborn")
# x=[100,200,300,400,500,10,50,250]
# y=[1000,2000,3000,4000,5000,300,500,3000]
# plt.scatter(x,y,marker='o',s=50) #size of marker
# plt.show()

#xtest[xtest.columns[0]][:30]


# Scatter the points of both classes
# Assuming X_test contains the feature vectors of your test data
# Replace X_test with your actual test data
plt.figure(figsize=(10, 6))

# Scatter the points for Class 1 (e.g., label 1)
# plt.scatter(X_test[y_test == 1][:, 0], X_test[y_test == 1][:, 1], color='blue', label='Class 1')

# # Scatter the points for Class -1 (e.g., label -1)
# plt.scatter(X_test[y_test == -1][:, 0], X_test[y_test == -1][:, 1], color='red', label='Class -1')

# # Plot the decision boundary
# # This example assumes a linear decision boundary
# x_values = np.linspace(X_test[:, 0].min(), X_test[:, 0].max(), 100)
# y_values = (-w[0] * x_values - b) / w[1]
# plt.plot(x_values, y_values, label='Decision Boundary', color='green', linestyle='--')

# plt.xlabel('Feature 1')
# plt.ylabel('Feature 2')
# plt.legend()
# plt.show()


#Create Grid of Points: Create a grid of points covering the range of data points to visualize the decision boundary.

# x_min, x_max = X.min() - 1, X.max() + 1
# y_min, y_max = Y.min() - 1, Y.max() + 1
# xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),np.arange(y_min, y_max, 0.1))
# #Predict Class for Grid Points: Use the trained SLP model to predict the class for each point in the grid.
# clf = Perceptron()
# clf.fit(X[:,np.newaxis],Y)
# Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
# Z = Z.reshape(xx.shape)
# #Plot Decision Boundary: Plot the decision boundary using the predicted class values.

# plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.2)