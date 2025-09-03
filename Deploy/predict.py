#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle



# In[2]:


out_file='model_C=1.0.bin'


# In[3]:


with open(out_file, 'rb') as f_in:
    dv, model=pickle.load(f_in)


# In[4]:





# In[ ]:


from flask import Flask
from flask import request
from flask import jsonify

app=Flask('churn')

@app.route('/predict',methods=['POST'])
def predict ():
    customer= request.get_json()

    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0, 1]
    churn=y_pred>=.5
    result={
        'churn_probabilty':float(y_pred),
        'churn':bool(churn)

    }
    return jsonify(result) 


# In[ ]:


if __name__=="__main__":
    app.run(debug=True ,host='0.0.0.0', port =9696 )


# In[ ]:




