import numpy as np
import matplotlib.pyplot as plt

shap_result = np.load("shap_exp_result_1.npy",allow_pickle=True)
#print(shap_result[9].shape)
shap = shap_result[0]
for i in range(1,10):
    shap = np.vstack((shap,shap_result[i]))
print(shap.shape)
'''
for i in range(shap.shape[1]):
    if np.count_nonzero(shap[:,i]) > 25:
        print(i)
        print(max(shap[:,i]))
'''
#shap = [abs(i) for i in shap]
result = list(np.mean(shap,axis=0))
print(max(result))
result = [abs(i) for i in result]
#print(result.shape)
result.sort(reverse=True)
x = np.arange(4917)
print(max(result))

plt.xlabel("feature")
plt.ylabel("shap value")
plt.plot(x,result)
plt.show()

for i in range(len(result)):
    print(result[i])
    print(i)
