#作业要求
#3.测试：
# 导入模型
# 用经过参数查找后的模型进行测试
# 输出分类精度
import model 


twolayermodel =model.Two_Layer_Net()
test_images,test_labels = model.load_mnist('./mnist',kind = 't10k')
file = './besttwolayermodel.npz'
twolayermodel.load_model(file)

Best_accuracy = (twolayermodel.predict(test_images) == test_labels).mean()
print('Best Accuracy:',Best_accuracy)