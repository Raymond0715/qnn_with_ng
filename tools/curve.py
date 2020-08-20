# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib import rc
# rc('mathtext', default='regular')

# a = []
# b = []
# with open("train_log.txt", "r") as f:
    # for line in f.readlines():
        # line = line.strip('\n')
        # # print(line)
        # if line.find('Test Accuracy_Top1') == 0:
            # a.append(line.split(' ')[2])
        # elif line.find('Training Loss') == 0:
            # b.append(line.split(' ')[2])
# a_float = []
# for num in a:
    # a_float.append(100-float(num)*100)
# # print(a_float)
# # plt.plot(a_float, color='red', linewidth=1.5, linestyle='-', label='Test error: general gradient')
# # plt.plot(b, color='red', linewidth=1.5, linestyle='--', label='Training loss: general gradient')


# c = []
# d = []
# with open("train_log_ste_without_weight_decay.txt", "r") as f:
    # for line in f.readlines():
        # line = line.strip('\n')
        # # print(line)
        # if line.find('Test Accuracy_Top1') == 0:
            # c.append(line.split(' ')[2])
        # elif line.find('Training Loss') == 0:
            # d.append(line.split(' ')[2])
# c_float = []
# for num in c:
    # c_float.append(100-float(num)*100)
# # print(c_float)
# # plt.plot(c_float, color='blue', linewidth=1.5, linestyle='-', label='Test error: natural gradient')
# # plt.plot(d, color='blue', linewidth=1.5, linestyle='--', label='Training loss: natural gradient')
# #
# # plt.axis([0, 160, 0.1, 0.8])
# # plt.grid()
# # plt.legend()
# # plt.xlabel('epoch')
# # plt.ylabel('test error')
# # plt.twinx()
# # plt.ylabel('training loss')
# # plt.show()

# fig = plt.figure()
# ax = fig.add_subplot(111)

# lns1 = ax.plot(a_float, '-', linewidth=1.5, color='red', label='Test error: general gradient')
# lns2 = ax.plot(c_float, '-', linewidth=1.5, color='blue', label='Test error: natural gradient')
# ax2 = ax.twinx()
# lns3 = ax2.plot(b, '--', linewidth=1.5, color='red', label='Training loss: general gradient')
# lns4 = ax2.plot(d, '--', linewidth=1.5, color='blue', label='Training loss: natural gradient')

# # added these three lines
# lns = lns1+lns2+lns3+lns4
# labs = [l.get_label() for l in lns]
# ax.legend(lns, labs, loc=0)

# ax.grid()
# ax.set_xlabel("epoch")
# ax.set_ylabel("test error (%)")
# ax2.set_ylabel("training loss")
# ax.set_ylim(10, 70)
# ax2.set_ylim(0.0, 1.0)
# plt.show()

import pandas as pd
from matplotlib import rc
import matplotlib.pyplot as plt

ng_path = '../log/vgg16_cifar10/ng_1.csv'
ste_path = '../log/vgg16_cifar10/ste_1.csv'

ng_csv = pd.read_csv(ng_path)
# ng val accuracy
ng_val_accuracy = ng_csv.val_accuracy.to_list()
ng_val_error_rate = [100 - 100 * val for val in ng_val_accuracy]
# ng loss
ng_loss = ng_csv.val_loss.to_list()

ste_csv = pd.read_csv(ste_path)
# ste val accuracy
ste_val_accuracy = ste_csv.val_accuracy.to_list()
ste_val_error_rate = [100 - 100 * val for val in ste_val_accuracy]
# ste loss
ste_loss = ste_csv.val_loss.to_list()

rc('mathtext', default = 'regular')

fig = plt.figure()
ax = fig.add_subplot(111)

lns1 = ax.plot(ste_val_error_rate, '-', linewidth=1.5, color='red', label='Test error: general gradient')
lns2 = ax.plot(ng_val_error_rate, '-', linewidth=1.5, color='blue', label='Test error: natural gradient')
ax2 = ax.twinx()
lns3 = ax2.plot(ste_loss, '--', linewidth=1.5, color='red', label='Training loss: general gradient')
lns4 = ax2.plot(ng_loss, '--', linewidth=1.5, color='blue', label='Training loss: natural gradient')

# added these three lines
lns = lns1+lns2+lns3+lns4
labs = [l.get_label() for l in lns]
ax.legend(lns, labs, loc=0)

ax.grid()
ax.set_xlabel("epoch")
ax.set_ylabel("test error (%)")
ax2.set_ylabel("training loss")
ax.set_ylim(10, 100)
ax2.set_ylim(0.0, 1.5)
plt.show()
