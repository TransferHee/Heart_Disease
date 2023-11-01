from ex import experiment

model_names = ['BaseCNN', 'CNN_2', 'CNN_3', 'SE_CNN', 'M_resnet', 'M_vgg16']
lrs = [1e-4, 1e-5, 1e-6, 1e-7]
weights = [0, 1e-8, 1e-7, 1e-6]
is_balanced = [True, False]

for m in model_names:
    for b in is_balanced:
        best_acc = 0
        best_lr = None
        best_wd = None
        best_b = None
        for lr in lrs:
            for w in weights:
                print('Current hyperparameter:')
                print(m)
                print('lr:',lr)
                print('weight decay:',w)
                print('b:',b)
                test_acc = experiment(lr=lr, weight_decay=w, batch_size=128, model_name=m, best_acc=best_acc, balanced=b)
                if test_acc > best_acc:
                    best_acc = test_acc
                    best_lr = lr
                    best_wd = w
                    best_b = b
                
        print('\n\n Result of model\n\n')
        print(m)
        print('acc:',best_acc)
        print('lr:',best_lr)
        print('wd:',best_wd)
        print('isbal', best_b)
