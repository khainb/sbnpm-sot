import numpy as np

repetitions = 25
n = 200
L = 100
K = 100
for method in ['Binder', 'VI', 'omARI']:
    all_TV = np.load('saved/{}_TV_n{}_K{}.npy'.format(method, n, K))
    all_SW = np.load('saved/{}_SW_n{}_K{}.npy'.format(method, n, K))
    all_TVtrue = np.load('saved/{}_trueTV_n{}_K{}.npy'.format(method, n, K, ))
    all_SWtrue = np.load('saved/{}_trueSW_n{}_K{}.npy'.format(method, n, K))

    print('Method {}'.format(method))
    print('TV {}+-{}'.format(np.round(np.mean(all_TV), 4), np.round(np.std(all_TV), 4)))
    print('True TV {}+-{}'.format(np.round(np.mean(all_TVtrue), 4), np.round(np.std(all_TVtrue), 4)))
    print('SW {}+-{}'.format(np.round(np.mean(all_SW), 4), np.round(np.std(all_SW), 4)))
    print('True SW {}+-{}'.format(np.round(np.mean(all_SWtrue), 4), np.round(np.std(all_SWtrue), 4)))
for method in ['SW', 'MixSW', 'SMixW']:
    all_TV = []
    all_SW = []
    all_TVtrue = []
    all_SWtrue = []
    for time in range(repetitions):
        TV = np.load('saved/{}_TV_n{}_L{}_K{}_repeat{}.npy'.format(method, n, L, K, time))
        SW = np.load('saved/{}_SW_n{}_L{}_K{}_repeat{}.npy'.format(method, n, L, K, time))
        trueTV = np.load('saved/{}_trueTV_n{}_L{}_K{}_repeat{}.npy'.format(method, n, L, K, time))
        trueSW = np.load('saved/{}_trueSW_n{}_L{}_K{}_repeat{}.npy'.format(method, n, L, K, time))
        all_TV.append(np.mean(TV))
        all_SW.append(np.mean(SW))
        all_TVtrue.append(np.mean(trueTV))
        all_SWtrue.append(np.mean(trueSW))

    print('Method {}'.format(method))
    print('TV {}+-{}'.format(np.round(np.mean(all_TV), 4), np.round(np.std(all_TV), 4)))
    print('True TV {}+-{}'.format(np.round(np.mean(all_TVtrue), 4), np.round(np.std(all_TVtrue), 4)))
    print('SW {}+-{}'.format(np.round(np.mean(all_SW), 4), np.round(np.std(all_SW), 4)))
    print('True SW {}+-{}'.format(np.round(np.mean(all_SWtrue), 4), np.round(np.std(all_SWtrue), 4)))
