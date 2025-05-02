import numpy as np

L = 100
for method in ['Binder', 'VI', 'omARI']:
    all_TV = np.load('saved/{}_TV.npy'.format(method))
    all_SW = np.load('saved/{}_SW.npy'.format(method))

    print('Method {}'.format(method))
    print('TV {}+-{}'.format(np.round(np.mean(all_TV), 4), np.round(np.std(all_TV), 4)))
    print('SW {}+-{}'.format(np.round(np.mean(all_SW), 4), np.round(np.std(all_SW), 4)))
for method in ['SW', 'MixSW', 'SMixW']:
    all_TV = []
    all_SW = []
    all_TVtrue = []
    all_SWtrue = []
    TV = np.load('saved/{}_TV_L{}.npy'.format(method, L))
    SW = np.load('saved/{}_SW_L{}.npy'.format(method, L))

    all_TV.append(np.mean(TV))
    all_SW.append(np.mean(SW))
    print('Method {}'.format(method))
    print('TV {}+-{}'.format(np.round(np.mean(all_TV), 4), np.round(np.std(all_TV), 4)))
    print('SW {}+-{}'.format(np.round(np.mean(all_SW), 4), np.round(np.std(all_SW), 4)))
