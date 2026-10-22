import numpy as np, sys
sys.path.insert(0,'/home/rbaba/taukin-gap-20261023/code/analysis/atlas_classical_posterior')
from evaluate import GEN, wauc, lr_of
from hquality import wcorr
I=np.load('/home/rbaba/taukin-gap-20261023/out/learner_intervene.npz')
ref=np.load(GEN/'p11'/'full22_s42'/'validation_predictions.npz')
print('mean |identity - stored|', np.abs(I['identity']-ref['h_pred']).mean(), np.quantile(np.abs(I['identity']-ref['h_pred']),.999))
P=np.load('/home/rbaba/taukin-gap-20261023/out/val_base/posterior.npz')
assert np.array_equal(P['global_indices'],I['global_indices'])
w=P['weights'].astype(float); hx=ref['h'].astype(float); modes,rm,av=P['modes'],P['reco_h_mode'].astype(int),P['reco_h_available_side']
W=np.broadcast_to(w[:,None],av.shape)
for g,(tm,rmode) in {'pi':(0,0),'rho':(1,1),'3pi':(3,3)}.items():
    for v in ('identity','shuffle_other','shuffle_own'):
        out=[]
        for s in (0,1):
            pass
        msk=av&(rm==rmode)&(modes==tm)
        # compose: for own-side s, use variant_s's prediction of side s
        if v=='identity': h=I['identity']
        else: h=np.stack([I[f'{v}_0'][:,0],I[f'{v}_1'][:,1]],1)
        print(g, f'{v:14s}', '%.3f/%.3f/%.3f'%tuple(wcorr(h[...,c][msk],hx[...,c][msk],W[msk]) for c in range(3)))
