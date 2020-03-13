from sklearn.cluster import KMeans
import numpy as np, pandas as pd, brambox as bb
import pickle, argparse

p=argparse.ArgumentParser()
p.add_argument('--patch_size',default=512,type=int)
p.add_argument('--n_anchors',default=20,type=int)
p.add_argument('--sample_p',default=1.,type=float)

args=p.parse_args()
np.random.seed(42)
patch_size=args.patch_size
n_anchors=args.n_anchors
sample_p=args.sample_p
annotation_file = 'annotations_bbox_{}.pkl'.format(patch_size)
annotations=bb.io.load('pandas',annotation_file)
if sample_p<1.:
    annotations=annotations.sample(frac=sample_p)

X=annotations[['x_top_left','y_top_left']].astype(float).values+(annotations['width']/2.).astype(float).values.reshape(-1,1)
km=KMeans(n_clusters=n_anchors,n_jobs=-1).fit(X)
anchors=km.cluster_centers_
pickle.dump(anchors,open('anchors.pkl','wb'))
