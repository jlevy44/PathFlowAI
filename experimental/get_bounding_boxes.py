import brambox as bb
import os
from pathflowai.utils import load_sql_df, npy2da
import skimage
import dask, dask.array as da, pandas as pd, numpy as np
import argparse

def get_box(i,prop):
    c=[prop.centroid[1], prop.centroid[0]]
    l=rev_label[i+1]
    width = prop.bbox[3] - prop.bbox[1] + 1
    height = prop.bbox[2] - prop.bbox[0] + 1
    wh=max(width,height)
    #c = [ci-wh/2 for ci in c]
    return [l]+c+[wh]

def get_boxes(m,ID='test',x='x',y='y',patch_size='patchsize'):
    lbls,n_lbl=label(m)
    obj_labels={}
    for i in range(1,5):
        obj_labels[i]=np.unique(lbls[m==i].flatten())
    rev_label={}
    for k in obj_labels:
        for i in obj_labels[k]:
            rev_label[i]=k
    objProps = skimage.measure.regionprops(lbls)
    boxes=dask.compute(*[dask.delayed(get_box)(i,prop) for i,prop in enumerate(objProps)],scheduler='threading')
    #print(boxes)
    boxes=pd.DataFrame(np.array(boxes).astype(int),columns=['class_label','x_top_left','y_top_left','width'])
    boxes['height']=boxes['width']
    boxes['image']='{}_{}_{}_{}'.format(ID,x,y,patch_size)
    boxes=boxes[['image','class_label','x_top_left','y_top_left','width','height']]
    boxes.loc[:,'x_top_left']=np.clip(boxes.loc[:,'x_top_left'],0,m.shape[1])
    boxes.loc[:,'y_top_left']=np.clip(boxes.loc[:,'y_top_left'],0,m.shape[0])
    bbox_df=bb.util.new('annotation').drop(columns=['difficult','ignore','lost','occluded','truncated'])[['image','class_label','x_top_left','y_top_left','width','height']]
    bbox_df=bbox_df.append(boxes)
    return boxes

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--num_classes',default=3,type=int)
    p.add_argument('--patch_size',default=256,type=int)
    p.add_argument('--p_sample',default=0.7,type=float)
    p.add_argument('--input_dir',default='inputs',type=str)
    p.add_argument('--patch_info_file',default='cell_info.db',type=str)
    p.add_argument('--reference_mask',default='reference_mask.npy',type=str)

    args=p.parse_args()
    input_dir=args.input_dir
    patch_info_file=args.patch_info_file
    patch_size=args.patch_size
    p_sample=args.p_sample
    annotation_file = 'annotations_bbox_{}.pkl'.format(patch_size)
    reference_mask=args.reference_mask
    if not os.path.exists('widths.pkl'):
        m=np.load(reference_mask)
        bbox_df=get_boxes(m)
        official_widths=dict(bbox_df.groupby('class_label')['width'].mean()+2*bbox_df.groupby('class_label')['width'].std())
        pickle.dump(official_widths,open('widths.pkl','wb'))
    else:
        official_widths=pickle.load(open('widths.pkl','rb'))

    patch_info=load_sql_df(patch_info_file, patch_size)
    IDs=patch_info['ID'].unique()
    #slides = {slide:da.from_zarr(join(input_dir,'{}.zarr'.format(slide))) for slide in IDs}
    masks = {mask:npy2da(join(input_dir,'{}_mask.npy'.format(slide))) for slide in IDs}

    if p_sample < 1.:
        patch_info=patch_info.sample(frac=p_sample)

    bbox_dfs=[]

    if not os.path.exists(annotation_file):
        bbox_df=bb.util.new('annotation').drop(columns=['difficult','ignore','lost','occluded','truncated'])[['image','class_label','x_top_left','y_top_left','width','height']]
    else:
        bbox_df=bb.io.load('pandas',annotation_file)

    patch_info=patch_info[~np.isin(np.vectorize(lambda i: '_'.join(patch_info.iloc[i][['ID','x','y','patch_size']].astype(str).tolist())(patch_info.shape[0]),set(bbox_df.image.cat.categories))]

    for i in patch_info.shape[0]:
        print(i)
        patch=patch_info.iloc[i]
        ID,x,y,patch_size2=patch[['ID','x','y','patch_size']].tolist()
        m=masks[ID][x:x+patch_size2,y:y+patch_size2]
        bbox_dff=get_boxes(m,ID=ID,x=x,y=y,patch_size=patch_size2)
        for i in official_widths.index:
            bbox_dff.loc[bbox_dff['class_label']==i,'width']=int(official_widths[i])
        bbox_dff.loc[:,'x_top_left']=boxes.loc[:,'x_top_left']-bbox_df['width']/2
        bbox_dff.loc[:,'y_top_left']=boxes.loc[:,'y_top_left']-bbox_df['width']/2
        bbox_dff.loc[:,'x_top_left']=np.clip(bbox_dff.loc[:,'x_top_left'],0,m.shape[1])
        bbox_dff.loc[:,'y_top_left']=np.clip(bbox_dff.loc[:,'y_top_left'],0,m.shape[0])
        bbox_dfs.append(bbox_dff)

    bbox_df=pd.concat([bbox_df]+bbox_dfs)


    bbox_df.loc[:,'height']=bbox_df['width']


    bb.io.save(bbox_df,'pandas',annotation_file)
