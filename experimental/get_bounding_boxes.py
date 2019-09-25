import brambox as bb
import os
from pathflowai.utils import load_sql_df, npy2da

def get_box(i,prop):
    c=[prop.centroid[1], prop.centroid[0]]
    l=rev_label[i+1]
    width = prop.bbox[3] - prop.bbox[1] + 1
    height = prop.bbox[2] - prop.bbox[0] + 1
    wh=max(width,height)
    c = [ci-wh/2 for ci in c]
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
    input_dir='inputs'
    patch_info_file='cell_info.db'
    patch_size=256
    p_sample=0.7
    annotation_file = 'annotations_bbox_{}.pkl'.format(patch_size)
    reference_mask='reference_mask.npy'
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

    patch_info=patch_info.sample(frac=p_sample)

    bbox_dfs=[]
    for i in patch_info.shape[0]:
        patch=patch_info.iloc[i]
        bbox_dff=get_boxes(masks[patch['ID']],ID=patch['ID'],x=patch['x'],y=patch['y'],patch_size=patch['patch_size'])

    if not os.path.exists(annotation_file):
        bbox_df=bb.util.new('annotation').drop(columns=['difficult','ignore','lost','occluded','truncated'])[['image','class_label','x_top_left','y_top_left','width','height']]
    else:
        bbox_df=bb.io.load('pandas',annotation_file)

    bbox_df=pd.concat([bbox_df]+bbox_dfs)

    for i in official_widths.index:
        bbox_df.loc[bbox_df['class_label']==i,'width']=int(official_widths[i])
    bbox_df.loc[:,'height']=bbox_df['width']

    bb.io.save(bbox_df,'pandas',annotation_file)
