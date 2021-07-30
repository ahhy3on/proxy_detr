import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

class Aggregator(nn.Module):
    """Some Information about Aggregator"""
    def __init__(self,d_model):
        super(Aggregator, self).__init__()
        self.subtraction_fc = nn.Linear(d_model,d_model)
        self.multiplication_fc = nn.Linear(d_model,d_model)
        self.category_specific_features_fc = nn.Linear(d_model*3,d_model)

    def forward(self, category_codes, query_features):
        print(query_features.shape, category_codes.shape)
        subtraction = F.relu(self.subtraction_fc(query_features-category_codes))
        multiplication = F.relu(self.multiplication_fc(query_features * category_codes))
        concat_feature = torch.cat((subtraction,multiplication,query_features),dim=-1)
        category_specific_features = F.relu(self.category_specific_features_fc(concat_feature))

        return category_specific_features

class CCE(nn.Module):
    """Some Information about CCE"""
    def __init__(self):
        super(CCE, self).__init__()
        #self.roi_align = torchvision.ops.roi_align
        self.gap = nn.AdaptiveAvgPool2d((1,1))
    
    def forward(self, src_flatten, spatial_shapes, support_anno):

        # 1) restore feature's spatial dimension
        batch, _ ,dim = src_flatten.shape
        src_split = src_flatten.split([H_ * W_ for H_, W_ in spatial_shapes], dim=1)
        feature_code = []
        bbox=[]
        #print(support_anno)
        for anno in support_anno:
            bbox.append(anno['boxes'])
        #print(bbox)
        for idx, (H_, W_) in enumerate(spatial_shapes):
            #[batch,length,dim] -> [batch, dim, H,W]
            feature_map = src_split[idx].transpose(1,2).reshape(batch,dim,H_,W_)
            #interpolation이 이뤄지더라도 GAP 사용하기 떄문에 큰 차이가 없지 않을까?
            # 2) roi_align
            aligned_support_bbox = torchvision.ops.roi_align(feature_map,bbox,(H_,W_))
            #[K,dim,H_,W_]
            # 3) GAP and sigmoid
            #[K,dim]
            feature_code.append(F.sigmoid(self.gap(aligned_support_bbox).squeeze()))
            #print(feature_code.shape)
       
        return torch.mean(torch.stack(feature_code,dim=-1),dim=-1).squeeze()
