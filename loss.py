import torch
from torch import nn


INF = 100000000


class IOULoss(nn.Module):
    def __init__(self, loc_loss_type):
        super().__init__()

        self.loc_loss_type = loc_loss_type

    def forward(self, out, target, weight=None):
        pred_left, pred_top, pred_right, pred_bottom = out.unbind(1)
        target_left, target_top, target_right, target_bottom = target.unbind(1)

        target_area = (target_left + target_right) * (target_top + target_bottom)
        pred_area = (pred_left + pred_right) * (pred_top + pred_bottom)

        w_intersect = torch.min(pred_left, target_left) + torch.min(
            pred_right, target_right
        )
        h_intersect = torch.min(pred_bottom, target_bottom) + torch.min(
            pred_top, target_top
        )

        area_intersect = w_intersect * h_intersect
        area_union = target_area + pred_area - area_intersect

        ious = (area_intersect + 1) / (area_union + 1)

        if self.loc_loss_type == 'iou':
            loss = -torch.log(ious)

        elif self.loc_loss_type == 'giou':
            g_w_intersect = torch.max(pred_left, target_left) + torch.max(
                pred_right, target_right
            )
            g_h_intersect = torch.max(pred_bottom, target_bottom) + torch.max(
                pred_top, target_top
            )
            g_intersect = g_w_intersect * g_h_intersect + 1e-7
            gious = ious - (g_intersect - area_union) / g_intersect

            loss = 1 - gious

        if weight is not None and weight.sum() > 0:
            return (loss * weight).sum() / weight.sum()

        else:
            return loss.mean()


def clip_sigmoid(input):
    out = torch.clamp(torch.sigmoid(input), min=1e-4, max=1 - 1e-4)

    return out


class SigmoidFocalLoss(nn.Module):
    def __init__(self, gamma, alpha):
        super().__init__()

        self.gamma = gamma
        self.alpha = alpha

    def forward(self, out, target):
        n_class = out.shape[1]
        class_ids = torch.arange(
            1, n_class + 1, dtype=target.dtype, device=target.device
        ).unsqueeze(0)

        t = target.unsqueeze(1)
        p = torch.sigmoid(out)

        gamma = self.gamma
        alpha = self.alpha

        term1 = (1 - p) ** gamma * torch.log(p)
        term2 = p ** gamma * torch.log(1 - p)

        # print(term1.sum(), term2.sum())

        loss = (
            -(t == class_ids).float() * alpha * term1
            - ((t != class_ids) * (t >= 0)).float() * (1 - alpha) * term2
        )

        return loss.sum()



# keypoint

class FCOSLoss(nn.Module):
    def __init__(
        self, sizes, gamma, alpha, iou_loss_type, center_sample, fpn_strides, pos_radius
    ):
        super().__init__()

        self.sizes = sizes

        self.cls_loss = SigmoidFocalLoss(gamma, alpha)
        self.box_loss = IOULoss(iou_loss_type)
        self.center_loss = nn.BCEWithLogitsLoss()

        self.center_sample = center_sample
        self.strides = fpn_strides
        self.radius = pos_radius

    def prepare_target(self, points, targets):
        ex_size_of_interest = []

        for i, point_per_level in enumerate(points):
            size_of_interest_per_level = point_per_level.new_tensor(self.sizes[i])
            ex_size_of_interest.append(
                size_of_interest_per_level[None].expand(len(point_per_level), -1)
            )

        ex_size_of_interest = torch.cat(ex_size_of_interest, 0)
        n_point_per_level = [len(point_per_level) for point_per_level in points]
        point_all = torch.cat(points, dim=0)
        #label, box_target = self.compute_target_for_location(
        #    point_all, targets, ex_size_of_interest, n_point_per_level #################
        #)

        label, box_target, key_stacks = self.compute_target_for_location(   ############i add this
            point_all, targets, ex_size_of_interest, n_point_per_level
        )

        #print("PREPAPRPRRPRPEE", len(box_target), len(key_stacks))
        #print(box_target, "\n", key_stacks)
        #print(box_target[0].shape, key_stacks[0].shape)
        #print(box_target[1].shape, key_stacks[1].shape)
        #print(box_target[2].shape, key_stacks[2].shape)
        #print(box_target[3].shape, key_stacks[3].shape)

        for i in range(len(label)):
            label[i] = torch.split(label[i], n_point_per_level, 0)
            box_target[i] = torch.split(box_target[i], n_point_per_level, 0)
            key_stacks[i] = torch.split(key_stacks[i], n_point_per_level, 0)  ######i add this

        label_level_first = []
        box_target_level_first = []
        key_stacks_level_first = []  #########i add this

        for level in range(len(points)):
            label_level_first.append(
                torch.cat([label_per_img[level] for label_per_img in label], 0)
            )
            box_target_level_first.append(
                torch.cat(
                    [box_target_per_img[level] for box_target_per_img in box_target], 0
                )
            )
            key_stacks_level_first.append( ################### i addd this
                torch.cat(########
                    [key_stack[level] for key_stack in key_stacks], 0#######
                )#############
            )##########
        #print(len(box_target_level_first), len(key_stacks_level_first))
        #return label_level_first, box_target_level_first
        return label_level_first, box_target_level_first, key_stacks_level_first  ###########i add this for keypoint

    def get_sample_region(self, gt, strides, n_point_per_level, xs, ys, radius=1):
        n_gt = gt.shape[0]
        n_loc = len(xs)
        gt = gt[None].expand(n_loc, n_gt, 4)
        center_x = (gt[..., 0] + gt[..., 2]) / 2
        center_y = (gt[..., 1] + gt[..., 3]) / 2
        #print("cenenenenterrrer", type(center_x), center_x.shape, center_y.shape)



        if center_x[..., 0].sum() == 0:
            return xs.new_zeros(xs.shape, dtype=torch.uint8)

        begin = 0

        center_gt = gt.new_zeros(gt.shape)

        for level, n_p in enumerate(n_point_per_level):
            end = begin + n_p
            stride = strides[level] * radius

            x_min = center_x[begin:end] - stride
            y_min = center_y[begin:end] - stride
            x_max = center_x[begin:end] + stride
            y_max = center_y[begin:end] + stride

            center_gt[begin:end, :, 0] = torch.where(
                x_min > gt[begin:end, :, 0], x_min, gt[begin:end, :, 0]
            )
            center_gt[begin:end, :, 1] = torch.where(
                y_min > gt[begin:end, :, 1], y_min, gt[begin:end, :, 1]
            )
            center_gt[begin:end, :, 2] = torch.where(
                x_max > gt[begin:end, :, 2], gt[begin:end, :, 2], x_max
            )
            center_gt[begin:end, :, 3] = torch.where(
                y_max > gt[begin:end, :, 3], gt[begin:end, :, 3], y_max
            )

            begin = end

        left = xs[:, None] - center_gt[..., 0]
        right = center_gt[..., 2] - xs[:, None]
        top = ys[:, None] - center_gt[..., 1]
        bottom = center_gt[..., 3] - ys[:, None]

        center_bbox = torch.stack((left, top, right, bottom), -1)
        is_in_boxes = center_bbox.min(-1)[0] > 0

        return is_in_boxes

    def compute_target_for_location(
        self, locations, targets, sizes_of_interest, n_point_per_level
    ):
        labels = []
        box_targets = []
        key_stack_list = [] #############i add this for keypoint

        xs, ys = locations[:, 0], locations[:, 1]
        #print("xs$$$$$$$$$$$$$$", type(xs), len(xs), xs)
        #print("ys$$$$$$$$$$$$$$", type(ys), len(ys), ys)
        #print(len(targets))
        #print("xs[:, none]", type(xs[:, None]), len(xs[:, None]), xs[:, None])
       

        for i in range(len(targets)):
            targets_per_img = targets[i]
            #print("targetperimg", type(targets_per_img), len(targets_per_img), targets_per_img.box)
            assert targets_per_img.mode == 'xyxy'
            bboxes = targets_per_img.box
            #print("bboxes", type(bboxes), len(bboxes), bboxes)
            #print("bboxes[:,0][none]", type(bboxes[:, 0][None]), bboxes[:, 0][None].shape, bboxes[:, 0][None])


            key_custom = targets_per_img.key   ################# i add this for keypoint

            #print(targets_per_img)
            #print("key!!", type(key_custom), len(key_custom), key_custom)


            labels_per_img = targets_per_img.fields['labels']
            area = targets_per_img.area()

            l = xs[:, None] - bboxes[:, 0][None]
            t = ys[:, None] - bboxes[:, 1][None]
            r = bboxes[:, 2][None] - xs[:, None]
            b = bboxes[:, 3][None] - ys[:, None]

            #print(xs[:, None].shape, xs[:, None])
            #print(l.shape)
            #print(l)
            #print(key_custom[:, 0][None])

            l_key = key_custom[:, 0][None] - bboxes[:, 0][None] ############i add this
            t_key = key_custom[:, 1][None] - bboxes[:, 1][None] ##
            r_key = bboxes[:, 2][None] - key_custom[:, 0][None] ##
            b_key = bboxes[:, 3][None] - key_custom[:, 1][None] ##

            #print("LENGTJHTJHTJHT", l_key[0].shape, len(l), l_key)
            #print(l_key.expand(l.shape[0],-1))
            l_key_stack = l_key.expand(l.shape[0],-1)
            t_key_stack = t_key.expand(l.shape[0],-1)
            r_key_stack = r_key.expand(l.shape[0],-1)
            b_key_stack = b_key.expand(l.shape[0],-1)

            #print(l_key_stack.shape)

            key_stack = torch.stack([l_key_stack, t_key_stack, r_key_stack, b_key_stack], dim=2)
            #print("KEYYSSTTACK", key_stack.shape, key_stack)

            



            box_targets_per_img = torch.stack([l, t, r, b], 2)
            #print("BOXTARGETIMGaeeee", box_targets_per_img.shape, box_targets_per_img)
            if self.center_sample:
                is_in_boxes = self.get_sample_region(
                    bboxes, self.strides, n_point_per_level, xs, ys, radius=self.radius
                )

            else:
                is_in_boxes = box_targets_per_img.min(2)[0] > 0

            max_box_targets_per_img = box_targets_per_img.max(2)[0]
            #print("maxboxtargetperimg", max_box_targets_per_img.shape)

            is_cared_in_level = (
                max_box_targets_per_img >= sizes_of_interest[:, [0]]
            ) & (max_box_targets_per_img <= sizes_of_interest[:, [1]])
            #print(is_cared_in_level)  


            locations_to_gt_area = area[None].repeat(len(locations), 1)
            locations_to_gt_area[is_in_boxes == 0] = INF
            locations_to_gt_area[is_cared_in_level == 0] = INF

            locations_to_min_area, locations_to_gt_id = locations_to_gt_area.min(1)

            #print("BOXTAR", box_targets_per_img.shape, box_targets_per_img)

            box_targets_per_img = box_targets_per_img[
                range(len(locations)), locations_to_gt_id
            ]
            #print("BOXTAR", box_targets_per_img.shape, box_targets_per_img)
            
            key_stack = key_stack[range(len(locations)), locations_to_gt_id]  #####i add this for 
            
            #print("KJEYSTACKKKK", key_stack.shape, key_stack)


            labels_per_img = labels_per_img[locations_to_gt_id]
            labels_per_img[locations_to_min_area == INF] = 0

            labels.append(labels_per_img)
            box_targets.append(box_targets_per_img)
            key_stack_list.append(key_stack)   #######i add this for keypoinyt


            #print("BOXTARGETIMGaeeee", box_targets_per_img.shape, box_targets_per_img)
            #print(labels_per_img, labels)
            #print("AAAAAAAAAAAA", type(box_targets), len(box_targets), box_targets)



            
        #return labels, box_targets
        return labels, box_targets, key_stack_list ########### i add this

    #def compute_centerness_targets(self, box_targets):
    def compute_centerness_targets(self, box_targets, key_stacks):  ###########i add this

        left_right = box_targets[:, [0, 2]]
        top_bottom = box_targets[:, [1, 3]]

        left_right_key = key_stacks[:, [0, 2]] ######## i add this
        top_bottom_key = key_stacks[:, [1, 3]]#### 


        #print("BOXTARGET CENTERNESS", box_targets, len(box_targets))
        #print("KEYSTACK CENTERNESSS", key_stacks, len(key_stacks))
        #print(len(key_stacks[0]), len(key_stacks[1]), len(key_stacks[2]), len(key_stacks[3]), len(key_stacks[4]))
    
        #print("909090190901909102c@@en" , type(left_right.min(-1)[0]))
        
        print("##########", left_right[:,0].shape, top_bottom[:,0].shape)

        l_over_r = left_right[:,0] / left_right[:,1]   # l/r ... i add this for keypoint
        t_over_b = top_bottom[:,0] / top_bottom[:,1]   # t/b ...

        l_over_r_key = left_right_key[:,0] / left_right_key[:,1]   # l/r_key ... i add this for keypoint
        t_over_b_key = top_bottom_key[:,0] / top_bottom_key[:,1]   # t/b_key ...

        #print(l_over_r.shape, l_over_r_key.shape)

        ppp = l_over_r_key / l_over_r
        qqq = t_over_b_key / t_over_b
        
        #####
        l_over_t = left_right[:,0] / top_bottom[:,0]
        l_over_t_key = left_right_key[:,0] / top_bottom_key[:,0]
        rrr = l_over_t_key / l_over_t
        rrr_min = torch.stack([abs(1/i) if abs(i)>1 else abs(i) for i in rrr])
        ##### 

        ppp_min = torch.stack([abs(1/i) if abs(i)>1 else abs(i) for i in ppp])#########
        qqq_min = torch.stack([abs(1/i) if abs(i)>1 else abs(i) for i in qqq])######
        
        centerness = ppp_min * qqq_min * rrr_min

        #l_over_r_key_min = torch.stack([abs(1/i) if abs(i)>1 else abs(i) for i in l_over_r_key])#########
        #t_over_b_key_min = torch.stack([abs(1/i) if abs(i)>1 else abs(i) for i in t_over_b_key])######

        #centerness = l_over_r_key_min * t_over_b_key_min###############

             



       



        #centerness = (left_right.min(-1)[0] / left_right.max(-1)[0]) * (
        #    top_bottom.min(-1)[0] / top_bottom.max(-1)[0]
        #)





        #print("!!!!!!!!!!!!!!!!!!!!!" , type(centerness), len(centerness), centerness)

        return torch.sqrt(centerness)

    def forward(self, locations, cls_pred, box_pred, center_pred, targets):
        batch = cls_pred[0].shape[0]
        n_class = cls_pred[0].shape[1]

        #print("loss.pyyy@@@@@@@@@@@",targets[0].key)
        #labels, box_targets = self.prepare_target(locations, targets)
        labels, box_targets, key_stacks = self.prepare_target(locations, targets) #######i add this for keypoint
        #print("FORWARDDDD", len(box_targets), len(key_stacks))
        cls_flat = []
        box_flat = []
        center_flat = []

        labels_flat = []
        box_targets_flat = []
        key_stacks_flat = []  ##################i add this
        

        for i in range(len(labels)):
            cls_flat.append(cls_pred[i].permute(0, 2, 3, 1).reshape(-1, n_class))
            box_flat.append(box_pred[i].permute(0, 2, 3, 1).reshape(-1, 4))
            center_flat.append(center_pred[i].permute(0, 2, 3, 1).reshape(-1))

            labels_flat.append(labels[i].reshape(-1))
            box_targets_flat.append(box_targets[i].reshape(-1, 4))
            key_stacks_flat.append(key_stacks[i].reshape(-1, 4)) ############## i add this

        cls_flat = torch.cat(cls_flat, 0)
        box_flat = torch.cat(box_flat, 0)
        center_flat = torch.cat(center_flat, 0)

        labels_flat = torch.cat(labels_flat, 0)
        box_targets_flat = torch.cat(box_targets_flat, 0)
        key_stacks_flat = torch.cat(key_stacks_flat, 0) ###############i add this

        pos_id = torch.nonzero(labels_flat > 0).squeeze(1)

        cls_loss = self.cls_loss(cls_flat, labels_flat.int()) / (pos_id.numel() + batch)

        box_flat = box_flat[pos_id]
        center_flat = center_flat[pos_id]

        box_targets_flat = box_targets_flat[pos_id]
        key_stacks_flat = key_stacks_flat[pos_id] ##########i add this

        if pos_id.numel() > 0:
            center_targets = self.compute_centerness_targets(box_targets_flat, key_stacks_flat)

            box_loss = self.box_loss(box_flat, box_targets_flat, center_targets)
            center_loss = self.center_loss(center_flat, center_targets)
            #print("center_loss233232232332", center_loss)

        else:
            box_loss = box_flat.sum()
            center_loss = center_flat.sum()



        return cls_loss, box_loss, center_loss

