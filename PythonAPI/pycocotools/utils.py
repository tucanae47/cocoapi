import copy
import numpy as np
from .coco import COCO
from .cocoeval import COCOeval

class COCOAnalyzer:
    def __init__(self, cocoGt, cocoDt):
        self.cocoGt = copy.copy(cocoGt)
        self.cocoDt = copy.copy(cocoDt)
        imgIds = cocoGt.getImgIds()
        iou_type = "bbox"
        self.cocoEval = COCOeval(copy.deepcopy(cocoGt), copy.deepcopy(cocoDt), iou_type)
        self.cocoEval.params.imgIds = imgIds
        self.cocoEval.params.iouThrs = [.75, .5, .1]
        self.cocoEval.params.maxDets = [100]#, 100, 100]
        self.cocoEval.evaluate()
        self.cocoEval.accumulate()
        #self.cocoEval.summarize()
        self.ps = self.cocoEval.eval['precision']
        self.ps = np.vstack([self.ps, np.zeros((4, *self.ps.shape[1:]))])
        self.catIds = cocoGt.getCatIds()
        self.recThrs = self.cocoEval.params.recThrs        

        for k, catId in enumerate(self.catIds):
            _, ps_ = self.analyze_individual_category(catId)
            ps_supercategory = ps_['ps_supercategory']
            ps_allcategory = ps_['ps_allcategory']
            # compute precision but ignore superclass confusion
            self.ps[3, :, k, :, :] = ps_supercategory
            # compute precision but ignore any class confusion
            self.ps[4, :, k, :, :] = ps_allcategory
            # fill in background and false negative errors and plot
            self.ps[self.ps == -1] = 0
            self.ps[5, :, k, :, :] = (self.ps[4, :, k, :, :] > 0)
            self.ps[6, :, k, :, :] = 1.0

    def analyze_individual_category(self, catId, iou_type="bbox"):
        nm = self.cocoGt.loadCats(catId)[0]    
        ps_ = {}
        dt = copy.deepcopy(self.cocoDt)
        nm = self.cocoGt.loadCats(catId)[0]
        imgIds = self.cocoGt.getImgIds()
        dt_anns = dt.dataset['annotations']
        select_dt_anns = []
        for ann in dt_anns:
            if ann['category_id'] == catId:
                select_dt_anns.append(ann)
        dt.dataset['annotations'] = select_dt_anns
        dt.createIndex()
        # compute precision but ignore superclass confusion
        gt = copy.deepcopy(self.cocoGt)
        child_catIds = gt.getCatIds(supNms=[nm['supercategory']])
        for idx, ann in enumerate(gt.dataset['annotations']):
            if (ann['category_id'] in child_catIds and ann['category_id'] != catId):
                gt.dataset['annotations'][idx]['ignore'] = 1
                gt.dataset['annotations'][idx]['iscrowd'] = 1
                gt.dataset['annotations'][idx]['category_id'] = catId
    
        cocoEval = COCOeval(gt, copy.deepcopy(dt), iou_type)
        cocoEval.params.imgIds = imgIds
        cocoEval.params.maxDets = [100]
        cocoEval.params.iouThrs = [.1]
        cocoEval.params.useCats = 1
        cocoEval.evaluate()
        cocoEval.accumulate()
        k = self.catIds.index(catId)
        ps_supercategory = cocoEval.eval['precision'][0, :, k, :, :]
        ps_['ps_supercategory'] = ps_supercategory

       # compute precision but ignore any class confusion
        gt = copy.deepcopy(self.cocoGt)
        for idx, ann in enumerate(gt.dataset['annotations']):
            if ann['category_id'] != catId:
                gt.dataset['annotations'][idx]['ignore'] = 1
                gt.dataset['annotations'][idx]['iscrowd'] = 1
                gt.dataset['annotations'][idx]['category_id'] = catId
        cocoEval = COCOeval(gt, copy.deepcopy(dt), iou_type)
        cocoEval.params.imgIds = imgIds
        cocoEval.params.maxDets = [100]
        cocoEval.params.iouThrs = [.1]
        cocoEval.params.useCats = 1
        cocoEval.evaluate()
        cocoEval.accumulate()
        ps_allcategory = cocoEval.eval['precision'][0, :, k, :, :]
        ps_['ps_allcategory'] = ps_allcategory
        return k, ps_

    def makeplot(self, catId=None, ax=None, area="all", info="basic"):
        import matplotlib.pyplot as plt
        rs = self.recThrs
        if catId is None:
            ps = self.ps            
            class_name = "all clases"
        else:
            k = self.catIds.index(catId)
            ps = self.ps[:,:,k]
            class_name  = self.cocoGt.loadCats(catId)[0]
            class_name  = class_name["name"]

        cs = np.vstack([
            np.ones((2, 3)),
            np.array([.31, .51, .74]),
            np.array([.75, .31, .30]),
            np.array([.36, .90, .38]),
            np.array([.50, .39, .64]),
            np.array([1, .6, 0])
        ])

        areaNames = ['all', 'small', 'medium', 'large']
        types = ['C75', 'C50', 'Loc', 'Sim', 'Oth', 'BG', 'FN']    
        if info=="basic":
            types = types[:2]
            cs = ["#a6bbff", "#5e66ff"]

        i = areaNames.index(area)
        area_ps = ps[..., i, 0]
        aps = [ps_.mean() for ps_ in area_ps]
        ps_curve = [ ps_.mean(axis=1) if ps_.ndim > 1 else ps_ for ps_ in area_ps]
        ps_curve.insert(0, np.zeros(ps_curve[0].shape))
        if ax is None:
            ax = plt.gca()
        result={}
        for k in range(len(types)):
            ax.plot(rs, ps_curve[k + 1], color=[0, 0, 0], linewidth=0.5)
            ax.fill_between( rs, ps_curve[k], ps_curve[k + 1], color=cs[k], label=str('[{:.3f}'.format(aps[k]) + ']' + types[k]))
            result[types[k]] = aps[k]
        plt.xlabel('recall')
        plt.ylabel('precision')
        plt.xlim(0, 1.)
        plt.ylim(0, 1.)
        figure_tile = "Category: {}     ( Area: {} )     mAp(IoU=0.5)={:.2f}".format(class_name, areaNames[i], result["C50"])
        plt.title(figure_tile)
        plt.legend()
        return result
    

