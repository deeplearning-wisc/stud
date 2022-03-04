from os import X_OK, pardir
import torch 
import argparse

def process(ckpt_path, save_path):
    ckpt = torch.load(ckpt_path)
    new_order = [0, 1, 2, 4, 3, 7, 6, 5, 8, 9, 10]
    ckpt['model']['roi_heads.box_predictor.cls_score.weight'] =ckpt['model']['roi_heads.box_predictor.cls_score.weight'][new_order]
    ckpt['model']['roi_heads.box_predictor.cls_score.bias'] = ckpt['model']['roi_heads.box_predictor.cls_score.bias'][new_order]

    new_order4 = []
    for x in new_order[:-1]:
        for i in range(4):
            new_order4.append(4*x+i)

    ckpt['model']['roi_heads.box_predictor.bbox_pred.weight'] = ckpt['model']['roi_heads.box_predictor.bbox_pred.weight'][new_order4]
    ckpt['model']['roi_heads.box_predictor.bbox_pred.bias'] = ckpt['model']['roi_heads.box_predictor.bbox_pred.bias'][new_order4]

    torch.save(ckpt, save_path)
    print('done!')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt-path', '-i', type=str,  help='input ckpt path')
    parser.add_argument('--save-path', '-o', type=str, help='output ckpt path')
    args = parser.parse_args()

    process(args.ckpt_path, args.save_path)

if __name__ == '__main__':
    main()

    
    

