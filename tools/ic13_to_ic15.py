import glob
import os

gt_dir = './Challenge2_Training_Task1_GT'
output_dir = './test'

txt_files = glob.glob(gt_dir + "/*.*")
for txt_file in txt_files:
    out_file = os.path.join(output_dir, txt_file.split('/')[-1])
    with open(txt_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        lines = [line.strip().replace('"', '').split(' ') for line in lines]

    with open(out_file, 'w', encoding='utf-8') as f:
        for l in lines:
            pnts = [int(x) for x in l[:4]]
            w = pnts[2] - pnts[0]
            h = pnts[3] - pnts[1]
            x1 = pnts[0]
            y1 = pnts[1]
            x2 = x1 + w
            y2 = y1
            x3 = x1 + w
            y3 = y1 + h
            x4 = x1
            y4 = y1 + h
            f.write('{},{},{},{},{},{},{},{},{}\n'.format(x1, y1, x2, y2, x3, y3, x4, y4, l[4]))
