#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, csv, os
import numpy as np
import matplotlib.pyplot as plt

def load_xy(path):
    xs, ys = [], []
    with open(path, newline='') as f:
        r = csv.reader(f)
        for row in r:
            if not row or len(row) < 2:
                continue
            try:
                x = float(row[0].strip()); y = float(row[1].strip())
                xs.append(x); ys.append(y)
            except ValueError:
                # 헤더/주석 등 숫자 변환 불가 행은 건너뜀
                continue
    if not xs:
        raise ValueError(f"'{path}'에서 (x,y) 데이터를 읽지 못했습니다. CSV 형식(두 열) 확인 필요.")
    return np.array(xs), np.array(ys)

def main():
    ap = argparse.ArgumentParser(description="Track & racing line visualizer (inner/outer/center/racing)")
    ap.add_argument("--inner",  required=True, help="inner.csv 경로")
    ap.add_argument("--outer",  required=True, help="outer.csv 경로")
    ap.add_argument("--center", help="centerline.csv 경로(선택)")
    ap.add_argument("--racing", help="racingline.csv 경로(선택)")
    ap.add_argument("--save",   help="결과 저장 파일명(예: track.png). 없으면 창에 표시")
    ap.add_argument("--title",  default="Track Visualization", help="플롯 제목")
    args = ap.parse_args()

    inner_x, inner_y = load_xy(args.inner)
    outer_x, outer_y = load_xy(args.outer)

    plt.figure(figsize=(8, 8))
    plt.plot(inner_x, inner_y, '-',  linewidth=1.5, label='Inner')
    plt.plot(outer_x, outer_y, '-',  linewidth=1.5, label='Outer')

    if args.center and os.path.exists(args.center):
        cx, cy = load_xy(args.center)
        plt.plot(cx, cy, '-', linewidth=2.2, label='Centerline')

    if args.racing and os.path.exists(args.racing):
        rx, ry = load_xy(args.racing)
        # 레이싱 라인은 눈에 띄게
        plt.plot(rx, ry, '-', linewidth=2.2, label='Racing Line')

    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.title(args.title)
    plt.legend()

    if args.save:
        plt.savefig(args.save, dpi=300, bbox_inches='tight')
        print(f"Saved to {args.save}")
    else:
        plt.show()

if __name__ == "__main__":
    main()
