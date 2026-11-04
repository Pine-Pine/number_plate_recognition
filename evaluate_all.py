#!/usr/bin/env python3
# number_plates内の全30枚の画像に対してFlorence-2（base / large）の認識精度・処理時間を評価するスクリプト

import argparse
import json
import statistics
from pathlib import Path

from PIL import Image, ImageOps

from florence2_plate_ocr import (
    DEFAULT_TEXT,
    detect_and_recognize,
    load_model,
    render_result,
    run_florence_task,
)

# 10枚のナンバープレートの正解ラベル順序（条件1〜3の各10枚に対応）
EXPECTED_SEQUENCE = [
    '054', '069', '097', '111', '239',
    '334', '598', '666', '784', '893',
]

CONDITION_NAMES = {
    1: '中距離・右側配置 (092301-092543)',
    2: '近距離・右側配置 (092917-093156)',
    3: '中遠距離・左側配置 (093521-093751)',
}


def evaluate_model_on_dataset(model_id, image_paths, device_arg, text_prompt, save_dir=None) :
    # 指定したモデルで全30枚の認識精度と処理時間を計測する
    processor, model, device, dtype = load_model(model_id, device_arg)
    short_name = str(model_id).split('/')[-1]

    # ウォームアップ推論（初回コンパイル・キャッシュのオーバーヘッドを除外）
    dummy_img = Image.new('RGB', (640, 480), 'white')
    run_florence_task(processor, model, device, dtype, dummy_img, '<OCR>', max_new_tokens=32)

    records = []
    for idx, img_path in enumerate(image_paths) :
        gt = EXPECTED_SEQUENCE[idx % 10]
        cond_id = (idx // 10) + 1
        with Image.open(img_path) as src :
            image = ImageOps.exif_transpose(src).convert('RGB')

        res = detect_and_recognize(image, processor, model, device, dtype, text_prompt=text_prompt)
        pred = res['plate_number']
        is_correct = (pred == gt)

        if save_dir is not None :
            model_out_dir = save_dir / short_name
            model_out_dir.mkdir(parents=True, exist_ok=True)
            vis = render_result(image, res, text_prompt=text_prompt)
            vis.save(model_out_dir / f'{img_path.stem}_plate_ocr.png', format='PNG')

        record = {
            'index': idx + 1,
            'file': img_path.name,
            'condition': cond_id,
            'condition_name': CONDITION_NAMES[cond_id],
            'ground_truth': gt,
            'predicted': pred,
            'correct': is_correct,
            'plate_bbox': res['plate_bbox'],
            'timing_s': res['timing_s'],
        }
        records.append(record)
        status_mark = 'OK' if is_correct else 'NG'
        print(
            f"[{short_name}] [{idx + 1:02d}/30] {img_path.name} | "
            f"GT={gt} Pred={pred} ({status_mark}) | "
            f"Grounding={res['timing_s']['grounding']:.3f}s "
            f"OCR={res['timing_s']['ocr']:.3f}s "
            f"Total={res['timing_s']['total']:.3f}s"
        )

    correct_count = sum(1 for r in records if r['correct'])
    ground_times = [r['timing_s']['grounding'] for r in records]
    ocr_times = [r['timing_s']['ocr'] for r in records]
    total_times = [r['timing_s']['total'] for r in records]

    by_condition = {}
    for cid, cname in CONDITION_NAMES.items() :
        sub = [r for r in records if r['condition'] == cid]
        c_ok = sum(1 for r in sub if r['correct'])
        c_tot = [r['timing_s']['total'] for r in sub]
        by_condition[str(cid)] = {
            'name': cname,
            'correct': c_ok,
            'total': len(sub),
            'accuracy_pct': round(c_ok / len(sub) * 100.0, 2),
            'mean_total_time_s': round(statistics.mean(c_tot), 4),
        }

    summary = {
        'model': model_id,
        'device': device,
        'total_images': len(records),
        'correct_images': correct_count,
        'accuracy_pct': round(correct_count / len(records) * 100.0, 2),
        'timing_summary_s': {
            'mean_grounding': round(statistics.mean(ground_times), 4),
            'mean_ocr': round(statistics.mean(ocr_times), 4),
            'mean_total': round(statistics.mean(total_times), 4),
            'median_total': round(statistics.median(total_times), 4),
            'min_total': round(min(total_times), 4),
            'max_total': round(max(total_times), 4),
        },
        'by_condition': by_condition,
        'records': records,
    }
    return summary


def main() :
    parser = argparse.ArgumentParser(description='全30枚のナンバープレート画像に対する評価スクリプト')
    parser.add_argument('--image-dir', type=Path, default=Path('number_plates'), help='画像フォルダのパス')
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda', 'mps'), default='auto')
    parser.add_argument('--text', default=DEFAULT_TEXT, help='Grounding用の英語プロンプト')
    parser.add_argument('--output-json', type=Path, default=Path('results/evaluation_results.json'))
    parser.add_argument('--save-images-dir', type=Path, help='アノテーション結果画像の保存先フォルダ')
    args = parser.parse_args()

    image_paths = sorted(args.image_dir.glob('*.jpg'))
    if not image_paths :
        raise SystemExit(f'画像が見つかりません: {args.image_dir}')

    models = [
        'florence-community/Florence-2-base',
        'florence-community/Florence-2-large',
    ]

    all_summaries = {}
    for model_id in models :
        short_name = model_id.split('/')[-1]
        print(f'\n================ Evaluating {short_name} ================')
        summary = evaluate_model_on_dataset(
            model_id, image_paths, args.device, args.text, args.save_images_dir
        )
        all_summaries[short_name] = summary
        ts = summary['timing_summary_s']
        print(
            f"=> {short_name}: 正解率 {summary['correct_images']}/{summary['total_images']} "
            f"({summary['accuracy_pct']:.1f}%) | "
            f"平均処理時間 {ts['mean_total']:.3f}s "
            f"(Grounding {ts['mean_grounding']:.3f}s + OCR {ts['mean_ocr']:.3f}s)"
        )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(all_summaries, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    print(f'\nSaved full evaluation report to: {args.output_json.resolve()}')


if __name__ == '__main__' :
    main()
