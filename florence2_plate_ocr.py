#!/usr/bin/env python3
# Florence-2を用いたナンバープレート領域検出（Bounding Box取得）および3桁ナンバーOCRスクリプト
# 参考: https://github.com/Shuma-s58/florence2_work

import argparse
import json
import math
import re
import sys
import time
import unicodedata
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

GROUNDING_TASK = '<CAPTION_TO_PHRASE_GROUNDING>'
OCR_TASK = '<OCR_WITH_REGION>'
DEFAULT_TEXT = 'A blue sign with numbers.'
DEFAULT_MODEL = 'florence-community/Florence-2-large'

# ローカルモデル配置用のディレクトリ（存在すれば優先利用）
LOCAL_MODELS_DIR = Path(__file__).resolve().parent / 'models'


def resolve_model_path(model_arg) :
    # 指定されたモデルIDまたはローカルパスを解決する
    import os
    p = Path(model_arg)
    if p.exists() :
        return str(p)
    short_name = str(model_arg).split('/')[-1]
    env_cache = os.environ.get('FLORENCE2_CACHE_DIR')
    if env_cache :
        env_path = Path(env_cache) / short_name
        if env_path.exists() :
            return str(env_path)
    local_path = LOCAL_MODELS_DIR / short_name
    if local_path.exists() :
        return str(local_path)
    return str(model_arg)


def select_device(device_arg) :
    # 実行デバイス（mps / cuda / cpu）を選択する
    import torch
    if device_arg == 'auto' :
        if torch.cuda.is_available() :
            return 'cuda'
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() :
            return 'mps'
        return 'cpu'
    if device_arg == 'cuda' and not torch.cuda.is_available() :
        raise RuntimeError('CUDAが利用できません。--device cpu または mps を指定してください。')
    if device_arg == 'mps' and not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()) :
        raise RuntimeError('MPSが利用できません。--device cpu を指定してください。')
    return device_arg


def load_font(path, size) :
    # 結果描画用のフォントを読み込む
    if path :
        return ImageFont.truetype(str(path), size)
    candidates = (
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
        'C:/Windows/Fonts/arial.ttf',
        '/System/Library/Fonts/Supplemental/Arial.ttf',
    )
    for candidate in candidates :
        if Path(candidate).is_file() :
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default(size=size)


def wrap_line(text, font, max_width) :
    # 指定したピクセル幅に収まるように文字列を折り返す
    lines = []
    for paragraph in str(text).splitlines() or [''] :
        current = ''
        for character in paragraph :
            if current and font.getlength(current + character) > max_width :
                lines.append(current)
                current = character
            else :
                current += character
        lines.append(current)
    return lines


def load_model(model_arg, device_arg='auto') :
    # Florence-2のプロセッサとモデルを読み込む
    import torch
    from transformers import AutoProcessor, Florence2ForConditionalGeneration

    device = select_device(device_arg)
    dtype = torch.float16 if device in ('cuda', 'mps') else torch.float32
    model_path = resolve_model_path(model_arg)
    processor = AutoProcessor.from_pretrained(model_path)
    model = Florence2ForConditionalGeneration.from_pretrained(
        model_path, dtype=dtype
    ).to(device).eval()
    return processor, model, device, dtype


def run_florence_task(processor, model, device, dtype, image, task, text_input='', max_new_tokens=512) :
    # Florence-2の単一タスク推論を実行し、パース済み辞書を返す
    import torch

    prompt = task + text_input
    inputs = processor(text=prompt, images=image, return_tensors='pt')
    inputs = {
        key: value.to(device=device, dtype=dtype) if value.is_floating_point()
        else value.to(device=device)
        for key, value in inputs.items()
    }
    with torch.inference_mode() :
        ids = model.generate(
            **inputs, max_new_tokens=max_new_tokens, num_beams=3, do_sample=False
        )
    if device == 'mps' :
        torch.mps.synchronize()
    elif device == 'cuda' :
        torch.cuda.synchronize()
    raw = processor.batch_decode(ids, skip_special_tokens=False)[0]
    parsed = processor.post_process_generation(raw, task=task, image_size=image.size)
    if task not in parsed :
        raise RuntimeError(f'モデル出力にタスク {task} の結果が含まれていません。')
    return parsed[task]


def parse_grounding_regions(grounding_result, image_size) :
    # Groundingの結果から有効なBounding Boxリストを作成する
    labels = grounding_result.get('labels', [])
    boxes = grounding_result.get('bboxes', [])
    width, height = image_size
    regions = []
    for index, (label, box) in enumerate(zip(labels, boxes), 1) :
        coords = [float(v) for v in box]
        if len(coords) != 4 or not all(map(math.isfinite, coords)) :
            continue
        x1, y1, x2, y2 = coords
        x1 = max(0.0, min(x1, width))
        y1 = max(0.0, min(y1, height))
        x2 = max(0.0, min(x2, width))
        y2 = max(0.0, min(y2, height))
        if x2 <= x1 or y2 <= y1 :
            continue
        area = (x2 - x1) * (y2 - y1)
        regions.append({
            'id': index,
            'label': str(label),
            'bbox': [x1, y1, x2, y2],
            'area_px2': area,
        })
    return regions


def choose_plate_region(regions) :
    # 検出された候補領域からナンバープレート全体のBounding Boxを選択する
    if not regions :
        return None
    # ラベルに sign または plate を含む候補を優先し、最大面積のものを選択する
    plate_candidates = [
        r for r in regions
        if 'sign' in r['label'].lower() or 'plate' in r['label'].lower()
    ]
    target_list = plate_candidates if plate_candidates else regions
    return max(target_list, key=lambda r : r['area_px2'])


def recognize_plate_from_crop(processor, model, device, dtype, image, plate_bbox, max_new_tokens=256) :
    # プレート領域を切り出し、リサイズ歪み防止用の余白を付与して3桁の数字をOCR認識する
    x1, y1, x2, y2 = [int(round(v)) for v in plate_bbox]
    x1 = max(0, min(x1, image.width - 1))
    y1 = max(0, min(y1, image.height - 1))
    x2 = max(x1 + 1, min(x2, image.width))
    y2 = max(y1 + 1, min(y2, image.height))
    crop = image.crop((x1, y1, x2, y2))

    # Florence-2の768x768リサイズで文字が過度に拡大されるのを防ぐため白背景キャンバスの中央に配置
    cw, ch = crop.size
    canvas = Image.new('RGB', (cw * 3, ch * 3), 'white')
    offset_x, offset_y = cw, ch
    canvas.paste(crop, (offset_x, offset_y))

    ocr_res = run_florence_task(
        processor, model, device, dtype, canvas, OCR_TASK, max_new_tokens=max_new_tokens
    )
    labels = ocr_res.get('labels', [])
    quad_boxes = ocr_res.get('quad_boxes', [])

    ocr_regions = []
    three_digit_candidates = []
    for idx, (raw_text, box) in enumerate(zip(labels, quad_boxes), 1) :
        coords = [float(v) for v in box]
        if len(coords) != 8 :
            continue
        # キャンバス上の座標を元画像のピクセル座標に変換
        orig_quad = []
        for i in range(0, 8, 2) :
            ox = coords[i] - offset_x + x1
            oy = coords[i + 1] - offset_y + y1
            orig_quad.extend([ox, oy])
        xs = orig_quad[0::2]
        ys = orig_quad[1::2]
        norm_text = unicodedata.normalize('NFKC', str(raw_text)).strip()
        region_info = {
            'id': idx,
            'text': norm_text,
            'raw_text': str(raw_text),
            'bbox': [min(xs), min(ys), max(xs), max(ys)],
            'quad_box': orig_quad,
        }
        ocr_regions.append(region_info)
        # 3桁の数字のみで構成される文字列（先頭0を含む）を抽出
        if re.fullmatch(r'[0-9]{3}', norm_text) is not None :
            three_digit_candidates.append(norm_text)

    # 完全一致がない場合は領域内テキストから3桁数字の部分一致を探す（2026等の4桁は除外）
    if not three_digit_candidates :
        for region_info in ocr_regions :
            matches = re.findall(r'(?<![0-9])[0-9]{3}(?![0-9])', region_info['text'])
            three_digit_candidates.extend(matches)

    plate_number = three_digit_candidates[0] if three_digit_candidates else None
    return plate_number, ocr_regions


def detect_and_recognize(image, processor, model, device, dtype, text_prompt=DEFAULT_TEXT, max_new_tokens=512) :
    # プレート領域検出（Grounding）＋OCRを一貫実行して結果辞書を返す
    t0 = time.perf_counter()
    grounding_raw = run_florence_task(
        processor, model, device, dtype, image, GROUNDING_TASK, text_prompt, max_new_tokens=max_new_tokens
    )
    t_ground = time.perf_counter() - t0

    grounding_regions = parse_grounding_regions(grounding_raw, image.size)
    selected_plate = choose_plate_region(grounding_regions)

    t1 = time.perf_counter()
    plate_number = None
    ocr_regions = []
    if selected_plate is not None :
        plate_number, ocr_regions = recognize_plate_from_crop(
            processor, model, device, dtype, image, selected_plate['bbox'], max_new_tokens=max_new_tokens
        )
    t_ocr = time.perf_counter() - t1

    return {
        'detected': selected_plate is not None,
        'plate_number': plate_number,
        'plate_bbox': selected_plate['bbox'] if selected_plate else None,
        'grounding_candidates': grounding_regions,
        'ocr_regions': ocr_regions,
        'timing_s': {
            'grounding': round(t_ground, 4),
            'ocr': round(t_ocr, 4),
            'total': round(t_ground + t_ocr, 4),
        },
    }


def render_result(image, result, font_path=None, font_size=20, text_prompt=DEFAULT_TEXT) :
    # 元画像に検出枠と認識結果パネルを描画した画像を生成する
    font = load_font(font_path, font_size)
    width = max(image.width, 800)
    margin = 20
    line_height = font_size + 10

    plate_num_str = result['plate_number'] if result['plate_number'] is not None else 'Not recognized'
    bbox_str = (
        ', '.join(f'{v:.1f}' for v in result['plate_bbox'])
        if result['plate_bbox'] is not None else 'None'
    )
    timing = result['timing_s']
    lines = [
        'Florence-2 Plate Detection + OCR | Origin: top-left (pixels)',
        *wrap_line('Target Prompt: ' + text_prompt, font, width - 2 * margin),
        f'Recognized Plate Number: {plate_num_str}',
        f'Plate BBox [xmin, ymin, xmax, ymax]: [{bbox_str}]',
        f"Time: Grounding={timing['grounding']:.3f}s, OCR={timing['ocr']:.3f}s, Total={timing['total']:.3f}s",
        '',
    ]
    for reg in result['ocr_regions'] :
        r_box = ', '.join(f'{v:.1f}' for v in reg['bbox'])
        lines.extend(wrap_line(
            f"[OCR {reg['id']}] '{reg['text']}' | BBox: [{r_box}]",
            font, width - 2 * margin
        ))

    output = Image.new('RGB', (width, image.height + 2 * margin + line_height * len(lines)), 'white')
    output.paste(image, (0, 0))
    draw = ImageDraw.Draw(output)

    # ナンバープレート領域のBBoxを描画
    if result['plate_bbox'] is not None :
        box = result['plate_bbox']
        draw.rectangle(box, outline='#0077b6', width=4)
        label = f'Plate: {plate_num_str}'
        bounds = draw.textbbox((0, 0), label, font=font)
        lw = bounds[2] - bounds[0] + 12
        lh = bounds[3] - bounds[1] + 8
        lx = max(0, min(box[0], image.width - lw))
        ly = max(0, min(box[1] - lh, image.height - lh))
        draw.rectangle((lx, ly, lx + lw, ly + lh), fill='#0077b6')
        draw.text((lx + 6 - bounds[0], ly + 4 - bounds[1]), label, font=font, fill='white')

    # 数字OCR領域のBBoxを描画
    for reg in result['ocr_regions'] :
        if re.fullmatch(r'[0-9]{3}', reg['text']) is not None :
            draw.rectangle(reg['bbox'], outline='#d62828', width=3)

    draw.line((0, image.height, width, image.height), fill='#aaaaaa', width=2)
    y = image.height + margin
    for line in lines :
        draw.text((margin, y), line, font=font, fill='#202020')
        y += line_height
    return output


def main() :
    parser = argparse.ArgumentParser(description='Florence-2によるナンバープレート領域検出＋3桁数字OCR')
    parser.add_argument('image', type=Path, help='入力画像のパス')
    parser.add_argument('-o', '--output', type=Path, help='結果画像（PNG）の出力パス')
    parser.add_argument('--model', default=DEFAULT_MODEL, help='Florence-2モデルIDまたはローカルパス')
    parser.add_argument('--text', default=DEFAULT_TEXT, help='プレート検出用の英語プロンプト')
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda', 'mps'), default='auto')
    parser.add_argument('--font', type=Path, help='描画用フォントファイル（TTF/OTF/TTC）')
    parser.add_argument('--font-size', type=int, default=20)
    parser.add_argument('--max-new-tokens', type=int, default=512)
    parser.add_argument('--json', type=Path, help='認識結果を保存するJSONファイルパス')
    args = parser.parse_args()

    output_path = args.output or args.image.with_name(args.image.stem + '_plate_ocr.png')
    if output_path.suffix.lower() != '.png' :
        parser.error('出力画像の拡張子は .png にしてください。')

    try :
        with Image.open(args.image) as source :
            image = ImageOps.exif_transpose(source).convert('RGB')
        processor, model, device, dtype = load_model(args.model, args.device)
        result = detect_and_recognize(
            image, processor, model, device, dtype, args.text, args.max_new_tokens
        )
        annotated = render_result(image, result, args.font, args.font_size, args.text)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        annotated.save(output_path, format='PNG')

        payload = {
            'file': args.image.name,
            'model': args.model,
            'device': device,
            'image_size': list(image.size),
            **result,
        }
        if args.json :
            args.json.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps(payload, ensure_ascii=False))
        print(f'Saved: {output_path.resolve()}', file=sys.stderr)
    except (OSError, RuntimeError, ValueError, KeyError) as exc :
        parser.exit(1, f'Error: {exc}\n')


if __name__ == '__main__' :
    main()
