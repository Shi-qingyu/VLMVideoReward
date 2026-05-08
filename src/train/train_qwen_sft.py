import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.dataset.ms_swift import prepare_ms_swift_datasets

DEFAULT_NEW_SPECIAL_TOKENS = ['<answer>', '</answer>', '<box>', '</box>', '<t>', '</t>']


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {'true', '1', 'yes', 'y'}:
        return True
    if value in {'false', '0', 'no', 'n'}:
        return False
    raise argparse.ArgumentTypeError(f'Invalid boolean value: {value}')


def build_legacy_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Compatibility wrapper that migrates the legacy SFT entrypoint onto ms-swift.',
    )
    parser.add_argument('--model_name_or_path', required=True)
    parser.add_argument('--dataset_use', required=True)
    parser.add_argument('--cache_dir', default=None)
    parser.add_argument('--data_flatten', nargs='?', const=True, default=False, type=str2bool)
    parser.add_argument('--data_packing', nargs='?', const=True, default=False, type=str2bool)
    parser.add_argument('--using_cot', nargs='?', const=True, default=True, type=str2bool)
    parser.add_argument('--tune_mm_llm', nargs='?', const=True, default=False, type=str2bool)
    parser.add_argument('--tune_mm_mlp', nargs='?', const=True, default=False, type=str2bool)
    parser.add_argument('--tune_mm_vision', nargs='?', const=True, default=False, type=str2bool)
    parser.add_argument('--lora_enable', nargs='?', const=True, default=False, type=str2bool)
    parser.add_argument('--lora_r', type=int, default=64)
    parser.add_argument('--lora_alpha', type=int, default=128)
    parser.add_argument('--lora_dropout', type=float, default=0.0)
    parser.add_argument('--model_max_length', type=int, default=None)
    parser.add_argument('--mm_projector_lr', type=float, default=None)
    parser.add_argument('--vision_tower_lr', type=float, default=None)
    parser.add_argument('--max_pixels', type=int, default=None)
    parser.add_argument('--min_pixels', type=int, default=None)
    parser.add_argument('--video_max_frames', type=int, default=None)
    parser.add_argument('--video_min_frames', type=int, default=None)
    parser.add_argument('--video_max_pixels', type=int, default=None)
    parser.add_argument('--video_min_pixels', type=int, default=None)
    parser.add_argument('--video_fps', type=float, default=None)
    parser.add_argument('--base_interval', type=int, default=None)
    return parser


def has_flag(args: Sequence[str], flag: str) -> bool:
    return flag in args


def extract_option(args: List[str], option: str) -> Tuple[Optional[str], List[str]]:
    extracted: Optional[str] = None
    kept: List[str] = []
    skip_next = False
    for idx, arg in enumerate(args):
        if skip_next:
            skip_next = False
            continue
        if arg == option:
            if idx + 1 >= len(args):
                raise ValueError(f'Expected a value after {option}.')
            extracted = args[idx + 1]
            skip_next = True
            continue
        kept.append(arg)
    return extracted, kept


def to_json_flag(value: Any) -> str:
    return 'true' if value else 'false'


def build_model_kwargs(legacy_args, passthrough_args: List[str]) -> List[str]:
    existing_model_kwargs, passthrough_args[:] = extract_option(passthrough_args, '--model_kwargs')
    model_kwargs: Dict[str, Any] = {}
    if existing_model_kwargs:
        model_kwargs.update(json.loads(existing_model_kwargs))

    if legacy_args.min_pixels is not None:
        model_kwargs['min_pixels'] = legacy_args.min_pixels
    if legacy_args.max_pixels is not None:
        model_kwargs['max_pixels'] = legacy_args.max_pixels
    if legacy_args.video_min_pixels is not None:
        model_kwargs['video_min_pixels'] = legacy_args.video_min_pixels
    if legacy_args.video_max_pixels is not None:
        model_kwargs['video_max_pixels'] = legacy_args.video_max_pixels
    if legacy_args.video_min_frames is not None:
        model_kwargs['fps_min_frames'] = legacy_args.video_min_frames
    if legacy_args.video_max_frames is not None:
        model_kwargs['fps_max_frames'] = legacy_args.video_max_frames
    if legacy_args.video_fps is not None:
        model_kwargs['fps'] = legacy_args.video_fps

    if not model_kwargs:
        return []
    return ['--model_kwargs', json.dumps(model_kwargs)]


def build_swift_args(legacy_args, passthrough_args: List[str]) -> List[str]:
    swift_args = list(passthrough_args)
    output_dir, _ = extract_option(list(swift_args), '--output_dir')
    dataset_cache_root = Path(output_dir).resolve() / 'ms_swift_dataset' if output_dir else project_root / '.ms_swift_dataset'

    media_settings = {
        'min_pixels': legacy_args.min_pixels,
        'max_pixels': legacy_args.max_pixels,
        'video_min_pixels': legacy_args.video_min_pixels,
        'video_max_pixels': legacy_args.video_max_pixels,
        'video_min_frames': legacy_args.video_min_frames,
        'video_max_frames': legacy_args.video_max_frames,
        'video_fps': legacy_args.video_fps,
    }
    dataset_paths = prepare_ms_swift_datasets(
        dataset_use=legacy_args.dataset_use,
        model_name_or_path=legacy_args.model_name_or_path,
        using_cot=legacy_args.using_cot,
        media_settings=media_settings,
        output_root=dataset_cache_root,
        cache_dir=legacy_args.cache_dir,
    )

    if not has_flag(swift_args, '--use_hf'):
        swift_args.extend(['--use_hf', 'true'])
    if not has_flag(swift_args, '--add_version'):
        swift_args.extend(['--add_version', 'false'])

    swift_args.extend(['--model', legacy_args.model_name_or_path])
    swift_args.extend(['--dataset', *dataset_paths])

    if legacy_args.model_max_length is not None:
        swift_args.extend(['--max_length', str(legacy_args.model_max_length)])
    if legacy_args.max_pixels is not None:
        swift_args.extend(['--max_pixels', str(legacy_args.max_pixels)])

    swift_args.extend(['--tuner_type', 'lora' if legacy_args.lora_enable else 'full'])
    swift_args.extend(['--freeze_llm', to_json_flag(not legacy_args.tune_mm_llm)])
    swift_args.extend(['--freeze_aligner', to_json_flag(not legacy_args.tune_mm_mlp)])
    swift_args.extend(['--freeze_vit', to_json_flag(not legacy_args.tune_mm_vision)])

    if legacy_args.lora_enable:
        swift_args.extend([
            '--lora_rank',
            str(legacy_args.lora_r),
            '--lora_alpha',
            str(legacy_args.lora_alpha),
            '--lora_dropout',
            str(legacy_args.lora_dropout),
            '--target_modules',
            'q_proj',
            'k_proj',
            'v_proj',
            'o_proj',
        ])

    if legacy_args.mm_projector_lr is not None:
        swift_args.extend(['--aligner_lr', str(legacy_args.mm_projector_lr)])
    if legacy_args.vision_tower_lr is not None:
        swift_args.extend(['--vit_lr', str(legacy_args.vision_tower_lr)])
    if legacy_args.data_flatten or legacy_args.data_packing:
        swift_args.extend(['--packing', 'true'])

    swift_args.extend(['--new_special_tokens', *DEFAULT_NEW_SPECIAL_TOKENS])
    swift_args.extend(build_model_kwargs(legacy_args, swift_args))
    return swift_args


def main():
    parser = build_legacy_parser()
    legacy_args, passthrough_args = parser.parse_known_args()

    try:
        from swift.pipelines import sft_main
    except ImportError as exc:
        raise ImportError(
            'Failed to import ms-swift. Please install the dependencies from `requirements.txt` '
            'or run `pip install ms-swift`.'
        ) from exc

    swift_args = build_swift_args(legacy_args, passthrough_args)
    return sft_main(swift_args)


if __name__ == '__main__':
    main()
