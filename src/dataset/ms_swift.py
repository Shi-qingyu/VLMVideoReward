import contextlib
import fcntl
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from transformers import AutoProcessor

from . import data_list

THINK_PATTERN = re.compile(r'<think>.*?</think>', re.DOTALL)
PLACEHOLDER_PATTERN = re.compile(r'(<image>|<video>)')
ROLE_MAPPING = {
    'human': 'user',
    'user': 'user',
    'gpt': 'assistant',
    'assistant': 'assistant',
}


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open('r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]


def load_annotations(path: Path) -> List[Any]:
    if path.suffix == '.jsonl':
        return read_jsonl(path)
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def normalize_media_list(media: Any) -> List[Any]:
    if media is None:
        return []
    if isinstance(media, str):
        return [media]
    return list(media)


def make_abs_path(base_path: Path, rel_path: str) -> str:
    return str((base_path / rel_path).resolve())


def strip_cot(text: str) -> str:
    return THINK_PATTERN.sub('', text).strip()


def inject_time_instruction(text: str, time_instruction: str) -> str:
    if not time_instruction:
        return text.strip()

    blocks: List[str] = []
    inserted = False
    for segment in PLACEHOLDER_PATTERN.split(text):
        if segment in {'<image>', '<video>'}:
            blocks.append(segment)
            continue
        segment = segment.strip()
        if not segment:
            continue
        if not inserted:
            blocks.append(f'{time_instruction}\n{segment}')
            inserted = True
        else:
            blocks.append(segment)

    return '\n'.join(blocks).strip()


def configure_processor_media(processor, media_settings: Dict[str, Any]) -> None:
    image_processor = getattr(processor, 'image_processor', None)
    if image_processor is not None:
        if media_settings.get('min_pixels') is not None and hasattr(image_processor, 'min_pixels'):
            image_processor.min_pixels = media_settings['min_pixels']
        if media_settings.get('max_pixels') is not None and hasattr(image_processor, 'max_pixels'):
            image_processor.max_pixels = media_settings['max_pixels']

    video_processor = getattr(processor, 'video_processor', None)
    if video_processor is not None:
        if media_settings.get('video_min_pixels') is not None and hasattr(video_processor, 'min_pixels'):
            video_processor.min_pixels = media_settings['video_min_pixels']
        if media_settings.get('video_max_pixels') is not None and hasattr(video_processor, 'max_pixels'):
            video_processor.max_pixels = media_settings['video_max_pixels']
        if media_settings.get('video_min_frames') is not None and hasattr(video_processor, 'min_frames'):
            video_processor.min_frames = media_settings['video_min_frames']
        if media_settings.get('video_max_frames') is not None and hasattr(video_processor, 'max_frames'):
            video_processor.max_frames = media_settings['video_max_frames']
        if media_settings.get('video_fps') is not None and hasattr(video_processor, 'fps'):
            video_processor.fps = media_settings['video_fps']


def build_processor(
    model_name_or_path: str,
    media_settings: Dict[str, Any],
    cache_dir: Optional[str] = None,
):
    processor = AutoProcessor.from_pretrained(model_name_or_path, cache_dir=cache_dir)
    configure_processor_media(processor, media_settings)
    return processor


def build_video_time_instruction(item: Dict[str, Any], processor) -> str:
    videos = normalize_media_list(item.get('videos'))
    video_processor = getattr(processor, 'video_processor', None)
    if not videos or video_processor is None:
        return ''

    base_path = Path(item.get('data_path', ''))
    abs_videos = [make_abs_path(base_path, video) for video in videos]
    vp_output = video_processor(videos=abs_videos, return_metadata=True)

    video_metadata = vp_output.video_metadata[0]
    video_grid_thw = vp_output.video_grid_thw
    sample_fps = float(getattr(video_processor, 'fps', 0.0))
    temporal_patch_size = int(getattr(video_processor, 'temporal_patch_size', 1))
    total_frames = int(video_grid_thw[0][0] * temporal_patch_size)
    duration = float(video_metadata.get('duration', 0.0))
    return (
        f'This video is uniformly sampled at {sample_fps:.2f} fps, contains {total_frames} frames '
        f'from 0 seconds to {duration:.1f} seconds.'
    )


def convert_conversations(
    item: Dict[str, Any],
    using_cot: bool,
    time_instruction: str,
) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = []
    for turn in item['conversations']:
        role = ROLE_MAPPING.get(turn['from'])
        if role is None:
            raise ValueError(f"Unsupported conversation role: {turn['from']}")

        text = turn['value']
        if role == 'assistant' and not using_cot:
            text = strip_cot(text)
        elif role == 'user' and time_instruction:
            text = inject_time_instruction(text, time_instruction)
            time_instruction = ''
        else:
            text = text.strip()

        messages.append({'role': role, 'content': text})
    return messages


def convert_sample(
    item: Dict[str, Any],
    using_cot: bool,
    processor,
) -> Dict[str, Any]:
    base_path = Path(item.get('data_path', ''))
    images = [make_abs_path(base_path, image) for image in normalize_media_list(item.get('images'))]
    videos = [make_abs_path(base_path, video) for video in normalize_media_list(item.get('videos'))]
    time_instruction = build_video_time_instruction(item, processor) if videos else ''

    row: Dict[str, Any] = {
        'messages': convert_conversations(item, using_cot=using_cot, time_instruction=time_instruction),
    }
    if images:
        row['images'] = images
    if videos:
        row['videos'] = videos
    if item.get('objects') is not None:
        row['objects'] = item['objects']
    if item.get('tools') is not None:
        row['tools'] = item['tools']
    if item.get('chat_template_kwargs') is not None:
        row['chat_template_kwargs'] = item['chat_template_kwargs']
    return row


def iter_samples(annotations: Iterable[Any], data_path: str) -> Iterable[Dict[str, Any]]:
    for ann in annotations:
        if isinstance(ann, list):
            for sub_ann in ann:
                sub_ann = dict(sub_ann)
                sub_ann['data_path'] = data_path
                yield sub_ann
        else:
            ann = dict(ann)
            ann['data_path'] = data_path
            yield ann


@contextlib.contextmanager
def file_lock(lock_path: Path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('w') as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def prepare_ms_swift_datasets(
    *,
    dataset_use: str,
    model_name_or_path: str,
    using_cot: bool,
    media_settings: Dict[str, Any],
    output_root: Path,
    cache_dir: Optional[str] = None,
) -> List[str]:
    dataset_tokens = [token.strip() for token in dataset_use.split(',') if token.strip()]
    if not dataset_tokens:
        raise ValueError('dataset_use must not be empty.')

    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    lock_path = output_root / '.prepare.lock'
    manifest_path = output_root / 'manifest.json'
    with file_lock(lock_path):
        if manifest_path.exists():
            with manifest_path.open('r', encoding='utf-8') as f:
                manifest = json.load(f)
            dataset_paths = manifest.get('dataset_paths', [])
            if (
                manifest.get('dataset_use') == dataset_use
                and manifest.get('model_name_or_path') == model_name_or_path
                and manifest.get('using_cot') == using_cot
                and manifest.get('media_settings') == media_settings
                and dataset_paths
                and all(Path(path).exists() for path in dataset_paths)
            ):
                return dataset_paths

        processor = build_processor(
            model_name_or_path=model_name_or_path,
            media_settings=media_settings,
            cache_dir=cache_dir,
        )
        dataset_configs = data_list(dataset_tokens)
        dataset_paths: List[str] = []

        for dataset_token, dataset_config in zip(dataset_tokens, dataset_configs):
            annotation_path = Path(dataset_config['annotation_path']).resolve()
            data_path = str(Path(dataset_config['data_path']).resolve())
            annotations = load_annotations(annotation_path)

            sampling_rate = dataset_config.get('sampling_rate', 1.0)
            if sampling_rate < 1.0:
                sample_size = int(len(annotations) * sampling_rate)
                annotations = random.sample(annotations, sample_size)

            safe_name = re.sub(r'[^a-zA-Z0-9._-]+', '_', dataset_token)
            output_path = output_root / f'{safe_name}.jsonl'
            with output_path.open('w', encoding='utf-8') as f:
                for sample in iter_samples(annotations, data_path=data_path):
                    row = convert_sample(sample, using_cot=using_cot, processor=processor)
                    f.write(json.dumps(row, ensure_ascii=False) + '\n')
            dataset_paths.append(str(output_path))

        manifest = {
            'dataset_use': dataset_use,
            'model_name_or_path': model_name_or_path,
            'using_cot': using_cot,
            'media_settings': media_settings,
            'dataset_paths': dataset_paths,
        }
        with manifest_path.open('w', encoding='utf-8') as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        return dataset_paths
