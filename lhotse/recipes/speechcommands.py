"""
About the speech_commands corpus
This is a set of one-second .wav audio files, each containing a single spoken English word or background noise.
These words are from a small set of commands, and are spoken by a variety of different speakers.
This data set is designed to help train simple machine learning models.
It is covered in more detail at https://arxiv.org/abs/1804.03209.

Version 0.01 of the data set (configuration "v0.01") was released on August 3rd 2017 and contains 64,727 audio files.
Version 0.02 of the data set (configuration "v0.02") was released on April 11th 2018 and contains 105,829 audio files.
"""

import logging
import os
import shutil
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

from tqdm.auto import tqdm

from lhotse import validate_recordings_and_supervisions
from lhotse.audio import Recording, RecordingSet
from lhotse.recipes.utils import manifests_exist, read_manifests_if_cached
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import (
    Pathlike,
    is_module_available,
    safe_extract,
    urlretrieve_progress,
)

_DOWNLOAD_PATH_V1 = "http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz"
_TEST_DOWNLOAD_PATH_V1 = (
    "http://download.tensorflow.org/data/speech_commands_test_set_v0.01.tar.gz"
)
_DOWNLOAD_PATH_V2 = "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
_TEST_DOWNLOAD_PATH_V2 = (
    "http://download.tensorflow.org/data/speech_commands_test_set_v0.02.tar.gz"
)

_SPLITS = ["train", "valid", "test"]

WORDS = ["down", "go", "left", "no", "off", "on", "right", "stop", "up", "yes"]
SILENCE = "_silence_"
UNKNOWN = "_unknown_"
BACKGROUND_NOISE = "_background_noise_"


def _download_speechcommands(
    speechcommands_version: str,
    target_dir: Pathlike = ".",
    force_download: bool = False,
) -> Path:
    """
    Download and unzip Speech Commands dataset

    :param speechcommands_version: str, dataset version.
    :param target_dir: Pathlike, the path of the dir to store the dataset.
    :param force_download: bool, if True, download the archive even if it already exists.

    :return: the path to downloaded and extracted directory with data.
    """
    target_dir = Path(target_dir)
    corpus_dir = target_dir / f"SpeechCommands{speechcommands_version}"
    corpus_dir.mkdir(parents=True, exist_ok=True)

    dataset_parts = (
        f"speech_commands_v0.0{speechcommands_version}",
        f"speech_commands_test_set_v0.0{speechcommands_version}",
    )

    for part in tqdm(
        dataset_parts, desc=f"Downloading Speech Commands v0.0{speechcommands_version}"
    ):
        logging.info(f"Processing split: {part}")
        part_dir = corpus_dir / part
        completed_detector = part_dir / ".completed"
        if completed_detector.is_file():
            logging.info(f"Skipping {part} because {completed_detector} exists.")
            continue
        # Process the archive.
        tar_name = f"{part}.tar.gz"
        tar_path = corpus_dir / tar_name
        if force_download or not tar_path.is_file():
            urlretrieve_progress(
                f"http://download.tensorflow.org/data/{tar_name}",
                filename=tar_path,
                desc=f"Downloading {tar_name}",
            )
        # Remove partial unpacked files, if any, and unpack everything.
        shutil.rmtree(part_dir, ignore_errors=True)
        with tarfile.open(tar_path) as tar:
            tar.extractall(part_dir)
        completed_detector.touch()

    return corpus_dir


def download_speechcommands1(
    target_dir: Pathlike = ".",
    force_download: bool = False,
) -> Path:
    """
    Download and unzip the Speech Commands v0.01 data.
    :param target_dir: Pathlike, the path of the dir to store the dataset.
    :param force_download: bool, if True, download the archive even if it already exists.
    :return: the path to downloaded and extracted directory with data.
    """

    return _download_speechcommands(
        speechcommands_version="1",
        target_dir=target_dir,
        force_download=force_download,
    )


def download_speechcommands2(
    target_dir: Pathlike = ".",
    force_download: bool = False,
) -> Path:
    """
    Download and unzip the Speech Commands v0.02 data.
    :param target_dir: Pathlike, the path of the dir to store the dataset.
    :param force_download: bool, if True, download the archive even if it already exists.
    :return: the path to downloaded and extracted directory with data.
    """

    return _download_speechcommands(
        speechcommands_version="2",
        target_dir=target_dir,
        force_download=force_download,
    )


def _prepare_train_valid(
    speechcommands_version: str,
    corpus_dir: Pathlike,
) -> Tuple[RecordingSet, SupervisionSet]:
    """
    Returns the RecodingSet and SupervisionSet given a dataset part.
    :param speechcommands_version: str, dataset version.
    :param corpus_dir: Pathlike, the path of the data dir.
    :return: the RecodingSet and SupervisionSet for train and valid.
    """
    corpus_dir = Path(corpus_dir)
    part_path = corpus_dir / f"speech_commands_v0.0{speechcommands_version}"

    # Split dataset into train and valid
    train_paths = []
    for root in os.listdir(part_path):
        if root == "testing_list.txt":
            with open(part_path / root) as file_obj:
                train_test_paths = file_obj.read().strip().splitlines()
        elif root == "validation_list.txt":
            with open(part_path / root) as file_obj:
                valid_paths = file_obj.read().strip().splitlines()
        elif os.path.isdir(part_path / root):
            file_paths = os.listdir(part_path / root)
            train_paths += [
                os.path.join(root, file_path)
                for file_path in file_paths
                if file_path.endswith(".wav")
            ]

    # Original validation files don't include silence - we add them manually here
    valid_paths.append(os.path.join(BACKGROUND_NOISE, "running_tap.wav"))

    # The paths for the train set is just whichever paths that do not exist in either the test or validation splits.
    print(len(train_paths), len(valid_paths), len(train_test_paths))
    train_paths = set(train_paths) - set(valid_paths) - set(train_test_paths)
    print(len(train_paths), len(valid_paths))

    train_recordings = []
    train_supervisions = []
    for train_path in train_paths:
        audio_path = part_path / train_path
        audio_path = audio_path.resolve()
        train_path_splits = train_path.split("/")
        audio_file_name = train_path.replace("/", "_").replace(".wav", "")
        audio_file_name_splits = train_path_splits[1].split("_")

        if train_path_splits[0] == BACKGROUND_NOISE:
            speaker = None
            text = None
        else:
            speaker = audio_file_name_splits[0]
            text = train_path_splits[0].strip()
        if not audio_path.is_file():
            logging.warning(f"No such file: {audio_path}")
            continue
        train_recording = Recording.from_file(
            path=audio_path,
            recording_id=audio_file_name,
        )
        train_recordings.append(train_recording)
        train_segment = SupervisionSegment(
            id=audio_file_name,
            recording_id=audio_file_name,
            start=0.0,
            duration=train_recording.duration,
            channel=0,
            language="English",
            speaker=speaker,
            text=text,
        )
        train_supervisions.append(train_segment)
    train_recording_set = RecordingSet.from_recordings(train_recordings)
    train_supervision_set = SupervisionSet.from_segments(train_supervisions)

    yield train_recording_set, train_supervision_set

    valid_recordings = []
    valid_supervisions = []
    for valid_path in valid_paths:
        audio_path = part_path / valid_path
        audio_path = audio_path.resolve()
        valid_path_splits = valid_path.split("/")
        audio_file_name = valid_path.replace("/", "_").replace(".wav", "")
        audio_file_name_splits = valid_path_splits[1].split("_")

        if valid_path_splits[0] == BACKGROUND_NOISE:
            speaker = None
            text = None
        else:
            speaker = audio_file_name_splits[0]
            text = valid_path_splits[0].strip()

        if not audio_path.is_file():
            logging.warning(f"No such file: {audio_path}")
            continue
        valid_recording = Recording.from_file(
            path=audio_path,
            recording_id=audio_file_name,
        )
        valid_recordings.append(valid_recording)
        valid_segment = SupervisionSegment(
            id=audio_file_name,
            recording_id=audio_file_name,
            start=0.0,
            duration=valid_recording.duration,
            channel=0,
            language="English",
            speaker=speaker,
            text=text,
        )
        valid_supervisions.append(valid_segment)
    valid_recording_set = RecordingSet.from_recordings(valid_recordings)
    valid_supervision_set = SupervisionSet.from_segments(valid_supervisions)

    yield valid_recording_set, valid_supervision_set


def _prepare_test(
    speechcommands_version: str,
    corpus_dir: Pathlike,
) -> Tuple[RecordingSet, SupervisionSet]:
    """
    Returns the RecodingSet and SupervisionSet given a dataset part.
    :param speechcommands_version: str, dataset version.
    :param corpus_dir: Pathlike, the path of the data dir.
    :return: the RecodingSet and SupervisionSet for train and valid.
    """
    corpus_dir = Path(corpus_dir)
    part_path = corpus_dir / f"speech_commands_test_set_v0.0{speechcommands_version}"
    test_paths = []
    for root in os.listdir(part_path):
        if os.path.isdir(part_path / root):
            file_paths = os.listdir(part_path / root)
            test_paths += [
                os.path.join(root, file_path)
                for file_path in file_paths
                if file_path.endswith(".wav")
            ]

    test_paths = set(test_paths)

    test_recordings = []
    test_supervisions = []
    for test_path in test_paths:
        audio_path = part_path / test_path
        audio_path = audio_path.resolve()
        test_path_splits = test_path.split("/")
        audio_file_name = test_path.replace("/", "_").replace(".wav", "")
        audio_file_name_splits = test_path_splits[1].split("_")

        if test_path_splits[0] in WORDS:
            speaker = audio_file_name_splits[0]
            text = test_path_splits[0].strip()
        elif test_path_splits[0] == SILENCE:
            speaker = None
            text = None
        elif test_path_splits[0] == UNKNOWN:
            speaker = audio_file_name_splits[1]
            text = audio_file_name_splits[0].strip()

        if not audio_path.is_file():
            logging.warning(f"No such file: {audio_path}")
            continue

        test_recording = Recording.from_file(
            path=audio_path,
            recording_id=audio_file_name,
        )
        test_recordings.append(test_recording)
        test_segment = SupervisionSegment(
            id=audio_file_name,
            recording_id=audio_file_name,
            start=0.0,
            duration=test_recording.duration,
            channel=0,
            language="English",
            speaker=speaker,
            text=text,
        )
        test_supervisions.append(test_segment)
    test_recording_set = RecordingSet.from_recordings(test_recordings)
    test_supervision_set = SupervisionSet.from_segments(test_supervisions)

    return test_recording_set, test_supervision_set


def _prepare_speechcommands1(
    corpus_dir: Pathlike,
    output_dir: Optional[Pathlike] = None,
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Prepare manifests for the Speech Commands v0.01 corpus.
    :param corpus_dir: Path to the Speech Commands v0.01 dataset.
    :param output_dir: Pathlike, the path where to write the manifests.
    :return: a Dict whose key is the dataset part, and the value is Dicts with the keys 'recordings' and 'supervisions'.
    """
    speechcommands_version = "1"
    corpus_dir = Path(corpus_dir)
    assert corpus_dir.is_dir(), f"No such directory: {corpus_dir}"

    dataset_parts = ("speech_commands_v0.01", "speech_commands_test_set_v0.01")
    subsets = _SPLITS

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    manifests = defaultdict(dict)

    for part in tqdm(subsets, desc="Dataset parts"):
        logging.info(f"Processing Speech Commands v0.01 subset: {part}")
        if manifests_exist(
            part=part,
            output_dir=output_dir,
            prefix="speechcommands1",
            suffix="jsonl.gz",
        ):
            logging.info(
                f"Speech Commands v0.01 subset: {part} already prepared - skipping."
            )
            continue
        if part == "train":
            g = _prepare_train_valid(speechcommands_version, corpus_dir)
            recording_set, supervision_set = next(g)
        elif part == "valid":
            recording_set, supervision_set = next(g)
        elif part == "test":
            recording_set, supervision_set = _prepare_test(
                speechcommands_version, corpus_dir
            )

        if output_dir is not None:
            supervision_set.to_file(
                output_dir / f"speechcommands1_supervisions_{part}.jsonl.gz"
            )
            recording_set.to_file(
                output_dir / f"speechcommands1_recordings_{part}.jsonl.gz"
            )

        manifests[part] = {"recordings": recording_set, "supervisions": supervision_set}

    return manifests


def _prepare_speechcommands2(
    corpus_dir: Pathlike,
    output_dir: Optional[Pathlike] = None,
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Prepare manifests for the Speech Commands v0.02 corpus.
    :param corpus_dir: Path to the Speech Commands v0.02 dataset.
    :param output_dir: Pathlike, the path where to write the manifests.
    :return: a Dict whose key is the dataset part, and the value is Dicts with the keys 'recordings' and 'supervisions'.
    """
    speechcommands_version = "2"
    corpus_dir = Path(corpus_dir)
    assert corpus_dir.is_dir(), f"No such directory: {corpus_dir}"

    dataset_parts = ("speech_commands_v0.02", "speech_commands_test_set_v0.02")
    subsets = _SPLITS

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    manifests = defaultdict(dict)

    for part in tqdm(subsets, desc="Dataset parts"):
        logging.info(f"Processing Speech Commands v0.02 subset: {part}")
        if manifests_exist(
            part=part,
            output_dir=output_dir,
            prefix="speechcommands2",
            suffix="jsonl.gz",
        ):
            logging.info(
                f"Speech Commands v0.02 subset: {part} already prepared - skipping."
            )
            continue
        if part == "train":
            g = _prepare_train_valid(speechcommands_version, corpus_dir)
            recording_set, supervision_set = next(g)
        elif part == "valid":
            recording_set, supervision_set = next(g)
        elif part == "test":
            recording_set, supervision_set = _prepare_test(
                speechcommands_version, corpus_dir
            )

        if output_dir is not None:
            supervision_set.to_file(
                output_dir / f"speechcommands2_supervisions_{part}.jsonl.gz"
            )
            recording_set.to_file(
                output_dir / f"speechcommands2_recordings_{part}.jsonl.gz"
            )

        manifests[part] = {"recordings": recording_set, "supervisions": supervision_set}

    return manifests


def prepare_speechcommands(
    speechcommands1_root: Optional[Pathlike] = None,
    speechcommands2_root: Optional[Pathlike] = None,
    output_dir: Optional[Pathlike] = None,
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Returns the manifests which consist of the Recordings and Supervisions
    :param speechcommands1_root: Path to the Speech Commands v0.01 dataset.
    :param speechcommands2_root: Path to the Speech Commands v0.02 dataset.
    :param output_dir: Pathlike, the path where to write the manifests.
    :return: a Dict whose key is the dataset part, and the value is Dicts with the keys 'recordings' and 'supervisions'.
    """
    speechcommands1_root = Path(speechcommands1_root) if speechcommands1_root else None
    speechcommands2_root = Path(speechcommands2_root) if speechcommands2_root else None
    if not (speechcommands1_root or speechcommands2_root):
        raise ValueError(
            "Either Speech Commands v0.01 or Speech Commands v0.02 path must be provided."
        )

    output_dir = Path(output_dir) if output_dir is not None else None
    manifests = defaultdict(dict)
    if speechcommands1_root:
        logging.info("Preparing Speech Commands v0.01...")
        manifests.update(_prepare_speechcommands1(speechcommands1_root, output_dir))
    if speechcommands2_root:
        logging.info("Preparing Speech Commands v0.02...")
        manifests.update(_prepare_speechcommands2(speechcommands2_root, output_dir))

    return manifests
