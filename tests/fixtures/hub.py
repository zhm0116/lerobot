from pathlib import Path

import datasets
import pytest
from huggingface_hub.utils import filter_repo_objects

from lerobot.common.datasets.utils import EPISODES_PATH, INFO_PATH, STATS_PATH, TASKS_PATH
from tests.fixtures.constants import LEROBOT_TEST_DIR


@pytest.fixture(scope="session")
def mock_snapshot_download_factory(
    info_factory,
    info_path,
    stats_factory,
    stats_path,
    tasks_factory,
    tasks_path,
    episodes_factory,
    episode_path,
    single_episode_parquet_path,
    hf_dataset_factory,
):
    """
    This factory allows to patch snapshot_download such that when called, it will create expected files rather
    than making calls to the hub api. Its design allows to pass explicitly files which you want to be created.
    """

    def _mock_snapshot_download_func(
        info: dict | None = None,
        stats: dict | None = None,
        tasks: list[dict] | None = None,
        episodes: list[dict] | None = None,
        hf_dataset: datasets.Dataset | None = None,
    ):
        if not info:
            info = info_factory()
        if not stats:
            stats = stats_factory(features=info["features"])
        if not tasks:
            tasks = tasks_factory(total_tasks=info["total_tasks"])
        if not episodes:
            episodes = episodes_factory(
                total_episodes=info["total_episodes"], total_frames=info["total_frames"], tasks=tasks
            )
        if not hf_dataset:
            hf_dataset = hf_dataset_factory(tasks=tasks, episodes=episodes, fps=info["fps"])

        def _extract_episode_index_from_path(fpath: str) -> int:
            path = Path(fpath)
            if path.suffix == ".parquet" and path.stem.startswith("episode_"):
                episode_index = int(path.stem[len("episode_") :])  # 'episode_000000' -> 0
                return episode_index
            else:
                return None

        def _mock_snapshot_download(
            repo_id: str,
            local_dir: str | Path | None = None,
            allow_patterns: str | list[str] | None = None,
            ignore_patterns: str | list[str] | None = None,
            *args,
            **kwargs,
        ) -> str:
            if not local_dir:
                local_dir = LEROBOT_TEST_DIR

            # List all possible files
            all_files = []
            meta_files = [INFO_PATH, STATS_PATH, TASKS_PATH, EPISODES_PATH]
            all_files.extend(meta_files)

            data_files = []
            for episode_dict in episodes:
                ep_idx = episode_dict["episode_index"]
                ep_chunk = ep_idx // info["chunks_size"]
                data_path = info["data_path"].format(episode_chunk=ep_chunk, episode_index=ep_idx)
                data_files.append(data_path)
            all_files.extend(data_files)

            allowed_files = filter_repo_objects(
                all_files, allow_patterns=allow_patterns, ignore_patterns=ignore_patterns
            )

            # Create allowed files
            for rel_path in allowed_files:
                if rel_path.startswith("data/"):
                    episode_index = _extract_episode_index_from_path(rel_path)
                    if episode_index is not None:
                        _ = single_episode_parquet_path(local_dir, episode_index, hf_dataset, info)
                if rel_path == INFO_PATH:
                    _ = info_path(local_dir, info)
                elif rel_path == STATS_PATH:
                    _ = stats_path(local_dir, stats)
                elif rel_path == TASKS_PATH:
                    _ = tasks_path(local_dir, tasks)
                elif rel_path == EPISODES_PATH:
                    _ = episode_path(local_dir, episodes)
                else:
                    pass
            return str(local_dir)

        return _mock_snapshot_download

    return _mock_snapshot_download_func
