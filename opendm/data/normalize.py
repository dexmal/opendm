import json
import pathlib
from enum import Enum

import megfile
import numpy as np
import pydantic
from loguru import logger


class NormStats(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(
        arbitrary_types_allowed=True,
        json_encoders={np.ndarray: lambda value: value.tolist()},
    )

    mean: np.ndarray
    std: np.ndarray
    q01: np.ndarray | None = None  # 1st quantile
    q99: np.ndarray | None = None  # 99th quantile
    min: np.ndarray | None = None
    max: np.ndarray | None = None

    @pydantic.field_validator("mean", "std", "q01", "q99", "min", "max", mode="before")
    @classmethod
    def _coerce_ndarray(cls, value):
        if value is None:
            return None
        return np.asarray(value)


class RunningStats:
    """Compute running statistics of a batch of vectors."""

    def __init__(self):
        self._count = 0
        self._mean = None
        self._mean_of_squares = None
        self._min = None
        self._max = None
        self._histograms = None
        self._bin_edges = None
        self._num_quantile_bins = 5000  # for computing quantiles on the fly

    def update(self, batch: np.ndarray) -> None:
        """
        Update the running statistics with a batch of vectors.

        Args:
            vectors (np.ndarray): A 2D array where each row is a new vector.
        """
        if batch.ndim == 1:
            batch = batch.reshape(-1, 1)
        num_elements, vector_length = batch.shape
        if self._count == 0:
            self._mean = np.mean(batch, axis=0)
            self._mean_of_squares = np.mean(batch**2, axis=0)
            self._min = np.min(batch, axis=0)
            self._max = np.max(batch, axis=0)
            self._histograms = [
                np.zeros(self._num_quantile_bins) for _ in range(vector_length)
            ]
            self._bin_edges = [
                np.linspace(
                    self._min[i] - 1e-10,
                    self._max[i] + 1e-10,
                    self._num_quantile_bins + 1,
                )
                for i in range(vector_length)
            ]
        else:
            if vector_length != self._mean.size:
                raise ValueError(
                    "The length of new vectors does not match the initialized vector length."
                )
            new_max = np.max(batch, axis=0)
            new_min = np.min(batch, axis=0)
            max_changed = np.any(new_max > self._max)
            min_changed = np.any(new_min < self._min)
            self._max = np.maximum(self._max, new_max)
            self._min = np.minimum(self._min, new_min)

            if max_changed or min_changed:
                self._adjust_histograms()

        self._count += num_elements

        batch_mean = np.mean(batch, axis=0)
        batch_mean_of_squares = np.mean(batch**2, axis=0)

        # Update running mean and mean of squares.
        self._mean += (batch_mean - self._mean) * (num_elements / self._count)
        self._mean_of_squares += (batch_mean_of_squares - self._mean_of_squares) * (
            num_elements / self._count
        )

        self._update_histograms(batch)

    def get_statistics(self) -> NormStats:
        """
        Compute and return the statistics of the vectors processed so far.

        Returns:
            dict: A dictionary containing the computed statistics.
        """
        if self._count < 2:
            raise ValueError("Cannot compute statistics for less than 2 vectors.")

        variance = self._mean_of_squares - self._mean**2
        stddev = np.sqrt(np.maximum(0, variance))
        q01, q99 = self._compute_quantiles([0.01, 0.99])
        return NormStats(
            mean=self._mean, std=stddev, q01=q01, q99=q99, min=self._min, max=self._max
        )

    def _adjust_histograms(self):
        """Adjust histograms when min or max changes."""
        for i in range(len(self._histograms)):
            old_edges = self._bin_edges[i]
            new_edges = np.linspace(
                self._min[i], self._max[i], self._num_quantile_bins + 1
            )

            # Redistribute the existing histogram counts to the new bins
            new_hist, _ = np.histogram(
                old_edges[:-1], bins=new_edges, weights=self._histograms[i]
            )

            self._histograms[i] = new_hist
            self._bin_edges[i] = new_edges

    def _update_histograms(self, batch: np.ndarray) -> None:
        """Update histograms with new vectors."""
        for i in range(batch.shape[1]):
            hist, _ = np.histogram(batch[:, i], bins=self._bin_edges[i])
            self._histograms[i] += hist

    def _compute_quantiles(self, quantiles):
        """Compute quantiles based on histograms."""
        results = []
        for q in quantiles:
            target_count = q * self._count
            q_values = []
            for hist, edges in zip(self._histograms, self._bin_edges, strict=True):
                cumsum = np.cumsum(hist)
                idx = np.searchsorted(cumsum, target_count)
                q_values.append(edges[idx])
            results.append(np.array(q_values))
        return results


class _NormStatsDict(pydantic.BaseModel):
    norm_stats: dict[str, NormStats]


class _MultiRobotNormStatsDict(pydantic.BaseModel):
    default_robot_type: str
    norm_stats: dict[str, NormStats]
    norm_stats_by_robot: dict[str, dict[str, NormStats]]


def _robot_type_value(robot_type: str | Enum | None) -> str | None:
    if robot_type is None:
        return None
    if isinstance(robot_type, Enum):
        robot_type = robot_type.value
    return str(robot_type)


def _normalize_profile_keys(
    profiles: dict[str, dict[str, NormStats]],
) -> dict[str, dict[str, NormStats]]:
    normalized = {}
    for robot_type, profile in profiles.items():
        normalized_robot_type = _robot_type_value(robot_type)
        if normalized_robot_type in normalized:
            raise ValueError(
                f"Duplicate norm stats profile for robot_type {normalized_robot_type!r}"
            )
        normalized[normalized_robot_type] = dict(profile)
    return normalized


def _norm_stats_equal(
    left: dict[str, NormStats],
    right: dict[str, NormStats],
) -> bool:
    if left.keys() != right.keys():
        return False
    for key in left:
        for field_name in NormStats.model_fields:
            left_value = getattr(left[key], field_name)
            right_value = getattr(right[key], field_name)
            if left_value is None or right_value is None:
                if left_value is not None or right_value is not None:
                    return False
                continue
            if not np.array_equal(left_value, right_value):
                return False
    return True


def _validate_robot_profile(
    robot_type: str,
    profile: dict[str, NormStats],
) -> None:
    if "action" not in profile:
        raise ValueError(f"Norm stats for robot_type {robot_type!r} are missing action")
    for key, stats in profile.items():
        expected_shape = stats.mean.shape
        for field_name in ("std", "q01", "q99", "min", "max"):
            value = getattr(stats, field_name)
            if value is not None and value.shape != expected_shape:
                raise ValueError(
                    f"Norm stats for robot_type {robot_type!r}, field {key!r} "
                    f"have inconsistent {field_name} shape {value.shape}; "
                    f"expected {expected_shape}"
                )


class NormStatsFile:
    """Normalization statistics loaded from a legacy or multi-robot file."""

    def __init__(
        self,
        norm_stats: dict[str, NormStats],
        default_robot_type: str | None = None,
        norm_stats_by_robot: dict[
            str,
            dict[str, NormStats],
        ]
        | None = None,
    ):
        self.norm_stats = dict(norm_stats)
        self.default_robot_type = default_robot_type
        self.norm_stats_by_robot = (
            {
                robot_type: dict(stats)
                for robot_type, stats in norm_stats_by_robot.items()
            }
            if norm_stats_by_robot is not None
            else None
        )
        self._legacy_warning_emitted = False

    @property
    def is_multi_robot(self) -> bool:
        return self.norm_stats_by_robot is not None

    def select(
        self,
        robot_type: str | Enum | None = None,
    ) -> dict[str, NormStats]:
        """Select statistics for a robot while preserving legacy behavior."""
        robot_type_value = _robot_type_value(robot_type)
        if self.norm_stats_by_robot is None:
            if robot_type_value is not None and not self._legacy_warning_emitted:
                logger.warning(
                    "The norm stats file does not identify a robot type; "
                    "using its legacy global statistics for {}.",
                    robot_type_value,
                )
                self._legacy_warning_emitted = True
            return self.norm_stats

        selected_robot_type = robot_type_value or self.default_robot_type
        if selected_robot_type not in self.norm_stats_by_robot:
            supported = sorted(self.norm_stats_by_robot)
            raise ValueError(
                f"Norm stats are unavailable for robot_type "
                f"{selected_robot_type!r}. Available: {supported}"
            )
        return self.norm_stats_by_robot[selected_robot_type]


def deserialize_norm_stats_file(data: str) -> NormStatsFile:
    """Deserialize either the historical or multi-robot file structure."""
    loaded = json.loads(data)
    if "norm_stats_by_robot" not in loaded:
        legacy = _NormStatsDict(**loaded)
        return NormStatsFile(norm_stats=legacy.norm_stats)

    parsed = _MultiRobotNormStatsDict(**loaded)
    profiles = _normalize_profile_keys(parsed.norm_stats_by_robot)
    if not profiles:
        raise ValueError("norm_stats_by_robot must contain at least one robot")
    default_robot_type = _robot_type_value(parsed.default_robot_type)
    if default_robot_type not in profiles:
        supported = sorted(profiles)
        raise ValueError(
            f"default_robot_type {default_robot_type!r} is unavailable. "
            f"Available: {supported}"
        )
    for robot_type, profile in profiles.items():
        _validate_robot_profile(robot_type, profile)
    default_stats = profiles[default_robot_type]
    if not _norm_stats_equal(parsed.norm_stats, default_stats):
        raise ValueError(
            "norm_stats must equal the profile selected by default_robot_type"
        )
    return NormStatsFile(
        norm_stats=parsed.norm_stats,
        default_robot_type=default_robot_type,
        norm_stats_by_robot=profiles,
    )


def serialize_norm_stats_file(
    norm_stats_by_robot: dict[
        str | Enum,
        dict[str, NormStats],
    ],
    default_robot_type: str | Enum,
) -> str:
    """Serialize multi-robot stats with a legacy-compatible default profile."""
    normalized_profiles = {}
    for robot_type, profile in norm_stats_by_robot.items():
        normalized_robot_type = _robot_type_value(robot_type)
        if normalized_robot_type is None:
            raise ValueError("robot_type cannot be None in a multi-robot norm file")
        if normalized_robot_type in normalized_profiles:
            raise ValueError(
                f"Duplicate norm stats profile for robot_type {normalized_robot_type!r}"
            )
        normalized_profiles[normalized_robot_type] = profile

    default_robot_type_value = _robot_type_value(default_robot_type)
    if default_robot_type_value not in normalized_profiles:
        supported = sorted(normalized_profiles)
        raise ValueError(
            f"default_robot_type {default_robot_type_value!r} is unavailable. "
            f"Available: {supported}"
        )
    for robot_type, profile in normalized_profiles.items():
        _validate_robot_profile(robot_type, profile)
    model = _MultiRobotNormStatsDict(
        default_robot_type=default_robot_type_value,
        norm_stats=normalized_profiles[default_robot_type_value],
        norm_stats_by_robot=normalized_profiles,
    )
    return model.model_dump_json(indent=2, exclude_none=True)


def load_norm_stats_file(path: pathlib.Path | str) -> NormStatsFile:
    """Load a complete normalization statistics file from any smart path."""
    with megfile.smart_open(str(path), "r") as file:
        return deserialize_norm_stats_file(file.read())


def serialize_json(
    norm_stats: dict[str, NormStats],
    *,
    exclude_none: bool = False,
) -> str:
    """Serialize the running statistics to a JSON string."""
    return _NormStatsDict(norm_stats=norm_stats).model_dump_json(
        indent=2,
        exclude_none=exclude_none,
    )


def deserialize_json(data: str) -> dict[str, NormStats]:
    """Deserialize the running statistics from a JSON string."""
    return _NormStatsDict(**json.loads(data)).norm_stats


def load(directory: pathlib.Path | str) -> dict[str, NormStats]:
    """Load the normalization stats from a directory."""
    path = pathlib.Path(directory) / "norm_stats.json"
    if not path.exists():
        raise FileNotFoundError(f"Norm stats file not found at: {path}")
    return deserialize_json(path.read_text())
