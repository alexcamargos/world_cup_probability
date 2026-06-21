"""Download external datasets into the local raw landing area."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from uuid import uuid4

try:
    from .settings import (
        DEFAULT_INGESTION_USER_AGENT,
        DEFAULT_RAW_DIR,
        DEFAULT_TIMEOUT_SECONDS,
    )
except ImportError:  # pragma: no cover - supports direct script execution.
    from settings import (
        DEFAULT_INGESTION_USER_AGENT,
        DEFAULT_RAW_DIR,
        DEFAULT_TIMEOUT_SECONDS,
    )

LOGGER = logging.getLogger(__name__)

DEFAULT_USER_AGENT = DEFAULT_INGESTION_USER_AGENT
CHUNK_SIZE_BYTES = 1024 * 1024
KAGGLE_DATASETS: tuple[str, ...] = (
    "abecklas/fifa-world-cup",
    "martj42/international-football-results-from-1872-to-2017",
)

DownloadStatus = Literal["downloaded", "skipped"]


class IngestionError(Exception):
    """Base exception for ingestion failures."""


class SourceValidationError(IngestionError):
    """Raised when a source declaration is invalid."""


class DownloadError(IngestionError):
    """Raised when a source cannot be downloaded."""


class ChecksumMismatchError(IngestionError):
    """Raised when a downloaded file does not match its expected checksum."""


class KaggleCredentialsError(IngestionError):
    """Raised when Kaggle API credentials are missing or invalid."""


class KaggleDatasetClient(Protocol):
    """Subset of the official Kaggle API used by this module."""

    def authenticate(self) -> None:
        """Authenticate the client."""

    def dataset_download_files(
        self,
        dataset: str,
        path: str | None = None,
        force: bool = False,
        quiet: bool = True,
        unzip: bool = False,
        licenses: Sequence[str] = (),
    ) -> None:
        """Download all files for a Kaggle dataset."""


@dataclass(frozen=True, slots=True)
class DataSource:
    """External dataset declaration."""

    name: str
    url: str
    filename: str | None = None
    expected_sha256: str | None = None
    headers: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        """Validate static source fields."""
        if not self.name.strip():
            raise SourceValidationError("Source name cannot be empty.")

        parsed_url = urlparse(self.url)
        if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
            raise SourceValidationError(
                f"Source '{self.name}' must use an absolute HTTP(S) URL.",
            )

        if self.filename is not None and not self.filename.strip():
            raise SourceValidationError(
                f"Source '{self.name}' filename cannot be empty.",
            )

        if self.expected_sha256 is not None:
            checksum = self.expected_sha256.strip().lower()
            if len(checksum) != 64 or not all(
                character in "0123456789abcdef" for character in checksum
            ):
                raise SourceValidationError(
                    f"Source '{self.name}' has an invalid SHA-256 checksum.",
                )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> DataSource:
        """Build a data source from a JSON-compatible mapping."""
        headers = _read_optional_str_mapping(payload, "headers")

        return cls(
            name=_read_required_str(payload, "name"),
            url=_read_required_str(payload, "url"),
            filename=_read_optional_str(payload, "filename"),
            expected_sha256=_read_optional_str(payload, "sha256"),
            headers=headers,
        )

    @property
    def target_filename(self) -> str:
        """Return the declared file name or derive one from the URL."""
        if self.filename is not None:
            return self.filename

        url_path_name = Path(urlparse(self.url).path).name
        if url_path_name:
            return url_path_name

        return f"{self.name}.bin"


@dataclass(frozen=True, slots=True)
class DownloadResult:
    """Metadata for one completed ingestion operation."""

    source_name: str
    url: str
    path: Path
    bytes_written: int
    sha256: str
    downloaded_at: datetime
    status: DownloadStatus


def load_manifest(path: Path | str) -> list[DataSource]:
    """Load source declarations from a JSON manifest.

    The manifest can be either a list of source objects or an object with a
    top-level ``sources`` list.
    """
    manifest_path = Path(path)

    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SourceValidationError(f"Manifest not found: {manifest_path}") from exc
    except json.JSONDecodeError as exc:
        raise SourceValidationError(
            f"Manifest is not valid JSON: {manifest_path}",
        ) from exc

    raw_sources = _extract_manifest_sources(payload, manifest_path)
    return [DataSource.from_mapping(source) for source in raw_sources]


def ingest_sources(
    sources: Iterable[DataSource],
    raw_dir: Path | str = DEFAULT_RAW_DIR,
    *,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    overwrite: bool = True,
) -> list[DownloadResult]:
    """Download all sources into the raw landing directory."""
    landing_dir = Path(raw_dir)
    landing_dir.mkdir(parents=True, exist_ok=True)

    results: list[DownloadResult] = []
    for source in sources:
        results.append(
            download_source(
                source,
                landing_dir,
                timeout_seconds=timeout_seconds,
                overwrite=overwrite,
            ),
        )

    return results


def download_source(
    source: DataSource,
    raw_dir: Path | str = DEFAULT_RAW_DIR,
    *,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    overwrite: bool = True,
) -> DownloadResult:
    """Download one source into the raw landing directory."""
    landing_dir = Path(raw_dir)
    landing_dir.mkdir(parents=True, exist_ok=True)

    target_path = _resolve_target_path(landing_dir, source.target_filename)

    if target_path.exists() and not overwrite:
        bytes_written, checksum = _fingerprint_file(target_path)
        return DownloadResult(
            source_name=source.name,
            url=source.url,
            path=target_path,
            bytes_written=bytes_written,
            sha256=checksum,
            downloaded_at=datetime.now(UTC),
            status="skipped",
        )

    target_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = target_path.with_name(f".{target_path.name}.{uuid4().hex}.tmp")
    request = _build_request(source)

    LOGGER.info("Downloading %s from %s", source.name, source.url)

    try:
        bytes_written, checksum = _stream_download(
            request,
            temporary_path,
            timeout_seconds,
        )
        _validate_checksum(source, checksum)
        temporary_path.replace(target_path)
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        temporary_path.unlink(missing_ok=True)
        raise DownloadError(f"Failed to download source '{source.name}'.") from exc
    except ChecksumMismatchError:
        temporary_path.unlink(missing_ok=True)
        raise

    downloaded_at = datetime.now(UTC)
    _write_metadata(
        target_path,
        DownloadResult(
            source_name=source.name,
            url=source.url,
            path=target_path,
            bytes_written=bytes_written,
            sha256=checksum,
            downloaded_at=downloaded_at,
            status="downloaded",
        ),
    )

    return DownloadResult(
        source_name=source.name,
        url=source.url,
        path=target_path,
        bytes_written=bytes_written,
        sha256=checksum,
        downloaded_at=downloaded_at,
        status="downloaded",
    )


def ingest_kaggle_datasets(download_path: str) -> list[Path]:
    """Download and extract configured Kaggle datasets.

    Kaggle credentials must be configured through ``~/.kaggle/kaggle.json`` or
    the ``KAGGLE_USERNAME`` and ``KAGGLE_KEY`` environment variables.
    """
    destination_root = Path(download_path)
    destination_root.mkdir(parents=True, exist_ok=True)

    try:
        api = _create_kaggle_api_client()
    except KaggleCredentialsError:
        raise

    extracted_paths: list[Path] = []
    for dataset in KAGGLE_DATASETS:
        dataset_path = destination_root / _kaggle_dataset_directory_name(dataset)
        dataset_path.mkdir(parents=True, exist_ok=True)

        try:
            LOGGER.info("Downloading Kaggle dataset '%s' to %s", dataset, dataset_path)
            api.dataset_download_files(
                dataset,
                path=str(dataset_path),
                force=True,
                quiet=False,
                unzip=True,
            )
        except Exception as exc:
            raise DownloadError(
                f"Failed to download Kaggle dataset '{dataset}'.",
            ) from exc

        extracted_paths.append(dataset_path)

    return extracted_paths


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for manual ingestion runs."""
    parser = argparse.ArgumentParser(
        description="Download external datasets into data/raw.",
    )
    parser.add_argument(
        "--manifest",
        required=True,
        type=Path,
        help="Path to a JSON manifest with external data sources.",
    )
    parser.add_argument(
        "--raw-dir",
        default=DEFAULT_RAW_DIR,
        type=Path,
        help=f"Landing directory for raw files. Defaults to {DEFAULT_RAW_DIR}.",
    )
    parser.add_argument(
        "--timeout",
        default=DEFAULT_TIMEOUT_SECONDS,
        type=float,
        help="HTTP timeout in seconds.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Keep existing files instead of downloading them again.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run ingestion from the command line."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args(argv)

    sources = load_manifest(args.manifest)
    results = ingest_sources(
        sources,
        args.raw_dir,
        timeout_seconds=args.timeout,
        overwrite=not args.skip_existing,
    )

    for result in results:
        LOGGER.info(
            "%s %s -> %s (%d bytes, sha256=%s)",
            result.status,
            result.source_name,
            result.path,
            result.bytes_written,
            result.sha256,
        )

    return 0


def _build_request(source: DataSource) -> Request:
    headers = {
        "Accept": "*/*",
        "User-Agent": DEFAULT_USER_AGENT,
    }
    if source.headers is not None:
        headers.update(source.headers)

    return Request(source.url, headers=headers)


def _create_kaggle_api_client() -> KaggleDatasetClient:
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except SystemExit as exc:
        _log_kaggle_credentials_error()
        raise KaggleCredentialsError("Kaggle API credentials are not configured.") from exc
    except ModuleNotFoundError as exc:
        LOGGER.error(
            "The official Kaggle package is not installed. Install it with: uv add kaggle",
        )
        raise IngestionError("Kaggle package is not installed.") from exc

    api = KaggleApi()

    try:
        api.authenticate()
    except SystemExit as exc:
        _log_kaggle_credentials_error()
        raise KaggleCredentialsError("Kaggle API credentials are not configured.") from exc
    except (OSError, ValueError) as exc:
        _log_kaggle_credentials_error()
        raise KaggleCredentialsError("Kaggle API credentials are not configured.") from exc

    return api


def _log_kaggle_credentials_error() -> None:
    kaggle_config_path = Path(os.environ.get("KAGGLE_CONFIG_DIR", "~/.kaggle"))
    kaggle_json_path = kaggle_config_path.expanduser() / "kaggle.json"

    LOGGER.error(
        "Kaggle API credentials are not configured. Create an API token in "
        "Kaggle under Account > Settings > API > Create New Token, then place "
        "the downloaded kaggle.json file at %s. In Docker, mount that file or "
        "set KAGGLE_CONFIG_DIR to the directory that contains kaggle.json. "
        "Alternatively, set KAGGLE_USERNAME and KAGGLE_KEY as environment "
        "variables.",
        kaggle_json_path,
    )


def _kaggle_dataset_directory_name(dataset: str) -> str:
    return dataset.replace("/", "__")


def _stream_download(
    request: Request,
    temporary_path: Path,
    timeout_seconds: float,
) -> tuple[int, str]:
    hasher = hashlib.sha256()
    bytes_written = 0

    with urlopen(request, timeout=timeout_seconds) as response:
        with temporary_path.open("wb") as file:
            while chunk := response.read(CHUNK_SIZE_BYTES):
                bytes_written += len(chunk)
                hasher.update(chunk)
                file.write(chunk)

    return bytes_written, hasher.hexdigest()


def _validate_checksum(source: DataSource, actual_sha256: str) -> None:
    if source.expected_sha256 is None:
        return

    expected_sha256 = source.expected_sha256.strip().lower()
    if actual_sha256 != expected_sha256:
        raise ChecksumMismatchError(
            f"Source '{source.name}' checksum mismatch: expected "
            f"{expected_sha256}, got {actual_sha256}.",
        )


def _fingerprint_file(path: Path) -> tuple[int, str]:
    hasher = hashlib.sha256()
    bytes_read = 0

    with path.open("rb") as file:
        while chunk := file.read(CHUNK_SIZE_BYTES):
            bytes_read += len(chunk)
            hasher.update(chunk)

    return bytes_read, hasher.hexdigest()


def _write_metadata(path: Path, result: DownloadResult) -> None:
    metadata_path = path.with_suffix(f"{path.suffix}.metadata.json")
    metadata = {
        "source_name": result.source_name,
        "url": result.url,
        "path": str(result.path),
        "bytes_written": result.bytes_written,
        "sha256": result.sha256,
        "downloaded_at": result.downloaded_at.isoformat(),
        "status": result.status,
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _resolve_target_path(raw_dir: Path, filename: str) -> Path:
    base_dir = raw_dir.resolve()
    target_path = (base_dir / filename).resolve()

    if target_path == base_dir or base_dir not in target_path.parents:
        raise SourceValidationError(
            f"Target filename escapes the raw directory: {filename}",
        )

    return target_path


def _extract_manifest_sources(
    payload: object,
    manifest_path: Path,
) -> list[Mapping[str, object]]:
    if isinstance(payload, list):
        raw_sources = payload
    elif isinstance(payload, dict):
        sources = payload.get("sources")
        if not isinstance(sources, list):
            raise SourceValidationError(
                f"Manifest must contain a 'sources' list: {manifest_path}",
            )
        raw_sources = sources
    else:
        raise SourceValidationError(
            f"Manifest must be a list or object: {manifest_path}",
        )

    source_mappings: list[Mapping[str, object]] = []
    for index, source in enumerate(raw_sources):
        if not isinstance(source, dict):
            raise SourceValidationError(
                f"Manifest source at index {index} must be an object.",
            )
        source_mappings.append(source)

    return source_mappings


def _read_required_str(payload: Mapping[str, object], field_name: str) -> str:
    value = payload.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise SourceValidationError(f"Field '{field_name}' must be a non-empty string.")

    return value


def _read_optional_str(
    payload: Mapping[str, object],
    field_name: str,
) -> str | None:
    value = payload.get(field_name)
    if value is None:
        return None

    if not isinstance(value, str):
        raise SourceValidationError(f"Field '{field_name}' must be a string.")

    return value


def _read_optional_str_mapping(
    payload: Mapping[str, object],
    field_name: str,
) -> Mapping[str, str] | None:
    value = payload.get(field_name)
    if value is None:
        return None

    if not isinstance(value, dict):
        raise SourceValidationError(f"Field '{field_name}' must be an object.")

    headers: dict[str, str] = {}
    for header_name, header_value in value.items():
        if not isinstance(header_name, str) or not isinstance(header_value, str):
            raise SourceValidationError(
                f"Field '{field_name}' must only contain string pairs.",
            )
        headers[header_name] = header_value

    return headers


if __name__ == "__main__":
    raise SystemExit(main())
