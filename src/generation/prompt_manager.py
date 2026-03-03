"""
Versioned YAML prompt management.

Prompts are stored in config/prompts/{version}/{name}.yaml.
The active version is read from config/prompts/active.yaml.
Loaded prompts are cached in memory for the process lifetime.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, Optional

import yaml

logger = logging.getLogger(__name__)


class PromptManager:
    """
    Loads and caches versioned prompt configurations from YAML files.

    Directory layout expected:
        {config_dir}/
            active.yaml              ← {"active_version": "v1"}
            v1/
                rag_answer.yaml
                image_description.yaml
            v2/
                rag_answer.yaml      ← future version
    """

    def __init__(self, config_dir: str = "config/prompts") -> None:
        self.config_dir = config_dir
        self._cache: Dict[str, Dict] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_active_version(self) -> str:
        """
        Read the active prompt version from active.yaml.

        Returns:
            Version string, e.g. "v1".
        """
        active_path = os.path.join(self.config_dir, "active.yaml")
        with open(active_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        version = data.get("active_version", "v1")
        # Allow environment variable override
        env_override = os.environ.get("ACTIVE_PROMPT_VERSION")
        if env_override:
            version = env_override
        return version

    def load_prompt(
        self, prompt_name: str, version: Optional[str] = None
    ) -> Dict:
        """
        Load a prompt configuration by name.

        Args:
            prompt_name: Base name without extension, e.g. "rag_answer".
            version: Explicit version string. If None, uses the active version.

        Returns:
            Parsed YAML dict containing at minimum:
            - system_prompt (str)
            - user_template (str, optional)
            - model (str, optional)
            - temperature (float, optional)
            - max_tokens (int, optional)
        """
        if version is None:
            version = self.get_active_version()

        cache_key = f"{version}/{prompt_name}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        path = os.path.join(self.config_dir, version, f"{prompt_name}.yaml")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Prompt file not found: {path}. "
                f"Check that '{prompt_name}' exists in version '{version}'."
            )

        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        self._cache[cache_key] = config
        logger.debug("Loaded prompt '%s' (version=%s).", prompt_name, version)
        return config

    def render_template(self, template: str, **kwargs) -> str:
        """
        Fill a prompt template string using str.format_map.

        Args:
            template: Template string with {placeholder} markers.
            **kwargs: Values to substitute.

        Returns:
            Rendered string.
        """
        return template.format_map(kwargs)

    def list_available_versions(self) -> list:
        """Return all version directories found under config_dir."""
        return [
            name
            for name in os.listdir(self.config_dir)
            if os.path.isdir(os.path.join(self.config_dir, name))
        ]
