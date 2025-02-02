
import yaml
import os

class ConfigLoader:
    """
    A class to load and manage YAML configuration files for model and training
    configuration
    """

    def __init__(self, config_path: str, environment_prefix: str = None):
        """
        Initialize the ConfigLoader.

        Args:
            config_path (str): Path to the YAML configuration file.
            environment_prefix (str): Optional prefix for environment variable overrides.
        """
        self.config_path = config_path
        self.environment_prefix = environment_prefix
        self.config = self._load_config()

    def _load_config(self) -> dict:
        """Load the YAML configuration file."""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with open(self.config_path, "r") as f:
            config = yaml.safe_load(f)

        if self.environment_prefix:
            config = self._apply_env_overrides(config)

        return config

    def _apply_env_overrides(self, config: dict) -> dict:
        """
        Apply environment variable overrides to the configuration.

        Args:
            config (dict): The base configuration.

        Returns:
            dict: The updated configuration with environment variable overrides.
        """
        for key, value in config.items():
            env_var = f"{self.environment_prefix}_{key}".upper()
            if isinstance(value, dict):
                config[key] = self._apply_env_overrides(value)
            elif env_var in os.environ:
                # Convert string env vars back to the correct type
                config[key] = self._cast_env_value(os.environ[env_var], value)

        return config

    @staticmethod
    def _cast_env_value(env_value: str, default_value):
        """
        Cast the environment variable string to the same type as the default value.

        Args:
            env_value (str): The environment variable value as a string.
            default_value: The default value to infer the type.

        Returns:
            The value cast to the appropriate type.
        """
        if isinstance(default_value, bool):
            return env_value.lower() in ("true", "1", "yes")
        elif isinstance(default_value, int):
            return int(env_value)
        elif isinstance(default_value, float):
            return float(env_value)
        return env_value  # Assume string for other types

    def get(self, key: str, default=None):
        """
        Get a value from the configuration.

        Args:
            key (str): The key to retrieve.
            default: The default value if the key does not exist.

        Returns:
            The value from the configuration or the default.
        """
        keys = key.split(".")
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

    def reload(self):
        """Reload the configuration from the YAML file."""
        self.config = self._load_config()