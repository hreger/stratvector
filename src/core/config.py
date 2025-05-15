"""
Configuration management for the trading system.
"""

from typing import Dict, List, Optional, Union
from pathlib import Path
import os
import toml
from pydantic import BaseModel, Field, validator
from enum import Enum

class RiskProfile(str, Enum):
    """Risk profile types."""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"

class IBConfig(BaseModel):
    """Interactive Brokers connection configuration."""
    host: str = Field(default="127.0.0.1")
    port: int = Field(default=7496)
    client_id: int = Field(default=1)
    timeout: int = Field(default=20)
    readonly: bool = Field(default=False)
    
    @validator('port')
    def validate_port(cls, v):
        """Validate port number."""
        if not 1024 <= v <= 65535:
            raise ValueError("Port must be between 1024 and 65535")
        return v

class StrategyParameters(BaseModel):
    """Strategy parameters configuration."""
    lookback_period: int = Field(gt=0)
    holding_period: int = Field(gt=0)
    rebalance_frequency: str
    position_size: float = Field(gt=0, le=1)
    entry_threshold: float
    exit_threshold: float
    max_position_size: float = Field(gt=0, le=1)
    
    @validator('rebalance_frequency')
    def validate_frequency(cls, v):
        """Validate rebalance frequency format."""
        valid_frequencies = ['1m', '5m', '15m', '30m', '1h', '1d', '1w']
        if v not in valid_frequencies:
            raise ValueError(f"Invalid frequency. Must be one of {valid_frequencies}")
        return v

class UniverseConfig(BaseModel):
    """Trading universe configuration."""
    symbols: List[str]
    exchange: str
    currency: str
    
    @validator('symbols')
    def validate_symbols(cls, v):
        """Validate symbol list."""
        if not v:
            raise ValueError("Symbols list cannot be empty")
        return v

class RiskConfig(BaseModel):
    """Risk management configuration."""
    max_position_size: float = Field(gt=0, le=1)
    stop_loss: float = Field(gt=0, le=1)
    take_profit: float = Field(gt=0)
    max_drawdown: float = Field(gt=0, le=1)
    
    @validator('take_profit')
    def validate_take_profit(cls, v, values):
        """Validate take profit relative to stop loss."""
        if 'stop_loss' in values and v <= values['stop_loss']:
            raise ValueError("Take profit must be greater than stop loss")
        return v

class ExecutionConfig(BaseModel):
    """Execution configuration."""
    slippage: float = Field(ge=0, le=0.01)
    commission: float = Field(ge=0, le=0.01)
    
    @validator('slippage', 'commission')
    def validate_fees(cls, v):
        """Validate fee percentages."""
        if v > 0.01:  # 1%
            raise ValueError("Fees cannot exceed 1%")
        return v

class StrategyConfig(BaseModel):
    """Complete strategy configuration."""
    parameters: StrategyParameters
    universe: UniverseConfig
    risk: RiskConfig
    execution: ExecutionConfig

class ConfigManager:
    """
    Configuration manager for the trading system.
    
    Parameters
    ----------
    config_dir : Union[str, Path]
        Directory containing configuration files
    env_prefix : str, default="STRATVECTOR"
        Prefix for environment variables
        
    Attributes
    ----------
    config_dir : Path
        Path to configuration directory
    env_prefix : str
        Environment variable prefix
    """
    
    def __init__(
        self,
        config_dir: Union[str, Path],
        env_prefix: str = "STRATVECTOR"
    ) -> None:
        self.config_dir = Path(config_dir)
        self.env_prefix = env_prefix
        
    def _load_toml(self, file_path: Path) -> Dict:
        """Load and parse TOML file."""
        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
            
        with open(file_path, 'r') as f:
            return toml.load(f)
            
    def _get_env_value(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Get value from environment variable."""
        env_key = f"{self.env_prefix}_{key}"
        return os.getenv(env_key, default)
        
    def load_strategy_config(self, strategy_name: str) -> StrategyConfig:
        """
        Load strategy configuration.
        
        Parameters
        ----------
        strategy_name : str
            Name of the strategy configuration file (without extension)
            
        Returns
        -------
        StrategyConfig
            Validated strategy configuration
        """
        config_path = self.config_dir / "strategies" / f"{strategy_name}.toml"
        config_data = self._load_toml(config_path)
        return StrategyConfig(**config_data)
        
    def load_risk_profile(self, profile: RiskProfile) -> RiskConfig:
        """
        Load risk profile configuration.
        
        Parameters
        ----------
        profile : RiskProfile
            Risk profile type
            
        Returns
        -------
        RiskConfig
            Validated risk configuration
        """
        config_path = self.config_dir / "risk_profiles" / f"{profile.value}.toml"
        config_data = self._load_toml(config_path)
        return RiskConfig(**config_data)
        
    def get_ib_config(self) -> IBConfig:
        """
        Get Interactive Brokers configuration with environment overrides.
        
        Returns
        -------
        IBConfig
            Validated IB configuration
        """
        # Load base configuration
        config_path = self.config_dir / "ib_config.toml"
        config_data = self._load_toml(config_path)
        
        # Apply environment overrides
        config_data['host'] = self._get_env_value('IB_HOST', config_data.get('host'))
        config_data['port'] = int(self._get_env_value('IB_PORT', str(config_data.get('port', 7496))))
        config_data['client_id'] = int(self._get_env_value('IB_CLIENT_ID', str(config_data.get('client_id', 1))))
        
        return IBConfig(**config_data)
        
    def validate_config(self, config: Union[StrategyConfig, RiskConfig, IBConfig]) -> bool:
        """
        Validate configuration object.
        
        Parameters
        ----------
        config : Union[StrategyConfig, RiskConfig, IBConfig]
            Configuration object to validate
            
        Returns
        -------
        bool
            True if configuration is valid
        """
        try:
            if isinstance(config, (StrategyConfig, RiskConfig, IBConfig)):
                return True
            return False
        except Exception as e:
            raise ValueError(f"Configuration validation failed: {str(e)}")
            
    def save_config(self, config: Union[StrategyConfig, RiskConfig, IBConfig], file_path: Path) -> None:
        """
        Save configuration to TOML file.
        
        Parameters
        ----------
        config : Union[StrategyConfig, RiskConfig, IBConfig]
            Configuration object to save
        file_path : Path
            Path to save configuration file
        """
        if not self.validate_config(config):
            raise ValueError("Invalid configuration object")
            
        config_dict = config.dict()
        with open(file_path, 'w') as f:
            toml.dump(config_dict, f) 