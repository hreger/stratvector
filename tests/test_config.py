"""
Tests for the configuration system.
"""

import pytest
import os
from pathlib import Path
from src.core.config import (
    ConfigManager,
    RiskProfile,
    StrategyConfig,
    RiskConfig,
    IBConfig
)

@pytest.fixture
def config_dir(tmp_path):
    """Create temporary config directory with test files."""
    # Create directory structure
    strategies_dir = tmp_path / "strategies"
    risk_profiles_dir = tmp_path / "risk_profiles"
    strategies_dir.mkdir()
    risk_profiles_dir.mkdir()
    
    # Create test files
    momentum_config = """
    [parameters]
    lookback_period = 20
    holding_period = 10
    rebalance_frequency = "1d"
    position_size = 0.1
    entry_threshold = 0.02
    exit_threshold = -0.01
    max_position_size = 0.2

    [universe]
    symbols = ["AAPL", "MSFT", "GOOGL"]
    exchange = "NASDAQ"
    currency = "USD"

    [risk]
    max_position_size = 0.2
    stop_loss = 0.05
    take_profit = 0.1
    max_drawdown = 0.15

    [execution]
    slippage = 0.001
    commission = 0.001
    """
    
    moderate_risk = """
    max_position_size = 0.15
    stop_loss = 0.05
    take_profit = 0.1
    max_drawdown = 0.15
    """
    
    ib_config = """
    host = "127.0.0.1"
    port = 7496
    client_id = 1
    timeout = 20
    readonly = false
    """
    
    # Write test files
    (strategies_dir / "momentum.toml").write_text(momentum_config)
    (risk_profiles_dir / "moderate.toml").write_text(moderate_risk)
    (tmp_path / "ib_config.toml").write_text(ib_config)
    
    return tmp_path

@pytest.fixture
def config_manager(config_dir):
    """Create ConfigManager instance."""
    return ConfigManager(config_dir)

def test_load_strategy_config(config_manager):
    """Test loading strategy configuration."""
    config = config_manager.load_strategy_config("momentum")
    
    assert isinstance(config, StrategyConfig)
    assert config.parameters.lookback_period == 20
    assert config.parameters.holding_period == 10
    assert config.parameters.rebalance_frequency == "1d"
    assert config.universe.symbols == ["AAPL", "MSFT", "GOOGL"]
    assert config.risk.max_drawdown == 0.15

def test_load_risk_profile(config_manager):
    """Test loading risk profile."""
    config = config_manager.load_risk_profile(RiskProfile.MODERATE)
    
    assert isinstance(config, RiskConfig)
    assert config.max_position_size == 0.15
    assert config.stop_loss == 0.05
    assert config.take_profit == 0.1
    assert config.max_drawdown == 0.15

def test_get_ib_config(config_manager):
    """Test loading IB configuration."""
    config = config_manager.get_ib_config()
    
    assert isinstance(config, IBConfig)
    assert config.host == "127.0.0.1"
    assert config.port == 7496
    assert config.client_id == 1
    assert config.timeout == 20
    assert not config.readonly

def test_environment_overrides(config_manager):
    """Test environment variable overrides."""
    # Set environment variables
    os.environ["STRATVECTOR_IB_HOST"] = "192.168.1.1"
    os.environ["STRATVECTOR_IB_PORT"] = "4001"
    os.environ["STRATVECTOR_IB_CLIENT_ID"] = "2"
    
    config = config_manager.get_ib_config()
    
    assert config.host == "192.168.1.1"
    assert config.port == 4001
    assert config.client_id == 2
    
    # Clean up
    del os.environ["STRATVECTOR_IB_HOST"]
    del os.environ["STRATVECTOR_IB_PORT"]
    del os.environ["STRATVECTOR_IB_CLIENT_ID"]

def test_invalid_strategy_config(config_manager, config_dir):
    """Test invalid strategy configuration."""
    # Create invalid config
    invalid_config = """
    [parameters]
    lookback_period = -1  # Invalid: must be positive
    holding_period = 10
    rebalance_frequency = "1d"
    position_size = 0.1
    entry_threshold = 0.02
    exit_threshold = -0.01
    max_position_size = 0.2

    [universe]
    symbols = []  # Invalid: cannot be empty
    exchange = "NASDAQ"
    currency = "USD"

    [risk]
    max_position_size = 0.2
    stop_loss = 0.05
    take_profit = 0.1
    max_drawdown = 0.15

    [execution]
    slippage = 0.001
    commission = 0.001
    """
    
    (config_dir / "strategies" / "invalid.toml").write_text(invalid_config)
    
    with pytest.raises(ValueError):
        config_manager.load_strategy_config("invalid")

def test_invalid_risk_profile(config_manager, config_dir):
    """Test invalid risk profile configuration."""
    # Create invalid config
    invalid_config = """
    max_position_size = 1.5  # Invalid: must be <= 1
    stop_loss = 0.05
    take_profit = 0.01  # Invalid: must be > stop_loss
    max_drawdown = 0.15
    """
    
    (config_dir / "risk_profiles" / "invalid.toml").write_text(invalid_config)
    
    with pytest.raises(ValueError):
        config_manager.load_risk_profile(RiskProfile("invalid"))

def test_save_config(config_manager, tmp_path):
    """Test saving configuration."""
    # Create test config
    config = IBConfig(
        host="192.168.1.1",
        port=4001,
        client_id=2,
        timeout=30,
        readonly=True
    )
    
    # Save config
    save_path = tmp_path / "test_config.toml"
    config_manager.save_config(config, save_path)
    
    # Load saved config
    loaded_config = config_manager._load_toml(save_path)
    
    assert loaded_config["host"] == "192.168.1.1"
    assert loaded_config["port"] == 4001
    assert loaded_config["client_id"] == 2
    assert loaded_config["timeout"] == 30
    assert loaded_config["readonly"] is True

def test_validate_config(config_manager):
    """Test configuration validation."""
    # Valid config
    valid_config = IBConfig()
    assert config_manager.validate_config(valid_config)
    
    # Invalid config
    class InvalidConfig:
        pass
    
    with pytest.raises(ValueError):
        config_manager.validate_config(InvalidConfig()) 