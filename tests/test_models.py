"""
Tests for ML models
"""

import pytest
import numpy as np
from src.models.lstm_model import LSTMModel
from src.models.cnn_lstm_hybrid import CNNLSTMHybrid


def test_lstm_model_build():
    """Test LSTM model building."""
    model = LSTMModel(
        sequence_length=60,
        features=10,
        hidden_units=[64, 32],
        dropout=0.2
    )
    
    built_model = model.build_model(output_dim=1)
    assert built_model is not None
    assert model.model is not None


def test_lstm_prepare_sequences():
    """Test LSTM sequence preparation."""
    model = LSTMModel(sequence_length=10, features=5)
    
    X = np.random.randn(100, 5)
    y = np.random.randn(100)
    
    X_seq, y_seq = model.prepare_sequences(X, y)
    
    assert X_seq.shape[0] == 90  # 100 - 10
    assert X_seq.shape[1] == 10  # sequence_length
    assert X_seq.shape[2] == 5  # features
    assert y_seq.shape[0] == 90


def test_cnn_lstm_model_build():
    """Test CNN-LSTM model building."""
    model = CNNLSTMHybrid(
        sequence_length=60,
        spatial_features=5,
        temporal_features=10
    )
    
    built_model = model.build_model(output_dim=1)
    assert built_model is not None
    assert model.model is not None

