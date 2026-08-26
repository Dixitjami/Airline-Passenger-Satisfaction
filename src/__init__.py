from .preprocessing import (
    load_data,
    clean_data,
    preprocess_data,
    encode_target,
    decode_target,
    get_feature_columns,
    ID_COLUMNS,
    TARGET_COLUMN,
    NUMERICAL_COLUMNS,
    CATEGORICAL_COLUMNS,
    SERVICE_RATING_COLUMNS,
)
from .components.data_ingestion import DataIngestion, load_raw_data

# train.py imports xgboost conditionally; evaluate.py / predict.py are
# planned but not implemented yet. Import them only when present so the
# package stays usable in the meantime.
try:
    from .train import train_model, train_all_models
except ImportError:  # pragma: no cover
    train_model = None
    train_all_models = None

try:
    from .evaluate import evaluate_model, compare_models, get_classification_report
except ImportError:  # pragma: no cover
    pass

try:
    from .predict import predict_satisfaction, load_model, preprocess_single_passenger
except ImportError:  # pragma: no cover
    pass

__all__ = [
    "load_data",
    "load_raw_data",
    "DataIngestion",
    "clean_data",
    "preprocess_data",
    "encode_target",
    "decode_target",
    "get_feature_columns",
    "ID_COLUMNS",
    "TARGET_COLUMN",
    "NUMERICAL_COLUMNS",
    "CATEGORICAL_COLUMNS",
    "SERVICE_RATING_COLUMNS",
]
