import pandas as pd
from typing_extensions import Tuple, Annotated
from zenml import step, pipeline, ArtifactConfig
import data


@step(enable_cache=False)
def extract() -> Tuple[
    Annotated[
        pd.DataFrame,
        ArtifactConfig(name="extracted_data", tags=["data_preparation"]),
    ],
    Annotated[str, ArtifactConfig(name="data_version", tags=["data_preparation"])],
]:
    version = 'v1'
    df, version = data.extract_data()
    print(df.shape, version)
    return df, version


@step(enable_cache=False)
def transform(df: pd.DataFrame) -> Tuple[
    Annotated[
        pd.DataFrame, ArtifactConfig(name="input_features", tags=["data_preparation"])
    ],
    Annotated[
        pd.DataFrame, ArtifactConfig(name="input_target", tags=["data_preparation"])
    ],
]:
    X, y = data.transform_data(df=df)
    print("Data transformed successfully!")
    return X, y


@step(enable_cache=False)
def split_save(X: pd.DataFrame, y: pd.DataFrame, version: str) -> Tuple[
    Annotated[pd.DataFrame, ArtifactConfig(name="features", tags=["data_preparation"])],
    Annotated[pd.DataFrame, ArtifactConfig(name="target", tags=["data_preparation"])],
]:
    data.split_save_data(X, y, version)

    return X, y


@pipeline()
def prepare_data_pipeline():
    df, version = extract()
    X, y = transform(df)
    X, y = split_save(X, y, version)


if __name__ == "__main__":
    run = prepare_data_pipeline()