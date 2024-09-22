from datasets.data import extract_preprocessed_data
from preprocess import preprocess
from linear_regression import train, log_metadata


def run(version=None):
    if version == None:
        version = "v1"
    # Extract features and target for training data using the specified version
    train_df = extract_preprocessed_data(name="train_df", version=version)
    test_df = extract_preprocessed_data(name="test_df", version=version)
    print("Train dataset: ", train_df.info(), version)
    print("Test dataset: ", test_df.info(), version)

    
    X_train, y_train, X_test, y_test = preprocess(train_df, test_df)
    print(X_test[1])

    regressor = train(X_train, y_train)
    
    log_metadata(regressor, X_test, y_test)


def main():
    run()


if __name__ == "__main__":
    main()