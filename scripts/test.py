import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
import joblib
import argparse

def load_data(input_file_1, input_file_2, input_file_3):
    df1 = pd.read_csv(input_file_1, sep='\t', na_values='.', dtype=str)
    df1 = df1.dropna()  # Filter out rows containing NaN values
    df1['position'] = df1['position'].astype(int)  # Set position as integer

    df2 = pd.read_csv(input_file_2, sep='\t',header=None)
    df2.columns = ['sample_name', 'label']

    # Load positions to select
    positions = pd.read_csv(input_file_3, sep='\t', header=None, names=['chr', 'position'])
    positions['position'] = positions['position'].astype(int)  # Set position as integer

    return df1, df2, positions

def filter_data(df1, positions):
    filtered_df1 = df1[df1.set_index(['chr', 'position']).index.isin(positions.set_index(['chr', 'position']).index)]
    return filtered_df1

def extract_features_labels(df, tagged_samples):
    features = []
    labels = []

    for _, row in tagged_samples.iterrows():
        sample_name = row['sample_name']
        mat_col = f'{sample_name}.mat'
        pat_col = f'{sample_name}.pat'

        if mat_col in df.columns and pat_col in df.columns:
            mat_values = df[mat_col].astype(float).values
            pat_values = df[pat_col].astype(float).values

            # Create features based on differences
            features.append(mat_values - pat_values)  # Feature for maternal haplotype
            labels.append('mat')  # Label for maternal haplotype

            features.append(pat_values - mat_values)  # Feature for paternal haplotype
            labels.append('pat')  # Label for paternal haplotype

    return np.array(features), np.array(labels)

def train_model(X, y):
    # Ensure X is 2D
    if len(X.shape) == 1:
        X = X.reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = GaussianNB()
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    print(f'Model accuracy: {accuracy:.2f}')

    return model

def predict_haplotypes(model, filtered_df1, untagged_samples):
    predictions = []

    for _, row in untagged_samples.iterrows():
        sample_name = row['sample_name']
        mat_col = f'{sample_name}.mat'
        pat_col = f'{sample_name}.pat'

        if mat_col not in filtered_df1.columns or pat_col not in filtered_df1.columns:
            print(f"Error: Missing columns for sample {sample_name}.")
            continue

        mat_values = filtered_df1[mat_col].astype(float).values
        pat_values = filtered_df1[pat_col].astype(float).values

        if np.count_nonzero(~np.isnan(mat_values)) < 2 or np.count_nonzero(~np.isnan(pat_values)) < 2:
            print(f"Error: Less than two haplotypes found for sample {sample_name}.")
            continue

        # Calculate features
        feature_mat = mat_values - pat_values
        feature_pat = pat_values - mat_values

        # Ensure features are valid
        if np.any(np.isnan(feature_mat)) or np.any(np.isnan(feature_pat)):
            print(f"Error: Invalid features for sample {sample_name}.")
            continue

        # Reshape features to match training input shape
        feature_mat = feature_mat.reshape(1, -1)  # Reshape for single sample prediction
        feature_pat = feature_pat.reshape(1, -1)

        # Make predictions
        mat_pred = model.predict(feature_mat)
        pat_pred = model.predict(feature_pat)

        # Calculate confidence scores
        confidence_mat = model.predict_proba(feature_mat)
        confidence_pat = model.predict_proba(feature_pat)

        predictions.append([f'{sample_name}.mat', mat_pred[0], confidence_mat[0]])
        predictions.append([f'{sample_name}.pat', pat_pred[0], confidence_pat[0]])

    return pd.DataFrame(predictions, columns=['haplotype_name', 'inferred_origin', 'confidence'])

def save_model(model, model_filename):
    joblib.dump(model, model_filename)

def main(args):
    df1, df2, positions = load_data(args.input_file_1, args.input_file_2, args.input_file_3)

    filtered_df1 = filter_data(df1, positions)
    tagged_samples = df2[df2['label'] == 'tagged']
    X, y = extract_features_labels(filtered_df1, tagged_samples)

    model = None
    if args.train_model:
        model = train_model(X, y)
        save_model(model, args.model_filename)
    elif args.load_model:
        model = joblib.load(args.model_filename)

    if model is not None:
        untagged_samples = df2[df2['label'] == 'untagged']
        predictions = predict_haplotypes(model, filtered_df1, untagged_samples)

        results = pd.DataFrame(predictions, columns=['haplotype_name', 'inferred_origin', 'confidence'])
        results.to_csv(f'methylation_corrected_haplotype.{args.model_filename}.tsv', sep="\t", header=True, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file_1', required=True, help='Path to the first input TSV file with methylation data.')
    parser.add_argument('--input_file_2', required=True, help='Path to the second input TSV file with sample labels.')
    parser.add_argument('--input_file_3', required=True, help='Path to the third input file with chr and position.')
    parser.add_argument('--model_filename', help='Path to save or load the model.')
    parser.add_argument('--train_model', action='store_true', help='Flag to train a new model.')
    parser.add_argument('--load_model', action='store_true', help='Flag to load an existing model.')
    args = parser.parse_args()

    main(args)
