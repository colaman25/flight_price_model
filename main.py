import argparse
from train import train_model
from predict import run_prediction

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "predict"], help="Mode: train or predict")
    parser.add_argument("--input", required=True, help="Path to input CSV file")
    parser.add_argument("--model_path", required=True, help="Path to save/load the model")
    parser.add_argument("--output", help="Path to save prediction result")

    args = parser.parse_args()

    if args.mode == "train":
        train_model(args.input, args.model_path)
    elif args.mode == "predict":
        run_prediction(args.input, args.model_path, args.output)
