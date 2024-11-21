# How to Use This Repository

## Training the Model

1. Ensure you have all dependencies installed.
2. Execute the training script:
   ```bash
   python src/train.py
   ```

## Running Tests

1. Ensure pytest is installed.
2. Run the tests using:
   ```bash
   pytest
   ```

## GitHub Actions

- The workflow is triggered on git push.
- It checks out the code, sets up Python, installs dependencies, and runs tests. 