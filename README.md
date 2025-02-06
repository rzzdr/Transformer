# Transformer
Transformer implemented from scratch in PyTorch, for English to Russian Translation.

## Training
### Setup Python and Poetry
- Install `Python3.11.9` and `Poetry`:
  - Follow [these instructions](https://asdf-vm.com/#/core-manage-asdf-vm?id=install-asdf-vm) to install `asdf`
    - `asdf plugin add python`
    - `asdf plugin add poetry`
    - `asdf install`
  - ~NOTE: your machine must have a system version of Python installed. If you don't, run the following: `asdf install python 3.11.11` && `asdf global python 3.11.11`
- If you have `Python 3.11` and `Poetry`installed already, please feel free to skip.

#### Install Dependencies
```bash
poetry install
```

#### Initialize poetry shell
```bash
poetry shell
```

### Run the following command to start the training process.
```bash
poetry run task train
```