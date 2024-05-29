# MGAI-MarioRL

# Environment 

```bash
conda create -n [ENV NAME] python=3.11
conda activate [ENV NAME]
```

## MacOS
To enable `mps` acceleration

```bash
pip3 install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu
```

## Windows 
To enable `cuda` acceleration

```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```


