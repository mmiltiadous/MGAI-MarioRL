# MGAI-MarioRL

# Environment 

```bash
conda create -n [ENV NAME] python=3.8
conda activate [ENV NAME]
```

```bash
pip install pip==22.0.4 setuptools==59.8.0 wheel==0.37.1
pip install gym==0.21
pip install 'stable-baselines3[extra]'==1.6.0
pip install gym-super-mario-bros==7.3.0
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


