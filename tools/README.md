# Supporting python scripts

This folder contains python scripts that are not used in the
core library, but are useful for supporting tasks.

#### Pre-process the EchoNet dynamic dataset:
Make sure the CONST paths are correct and then run:
```bash
python tools/preprocess_echonet.py
```

#### Test run time:
```bash
taskset -c 0-7 python tools/test_runtime.py --option
```
