@echo off


echo ` ****  Install_PyTorch_CUDA  ****
echo.
echo ` You have to uninstall "Torch" before
echo ` IF it's NOT the " Torch CUDA " version !
echo ` else ...
echo ` Say 'NO' to uninstall answer !
echo.

IF EXIST C:\Windows\py.exe (
    echo **  Using PY launcher
    py -3.10 -m pip -V
    py -3.10 -m pip uninstall torch
    py -3.10 -m pip install torch --index-url https://download.pytorch.org/whl/cu118
) ELSE (
    echo **  Using PIP
    pip3.10 -V
    pip3.10 uninstall torch
    pip3.10 install torch --index-url https://download.pytorch.org/whl/cu118
)
