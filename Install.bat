@echo off

REM Install_PyTorch_CUDA

echo.
echo ** You have to uninstall "Torch" before IF it's NOT the " Torch CUDA " version !
echo ** else ...
echo ** say 'No' to uninstall answer !
echo.

pip3.10 uninstall torch

pip3.10 install torch --index-url https://download.pytorch.org/whl/cu118
