@echo off

echo.
echo ` ****  Install_PyTorch_CUDA  ****
echo.
echo ` You have to uninstall "Torch" before
echo ` IF it's NOT the " Torch CUDA " version !
echo ` else ...
echo ` Say 'NO' to uninstall answer !
echo.
echo.

pip3.10 uninstall torch
pip3.10 install torch --index-url https://download.pytorch.org/whl/cu121

pip3.10 install -r requirements.txt
pip3.10 install -r requirements_PC.txt

echo.
echo.
py -3.10 -m App.setup
echo.

pause
exit
