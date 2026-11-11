@echo off
chcp 65001 >nul
echo Instaluje biblioteke duckdb - gotowe obrysy pol z Fields of The World
echo.
where py >nul 2>nul
if errorlevel 1 goto python
py -3.12 -m pip install "duckdb>=1.1"
if errorlevel 1 goto python
py -3.12 -c "import duckdb; print(); print('duckdb', duckdb.__version__, '- zainstalowane poprawnie')"
goto koniec
:python
python -m pip install "duckdb>=1.1"
python -c "import duckdb; print(); print('duckdb', duckdb.__version__, '- zainstalowane poprawnie')"
:koniec
echo.
echo Uruchom aplikacje ponownie, zeby zaczela korzystac z nowej biblioteki.
pause
