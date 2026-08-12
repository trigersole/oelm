@echo off
cd /d "%~dp0"
echo Starting OELM site...
echo.
echo Open this in your browser:
echo http://localhost:8080
echo.
echo Keep this window open while using the site.
echo Press Ctrl+C to stop the server.
echo.
python -m http.server 8080
