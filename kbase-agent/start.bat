@echo off
rem kbase-agent 一键启动（Windows 双击入口）
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start.ps1"
pause
