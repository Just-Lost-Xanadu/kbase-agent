# ============================================================
#  kbase-agent 一键启动（Windows PowerShell）
#  用法：双击 start.bat，或在此目录执行  ./start.ps1
#  可选参数：-NoStart  只做检查/补装/建索引，不拉起服务
# ============================================================
param([switch]$NoStart)

$ErrorActionPreference = 'Continue'
$Root = $PSScriptRoot
$Port = if ($env:KA_PORT) { $env:KA_PORT } else { '8000' }
$Url  = "http://127.0.0.1:$Port"

Set-Location $Root
Write-Host ""
Write-Host "== kbase-agent 启动器 ==" -ForegroundColor Cyan

function Step($msg) { Write-Host ""; Write-Host ">>> $msg" -ForegroundColor Green }
function Warn($msg) { Write-Host "[!] $msg" -ForegroundColor Yellow }
function Die($msg) { Write-Host "[x] $msg" -ForegroundColor Red; exit 1 }

# ---------- 1. Python ----------
Step "1/5 检查 Python"
$python = if (Test-Path "$Root\.venv\Scripts\python.exe") {
    "$Root\.venv\Scripts\python.exe"
} else {
    $g = Get-Command python -ErrorAction SilentlyContinue
    if ($g) { (Get-Command python).Source } else { $null }
}
if (-not $python) {
    Write-Host "未找到 python，请先安装 Python 3.10+ 并加入 PATH"
    Read-Host "按回车退出"; exit 1
}
Write-Host "python: $python"
& $python --version 2>&1 | ForEach-Object { Write-Host "    $_" }

# ---------- 2. 虚拟环境 ----------
Step "2/5 检查虚拟环境"
if (-not (Test-Path "$Root\.venv\Scripts\python.exe")) {
    Write-Host "创建 .venv ..."
    & python -m venv "$Root\.venv"
    if ($LASTEXITCODE -ne 0) { Die "创建 .venv 失败" }
    $python = "$Root\.venv\Scripts\python.exe"
} else {
    Write-Host ".venv 已存在"
}

# ---------- 3. 依赖 ----------
Step "3/5 检查依赖（缺失才安装，首次较久）"
& $python -c "import fastapi, langgraph, chromadb, mcp" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "安装依赖：pip install -e '.[dev]'"
    & $python -m pip install -e ".[dev]"
    if ($LASTEXITCODE -ne 0) { Die "依赖安装失败，请检查网络后重试" }
} else {
    Write-Host "依赖已就绪"
}

# ---------- 4. .env ----------
Step "4/5 检查环境变量 .env"
$envFile = "$Root\.env"
if (-not (Test-Path $envFile)) {
    Copy-Item "$Root\.env.example" $envFile
    Write-Host "已从 .env.example 生成 .env，请填入 DEEPSEEK_API_KEY 后重跑"
    Read-Host "按回车退出"; exit 1
}
$keyLine = Get-Content $envFile -Raw
if ($keyLine -match '(?m)^DEEPSEEK_API_KEY=(.+)$') {
    $key = $Matches[1].Trim()
} else { $key = '' }
if ([string]::IsNullOrWhiteSpace($key) -or $key -eq 'sk-your-key') {
    Warn "DEEPSEEK_API_KEY 为空或仍是占位符：网页/对话会报 503，但建索引与评测不受影响。"
    Warn "要体验 Agent 对话，请编辑 $envFile 填入真实 key 后重跑。"
} else {
    Write-Host "DEEPSEEK_API_KEY 已配置（${key}... 前4位）"
}

# ---------- 5. 索引 ----------
Step "5/5 检查检索索引（缺才建，首次会下载 embedding 模型）"
$chromaOk = (Test-Path "$Root\data\chroma\chunks.jsonl")
if (-not $chromaOk) {
    Write-Host "未发现索引，运行 python scripts/index_docs.py ..."
    & $python "$Root\scripts\index_docs.py"
    if ($LASTEXITCODE -ne 0) {
        Warn "建索引失败：多为无法下载 embedding 模型。"
        Warn "有模型缓存的机器可先设置：FASTEMBED_CACHE_PATH=<缓存目录> 后重跑。"
    }
} else {
    Write-Host "索引已存在，跳过建库"
}

Write-Host ""
Write-Host "== 就绪 ==" -ForegroundColor Cyan
Write-Host "  URL : $Url"
Write-Host "  /api/chat (同步)  /api/chat/stream (SSE)  /api/sessions (历史会话)"
Write-Host "  Ctrl+C 停止服务"

if ($NoStart) { Write-Host "(NoStart 模式：仅检查完毕，未启动服务)"; exit 0 }

# ---------- 启动 uvicorn（前台，Ctrl+C 停止）----------
Write-Host ""
Write-Host ">>> 启动服务并自动打开浏览器 ..." -ForegroundColor Green
Start-Process powershell -WindowStyle Hidden -ArgumentList @(
    '-NoProfile','-Command',
    "Start-Sleep -Seconds 6; Start-Process '$Url'"
)
& $python -m uvicorn app.main:app --host 127.0.0.1 --port $Port --reload
