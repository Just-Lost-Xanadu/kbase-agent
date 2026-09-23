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
# 预检必须 import 应用**真正会用到的模块**，不能只挑四个顶层包：aiosqlite 与
# langgraph-checkpoint-sqlite 是 app.store / app.agent.graph 的直接依赖，缺了照样能过
# "import fastapi, langgraph, chromadb, mcp" 这一关——于是脚本打印"依赖已就绪"，
# 服务也能起来、/api/health 也返回 200，但每个对话请求都 503（Agent 引擎未就绪）。
# 这里改成纯 import 应用模块：不启动服务，却能提前暴露这类"装了一半"的环境。
& $python -c "import app.main, app.agent.graph, app.mcp.servers" 2>$null
if ($LASTEXITCODE -ne 0) {
    # 先用 requirements.lock 复现"已验证可跑"的依赖组合：langgraph/langchain/chromadb 大版本
    # 变动快，直接 pip install -e . 可能装到不兼容的新版本；lock 装失败（如老版本在此平台
    # 没有轮子）再退回按 pyproject 约束安装，保证脚本不会因 lock 而彻底卡死。
    $lock = "$Root\requirements.lock"
    if (Test-Path $lock) {
        Write-Host "安装锁定依赖：pip install -r requirements.lock"
        & $python -m pip install -r $lock
        if ($LASTEXITCODE -ne 0) { Warn "按 requirements.lock 安装失败，退回按 pyproject 约束安装" }
    }
    Write-Host "安装项目依赖：pip install -e '.[dev]'"
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
# 环境变量优先：load_dotenv() 默认不覆盖已存在的环境变量，所以 key 放系统环境变量
# （setx DEEPSEEK_API_KEY ...）时 .env 留占位符也能正常跑，且 .env 可安全拷贝给别人。
if (-not [string]::IsNullOrWhiteSpace($env:DEEPSEEK_API_KEY)) {
    $keySource = '系统环境变量'
    $key = $env:DEEPSEEK_API_KEY
} else {
    $keySource = '.env'
}
if ([string]::IsNullOrWhiteSpace($key) -or $key -eq 'sk-your-key') {
    Warn "DEEPSEEK_API_KEY 未配置（.env 与进程环境变量里都没有）：网页/对话会报 503，但建索引与评测不受影响。"
    Warn "要体验 Agent 对话，请编辑 $envFile 填入真实 key（或 setx DEEPSEEK_API_KEY 后开新终端）再重跑。"
} else {
    # 不回显 key 的任何片段，避免截图/录屏时泄露
    Write-Host "DEEPSEEK_API_KEY 已配置（来源：$keySource，值不显示）"
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
