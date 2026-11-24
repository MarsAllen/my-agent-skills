# My Agent Skills

個人實戰沉澱的 **Agent Skills** 集合，採用標準 `SKILL.md` + `scripts/` + `references/` 結構，可直接搭配 **Google Antigravity**、**Claude Code**、**Codex**、**Gemini CLI** 等 AI Coding Agents 使用。

---

## 收錄的 Skills

### 1. [`colab-ai-video-production`](./skills/colab-ai-video-production/SKILL.md)
> **Colab AI 短影音與寵物／角色對嘴製作總指南（附本機 0-CCU 泛用 CLI 與 Colab V2V 精修腳本）**

從實戰製作「守宮對嘴深情唱情歌」迷因短影音中完整提煉出的兩段式製作流程與踩坑手冊：

- **四階段實戰工作流**（[SKILL.md](./skills/colab-ai-video-production/SKILL.md)）：
  1. **素材特寫與音訊分析**：大頭特寫構圖標準、30 fps 音節能量包絡抽取、參考影片節奏分析、無縫 SRT 驗證。
  2. **本機 0 CCU 精準對拍預覽**：以雙眼中心鎖定上臉、只驅動下巴開合（徹底消除「發條左右甩頭感」），搭配「由上往下物理眼皮闔上遮罩」（消除平面淡入鬼影），先在本機免費確認 100% 對嘴後再開 GPU。
  3. **Colab GPU 低強度 V2V 擴散精修**：以 `Wan2.1-T2V-1.3B-Diffusers`（`WanVideoToVideoPipeline`、`strength=0.28`，L4 24GB 僅佔 ~6 GB VRAM、約 3 分鐘）融合眼皮與口腔邊緣，內建防吐舌負向提示詞、可選 OpenCV `INTER_LANCZOS4` 亞像素高潮緩推鏡，以及 `-crf 15` 高畫質繁中字幕燒錄。
  4. **動態＋4 fps 細格強制驗收與 CCU 歸零關機**。
- **完整踩坑與調參手冊**（[pitfalls_and_tuning.md](./skills/colab-ai-video-production/references/pitfalls_and_tuning.md)）：
  - macOS Homebrew Python 3.15 (`openssl@4`) 缺 CA 憑證導致 `google-colab-cli` WebSocket 斷線修法
  - MCP 3 分鐘逾時背景 `Popen` 監控法、WebSocket 併行衝突、Colab GPU `503` 降級策略
  - `Wan2.2-S2V-14B` 免編譯 `flash_attn` 切 PyTorch SDPA 修法、A100 40GB OOM 參數組、移除硬編負向提示詞 `情绪激动剧烈` 與獨立 `AUDIO_CFG`
  - 避免模糊第二嘴型造成 V2V 吐舌幻覺、高畫質 `-crf 15` 字幕與 SRT 防閃爍檢查
- **附帶工具腳本**：
  - [`video_toolkit.py`](./skills/colab-ai-video-production/scripts/video_toolkit.py)：零第三方依賴（純 Python stdlib + `ffmpeg`/`ffprobe`），支援 `analyze-audio`、`analyze-ref`、`validate-srt`、`build-preview`、`audit-video` 五個子指令。
  - [`colab_v2v_polish.py`](./skills/colab-ai-video-production/scripts/colab_v2v_polish.py)：Colab GPU 端 V2V 精修＋緩推鏡＋繁中字幕一鍵腳本。

---

## 安裝方式（Installation）

### 1. Clone 本倉庫
```bash
git clone https://github.com/MarsAllen/my-agent-skills.git
```

### 2. 連結到你的 Agent 全域 Skills 目錄
依你使用的工具選擇對應路徑建立軟連結（symlink），之後 `git pull` 即可自動更新：

- **Google Antigravity**（全域）：
  ```bash
  mkdir -p ~/.gemini/config/skills
  ln -s "$(pwd)/my-agent-skills/skills/colab-ai-video-production" \
        ~/.gemini/config/skills/colab-ai-video-production
  ```
- **Claude Code**（全域）：
  ```bash
  mkdir -p ~/.claude/skills
  ln -s "$(pwd)/my-agent-skills/skills/colab-ai-video-production" \
        ~/.claude/skills/colab-ai-video-production
  ```
- **Codex**（全域）：
  ```bash
  mkdir -p ~/.agents/skills
  ln -s "$(pwd)/my-agent-skills/skills/colab-ai-video-production" \
        ~/.agents/skills/colab-ai-video-production
  ```
- **Gemini CLI**（全域）：
  ```bash
  mkdir -p ~/.gemini/skills
  ln -s "$(pwd)/my-agent-skills/skills/colab-ai-video-production" \
        ~/.gemini/skills/colab-ai-video-production
  ```

### 前置需求（Prerequisites）
- 本機工具：`uv`、`ffmpeg`、`ffprobe`
- Colab 遠端執行：搭配 Google Colab CLI / `colab` Skill 使用

---

## 快速測試 CLI（Quick Check）

```bash
TOOL=~/.gemini/config/skills/colab-ai-video-production/scripts/video_toolkit.py

# 分析音訊 30fps 音節能量與停頓點
uv run "$TOOL" analyze-audio --audio chorus.wav --fps 30 --output audio_env.json

# 檢查並修復 SRT 時間格式與閃爍縫隙
uv run "$TOOL" validate-srt --srt lyrics.srt --fix-gaps --output srt_report.json

# 產出 4fps 細格檢視圖與 ffprobe 品質報告
uv run "$TOOL" audit-video --video final.mp4 --fps 4 --output audit_4fps.jpg
```
