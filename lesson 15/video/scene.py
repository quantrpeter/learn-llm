"""
Lesson 15 – Serving & Deployment
Manim CE + Cantonese voiceover (Edge TTS)

Render (1080p):
    cd "lesson 15/video"
    manim render -qh scene.py ServingExplainer

Dependencies in the same directory:
    - edge_tts_service.py  (copy from any previous lesson's video/)
    - wallpaper1.jpg       (copy from lesson 1/video/)
"""

from manim import *
from manim_voiceover import VoiceoverScene

from edge_tts_service import EdgeTTSService

# ── colour palette ───────────────────────────────────────────────────────
C_BG = "#0f0f23"
C_BLUE = "#4fc3f7"
C_GREEN = "#81c784"
C_ORANGE = "#ffb74d"
C_PINK = "#f48fb1"
C_PURPLE = "#ce93d8"
C_YELLOW = "#fff176"
C_WHITE = "#e0e0e0"
C_RED = "#ef5350"
C_DIM = "#555577"
C_CYAN = "#4dd0e1"

FONT = "Noto Sans CJK HK"
MONO = "Menlo"


class ServingExplainer(VoiceoverScene):
    """Eight scenes explaining LLM serving & deployment in Cantonese."""

    def construct(self):
        self.set_speech_service(
            EdgeTTSService(voice="zh-HK-HiuMaanNeural", rate="+0%")
        )
        self.camera.background_color = BLACK

        self.bg_image = ImageMobject("wallpaper1.jpg")
        self.bg_image.height = config.frame_height
        self.bg_image.width = config.frame_width
        self.add(self.bg_image)

        self.bg_overlay = Rectangle(
            width=config.frame_width + 0.5,
            height=config.frame_height + 0.5,
            fill_color=BLACK,
            fill_opacity=0.55,
            stroke_width=0,
        )
        self.add(self.bg_overlay)

        self.watermark = self.zh(
            "香港編程學會", font_size=18, color="#9999bb"
        ).to_corner(UL, buff=0.3)
        self.add(self.watermark)

        self.scene_intro()
        self.scene_frameworks()
        self.scene_api_design()
        self.scene_streaming()
        self.scene_batching()
        self.scene_cost()
        self.scene_summary()
        self.scene_outro()

    # ── text helpers ─────────────────────────────────────────────────────

    def zh(self, txt, **kw):
        kw.setdefault("font", FONT)
        kw.setdefault("color", C_WHITE)
        return Text(txt, **kw)

    def en(self, txt, **kw):
        kw.setdefault("color", C_WHITE)
        return Text(txt, **kw)

    def mono(self, txt, **kw):
        kw.setdefault("font", MONO)
        kw.setdefault("color", C_WHITE)
        return Text(txt, **kw)

    def clear(self):
        keep = {self.bg_image, self.bg_overlay, self.watermark}
        to_fade = [m for m in self.mobjects if m not in keep]
        if to_fade:
            self.play(*[FadeOut(m) for m in to_fade], run_time=0.5)

    def make_box(self, label, color, width=2.5, height=0.8, font_size=24):
        rect = RoundedRectangle(
            corner_radius=0.15,
            width=width,
            height=height,
            fill_color=color,
            fill_opacity=0.25,
            stroke_color=color,
        )
        txt = self.zh(label, font_size=font_size, color=color).move_to(rect)
        return VGroup(rect, txt)

    def make_mono_box(self, label, color, width=2.5, height=0.8, font_size=22):
        rect = RoundedRectangle(
            corner_radius=0.15,
            width=width,
            height=height,
            fill_color=color,
            fill_opacity=0.25,
            stroke_color=color,
        )
        txt = self.mono(label, font_size=font_size, color=color).move_to(rect)
        return VGroup(rect, txt)

    def make_heading(self, text, color=C_BLUE, font_size=40):
        return self.zh(text, font_size=font_size, color=color).to_edge(UP, buff=0.5)

    # ── Scene 1 — Intro ──────────────────────────────────────────────────

    def scene_intro(self):
        title = self.zh("部署服務", font_size=56, color=C_BLUE)
        sub_en = self.en(
            "Serving & Deployment", font_size=24, color=C_WHITE
        ).next_to(title, DOWN, buff=0.4)
        sub_zh = self.zh(
            "第十五課", font_size=36, color=C_ORANGE
        ).next_to(sub_en, DOWN, buff=0.3)
        producer = self.zh(
            "制片人：Peter", font_size=24, color=C_DIM
        ).next_to(sub_zh, DOWN, buff=0.7)

        with self.voiceover(
            text="大家好！歡迎嚟到第十五課。"
            "上一課我哋學咗量化同推理優化，"
            "將模型壓縮到更細更快。"
            "今日我哋會學點樣將模型部署成真正嘅服務。"
            "包括三大 Serving 框架嘅比較、"
            "OpenAI 兼容嘅 API 設計、"
            "Streaming 串流回應、"
            "Continuous Batching 吞吐量優化、"
            "成本估算、同埋安全護欄。"
        ):
            self.play(Write(title), run_time=1.5)
            self.play(FadeIn(sub_en, shift=UP * 0.2), run_time=0.8)
            self.play(FadeIn(sub_zh, shift=UP * 0.2), run_time=0.8)
            self.play(FadeIn(producer, shift=UP * 0.2), run_time=0.5)

        self.wait(0.3)
        self.clear()

    # ── Scene 2 — Frameworks (3 comparison cards) ─────────────────────────

    def scene_frameworks(self):
        heading = self.make_heading("三大 Serving 框架")

        # --- vLLM card ---
        vllm_box = RoundedRectangle(
            corner_radius=0.15, width=3.6, height=3.8,
            fill_color=C_BLUE, fill_opacity=0.10,
            stroke_color=C_BLUE, stroke_width=2,
        )
        vllm_title = self.en("vLLM", font_size=28, color=C_BLUE)
        vllm_title.next_to(vllm_box, UP, buff=0.1)
        vllm_details = self.zh(
            "Python + C++ 核心\n"
            "PagedAttention\n"
            "最高吞吐量\n"
            "AWQ / GPTQ / FP8\n"
            "雲端 GPU 首選",
            font_size=14, color=C_WHITE,
            line_spacing=1.2,
        ).move_to(vllm_box)
        vllm_card = VGroup(vllm_box, vllm_title, vllm_details)

        # --- TGI card ---
        tgi_box = RoundedRectangle(
            corner_radius=0.15, width=3.6, height=3.8,
            fill_color=C_GREEN, fill_opacity=0.10,
            stroke_color=C_GREEN, stroke_width=2,
        )
        tgi_title = self.en("TGI", font_size=28, color=C_GREEN)
        tgi_title.next_to(tgi_box, UP, buff=0.1)
        tgi_details = self.zh(
            "Rust + Python\n"
            "HuggingFace 生態\n"
            "Flash Attention\n"
            "Docker 部署\n"
            "企業級穩定",
            font_size=14, color=C_WHITE,
            line_spacing=1.2,
        ).move_to(tgi_box)
        tgi_card = VGroup(tgi_box, tgi_title, tgi_details)

        # --- llama.cpp card ---
        lcpp_box = RoundedRectangle(
            corner_radius=0.15, width=3.6, height=3.8,
            fill_color=C_ORANGE, fill_opacity=0.10,
            stroke_color=C_ORANGE, stroke_width=2,
        )
        lcpp_title = self.en("llama.cpp", font_size=28, color=C_ORANGE)
        lcpp_title.next_to(lcpp_box, UP, buff=0.1)
        lcpp_details = self.zh(
            "純 C/C++\n"
            "CPU / Apple Metal\n"
            "GGUF 量化格式\n"
            "本地 / 邊緣裝置\n"
            "無需 GPU",
            font_size=14, color=C_WHITE,
            line_spacing=1.2,
        ).move_to(lcpp_box)
        lcpp_card = VGroup(lcpp_box, lcpp_title, lcpp_details)

        cards = VGroup(vllm_card, tgi_card, lcpp_card).arrange(RIGHT, buff=0.5)
        cards.shift(DOWN * 0.2)

        # Bottom summary
        summary = self.zh(
            "高吞吐量用 vLLM　｜　HuggingFace 生態用 TGI　｜　本地部署用 llama.cpp",
            font_size=18, color=C_YELLOW,
        ).to_edge(DOWN, buff=0.4)

        with self.voiceover(
            text="部署 LLM 最重要嘅就係揀啱 Serving 框架。"
            "第一個係 vLLM。"
            "佢用 Python 寫，但核心係 C++ kernel。"
            "最大特色係 PagedAttention，將 KV cache 當虛擬記憶體咁管理。"
            "吞吐量係三個框架入面最高嘅。"
            "第二個係 TGI，由 HuggingFace 開發。"
            "用 Rust 寫，所以好穩定。"
            "如果你已經用緊 HuggingFace 嘅模型同工具，TGI 係最自然嘅選擇。"
            "第三個係 llama.cpp。"
            "純 C++ 寫，可以喺 CPU 同 Apple Metal 上面跑。"
            "用 GGUF 格式做量化，唔需要 GPU 都可以用。"
            "最適合本地部署同邊緣裝置。"
            "簡單嚟講，高吞吐量用 vLLM，"
            "HuggingFace 生態用 TGI，本地部署用 llama.cpp。"
        ):
            self.play(Write(heading), run_time=0.6)
            self.play(FadeIn(vllm_card, shift=DOWN * 0.2), run_time=0.6)
            self.play(FadeIn(tgi_card, shift=DOWN * 0.2), run_time=0.6)
            self.play(FadeIn(lcpp_card, shift=DOWN * 0.2), run_time=0.6)
            self.play(FadeIn(summary, shift=UP * 0.2), run_time=0.5)

        self.wait(0.3)
        self.clear()

    # ── Scene 3 — API Design (JSON request → response) ────────────────────

    def scene_api_design(self):
        heading = self.make_heading("OpenAI 兼容 API")

        # Request box (left)
        req_box = RoundedRectangle(
            corner_radius=0.15, width=5.2, height=4.2,
            fill_color=C_BLUE, fill_opacity=0.10,
            stroke_color=C_BLUE, stroke_width=2,
        )
        req_title = self.zh("請求 Request", font_size=20, color=C_BLUE)
        req_title.next_to(req_box, UP, buff=0.1)
        req_json = self.mono(
            '{\n'
            '  "model": "llama-3-8b",\n'
            '  "messages": [\n'
            '    {"role": "user",\n'
            '     "content": "Hello"}\n'
            '  ],\n'
            '  "temperature": 0.7,\n'
            '  "max_tokens": 256,\n'
            '  "stream": false\n'
            '}',
            font_size=13, color=C_CYAN,
        ).move_to(req_box)
        req_group = VGroup(req_box, req_title, req_json)

        # Response box (right)
        resp_box = RoundedRectangle(
            corner_radius=0.15, width=5.2, height=4.2,
            fill_color=C_GREEN, fill_opacity=0.10,
            stroke_color=C_GREEN, stroke_width=2,
        )
        resp_title = self.zh("回應 Response", font_size=20, color=C_GREEN)
        resp_title.next_to(resp_box, UP, buff=0.1)
        resp_json = self.mono(
            '{\n'
            '  "choices": [{\n'
            '    "message": {\n'
            '      "role": "assistant",\n'
            '      "content": "Hi!"\n'
            '    },\n'
            '    "finish_reason": "stop"\n'
            '  }],\n'
            '  "usage": {"total": 12}\n'
            '}',
            font_size=13, color=C_GREEN,
        ).move_to(resp_box)
        resp_group = VGroup(resp_box, resp_title, resp_json)

        panels = VGroup(req_group, resp_group).arrange(RIGHT, buff=0.6)
        panels.shift(DOWN * 0.15)

        # Arrow between them
        arrow = Arrow(
            req_box.get_right(), resp_box.get_left(),
            buff=0.15, color=C_YELLOW, stroke_width=3,
        )
        arrow_label = self.zh("模型推理", font_size=16, color=C_YELLOW)
        arrow_label.next_to(arrow, UP, buff=0.1)

        # Bottom note
        note = self.zh(
            "同一格式適用於 vLLM、TGI、llama.cpp — 換 base URL 就得",
            font_size=16, color=C_DIM,
        ).to_edge(DOWN, buff=0.4)

        with self.voiceover(
            text="部署好模型之後，我哋需要一個 API 接口。"
            "而家最標準嘅格式係 OpenAI 嘅 Chat Completions API。"
            "左邊係請求。你傳一個 JSON 物件，"
            "入面有 model 名、messages 對話歷史、"
            "temperature 控制隨機性、max tokens 限制長度。"
            "模型做完推理之後，右邊就係回應。"
            "入面有 assistant 嘅回覆內容、"
            "finish reason 話你知點解停止，"
            "同埋 usage 記錄用咗幾多 token。"
            "最重要嘅係，vLLM、TGI、llama.cpp 全部都支援呢個格式。"
            "所以你換 serving 框架嘅時候，只需要換個 base URL 就得。"
        ):
            self.play(Write(heading), run_time=0.6)
            self.play(FadeIn(req_group, shift=DOWN * 0.2), run_time=0.6)
            self.play(GrowArrow(arrow), FadeIn(arrow_label), run_time=0.5)
            self.play(FadeIn(resp_group, shift=DOWN * 0.2), run_time=0.6)
            self.play(FadeIn(note, shift=UP * 0.1), run_time=0.4)

        self.wait(0.3)
        self.clear()

    # ── Scene 4 — Streaming (tokens appearing one by one) ─────────────────

    def scene_streaming(self):
        heading = self.make_heading("串流回應 Streaming")

        # The prompt at top
        prompt_box = RoundedRectangle(
            corner_radius=0.12, width=8, height=0.7,
            fill_color=C_BLUE, fill_opacity=0.12,
            stroke_color=C_BLUE, stroke_width=1.5,
        ).shift(UP * 1.8)
        prompt_text = self.zh(
            "用戶：解釋咩係梯度下降？", font_size=20, color=C_BLUE
        ).move_to(prompt_box)

        # Token area — tokens will appear one by one
        tokens = [
            "梯度", "下降", "係", "一種", "優化", "算法", "，",
            "透過", "沿住", "loss", "減少", "嘅", "方向",
            "更新", "參數", "。",
        ]

        token_mobs = []
        x_offset = -5.5
        y_pos = 0.5
        line_width = 0.0
        for t in tokens:
            tm = self.zh(t, font_size=22, color=C_GREEN)
            char_width = len(t) * 0.28 + 0.15
            if line_width + char_width > 11.0:
                x_offset = -5.5
                y_pos -= 0.5
                line_width = 0.0
            tm.move_to(RIGHT * (x_offset + char_width / 2) + UP * y_pos)
            x_offset += char_width
            line_width += char_width
            token_mobs.append(tm)

        # TTFT indicator
        ttft_label = self.mono(
            "TTFT = 120ms", font_size=18, color=C_YELLOW
        ).shift(DOWN * 1.5 + LEFT * 3)

        # Timing bar
        time_bar_bg = Rectangle(
            width=6, height=0.25,
            fill_color=C_DIM, fill_opacity=0.3,
            stroke_color=C_DIM, stroke_width=1,
        ).shift(DOWN * 2.2 + RIGHT * 0.5)
        time_bar_fill = Rectangle(
            width=0.01, height=0.25,
            fill_color=C_GREEN, fill_opacity=0.6,
            stroke_width=0,
        ).align_to(time_bar_bg, LEFT).shift(DOWN * 2.2 + RIGHT * 0.5)

        time_label_start = self.mono("0ms", font_size=14, color=C_DIM)
        time_label_start.next_to(time_bar_bg, LEFT, buff=0.15)
        time_label_end = self.mono("500ms", font_size=14, color=C_DIM)
        time_label_end.next_to(time_bar_bg, RIGHT, buff=0.15)

        # SSE format example at bottom
        sse_label = self.mono(
            'data: {"delta": {"content": "梯度"}}',
            font_size=14, color=C_ORANGE,
        ).to_edge(DOWN, buff=0.35)

        with self.voiceover(
            text="Streaming 係改善用戶體驗嘅關鍵技術。"
            "冇 streaming 嘅話，用戶要等成個回應生成晒先睇到。"
            "如果有 500 個 token，每秒 30 個，即係要等成 17 秒。"
            "有咗 streaming，第一個 token 大約 120 毫秒就出到。"
            "之後每個 token 逐個逐個出現。"
            "用嘅係 Server-Sent Events 協議，簡稱 SSE。"
            "每個 token 包裝成一個 JSON chunk 傳送。"
            "ChatGPT 同 Claude 都係用呢個方法。"
            "所以雖然成個回應要幾秒鐘，"
            "但用戶一百毫秒就開始睇到文字，感覺好快。"
        ):
            self.play(Write(heading), run_time=0.6)
            self.play(FadeIn(prompt_box), FadeIn(prompt_text), run_time=0.4)
            self.play(
                FadeIn(time_bar_bg), FadeIn(time_label_start),
                FadeIn(time_label_end), run_time=0.3,
            )

            # First token with TTFT label
            self.play(FadeIn(token_mobs[0], shift=UP * 0.1), run_time=0.15)
            self.play(FadeIn(ttft_label), run_time=0.3)

            # Remaining tokens appear one by one
            for i, tm in enumerate(token_mobs[1:], start=1):
                progress = i / len(token_mobs)
                new_bar = Rectangle(
                    width=6 * progress, height=0.25,
                    fill_color=C_GREEN, fill_opacity=0.6,
                    stroke_width=0,
                ).align_to(time_bar_bg, LEFT).move_to(
                    time_bar_bg.get_left() + RIGHT * 3 * progress + DOWN * 0.0
                )
                self.play(
                    FadeIn(tm, shift=UP * 0.1),
                    Transform(time_bar_fill, new_bar),
                    run_time=0.12,
                )

            self.play(FadeIn(sse_label, shift=UP * 0.1), run_time=0.4)

        self.wait(0.3)
        self.clear()

    # ── Scene 5 — Batching (naive vs continuous) ──────────────────────────

    def scene_batching(self):
        heading = self.make_heading("Naive vs Continuous Batching")

        # --- Naive batching (left) ---
        naive_title = self.zh("Naive Batching", font_size=20, color=C_RED)
        naive_title.shift(UP * 1.5 + LEFT * 3.5)

        naive_bars = VGroup()
        bar_lengths = [4.0, 1.5, 3.5, 1.0]
        bar_colors = [C_BLUE, C_GREEN, C_ORANGE, C_PINK]
        for i, (length, col) in enumerate(zip(bar_lengths, bar_colors)):
            bar = Rectangle(
                width=length, height=0.35,
                fill_color=col, fill_opacity=0.5,
                stroke_color=col, stroke_width=1.5,
            )
            waste = Rectangle(
                width=4.0 - length, height=0.35,
                fill_color=C_RED, fill_opacity=0.15,
                stroke_color=C_RED, stroke_width=0.5,
            )
            bar.move_to(LEFT * 3.5 + UP * (0.7 - i * 0.5))
            bar.align_to(LEFT * 5.5, LEFT)
            waste.next_to(bar, RIGHT, buff=0)
            naive_bars.add(bar, waste)

        naive_label = self.zh(
            "短請求等長請求\nGPU 利用率 ~55%",
            font_size=14, color=C_RED,
        ).next_to(naive_bars, DOWN, buff=0.2)

        # --- Continuous batching (right) ---
        cont_title = self.zh("Continuous Batching", font_size=20, color=C_GREEN)
        cont_title.shift(UP * 1.5 + RIGHT * 3.5)

        cont_bars = VGroup()
        # Show that short requests are replaced immediately
        cont_data = [
            (4.0, C_BLUE),
            (1.5, C_GREEN),
            (3.5, C_ORANGE),
            (1.0, C_PINK),
        ]
        for i, (length, col) in enumerate(cont_data):
            bar = Rectangle(
                width=length, height=0.35,
                fill_color=col, fill_opacity=0.5,
                stroke_color=col, stroke_width=1.5,
            )
            bar.move_to(RIGHT * 3.5 + UP * (0.7 - i * 0.5))
            bar.align_to(RIGHT * 1.5, LEFT)
            cont_bars.add(bar)

        # New requests fill gaps
        fill_data = [
            (1, 2.0, C_CYAN),
            (3, 2.5, C_PURPLE),
        ]
        fill_bars = VGroup()
        for row, length, col in fill_data:
            bar = Rectangle(
                width=length, height=0.35,
                fill_color=col, fill_opacity=0.5,
                stroke_color=col, stroke_width=1.5,
            )
            parent = cont_bars[row * 1]  # Adjust to target correct bar
            bar.next_to(cont_bars[row], RIGHT, buff=0.05)
            fill_bars.add(bar)

        cont_label = self.zh(
            "空位立即填入新請求\nGPU 利用率 ~90%",
            font_size=14, color=C_GREEN,
        ).next_to(cont_bars, DOWN, buff=0.3)

        # Divider
        divider = DashedLine(
            UP * 2, DOWN * 2.5,
            dash_length=0.1, color=C_DIM,
        )

        # Bottom comparison
        comparison = VGroup(
            self.zh("吞吐量提升 2-4 倍", font_size=22, color=C_YELLOW),
            self.zh(
                "vLLM / TGI / llama.cpp 全部使用 Continuous Batching",
                font_size=16, color=C_DIM,
            ),
        ).arrange(DOWN, buff=0.15).to_edge(DOWN, buff=0.4)

        with self.voiceover(
            text="吞吐量優化最重要嘅技術就係 continuous batching。"
            "左邊係 naive batching。"
            "四個請求一齊開始，但佢哋長度唔同。"
            "短嘅請求做完之後要等長嘅請求。"
            "紅色部分就係浪費嘅 GPU 計算。"
            "利用率只有大約 55%。"
            "右邊係 continuous batching。"
            "短請求一做完，佢嘅位就即刻俾新請求填滿。"
            "GPU 利用率可以去到 90%。"
            "同樣嘅硬件，吞吐量可以提升 2 到 4 倍。"
            "vLLM、TGI、llama.cpp 全部都有呢個功能。"
        ):
            self.play(Write(heading), run_time=0.6)
            self.play(FadeIn(divider), run_time=0.3)
            self.play(FadeIn(naive_title), run_time=0.3)
            for bar in naive_bars:
                self.play(GrowFromEdge(bar, LEFT), run_time=0.15)
            self.play(FadeIn(naive_label), run_time=0.4)

            self.play(FadeIn(cont_title), run_time=0.3)
            for bar in cont_bars:
                self.play(GrowFromEdge(bar, LEFT), run_time=0.15)
            for bar in fill_bars:
                self.play(GrowFromEdge(bar, LEFT), run_time=0.2)
            self.play(FadeIn(cont_label), run_time=0.4)

            self.play(FadeIn(comparison, shift=UP * 0.2), run_time=0.5)

        self.wait(0.3)
        self.clear()

    # ── Scene 6 — Cost (GPU pricing table) ────────────────────────────────

    def scene_cost(self):
        heading = self.make_heading("成本估算 GPU 定價")

        # Table header
        header = self.mono(
            "GPU      VRAM   $/hr   7B tok/s  $/1M tok",
            font_size=16, color=C_YELLOW,
        ).shift(UP * 1.5)

        header_line = Line(
            header.get_left() + DOWN * 0.2,
            header.get_right() + DOWN * 0.2,
            color=C_YELLOW, stroke_width=1,
        )

        # Table rows
        rows_data = [
            ("T4       16GB   $0.50      25    $5.56", C_DIM),
            ("L4       24GB   $0.80      45    $4.94", C_DIM),
            ("A10G     24GB   $1.20      55    $6.06", C_WHITE),
            ("L40S     48GB   $2.50      80    $8.68", C_WHITE),
            ("A100     80GB   $4.00     120    $9.26", C_GREEN),
            ("H100     80GB   $8.00     200   $11.11", C_CYAN),
        ]

        row_mobs = VGroup()
        for i, (text, col) in enumerate(rows_data):
            row = self.mono(text, font_size=15, color=col)
            row.shift(UP * (0.9 - i * 0.45))
            row_mobs.add(row)

        # Highlight box around cheapest per-token
        best_box = SurroundingRectangle(
            row_mobs[1], color=C_GREEN, buff=0.08,
            stroke_width=1.5, fill_opacity=0.08,
        )
        best_label = self.zh(
            "L4 每百萬 token 最平！", font_size=16, color=C_GREEN
        ).next_to(best_box, RIGHT, buff=0.3)

        # Comparison with API
        api_compare = VGroup(
            self.zh("對比 API 定價：", font_size=18, color=C_ORANGE),
            self.mono("GPT-4o input:  $2.50 / 1M tokens", font_size=15, color=C_WHITE),
            self.mono("Self-hosted 7B: $0.62 / 1M tokens (L4, batch 8)", font_size=15, color=C_GREEN),
            self.zh("高流量情況下，自建可以慳 75% 以上", font_size=16, color=C_YELLOW),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.15).to_edge(DOWN, buff=0.35)
        api_compare.shift(LEFT * 0.5)

        with self.voiceover(
            text="成本估算對選擇硬件好重要。"
            "呢個表列出咗六款常用嘅雲端 GPU。"
            "T4 最平，每小時 0.5 美金，但速度最慢。"
            "H100 最快，但每小時要 8 美金。"
            "如果計每百萬 token 嘅成本，"
            "L4 其實係最平嘅，大約 5 美金。"
            "但呢個係單一請求嘅成本。"
            "如果用 batch size 8，成本會除以 8。"
            "L4 用 batch 8 嘅話，每百萬 token 只需要大約 0.62 美金。"
            "對比 GPT-4o 嘅 2.5 美金 input，"
            "自建服務喺高流量情況下可以慳超過 75%。"
        ):
            self.play(Write(heading), run_time=0.6)
            self.play(FadeIn(header), FadeIn(header_line), run_time=0.4)
            for row in row_mobs:
                self.play(FadeIn(row, shift=DOWN * 0.1), run_time=0.2)
            self.play(Create(best_box), FadeIn(best_label), run_time=0.5)
            for item in api_compare:
                self.play(FadeIn(item, shift=RIGHT * 0.2), run_time=0.35)

        self.wait(0.3)
        self.clear()

    # ── Scene 7 — Summary ────────────────────────────────────────────────

    def scene_summary(self):
        heading = self.make_heading("總結", font_size=44)

        bullets = VGroup(
            self.zh("•  vLLM / TGI / llama.cpp：按需求揀框架", font_size=24, color=C_BLUE),
            self.zh("•  OpenAI 兼容 API：換 URL 即可切換後端", font_size=24, color=C_GREEN),
            self.zh("•  Streaming SSE：100ms 就見到第一個 token", font_size=24, color=C_ORANGE),
            self.zh("•  Continuous Batching：吞吐量提升 2-4 倍", font_size=24, color=C_PINK),
            self.zh("•  成本計算：token/秒/美元 係關鍵指標", font_size=24, color=C_CYAN),
            self.zh("•  安全護欄：輸入過濾 + 輸出過濾 + 速率限制", font_size=24, color=C_PURPLE),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.35).move_to(ORIGIN)

        with self.voiceover(
            text="總結一下今日學咗嘅重點。"
            "第一，三大框架各有優勢，按需求揀。"
            "vLLM 最快，TGI 最穩，llama.cpp 最靈活。"
            "第二，用 OpenAI 兼容 API 格式，換個 URL 就可以切換後端。"
            "第三，Streaming 用 SSE 協議，100 毫秒就出到第一個 token。"
            "第四，Continuous batching 可以將吞吐量提升 2 到 4 倍。"
            "第五，成本計算嘅核心指標係 token 每秒每美元。"
            "高流量情況下自建服務比 API 平好多。"
            "第六，安全護欄係必須嘅。"
            "輸入過濾防 prompt injection，"
            "輸出過濾防 PII 洩漏，速率限制防濫用。"
            "掌握咗呢啲，你就可以將自己嘅 LLM 部署上線喇！"
        ):
            self.play(Write(heading), run_time=0.6)
            for b in bullets:
                self.play(FadeIn(b, shift=RIGHT * 0.3), run_time=0.5)

        self.wait(0.5)
        self.clear()

    # ── Scene 8 — Outro ──────────────────────────────────────────────────

    def scene_outro(self):
        thanks = self.zh("多謝收睇！", font_size=52, color=C_YELLOW)
        next_lesson = self.zh(
            "下一課：RAG（檢索增強生成）",
            font_size=28, color=C_CYAN,
        ).next_to(thanks, DOWN, buff=0.6)

        with self.voiceover(
            text="多謝收睇！下一課我哋會學 RAG。"
            "即係 Retrieval-Augmented Generation，檢索增強生成。"
            "包括 embedding 模型、向量數據庫、"
            "chunking 策略同 retrieve-then-generate 流程。"
            "教你點樣用外部知識嚟增強 LLM 嘅回答。"
            "記得繼續跟住學喇！"
        ):
            self.play(Write(thanks), run_time=1)
            self.play(FadeIn(next_lesson, shift=UP * 0.2), run_time=0.8)

        self.wait(1)
        self.play(FadeOut(thanks), FadeOut(next_lesson), run_time=0.8)
