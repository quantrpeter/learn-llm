"""
Lesson 11 – Evaluation & Benchmarking
Manim CE + Cantonese voiceover (Edge TTS)

Render (1080p):
    cd "lesson 11/video"
    manim render -qh scene.py EvaluationExplainer

Dependencies in the same directory:
    - edge_tts_service.py  (copy from any previous lesson's video/)
    - wallpaper1.jpg       (copy from lesson 1/video/)
"""

import math

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


class EvaluationExplainer(VoiceoverScene):
    """Single scene explaining LLM evaluation & benchmarking in Cantonese."""

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
        self.scene_perplexity()
        self.scene_benchmarks()
        self.scene_few_shot()
        self.scene_human_eval()
        self.scene_contamination()
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

    # ── Scene 1 — Title / Intro ──────────────────────────────────────────

    def scene_intro(self):
        title = self.zh("模型評估", font_size=56, color=C_BLUE)
        sub_en = self.en(
            "Evaluation & Benchmarking", font_size=24, color=C_WHITE
        ).next_to(title, DOWN, buff=0.4)
        sub_zh = self.zh(
            "第十一課", font_size=36, color=C_ORANGE
        ).next_to(sub_en, DOWN, buff=0.3)
        producer = self.zh(
            "制片人：Peter", font_size=24, color=C_DIM
        ).next_to(sub_zh, DOWN, buff=0.7)

        with self.voiceover(
            text="大家好！歡迎嚟到第十一課。"
            "上幾課我哋學咗點樣訓練同生成文本。"
            "但係你個模型到底有幾好呢？"
            "今日我哋會學點樣評估一個語言模型。"
            "包括 Perplexity、標準 benchmark、"
            "few-shot 評估、人工評估同埋污染檢測。"
        ):
            self.play(Write(title), run_time=1.5)
            self.play(FadeIn(sub_en, shift=UP * 0.2), run_time=0.8)
            self.play(FadeIn(sub_zh, shift=UP * 0.2), run_time=0.8)
            self.play(FadeIn(producer, shift=UP * 0.2), run_time=0.5)

        self.wait(0.3)
        self.clear()

    # ── Scene 2 — Perplexity (visual scale low=good high=bad) ────────────

    def scene_perplexity(self):
        heading = self.make_heading("Perplexity 困惑度")

        # Formula
        formula = self.mono(
            "PPL = exp( avg cross-entropy loss )",
            font_size=22, color=C_CYAN,
        ).next_to(heading, DOWN, buff=0.5)

        # Horizontal scale bar
        scale_width = 10.0
        scale_bar = Rectangle(
            width=scale_width, height=0.5,
            fill_color=C_DIM, fill_opacity=0.2,
            stroke_color=C_DIM, stroke_width=1.5,
        ).shift(DOWN * 0.3)

        # Gradient overlay: green on left, red on right
        green_bar = Rectangle(
            width=scale_width / 2, height=0.5,
            fill_color=C_GREEN, fill_opacity=0.3,
            stroke_width=0,
        ).align_to(scale_bar, LEFT).align_to(scale_bar, UP)
        red_bar = Rectangle(
            width=scale_width / 2, height=0.5,
            fill_color=C_RED, fill_opacity=0.3,
            stroke_width=0,
        ).align_to(scale_bar, RIGHT).align_to(scale_bar, UP)

        good_label = self.zh(
            "好 (低)", font_size=20, color=C_GREEN
        ).next_to(scale_bar, DOWN, buff=0.15).align_to(scale_bar, LEFT).shift(RIGHT * 0.4)
        bad_label = self.zh(
            "差 (高)", font_size=20, color=C_RED
        ).next_to(scale_bar, DOWN, buff=0.15).align_to(scale_bar, RIGHT).shift(LEFT * 0.4)

        # Model markers on the scale
        models = [
            ("LLaMA-2 70B", 3.3,    C_GREEN),
            ("GPT-3 175B",  15.0,   C_CYAN),
            ("GPT-2 124M",  29.0,   C_ORANGE),
            ("Random",      50000,  C_RED),
        ]

        markers = VGroup()
        log_min, log_max = math.log(1.5), math.log(60000)
        for name, ppl, color in models:
            log_ppl = math.log(max(ppl, 1.5))
            t = (log_ppl - log_min) / (log_max - log_min)
            x_pos = scale_bar.get_left()[0] + t * scale_width

            dot = Dot(
                point=[x_pos, scale_bar.get_center()[1], 0],
                color=color, radius=0.12,
            )
            lbl_name = self.mono(
                name, font_size=14, color=color,
            )
            lbl_ppl = self.mono(
                f"PPL={ppl}", font_size=12, color=C_DIM,
            )
            lbl_group = VGroup(lbl_name, lbl_ppl).arrange(DOWN, buff=0.05)
            lbl_group.next_to(dot, UP, buff=0.2)
            markers.add(VGroup(dot, lbl_group))

        # Intuition box
        intuition = VGroup(
            RoundedRectangle(
                corner_radius=0.12, width=10, height=1.2,
                fill_color=C_PURPLE, fill_opacity=0.12,
                stroke_color=C_PURPLE, stroke_width=1.5,
            ),
            self.zh(
                "PPL = 25 意味住模型每步都喺 25 個同樣可能嘅 token 之間揀",
                font_size=20, color=C_PURPLE,
            ),
            self.zh(
                "越低越好 — 代表模型對正確 token 嘅信心越高",
                font_size=18, color=C_DIM,
            ),
        )
        intuition[1:].arrange(DOWN, buff=0.1)
        VGroup(*intuition[1:]).move_to(intuition[0])
        intuition.to_edge(DOWN, buff=0.35)

        with self.voiceover(
            text="Perplexity 係語言模型最基本嘅評估指標。"
            "公式好簡單：PPL 等於 e 嘅平均 cross-entropy loss 次方。"
            "Perplexity 越低，代表模型越好。"
            "一個 PPL 等於 25 嘅模型，"
            "就好似每一步都喺 25 個同樣可能嘅 token 之間揀。"
            "隨機模型嘅 PPL 等於詞彙量，大概五萬。"
            "GPT-2 有 124M 參數，PPL 大約 29。"
            "GPT-3 有 1750 億參數，PPL 大約 15。"
            "LLaMA-2 70B 嘅 PPL 低到 3.3，非常強。"
            "記住：越低越好，永遠係。"
        ):
            self.play(Write(heading), run_time=0.6)
            self.play(FadeIn(formula, shift=DOWN * 0.2), run_time=0.6)
            self.play(
                FadeIn(scale_bar), FadeIn(green_bar), FadeIn(red_bar),
                FadeIn(good_label), FadeIn(bad_label),
                run_time=0.6,
            )
            for marker in markers:
                self.play(FadeIn(marker, scale=1.3), run_time=0.5)
            self.play(FadeIn(intuition, shift=UP * 0.2), run_time=0.6)

        self.wait(0.3)
        self.clear()

    # ── Scene 3 — Benchmarks (benchmark cards) ───────────────────────────

    def scene_benchmarks(self):
        heading = self.make_heading("標準 Benchmark")

        benchmarks = [
            ("HellaSwag",   "常識推理",   "句子補完", C_GREEN),
            ("ARC",         "科學問答",   "多項選擇", C_ORANGE),
            ("MMLU",        "57 科知識",  "多項選擇", C_CYAN),
            ("TruthfulQA",  "抗誤導",     "真假判斷", C_PINK),
        ]

        cards = VGroup()
        for name, desc_zh, format_zh, color in benchmarks:
            card = RoundedRectangle(
                corner_radius=0.15, width=2.8, height=2.8,
                fill_color=color, fill_opacity=0.12,
                stroke_color=color, stroke_width=2,
            )
            title = self.mono(
                name, font_size=22, color=color,
            ).move_to(card).shift(UP * 0.7)
            desc = self.zh(
                desc_zh, font_size=18, color=C_WHITE,
            ).move_to(card)
            fmt = self.zh(
                format_zh, font_size=16, color=C_DIM,
            ).move_to(card).shift(DOWN * 0.5)
            cards.add(VGroup(card, title, desc, fmt))

        cards.arrange(RIGHT, buff=0.3).shift(UP * 0.2)

        # How scoring works
        scoring = VGroup(
            self.zh("評分方法：模型對每個選項計算 log P，選最高分嘅答案",
                     font_size=20, color=C_YELLOW),
            self.zh("準確率 = 正確選擇數 / 總題數",
                     font_size=18, color=C_DIM),
        ).arrange(DOWN, buff=0.12)
        scoring.to_edge(DOWN, buff=0.5)

        with self.voiceover(
            text="Perplexity 只能話俾你知模型預測下一個 token 嘅能力。"
            "但係模型識唔識推理？有冇知識？"
            "呢啲就要靠標準 benchmark 嚟測試。"
            "HellaSwag 測試常識推理，要求模型揀最合理嘅句子補完。"
            "ARC 係小學科學題，用多項選擇。"
            "MMLU 涵蓋 57 個科目，由中學數學到專業法律。"
            "TruthfulQA 測試模型會唔會被常見嘅謬誤誤導。"
            "所有 benchmark 嘅核心方法都一樣："
            "模型對每個選項計算 log probability，然後揀最高分嗰個。"
        ):
            self.play(Write(heading), run_time=0.6)
            for card in cards:
                self.play(FadeIn(card, shift=DOWN * 0.2), run_time=0.5)
            for s in scoring:
                self.play(FadeIn(s, shift=UP * 0.1), run_time=0.4)

        self.wait(0.3)
        self.clear()

    # ── Scene 4 — Few-Shot (prompt template visualization) ───────────────

    def scene_few_shot(self):
        heading = self.make_heading("Few-Shot 評估")

        # 0-shot prompt
        zero_title = self.zh("0-shot", font_size=22, color=C_RED)
        zero_box = RoundedRectangle(
            corner_radius=0.15, width=5.0, height=3.0,
            fill_color=C_RED, fill_opacity=0.08,
            stroke_color=C_RED, stroke_width=2,
        )

        zero_content = VGroup(
            self.mono("Question: What is H2O?", font_size=14, color=C_WHITE),
            self.mono("  A. Salt", font_size=14, color=C_DIM),
            self.mono("  B. Water", font_size=14, color=C_GREEN),
            self.mono("  C. Oil", font_size=14, color=C_DIM),
            self.mono("  D. Gold", font_size=14, color=C_DIM),
            self.mono("Answer:", font_size=14, color=C_YELLOW),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
        zero_content.move_to(zero_box)
        zero_title.next_to(zero_box, UP, buff=0.15)
        zero_group = VGroup(zero_box, zero_title, zero_content)

        # 5-shot prompt
        five_title = self.zh("5-shot", font_size=22, color=C_GREEN)
        five_box = RoundedRectangle(
            corner_radius=0.15, width=5.0, height=3.0,
            fill_color=C_GREEN, fill_opacity=0.08,
            stroke_color=C_GREEN, stroke_width=2,
        )

        five_content = VGroup(
            self.mono("Q: Capital of France?", font_size=12, color=C_DIM),
            self.mono("Answer: B (Paris)", font_size=12, color=C_DIM),
            self.mono("... (4 more examples) ...", font_size=12, color=C_DIM),
            self.mono("", font_size=6, color=C_DIM),
            self.mono("Question: What is H2O?", font_size=14, color=C_WHITE),
            self.mono("  A. Salt  B. Water", font_size=14, color=C_WHITE),
            self.mono("  C. Oil   D. Gold", font_size=14, color=C_WHITE),
            self.mono("Answer:", font_size=14, color=C_YELLOW),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.08)
        five_content.move_to(five_box)
        five_title.next_to(five_box, UP, buff=0.15)
        five_group = VGroup(five_box, five_title, five_content)

        prompts = VGroup(zero_group, five_group).arrange(RIGHT, buff=0.6)
        prompts.shift(UP * 0.1)

        # Arrow between them
        arrow = Arrow(
            zero_box.get_right() + RIGHT * 0.1,
            five_box.get_left() + LEFT * 0.1,
            buff=0.05, color=C_YELLOW, stroke_width=3,
        )
        arrow_label = self.zh(
            "加入示範", font_size=16, color=C_YELLOW,
        ).next_to(arrow, UP, buff=0.1)

        # Bottom note
        note = VGroup(
            RoundedRectangle(
                corner_radius=0.12, width=10, height=0.7,
                fill_color=C_CYAN, fill_opacity=0.12,
                stroke_color=C_CYAN, stroke_width=1.5,
            ),
            self.zh(
                "GPT-3 論文發現：5-shot 可以將某啲 task 嘅準確率提高一倍",
                font_size=20, color=C_CYAN,
            ),
        )
        note[1].move_to(note[0])
        note.to_edge(DOWN, buff=0.35)

        with self.voiceover(
            text="Few-shot 評估係 GPT-3 引入嘅重要概念。"
            "0-shot 就係直接問問題，唔俾任何例子。"
            "5-shot 就係喺問題之前俾五個例子。"
            "模型從未 fine-tune 過，佢純粹從 prompt 入面嘅例子學到答題模式。"
            "呢個就叫 in-context learning。"
            "GPT-3 論文發現，5-shot 可以將某啲任務嘅準確率"
            "提高一倍，對比 0-shot。"
            "較細嘅模型特別依賴 few-shot 例子。"
            "大模型就算 0-shot 都可以做得唔錯。"
        ):
            self.play(Write(heading), run_time=0.6)
            self.play(FadeIn(zero_group, shift=DOWN * 0.2), run_time=0.7)
            self.play(
                GrowArrow(arrow), FadeIn(arrow_label),
                run_time=0.4,
            )
            self.play(FadeIn(five_group, shift=DOWN * 0.2), run_time=0.7)
            self.play(FadeIn(note, shift=UP * 0.2), run_time=0.6)

        self.wait(0.3)
        self.clear()

    # ── Scene 5 — Human Eval (A/B comparison) ────────────────────────────

    def scene_human_eval(self):
        heading = self.make_heading("人工評估")

        # A/B test visualization
        prompt_box = RoundedRectangle(
            corner_radius=0.15, width=8, height=0.8,
            fill_color=C_YELLOW, fill_opacity=0.12,
            stroke_color=C_YELLOW, stroke_width=1.5,
        ).shift(UP * 1.5)
        prompt_label = self.zh(
            "用戶提問：「點樣學機器學習？」",
            font_size=20, color=C_YELLOW,
        ).move_to(prompt_box)

        # Two response columns
        resp_a_box = RoundedRectangle(
            corner_radius=0.15, width=4.5, height=2.2,
            fill_color=C_GREEN, fill_opacity=0.1,
            stroke_color=C_GREEN, stroke_width=2,
        )
        resp_a_title = self.mono(
            "Model A (anonymous)", font_size=16, color=C_GREEN,
        ).next_to(resp_a_box, UP, buff=0.1)
        resp_a_text = self.zh(
            "建議從線性代數同\n概率論開始，然後\n用 PyTorch 做練習...",
            font_size=16, color=C_WHITE,
        ).move_to(resp_a_box)
        resp_a = VGroup(resp_a_box, resp_a_title, resp_a_text)

        resp_b_box = RoundedRectangle(
            corner_radius=0.15, width=4.5, height=2.2,
            fill_color=C_ORANGE, fill_opacity=0.1,
            stroke_color=C_ORANGE, stroke_width=2,
        )
        resp_b_title = self.mono(
            "Model B (anonymous)", font_size=16, color=C_ORANGE,
        ).next_to(resp_b_box, UP, buff=0.1)
        resp_b_text = self.zh(
            "機器學習好複雜，\n你需要好多數學\n同編程知識...",
            font_size=16, color=C_WHITE,
        ).move_to(resp_b_box)
        resp_b = VGroup(resp_b_box, resp_b_title, resp_b_text)

        responses = VGroup(resp_a, resp_b).arrange(RIGHT, buff=0.4)
        responses.shift(DOWN * 0.3)

        # Vote arrows pointing down
        vote_text = self.zh(
            "用戶投票 → Elo 評分更新", font_size=20, color=C_PURPLE,
        )

        # Elo box
        elo_box = VGroup(
            RoundedRectangle(
                corner_radius=0.12, width=10, height=1.2,
                fill_color=C_PURPLE, fill_opacity=0.12,
                stroke_color=C_PURPLE, stroke_width=1.5,
            ),
            self.zh(
                "Chatbot Arena: 20 萬+ 投票 → 最可信嘅 LLM 排名",
                font_size=20, color=C_PURPLE,
            ),
            self.zh(
                "GPT-4o ~1287 | Claude 3.5 ~1271 | Gemini 1.5 ~1260",
                font_size=16, color=C_DIM,
            ),
        )
        elo_box[1:].arrange(DOWN, buff=0.08)
        VGroup(*elo_box[1:]).move_to(elo_box[0])
        elo_box.to_edge(DOWN, buff=0.3)

        vote_text.next_to(elo_box, UP, buff=0.25)

        with self.voiceover(
            text="自動化嘅 benchmark 只能測試特定能力。"
            "但用戶真正關心嘅係：個模型答得好唔好？"
            "呢個時候就需要人工評估。"
            "A/B testing 嘅做法係："
            "用戶提交一個問題，兩個匿名模型各自回答。"
            "用戶揀邊個答案好啲，或者打和。"
            "然後用 Elo 評分系統更新排名，同國際象棋嘅排名一樣。"
            "LMSYS 嘅 Chatbot Arena 收集咗超過 20 萬條投票。"
            "呢個係目前最可信嘅 LLM 排名。"
            "因為係盲測，模型冇辦法作弊。"
        ):
            self.play(Write(heading), run_time=0.6)
            self.play(FadeIn(prompt_box), FadeIn(prompt_label), run_time=0.5)
            self.play(
                FadeIn(resp_a, shift=DOWN * 0.2),
                FadeIn(resp_b, shift=DOWN * 0.2),
                run_time=0.6,
            )
            self.play(FadeIn(vote_text, shift=DOWN * 0.1), run_time=0.4)
            self.play(FadeIn(elo_box, shift=UP * 0.2), run_time=0.6)

        self.wait(0.3)
        self.clear()

    # ── Scene 6 — Contamination (overlap detection) ──────────────────────

    def scene_contamination(self):
        heading = self.make_heading("污染檢測 Contamination")

        # Training data box
        train_box = RoundedRectangle(
            corner_radius=0.15, width=5.2, height=2.5,
            fill_color=C_BLUE, fill_opacity=0.1,
            stroke_color=C_BLUE, stroke_width=2,
        )
        train_title = self.zh(
            "訓練數據", font_size=20, color=C_BLUE,
        ).next_to(train_box, UP, buff=0.1)
        train_lines = VGroup(
            self.mono("doc 1: the cat sat on ...", font_size=14, color=C_DIM),
            self.mono("doc 2: water boils at ...", font_size=14, color=C_RED),
            self.mono("doc 3: transformers use ...", font_size=14, color=C_DIM),
            self.mono("doc 4: capital of france", font_size=14, color=C_RED),
            self.mono("doc 5: deep learning ...", font_size=14, color=C_DIM),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
        train_lines.move_to(train_box)
        train_group = VGroup(train_box, train_title, train_lines)

        # Benchmark box
        bench_box = RoundedRectangle(
            corner_radius=0.15, width=5.2, height=2.5,
            fill_color=C_ORANGE, fill_opacity=0.1,
            stroke_color=C_ORANGE, stroke_width=2,
        )
        bench_title = self.zh(
            "Benchmark 題目", font_size=20, color=C_ORANGE,
        ).next_to(bench_box, UP, buff=0.1)
        bench_lines = VGroup(
            self.mono("Q1: water boils at ...", font_size=14, color=C_RED),
            self.mono("Q2: airspeed velocity ...", font_size=14, color=C_GREEN),
            self.mono("Q3: capital of france", font_size=14, color=C_RED),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
        bench_lines.move_to(bench_box)
        bench_group = VGroup(bench_box, bench_title, bench_lines)

        boxes = VGroup(train_group, bench_group).arrange(RIGHT, buff=0.6)
        boxes.shift(UP * 0.2)

        # Overlap arrows
        overlap_arrows = VGroup()
        arrow1 = Arrow(
            bench_lines[0].get_left() + LEFT * 0.1,
            train_lines[1].get_right() + RIGHT * 0.1,
            buff=0.05, color=C_RED, stroke_width=2.5,
        )
        arrow2 = Arrow(
            bench_lines[2].get_left() + LEFT * 0.1,
            train_lines[3].get_right() + RIGHT * 0.1,
            buff=0.05, color=C_RED, stroke_width=2.5,
        )
        overlap_arrows.add(arrow1, arrow2)

        # Detection method
        method = VGroup(
            RoundedRectangle(
                corner_radius=0.12, width=10.5, height=1.2,
                fill_color=C_YELLOW, fill_opacity=0.12,
                stroke_color=C_YELLOW, stroke_width=1.5,
            ),
            self.zh(
                "檢測方法：計算 n-gram 重疊率（n=10~13）",
                font_size=20, color=C_YELLOW,
            ),
            self.zh(
                "重疊率 > 50% → 有污染風險 → 移除或標記",
                font_size=18, color=C_DIM,
            ),
        )
        method[1:].arrange(DOWN, buff=0.08)
        VGroup(*method[1:]).move_to(method[0])
        method.to_edge(DOWN, buff=0.3)

        with self.voiceover(
            text="如果模型背咗 benchmark 嘅答案，"
            "高分就冇意義。呢個叫做污染。"
            "例如訓練數據入面有 benchmark 嘅原文。"
            "模型唔係識推理，只係記住咗答案。"
            "最常用嘅檢測方法係 n-gram 重疊。"
            "將 benchmark 同訓練數據嘅文本拆成 n-gram，"
            "通常 n 等於 10 到 13。"
            "如果重疊率超過一半，就有污染風險。"
            "GPT-3 用 13-gram，LLaMA 用 10-gram 做檢測。"
            "負責任嘅研究會報告清潔前同清潔後嘅分數。"
        ):
            self.play(Write(heading), run_time=0.6)
            self.play(FadeIn(train_group, shift=DOWN * 0.2), run_time=0.6)
            self.play(FadeIn(bench_group, shift=DOWN * 0.2), run_time=0.6)
            for a in overlap_arrows:
                self.play(GrowArrow(a), run_time=0.4)
            self.play(FadeIn(method, shift=UP * 0.2), run_time=0.6)

        self.wait(0.3)
        self.clear()

    # ── Scene 7 — Summary ────────────────────────────────────────────────

    def scene_summary(self):
        heading = self.make_heading("總結", font_size=44)

        bullets = VGroup(
            self.zh("•  Perplexity：exp(CE loss)，越低越好", font_size=24, color=C_GREEN),
            self.zh("•  Benchmark：HellaSwag / ARC / MMLU / TruthfulQA", font_size=24, color=C_ORANGE),
            self.zh("•  Few-Shot：0-shot 到 5-shot 提升準確率", font_size=24, color=C_CYAN),
            self.zh("•  Eval Harness：標準化評估工具（lm-eval）", font_size=24, color=C_PINK),
            self.zh("•  人工評估：A/B 測試、Elo 評分、Chatbot Arena", font_size=24, color=C_PURPLE),
            self.zh("•  污染檢測：n-gram 重疊確保分數可信", font_size=24, color=C_YELLOW),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.35).move_to(ORIGIN)

        with self.voiceover(
            text="總結一下今日學咗嘅六個重點。"
            "第一，Perplexity 係語言模型最基本嘅指標，越低越好。"
            "第二，標準 benchmark 測試推理、知識同抗誤導能力。"
            "第三，Few-shot 評估唔使 fine-tune，直接喺 prompt 俾例子。"
            "第四，lm-evaluation-harness 係標準化嘅評估框架。"
            "第五，人工評估用 A/B 測試同 Elo 評分捕捉 benchmark 遺漏嘅質素。"
            "第六，污染檢測確保 benchmark 分數係真實嘅，唔係記憶出嚟嘅。"
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
            "下一課：Fine-Tuning（微調）",
            font_size=28, color=C_CYAN,
        ).next_to(thanks, DOWN, buff=0.6)

        with self.voiceover(
            text="多謝收睇！下一課我哋會學 Fine-Tuning，"
            "即係點樣將一個 base model 變成"
            "識聽指令嘅 assistant。"
            "包括 SFT、LoRA 同 QLoRA。"
            "記得繼續跟住學喇！"
        ):
            self.play(Write(thanks), run_time=1)
            self.play(FadeIn(next_lesson, shift=UP * 0.2), run_time=0.8)

        self.wait(1)
        self.play(FadeOut(thanks), FadeOut(next_lesson), run_time=0.8)
