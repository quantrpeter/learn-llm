"""
Lesson 14 – Quantization & Inference Optimization
Manim CE + Cantonese voiceover (Edge TTS)

Render (1080p):
    cd "lesson 14/video"
    manim render -qh scene.py QuantizationExplainer

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


class QuantizationExplainer(VoiceoverScene):
    """Eight scenes explaining quantization & inference optimization in Cantonese."""

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
        self.scene_why_quantize()
        self.scene_quantization_demo()
        self.scene_gptq()
        self.scene_speculative()
        self.scene_kv_cache()
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
        title = self.zh("量化優化", font_size=56, color=C_BLUE)
        sub_en = self.en(
            "Quantization & Inference Optimization", font_size=24, color=C_WHITE
        ).next_to(title, DOWN, buff=0.4)
        sub_zh = self.zh(
            "第十四課", font_size=36, color=C_ORANGE
        ).next_to(sub_en, DOWN, buff=0.3)
        producer = self.zh(
            "制片人：Peter", font_size=24, color=C_DIM
        ).next_to(sub_zh, DOWN, buff=0.7)

        with self.voiceover(
            text="大家好！歡迎嚟到第十四課。"
            "上一課我哋學咗偏好對齊，用 DPO 訓練模型分辨好壞。"
            "但係一個對齊好嘅模型，如果太大，跑唔起，都冇用。"
            "今日我哋會學量化同推理優化。"
            "包括 INT8、INT4 量化點樣將 7B 模型壓縮到可以喺手提電腦跑。"
            "仲有 GPTQ 量化、Speculative Decoding、"
            "同埋 KV Cache 優化。"
        ):
            self.play(Write(title), run_time=1.5)
            self.play(FadeIn(sub_en, shift=UP * 0.2), run_time=0.8)
            self.play(FadeIn(sub_zh, shift=UP * 0.2), run_time=0.8)
            self.play(FadeIn(producer, shift=UP * 0.2), run_time=0.5)

        self.wait(0.3)
        self.clear()

    # ── Scene 2 — Why Quantize (memory bars FP32 → INT4) ─────────────────

    def scene_why_quantize(self):
        heading = self.make_heading("點解要量化？7B 模型記憶體")

        bar_data = [
            ("FP32", 28.0, C_RED),
            ("FP16", 14.0, C_ORANGE),
            ("INT8",  7.0, C_YELLOW),
            ("INT4",  3.5, C_GREEN),
        ]

        max_gb = 28.0
        max_bar_width = 9.0
        bar_height = 0.65
        bars = VGroup()
        labels = VGroup()
        size_labels = VGroup()

        for i, (name, gb, color) in enumerate(bar_data):
            bar_w = (gb / max_gb) * max_bar_width
            bar = Rectangle(
                width=bar_w, height=bar_height,
                fill_color=color, fill_opacity=0.5,
                stroke_color=color, stroke_width=2,
            )
            bar.align_to(LEFT * 4.5, LEFT)
            bar.shift(DOWN * (i * 1.05) + UP * 1.2)
            bars.add(bar)

            lbl = self.mono(name, font_size=22, color=color)
            lbl.next_to(bar, LEFT, buff=0.25)
            labels.add(lbl)

            sz = self.mono(f"{gb:.1f} GB", font_size=20, color=C_WHITE)
            sz.next_to(bar, RIGHT, buff=0.2)
            size_labels.add(sz)

        # Annotation: laptop threshold
        threshold_x = LEFT * 4.5 + RIGHT * (6.0 / max_gb) * max_bar_width
        threshold_line = DashedLine(
            threshold_x + UP * 2.0, threshold_x + DOWN * 2.6,
            color=C_CYAN, stroke_width=2, dash_length=0.1,
        )
        threshold_label = self.zh(
            "6GB GPU", font_size=16, color=C_CYAN
        ).next_to(threshold_line, UP, buff=0.1)

        # Compression factor labels
        compress_label = self.zh(
            "FP32 → INT4 = 8x 壓縮", font_size=22, color=C_GREEN
        ).to_edge(DOWN, buff=0.6)

        with self.voiceover(
            text="一個 7B 參數嘅模型，用 FP32 儲存需要 28 GB。"
            "呢個大小，好多 GPU 都放唔落。"
            "如果用 FP16，減半到 14 GB，但仲係好大。"
            "INT8 量化，每個權重用 8 個 bit，只需要 7 GB。"
            "INT4 量化，每個權重用 4 個 bit，只需要 3.5 GB。"
            "3.5 GB 可以放入一張 6GB 嘅消費級 GPU，"
            "甚至可以喺手提電腦嘅 CPU 上面跑。"
            "呢個就係 8 倍嘅壓縮。"
            "而且質量下降非常少，通常 perplexity 只升少少。"
        ):
            self.play(Write(heading), run_time=0.6)
            for i in range(4):
                self.play(
                    GrowFromEdge(bars[i], LEFT),
                    FadeIn(labels[i]),
                    FadeIn(size_labels[i]),
                    run_time=0.6,
                )
            self.play(
                Create(threshold_line), FadeIn(threshold_label),
                run_time=0.5,
            )
            self.play(FadeIn(compress_label, shift=UP * 0.2), run_time=0.5)

        self.wait(0.3)
        self.clear()

    # ── Scene 3 — Quantization Demo (number line showing error) ──────────

    def scene_quantization_demo(self):
        heading = self.make_heading("量化過程：浮點數 → 整數")

        line_start, line_end = LEFT * 5, RIGHT * 5
        line_mob = Line(line_start, line_end, color=C_DIM).shift(UP * 0.8)
        tick_labels = VGroup()
        for val in range(-3, 4):
            x_pos = line_mob.get_start() + (val + 3) / 6 * (line_mob.get_end() - line_mob.get_start())
            tick = Line(x_pos + UP * 0.1, x_pos + DOWN * 0.1, color=C_DIM)
            lbl = self.mono(str(val), font_size=16, color=C_DIM).next_to(tick, DOWN, buff=0.1)
            tick_labels.add(tick, lbl)
        line = VGroup(line_mob, tick_labels).shift(ORIGIN)

        class _FakeLine:
            def __init__(self, mob):
                self._mob = mob
            def n2p(self, val):
                s = self._mob[0].get_start()
                e = self._mob[0].get_end()
                return s + (val + 3) / 6 * (e - s)
        _line = _FakeLine(line)

        # Original float values (dots on number line)
        float_values = [-2.3, -0.7, 0.4, 1.1, 2.6]
        float_dots = VGroup()
        float_labels = VGroup()
        for val in float_values:
            dot = Dot(_line.n2p(val), color=C_BLUE, radius=0.1)
            lbl = self.mono(f"{val}", font_size=16, color=C_BLUE)
            lbl.next_to(dot, UP, buff=0.15)
            float_dots.add(dot)
            float_labels.add(lbl)

        # Quantized positions (snapped to nearest integer)
        quant_values = [round(v) for v in float_values]
        quant_dots = VGroup()
        quant_labels = VGroup()
        for i, (orig, qval) in enumerate(zip(float_values, quant_values)):
            dot = Dot(_line.n2p(qval), color=C_ORANGE, radius=0.1)
            dot.shift(DOWN * 0.3)
            lbl = self.mono(f"{qval}", font_size=16, color=C_ORANGE)
            lbl.next_to(dot, DOWN, buff=0.15)
            quant_dots.add(dot)
            quant_labels.add(lbl)

        # Error arrows
        error_arrows = VGroup()
        for i, (orig, qval) in enumerate(zip(float_values, quant_values)):
            if abs(orig - qval) > 0.05:
                arrow = Arrow(
                    _line.n2p(orig) + DOWN * 0.1,
                    _line.n2p(qval) + UP * 0.1 + DOWN * 0.3,
                    color=C_RED, stroke_width=2, buff=0.05,
                    max_tip_length_to_length_ratio=0.3,
                )
                error_arrows.add(arrow)

        # Error explanation
        error_box = RoundedRectangle(
            corner_radius=0.12, width=10, height=1.6,
            fill_color=C_RED, fill_opacity=0.08,
            stroke_color=C_RED, stroke_width=1.5,
        ).shift(DOWN * 2.0)
        error_text = VGroup(
            self.zh("量化誤差 = 原始值 - 量化值", font_size=20, color=C_RED),
            self.zh(
                "INT8 有 256 個級別，INT4 只有 16 個級別",
                font_size=18, color=C_ORANGE,
            ),
            self.zh(
                "級別越少，誤差越大，但記憶體越少",
                font_size=18, color=C_YELLOW,
            ),
        ).arrange(DOWN, buff=0.15).move_to(error_box)

        with self.voiceover(
            text="量化嘅過程好簡單。"
            "我哋有一啲浮點數值嘅權重，"
            "例如負 2.3、負 0.7、0.4、1.1、2.6。"
            "量化就係將佢哋映射到最近嘅整數。"
            "負 2.3 變成負 2，0.4 變成 0，2.6 變成 3。"
            "中間嘅差距就係量化誤差。"
            "紅色箭嘴顯示咗呢個誤差。"
            "INT8 有 256 個級別，誤差好細。"
            "INT4 只有 16 個級別，誤差大啲，但記憶體慳好多。"
            "好嘅量化方法嘅目標就係盡量減少呢個誤差。"
        ):
            self.play(Write(heading), run_time=0.6)
            self.play(Create(line), run_time=0.8)
            for d, l in zip(float_dots, float_labels):
                self.play(FadeIn(d), FadeIn(l), run_time=0.3)
            for d, l in zip(quant_dots, quant_labels):
                self.play(FadeIn(d), FadeIn(l), run_time=0.3)
            for arrow in error_arrows:
                self.play(GrowArrow(arrow), run_time=0.3)
            self.play(FadeIn(error_box), FadeIn(error_text), run_time=0.6)

        self.wait(0.3)
        self.clear()

    # ── Scene 4 — GPTQ (calibration flow) ────────────────────────────────

    def scene_gptq(self):
        heading = self.make_heading("GPTQ 量化流程")

        # Step boxes arranged vertically
        step1_box = self.make_box("1. 收集校準數據", C_BLUE, width=4.5, height=0.9, font_size=20)
        step1_detail = self.zh(
            "128-256 個真實文本樣本", font_size=14, color=C_DIM
        ).next_to(step1_box, RIGHT, buff=0.3)
        step1 = VGroup(step1_box, step1_detail)

        step2_box = self.make_box("2. 計算 Hessian", C_ORANGE, width=4.5, height=0.9, font_size=20)
        step2_detail = self.zh(
            "邊啲權重對輸出影響最大？", font_size=14, color=C_DIM
        ).next_to(step2_box, RIGHT, buff=0.3)
        step2 = VGroup(step2_box, step2_detail)

        step3_box = self.make_box("3. 逐列量化 + 補償", C_GREEN, width=4.5, height=0.9, font_size=20)
        step3_detail = self.zh(
            "量化一列，將誤差分配到其他列", font_size=14, color=C_DIM
        ).next_to(step3_box, RIGHT, buff=0.3)
        step3 = VGroup(step3_box, step3_detail)

        step4_box = self.make_box("4. 儲存量化權重", C_PINK, width=4.5, height=0.9, font_size=20)
        step4_detail = self.zh(
            "INT4 權重 + scale + zero_point", font_size=14, color=C_DIM
        ).next_to(step4_box, RIGHT, buff=0.3)
        step4 = VGroup(step4_box, step4_detail)

        steps = VGroup(step1, step2, step3, step4).arrange(DOWN, buff=0.5)
        steps.shift(DOWN * 0.2)

        # Arrows between steps
        arrows = VGroup()
        for i in range(3):
            src = [step1_box, step2_box, step3_box][i]
            dst = [step2_box, step3_box, step4_box][i]
            arrow = Arrow(
                src[0].get_bottom(), dst[0].get_top(),
                buff=0.15, color=C_YELLOW, stroke_width=2.5,
            )
            arrows.add(arrow)

        # Comparison at bottom
        compare_box = RoundedRectangle(
            corner_radius=0.12, width=10, height=0.7,
            fill_color=C_CYAN, fill_opacity=0.10,
            stroke_color=C_CYAN, stroke_width=1.5,
        ).to_edge(DOWN, buff=0.3)
        compare_text = self.zh(
            "GPTQ 比直接量化誤差低好多，因為佢識得邊啲權重重要",
            font_size=18, color=C_CYAN,
        ).move_to(compare_box)

        with self.voiceover(
            text="GPTQ 係最常用嘅後訓練量化方法。"
            "第一步，收集校準數據。"
            "大概 128 到 256 個真實文本樣本就夠。"
            "第二步，計算 Hessian 矩陣。"
            "呢個矩陣話俾我哋知，邊啲權重對輸出影響最大。"
            "影響大嘅權重要小心處理，影響細嘅可以粗略量化。"
            "第三步，逐列量化同補償。"
            "量化一列之後，將產生嘅誤差分配到其他未量化嘅列。"
            "呢個就係 GPTQ 嘅核心技巧。"
            "第四步，儲存量化之後嘅權重，"
            "包括 INT4 嘅值加上 scale 同 zero point。"
            "GPTQ 比直接量化嘅誤差低好多。"
        ):
            self.play(Write(heading), run_time=0.6)
            for i, step in enumerate([step1, step2, step3, step4]):
                self.play(FadeIn(step, shift=DOWN * 0.2), run_time=0.5)
                if i < 3:
                    self.play(GrowArrow(arrows[i]), run_time=0.3)
            self.play(FadeIn(compare_box), FadeIn(compare_text), run_time=0.5)

        self.wait(0.3)
        self.clear()

    # ── Scene 5 — Speculative Decoding (draft + verify) ──────────────────

    def scene_speculative(self):
        heading = self.make_heading("Speculative Decoding 投機解碼")

        # Standard generation (slow) — top section
        slow_label = self.zh(
            "傳統生成（慢）", font_size=20, color=C_RED
        ).shift(UP * 1.8 + LEFT * 3)

        slow_boxes = VGroup()
        for i in range(4):
            box = RoundedRectangle(
                corner_radius=0.1, width=1.8, height=0.7,
                fill_color=C_RED, fill_opacity=0.15,
                stroke_color=C_RED, stroke_width=2,
            )
            txt = self.zh(f"大模型\n第{i+1}步", font_size=14, color=C_RED).move_to(box)
            slow_boxes.add(VGroup(box, txt))
        slow_boxes.arrange(RIGHT, buff=0.3)
        slow_boxes.next_to(slow_label, DOWN, buff=0.2)

        slow_arrows = VGroup()
        for i in range(3):
            arrow = Arrow(
                slow_boxes[i][0].get_right(), slow_boxes[i + 1][0].get_left(),
                buff=0.05, color=C_DIM, stroke_width=2,
            )
            slow_arrows.add(arrow)

        # Speculative decoding (fast) — bottom section
        fast_label = self.zh(
            "投機解碼（快）", font_size=20, color=C_GREEN
        ).shift(DOWN * 0.3 + LEFT * 3)

        # Draft phase
        draft_boxes = VGroup()
        for i in range(4):
            box = RoundedRectangle(
                corner_radius=0.1, width=1.1, height=0.6,
                fill_color=C_BLUE, fill_opacity=0.15,
                stroke_color=C_BLUE, stroke_width=2,
            )
            txt = self.zh(f"小模型", font_size=12, color=C_BLUE).move_to(box)
            draft_boxes.add(VGroup(box, txt))
        draft_boxes.arrange(RIGHT, buff=0.15)
        draft_boxes.next_to(fast_label, DOWN, buff=0.2).shift(LEFT * 1.2)

        draft_label = self.zh(
            "草稿", font_size=16, color=C_BLUE
        ).next_to(draft_boxes, LEFT, buff=0.2)

        # Verify phase
        verify_box = RoundedRectangle(
            corner_radius=0.1, width=5.2, height=0.6,
            fill_color=C_GREEN, fill_opacity=0.15,
            stroke_color=C_GREEN, stroke_width=2,
        )
        verify_txt = self.zh(
            "大模型一次過驗證全部", font_size=16, color=C_GREEN
        ).move_to(verify_box)
        verify_group = VGroup(verify_box, verify_txt)
        verify_group.next_to(draft_boxes, DOWN, buff=0.4)

        verify_arrow = Arrow(
            draft_boxes.get_bottom(), verify_box.get_top(),
            buff=0.1, color=C_YELLOW, stroke_width=2.5,
        )

        # Result with checkmarks
        result_items = VGroup(
            self.mono("T1 ✓", font_size=20, color=C_GREEN),
            self.mono("T2 ✓", font_size=20, color=C_GREEN),
            self.mono("T3 ✓", font_size=20, color=C_GREEN),
            self.mono("T4 ✗", font_size=20, color=C_RED),
        ).arrange(RIGHT, buff=0.5)
        result_items.next_to(verify_group, DOWN, buff=0.3)

        speedup_label = self.zh(
            "3 個 token 只用咗 1 次大模型！ ~2-3x 加速",
            font_size=20, color=C_YELLOW,
        ).to_edge(DOWN, buff=0.35)

        with self.voiceover(
            text="Speculative Decoding 係一個好聰明嘅加速方法。"
            "傳統生成要每個 token 都經過大模型，好慢。"
            "四個 token 就要四次大模型嘅推理。"
            "投機解碼嘅做法唔同。"
            "首先用一個細嘅草稿模型，好快咁生成四個候選 token。"
            "然後將呢四個 token 一次過餵入大模型驗證。"
            "大模型可以平行處理所有 token，所以只需要一次推理。"
            "如果草稿模型猜得準，大部分 token 都會被接受。"
            "呢度三個 token 被接受，第四個被拒絕。"
            "即係用一次大模型嘅成本，攞到三個 token。"
            "理論上可以有 2 到 3 倍嘅加速。"
            "最重要係：輸出質量完全唔變！"
        ):
            self.play(Write(heading), run_time=0.6)
            self.play(FadeIn(slow_label), run_time=0.3)
            for i in range(4):
                self.play(FadeIn(slow_boxes[i], shift=RIGHT * 0.2), run_time=0.3)
                if i < 3:
                    self.play(GrowArrow(slow_arrows[i]), run_time=0.15)
            self.play(FadeIn(fast_label), FadeIn(draft_label), run_time=0.3)
            for box in draft_boxes:
                self.play(FadeIn(box, shift=RIGHT * 0.1), run_time=0.2)
            self.play(GrowArrow(verify_arrow), run_time=0.3)
            self.play(FadeIn(verify_group, shift=DOWN * 0.1), run_time=0.5)
            for item in result_items:
                self.play(FadeIn(item, shift=UP * 0.1), run_time=0.25)
            self.play(FadeIn(speedup_label, shift=UP * 0.1), run_time=0.5)

        self.wait(0.3)
        self.clear()

    # ── Scene 6 — KV Cache (GQA / MQA comparison) ────────────────────────

    def scene_kv_cache(self):
        heading = self.make_heading("KV Cache 優化：MHA vs GQA vs MQA")

        # Three columns for MHA, GQA, MQA
        col_data = [
            ("MHA", "32 KV heads", C_RED, [1.0] * 8),
            ("GQA", "8 KV heads", C_ORANGE, [1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]),
            ("MQA", "1 KV head", C_GREEN, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ]

        columns = VGroup()
        for col_idx, (name, subtitle, color, pattern) in enumerate(col_data):
            # Header
            header = self.zh(name, font_size=26, color=color)
            sub = self.zh(subtitle, font_size=14, color=C_DIM)
            sub.next_to(header, DOWN, buff=0.08)

            # Query heads row (all 8 active)
            q_label = self.zh("Q heads", font_size=12, color=C_DIM)
            q_boxes = VGroup()
            for i in range(8):
                box = Square(
                    side_length=0.3,
                    fill_color=C_BLUE, fill_opacity=0.5,
                    stroke_color=C_BLUE, stroke_width=1,
                )
                q_boxes.add(box)
            q_boxes.arrange(RIGHT, buff=0.05)
            q_label.next_to(q_boxes, LEFT, buff=0.15)
            q_row = VGroup(q_label, q_boxes)

            # KV heads row (varies by attention type)
            kv_label = self.zh("KV heads", font_size=12, color=C_DIM)
            kv_boxes = VGroup()
            for i in range(8):
                opacity = 0.5 if pattern[i] > 0 else 0.08
                stroke = color if pattern[i] > 0 else C_DIM
                box = Square(
                    side_length=0.3,
                    fill_color=color, fill_opacity=opacity,
                    stroke_color=stroke, stroke_width=1,
                )
                kv_boxes.add(box)
            kv_boxes.arrange(RIGHT, buff=0.05)
            kv_label.next_to(kv_boxes, LEFT, buff=0.15)
            kv_row = VGroup(kv_label, kv_boxes)

            col = VGroup(header, sub, q_row, kv_row).arrange(DOWN, buff=0.25)
            columns.add(col)

        columns.arrange(RIGHT, buff=0.8).shift(UP * 0.3)

        # Memory comparison table
        mem_box = RoundedRectangle(
            corner_radius=0.12, width=10.5, height=1.8,
            fill_color=BLACK, fill_opacity=0.3,
            stroke_color=C_DIM, stroke_width=1,
        ).shift(DOWN * 2.2)

        mem_title = self.zh(
            "7B 模型 KV Cache (4096 tokens, FP16)", font_size=18, color=C_WHITE
        ).move_to(mem_box).shift(UP * 0.55)

        mem_entries = VGroup(
            self.mono("MHA: 2.00 GB", font_size=18, color=C_RED),
            self.mono("GQA: 0.50 GB", font_size=18, color=C_ORANGE),
            self.mono("MQA: 0.06 GB", font_size=18, color=C_GREEN),
        ).arrange(RIGHT, buff=1.0).move_to(mem_box).shift(DOWN * 0.15)

        savings_label = self.zh(
            "GQA 比 MHA 慳 75% 記憶體，質量幾乎一樣",
            font_size=16, color=C_CYAN,
        ).move_to(mem_box).shift(DOWN * 0.65)

        with self.voiceover(
            text="最後我哋講 KV Cache 優化。"
            "生成 token 嘅時候，每一層都要儲存 Key 同 Value。"
            "呢個 KV Cache 對長序列嚟講，記憶體消耗好大。"
            "MHA 即係 Multi-Head Attention。"
            "每個 query head 都有自己嘅 KV head。"
            "32 個 heads 就要 32 組 KV，記憶體最大。"
            "GQA 即係 Grouped Query Attention。"
            "每幾個 query heads 共享一個 KV head。"
            "例如 32 個 query heads 共享 8 個 KV heads。"
            "MQA 即係 Multi-Query Attention。"
            "所有 query heads 共享一個 KV head。"
            "KV Cache 最細，但質量可能稍為下降。"
            "以 7B 模型生成 4096 個 token 為例。"
            "MHA 需要 2 GB 嘅 KV Cache。"
            "GQA 只需要 0.5 GB，慳咗 75%。"
            "MQA 更加只需要 0.06 GB。"
            "LLaMA-2 70B 就係用 GQA，平衡質量同效率。"
        ):
            self.play(Write(heading), run_time=0.6)
            for col in columns:
                self.play(FadeIn(col, shift=DOWN * 0.2), run_time=0.6)
            self.play(FadeIn(mem_box), FadeIn(mem_title), run_time=0.4)
            for entry in mem_entries:
                self.play(FadeIn(entry, shift=UP * 0.1), run_time=0.35)
            self.play(FadeIn(savings_label), run_time=0.4)

        self.wait(0.3)
        self.clear()

    # ── Scene 7 — Summary ────────────────────────────────────────────────

    def scene_summary(self):
        heading = self.make_heading("總結", font_size=44)

        bullets = VGroup(
            self.zh("•  7B FP32 = 28GB → INT4 = 3.5GB（8倍壓縮）", font_size=24, color=C_GREEN),
            self.zh("•  Per-channel 比 per-tensor 量化精度高好多", font_size=24, color=C_ORANGE),
            self.zh("•  GPTQ 用 Hessian 揾出重要權重，減少誤差", font_size=24, color=C_PINK),
            self.zh("•  Speculative Decoding：草稿+驗證，2-3x 加速", font_size=24, color=C_CYAN),
            self.zh("•  GQA 共享 KV heads，慳 75% cache 記憶體", font_size=24, color=C_PURPLE),
            self.zh("•  量化 + 快取優化 = 消費級硬件跑大模型", font_size=24, color=C_YELLOW),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.35).move_to(ORIGIN)

        with self.voiceover(
            text="總結一下今日學咗嘅重點。"
            "第一，量化可以將 7B 模型由 28 GB 壓縮到 3.5 GB。"
            "第二，per-channel 量化比 per-tensor 精度高好多。"
            "第三，GPTQ 用 Hessian 矩陣揾出重要嘅權重，精確咁量化。"
            "第四，Speculative Decoding 用草稿加驗證嘅方式，加速 2 到 3 倍。"
            "第五，GQA 共享 KV heads，慳 75% 嘅 cache 記憶體。"
            "第六，結合呢啲技術，消費級硬件都可以跑大型語言模型。"
            "掌握咗呢啲優化技巧，你就可以部署自己嘅 LLM 喇！"
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
            "下一課：Deployment（部署同服務）",
            font_size=28, color=C_CYAN,
        ).next_to(thanks, DOWN, buff=0.6)

        with self.voiceover(
            text="多謝收睇！下一課我哋會學部署。"
            "即係點樣將你訓練好嘅 LLM 變成一個可以用嘅 API。"
            "包括 vLLM 服務、streaming 回覆、"
            "同埋生產環境嘅監控。"
            "記得繼續跟住學喇！"
        ):
            self.play(Write(thanks), run_time=1)
            self.play(FadeIn(next_lesson, shift=UP * 0.2), run_time=0.8)

        self.wait(1)
        self.play(FadeOut(thanks), FadeOut(next_lesson), run_time=0.8)
