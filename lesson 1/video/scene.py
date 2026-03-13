"""
Lesson 1 – Mathematical Foundations for LLMs
Manim CE + Cantonese voiceover (Edge TTS)

Render (1080p):
    cd "lesson 1/video"
    manim render -qh scene.py MathFoundations
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


class MathFoundations(VoiceoverScene):
    """Single scene explaining LLM math foundations in Cantonese."""

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
        self.scene_why_math()
        self.scene_vectors()
        self.scene_dot_product()
        self.scene_matrix_multiply()
        self.scene_gradients()
        self.scene_cross_entropy()
        self.scene_softmax()
        self.scene_log_sum_exp()
        self.scene_forward_pass()
        self.scene_summary()

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

    def make_box(self, label, color, width=2.5, height=0.8):
        rect = RoundedRectangle(
            corner_radius=0.15,
            width=width,
            height=height,
            fill_color=color,
            fill_opacity=0.25,
            stroke_color=color,
        )
        txt = self.zh(label, font_size=24, color=color).move_to(rect)
        return VGroup(rect, txt)

    def make_heading(self, text, color=C_BLUE, font_size=40):
        return self.zh(text, font_size=font_size, color=color).to_edge(UP, buff=0.5)

    # ── Scene 1 — Title / Intro ──────────────────────────────────────────

    def scene_intro(self):
        title = self.zh("LLM 數學基礎", font_size=56, color=C_BLUE)
        sub_en = self.en(
            "Mathematical Foundations for LLMs", font_size=24, color=C_WHITE
        ).next_to(title, DOWN, buff=0.4)
        sub_zh = self.zh(
            "第一課", font_size=36, color=C_ORANGE
        ).next_to(sub_en, DOWN, buff=0.3)
        producer = self.zh(
            "制片人：Peter", font_size=24, color=C_DIM
        ).next_to(sub_zh, DOWN, buff=0.7)

        with self.voiceover(
            text="大家好！今日我哋嚟學 LLM 嘅數學基礎。"
            "好多人以為要好叻數學先可以學 AI，其實唔使㗎。"
            "今日我哋會學五個最重要嘅概念，學完之後你就會明白 LLM 入面嘅數學點運作。"
        ):
            self.play(Write(title), run_time=1.5)
            self.play(FadeIn(sub_en, shift=UP * 0.2), run_time=0.8)
            self.play(FadeIn(sub_zh, shift=UP * 0.2), run_time=0.8)
            self.play(FadeIn(producer, shift=UP * 0.2), run_time=0.5)

        self.wait(0.3)
        self.clear()

    # ── Scene 2 — Why Math Matters ───────────────────────────────────────

    def scene_why_math(self):
        heading = self.make_heading("點解要學數學？")

        topics = [
            ("1. 線性代數", "向量、矩陣乘法", C_GREEN),
            ("2. 微積分", "梯度、反向傳播", C_ORANGE),
            ("3. 概率論", "交叉熵、困惑度", C_PINK),
            ("4. Softmax", "將分數變成概率", C_PURPLE),
            ("5. 數值穩定性", "Log-Sum-Exp 技巧", C_CYAN),
        ]

        items = VGroup()
        for label, detail, col in topics:
            line = VGroup(
                self.zh(label, font_size=28, color=col),
                self.zh(f"— {detail}", font_size=22, color=C_DIM),
            ).arrange(RIGHT, buff=0.3)
            items.add(line)
        items.arrange(DOWN, aligned_edge=LEFT, buff=0.4).move_to(ORIGIN)

        with self.voiceover(
            text="今日嘅五個主題分別係："
            "第一，線性代數，包括向量同矩陣乘法。"
            "第二，微積分，主要係梯度同反向傳播。"
            "第三，概率論，包括交叉熵同困惑度。"
            "第四，Softmax 函數，將分數變成概率。"
            "第五，數值穩定性，即係 Log Sum Exp 技巧。"
            "呢五個概念係理解 LLM 嘅基礎，我哋逐個嚟睇。"
        ):
            self.play(Write(heading), run_time=0.6)
            for item in items:
                self.play(FadeIn(item, shift=RIGHT * 0.3), run_time=0.5)

        self.wait(0.3)
        self.clear()

    # ── Scene 3 — Vectors ────────────────────────────────────────────────

    def scene_vectors(self):
        heading = self.make_heading("向量 Vectors")

        plane = NumberPlane(
            x_range=[-1, 5, 1],
            y_range=[-1, 4, 1],
            x_length=5,
            y_length=3.5,
            background_line_style={"stroke_color": C_DIM, "stroke_opacity": 0.3},
            axis_config={"color": C_DIM},
        ).shift(DOWN * 0.3)

        v1_arrow = Arrow(
            plane.c2p(0, 0), plane.c2p(1, 2),
            buff=0, color=C_GREEN, stroke_width=4,
        )
        v1_label = self.mono(
            "v1 = [1, 2]", font_size=22, color=C_GREEN,
        ).next_to(v1_arrow.get_end(), RIGHT, buff=0.2)

        v2_arrow = Arrow(
            plane.c2p(0, 0), plane.c2p(4, 3),
            buff=0, color=C_ORANGE, stroke_width=4,
        )
        v2_label = self.mono(
            "v2 = [4, 3]", font_size=22, color=C_ORANGE,
        ).next_to(v2_arrow.get_end(), RIGHT, buff=0.2)

        explain = self.zh(
            "向量 = 一組數字，代表空間入面嘅方向同長度",
            font_size=22, color=C_WHITE,
        ).to_edge(DOWN, buff=0.5)

        with self.voiceover(
            text="首先嚟睇向量。向量就係一組數字，代表空間入面嘅一個方向同長度。"
            "喺 LLM 入面，每一個詞語都會變成一個向量，叫做 embedding。"
            "例如呢度有兩個向量，v1 等於 1 comma 2，v2 等於 4 comma 3。"
            "佢哋嘅方向同長度都唔同。"
        ):
            self.play(Write(heading), run_time=0.6)
            self.play(Create(plane), run_time=1)
            self.play(GrowArrow(v1_arrow), Write(v1_label), run_time=1)
            self.play(GrowArrow(v2_arrow), Write(v2_label), run_time=1)
            self.play(FadeIn(explain, shift=UP * 0.2), run_time=0.5)

        self.wait(0.3)
        self.clear()

    # ── Scene 4 — Dot Product ────────────────────────────────────────────

    def scene_dot_product(self):
        heading = self.make_heading("點積 Dot Product")

        formula = self.mono(
            "a · b = Σ(ai × bi)", font_size=32, color=C_WHITE,
        ).shift(UP * 1.8)

        example_title = self.zh("例子：", font_size=24, color=C_DIM).shift(UP * 0.7 + LEFT * 3)
        a_vec = self.mono("[1, 2, 3]", font_size=28, color=C_GREEN).next_to(example_title, RIGHT, buff=0.3)
        dot_sym = self.mono("·", font_size=28, color=C_WHITE).next_to(a_vec, RIGHT, buff=0.2)
        b_vec = self.mono("[4, 5, 6]", font_size=28, color=C_ORANGE).next_to(dot_sym, RIGHT, buff=0.2)

        step1 = self.mono(
            "= 1×4 + 2×5 + 3×6", font_size=26, color=C_WHITE,
        ).shift(DOWN * 0.0)

        step2 = VGroup(
            self.mono("= 4 + 10 + 18 = ", font_size=26, color=C_WHITE),
            self.mono("32", font_size=30, color=C_YELLOW),
        ).arrange(RIGHT, buff=0.1).shift(DOWN * 0.7)

        why_box = RoundedRectangle(
            corner_radius=0.15, width=10, height=0.8,
            fill_color=C_BLUE, fill_opacity=0.15, stroke_color=C_BLUE,
        ).shift(DOWN * 2.0)
        why_text = self.zh(
            "Attention 分數 = Query 同 Key 嘅點積",
            font_size=24, color=C_BLUE,
        ).move_to(why_box)

        with self.voiceover(
            text="點積，英文叫 dot product，係兩個向量對應位置相乘再加埋。"
            "例如 1 comma 2 comma 3 同 4 comma 5 comma 6 嘅點積，"
            "就係 1 乘 4 加 2 乘 5 加 3 乘 6 等於 32。"
            "點解呢個重要呢？因為 LLM 入面嘅 Attention 機制，"
            "就係靠計算 Query 同 Key 嘅點積嚟決定邊啲詞語最相關。"
            "點積越大，代表兩個向量越相似。"
        ):
            self.play(Write(heading), run_time=0.6)
            self.play(Write(formula), run_time=1.0)
            self.play(FadeIn(example_title), Write(a_vec), Write(dot_sym), Write(b_vec), run_time=1)
            self.play(Write(step1), run_time=1)
            self.play(Write(step2), run_time=1)
            self.play(FadeIn(why_box), Write(why_text), run_time=1)

        self.wait(0.3)
        self.clear()

    # ── Scene 5 — Matrix Multiplication ──────────────────────────────────

    def scene_matrix_multiply(self):
        heading = self.make_heading("矩陣乘法 Matrix Multiply")

        mat_a_vals = [["1", "2"], ["3", "4"]]
        mat_b_vals = [["5", "6"], ["7", "8"]]
        mat_c_vals = [["19", "22"], ["43", "50"]]

        def make_matrix_display(values, color, label_text):
            rows = []
            for row_vals in values:
                entries = [self.mono(v, font_size=28, color=color) for v in row_vals]
                row_group = VGroup(*entries).arrange(RIGHT, buff=0.5)
                rows.append(row_group)
            grid = VGroup(*rows).arrange(DOWN, buff=0.3)

            bracket_l = self.mono("[", font_size=40, color=color).next_to(grid, LEFT, buff=0.15)
            bracket_r = self.mono("]", font_size=40, color=color).next_to(grid, RIGHT, buff=0.15)
            label = self.mono(label_text, font_size=28, color=color)

            mat = VGroup(bracket_l, grid, bracket_r)
            label.next_to(mat, UP, buff=0.2)
            return VGroup(label, mat)

        mat_a = make_matrix_display(mat_a_vals, C_GREEN, "A")
        mat_b = make_matrix_display(mat_b_vals, C_ORANGE, "B")
        mat_c = make_matrix_display(mat_c_vals, C_YELLOW, "C")

        times_sym = self.mono("×", font_size=36, color=C_WHITE)
        eq_sym = self.mono("=", font_size=36, color=C_WHITE)

        equation = VGroup(mat_a, times_sym, mat_b, eq_sym, mat_c).arrange(
            RIGHT, buff=0.4
        ).shift(UP * 0.3)

        detail = VGroup(
            self.mono("C[0,0] = 1×5 + 2×7 = ", font_size=24, color=C_WHITE),
            self.mono("19", font_size=28, color=C_YELLOW),
        ).arrange(RIGHT, buff=0.1).shift(DOWN * 1.8)

        why_box = RoundedRectangle(
            corner_radius=0.15, width=10, height=0.8,
            fill_color=C_BLUE, fill_opacity=0.15, stroke_color=C_BLUE,
        ).shift(DOWN * 2.8)
        why_text = self.zh(
            "每一層 Neural Network = 矩陣乘法 + bias",
            font_size=24, color=C_BLUE,
        ).move_to(why_box)

        with self.voiceover(
            text="矩陣乘法就係 neural network 嘅核心運算。"
            "矩陣 A 乘矩陣 B，每個結果嘅元素都係一個點積。"
            "例如 C 嘅第一行第一列，就係 A 嘅第一行同 B 嘅第一列嘅點積，"
            "即係 1 乘 5 加 2 乘 7 等於 19。"
            "喺 LLM 入面，每一層都做一次矩陣乘法。"
            "一個有 32 層嘅模型就做 32 次矩陣乘法。"
        ):
            self.play(Write(heading), run_time=0.6)
            self.play(
                FadeIn(mat_a, shift=RIGHT * 0.2),
                FadeIn(times_sym),
                FadeIn(mat_b, shift=LEFT * 0.2),
                run_time=1.2,
            )
            self.play(FadeIn(eq_sym), run_time=0.3)
            self.play(FadeIn(mat_c, scale=1.1), run_time=1)
            self.play(Write(detail), run_time=1)
            self.play(FadeIn(why_box), Write(why_text), run_time=1)

        self.wait(0.3)
        self.clear()

    # ── Scene 6 — Gradients & Backprop ───────────────────────────────────

    def scene_gradients(self):
        heading = self.make_heading("梯度同反向傳播")

        func_label = self.mono("f(x) = x²", font_size=36, color=C_WHITE).shift(UP * 2)

        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[-1, 9, 2],
            x_length=5,
            y_length=3,
            axis_config={"color": C_DIM},
        ).shift(DOWN * 0.2)

        curve = axes.plot(lambda x: x**2, x_range=[-3, 3], color=C_BLUE)

        dot_x = 2.0
        dot = Dot(axes.c2p(dot_x, dot_x**2), color=C_YELLOW, radius=0.1)
        tangent = axes.plot(
            lambda x: 2 * dot_x * (x - dot_x) + dot_x**2,
            x_range=[dot_x - 1.5, dot_x + 1.5],
            color=C_RED,
        )

        grad_label = self.mono(
            "df/dx = 2x = 4", font_size=24, color=C_RED,
        ).next_to(dot, UR, buff=0.3)

        arrow_down = Arrow(
            axes.c2p(dot_x, dot_x**2),
            axes.c2p(dot_x - 1.2, (dot_x - 1.2)**2 + 0.3),
            buff=0.1, color=C_GREEN, stroke_width=3,
        )
        step_label = self.zh(
            "向低處行", font_size=20, color=C_GREEN,
        ).next_to(arrow_down, DOWN, buff=0.15)

        chain_text = self.mono(
            "dL/dw = dL/dy · dy/dw", font_size=28, color=C_WHITE,
        ).shift(DOWN * 2.8)
        chain_label = self.zh(
            "Chain Rule — 反向傳播嘅核心",
            font_size=22, color=C_DIM,
        ).next_to(chain_text, DOWN, buff=0.2)

        with self.voiceover(
            text="Neural network 點樣學嘢呢？靠梯度。"
            "梯度就係函數嘅斜率，話俾你知邊個方向 loss 會減少。"
            "例如 f of x 等於 x 嘅平方，喺 x 等於 2 嗰度，梯度等於 4。"
            "正數嘅梯度代表要向左行先會去到低處。"
            "喺 neural network 入面，我哋用 chain rule，"
            "即係鏈式法則，將梯度由最後一層傳返去第一層。"
            "呢個過程就叫做反向傳播，英文叫 backpropagation。"
        ):
            self.play(Write(heading), run_time=0.6)
            self.play(Write(func_label), run_time=0.8)
            self.play(Create(axes), Create(curve), run_time=1.2)
            self.play(FadeIn(dot, scale=1.5), run_time=0.5)
            self.play(Create(tangent), Write(grad_label), run_time=1)
            self.play(GrowArrow(arrow_down), FadeIn(step_label), run_time=0.8)
            self.play(Write(chain_text), FadeIn(chain_label), run_time=1.2)

        self.wait(0.3)
        self.clear()

    # ── Scene 7 — Cross-Entropy Loss ─────────────────────────────────────

    def scene_cross_entropy(self):
        heading = self.make_heading("交叉熵 Cross-Entropy")

        formula = self.mono(
            "H(p,q) = -Σ p(x)·log q(x)", font_size=28, color=C_WHITE,
        ).shift(UP * 2.0)

        vocab_words = ["the", "a", "cat", "dog", "sat"]
        target_idx = 2
        good_probs = [0.05, 0.05, 0.80, 0.05, 0.05]

        bars = VGroup()
        for i, (word, prob) in enumerate(zip(vocab_words, good_probs)):
            bar_h = max(prob * 3.5, 0.05)
            color = C_GREEN if i == target_idx else C_DIM
            bar = Rectangle(
                width=0.7, height=bar_h,
                fill_color=color, fill_opacity=0.7,
                stroke_color=color, stroke_width=1,
            )
            label = self.en(word, font_size=18, color=C_WHITE)
            prob_label = self.en(f"{prob:.0%}", font_size=16, color=color)
            bars.add(VGroup(bar, label, prob_label))

        bars.arrange(RIGHT, buff=0.4, aligned_edge=DOWN)
        for group in bars:
            bar, label, prob_label = group
            label.next_to(bar, DOWN, buff=0.15)
            prob_label.next_to(bar, UP, buff=0.1)
        bars.shift(DOWN * 0.3)

        target_arrow = Arrow(
            bars[target_idx][0].get_top() + UP * 0.6,
            bars[target_idx][0].get_top() + UP * 0.1,
            buff=0, color=C_YELLOW, stroke_width=3,
        )
        target_label = self.zh(
            "正確答案", font_size=18, color=C_YELLOW,
        ).next_to(target_arrow, UP, buff=0.1)

        loss_good = VGroup(
            self.mono("Loss = -log(0.80) = ", font_size=24, color=C_GREEN),
            self.mono("0.22", font_size=28, color=C_YELLOW),
        ).arrange(RIGHT, buff=0.1).shift(DOWN * 2.3)

        explain = self.zh(
            "Loss 越低 = 模型對正確答案越有信心",
            font_size=22, color=C_WHITE,
        ).shift(DOWN * 3.0)

        with self.voiceover(
            text="交叉熵係訓練 LLM 嘅 loss function。"
            "佢衡量嘅係：模型對正確答案有幾大信心？"
            "例如正確答案係 cat，好嘅模型會俾 cat 80% 嘅概率。"
            "Loss 就等於 negative log of 0.8，約等於 0.22。概率越高，loss 越低。"
            "如果模型亂估，每個詞語都係 20%，loss 就會好高。"
            "訓練嘅目標就係令 loss 越嚟越低。"
        ):
            self.play(Write(heading), run_time=0.6)
            self.play(Write(formula), run_time=1)
            for bar_group in bars:
                self.play(FadeIn(bar_group, shift=UP * 0.2), run_time=0.3)
            self.play(GrowArrow(target_arrow), FadeIn(target_label), run_time=0.6)
            self.play(Write(loss_good), run_time=1)
            self.play(FadeIn(explain, shift=UP * 0.2), run_time=0.6)

        self.wait(0.3)
        self.clear()

    # ── Scene 8 — Softmax ────────────────────────────────────────────────

    def scene_softmax(self):
        heading = self.make_heading("Softmax 函數")

        formula = self.mono(
            "softmax(zi) = exp(zi) / Σ exp(zj)",
            font_size=26, color=C_WHITE,
        ).shift(UP * 2)

        logits_label = self.zh("Logits（原始分數）", font_size=22, color=C_DIM).shift(UP * 1.0 + LEFT * 2.5)
        logits_vals = self.mono(
            "[2.0, 1.0, 0.1, -1.0, 3.0]", font_size=24, color=C_ORANGE,
        ).next_to(logits_label, DOWN, buff=0.2)

        arrow_sf = Arrow(LEFT * 0.5, RIGHT * 0.5, buff=0, color=C_YELLOW, stroke_width=3).shift(
            RIGHT * 1.5 + UP * 0.3
        )
        sf_label = self.en("softmax", font_size=18, color=C_YELLOW).next_to(arrow_sf, UP, buff=0.1)

        probs_vals = self.mono(
            "[0.23, 0.09, 0.03, 0.01, 0.63]", font_size=24, color=C_GREEN,
        ).next_to(arrow_sf, RIGHT, buff=0.4)

        probs_label = self.zh("概率（加埋 = 1）", font_size=22, color=C_DIM).next_to(
            probs_vals, DOWN, buff=0.2
        )

        temp_heading = self.zh(
            "Temperature 控制分佈嘅尖銳程度",
            font_size=22, color=C_WHITE,
        ).shift(DOWN * 1.0)

        def make_bar_chart(probs, color, label_text):
            group = VGroup()
            for p in probs:
                bar = Rectangle(
                    width=0.35, height=max(p * 3, 0.03),
                    fill_color=color, fill_opacity=0.7,
                    stroke_color=color, stroke_width=1,
                )
                group.add(bar)
            group.arrange(RIGHT, buff=0.08, aligned_edge=DOWN)
            label = self.en(label_text, font_size=16, color=color)
            label.next_to(group, DOWN, buff=0.15)
            return VGroup(group, label)

        probs_low = [0.12, 0.02, 0.00, 0.00, 0.86]
        probs_mid = [0.23, 0.09, 0.03, 0.01, 0.63]
        probs_high = [0.26, 0.16, 0.10, 0.06, 0.43]

        chart_low = make_bar_chart(probs_low, C_RED, "T=0.5")
        chart_mid = make_bar_chart(probs_mid, C_YELLOW, "T=1.0")
        chart_high = make_bar_chart(probs_high, C_CYAN, "T=2.0")

        label_sharp = self.zh("更確定", font_size=16, color=C_RED).next_to(chart_low, DOWN, buff=0.4)
        label_default = self.zh("預設", font_size=16, color=C_YELLOW).next_to(chart_mid, DOWN, buff=0.4)
        label_creative = self.zh("更有創意", font_size=16, color=C_CYAN).next_to(chart_high, DOWN, buff=0.4)

        charts = VGroup(
            VGroup(chart_low, label_sharp),
            VGroup(chart_mid, label_default),
            VGroup(chart_high, label_creative),
        ).arrange(RIGHT, buff=0.8)
        charts.shift(DOWN * 2.2)

        with self.voiceover(
            text="LLM 最後一層會輸出一組叫 logits 嘅分數，可以係任何數值。"
            "Softmax 函數將呢啲分數轉換成概率，令佢哋全部係正數，加埋等於 1。"
            "公式就係 e 嘅 z 次方除以所有 e 嘅 z 次方嘅總和。"
            "Temperature 可以控制分佈嘅尖銳程度。"
            "低 temperature 令分佈更加集中，模型更加確定。"
            "高 temperature 令分佈更加平均，模型更加有創意。"
        ):
            self.play(Write(heading), run_time=0.6)
            self.play(Write(formula), run_time=1.0)
            self.play(FadeIn(logits_label), Write(logits_vals), run_time=0.8)
            self.play(GrowArrow(arrow_sf), FadeIn(sf_label), run_time=0.5)
            self.play(Write(probs_vals), FadeIn(probs_label), run_time=0.8)
            self.play(FadeIn(temp_heading), run_time=0.5)
            self.play(
                FadeIn(charts, shift=UP * 0.2),
                run_time=1,
            )

        self.wait(0.3)
        self.clear()

    # ── Scene 9 — Log-Sum-Exp Trick ──────────────────────────────────────

    def scene_log_sum_exp(self):
        heading = self.make_heading("數值穩定性")

        problem_title = self.zh(
            "問題：exp(1000) = ∞  溢出！", font_size=28, color=C_RED,
        ).shift(UP * 1.5)

        naive = self.mono(
            "softmax([1000, 1001, 1002])", font_size=24, color=C_WHITE,
        ).shift(UP * 0.5)
        crash = self.zh("直接計算 → 溢出錯誤！", font_size=24, color=C_RED).next_to(
            naive, DOWN, buff=0.3
        )

        cross = self.en("✗", font_size=40, color=C_RED).next_to(crash, RIGHT, buff=0.3)

        solution_title = self.zh(
            "解決方法：減去最大值", font_size=28, color=C_GREEN,
        ).shift(DOWN * 0.8)

        trick = self.mono(
            "softmax(zi) = softmax(zi - max(z))",
            font_size=24, color=C_GREEN,
        ).next_to(solution_title, DOWN, buff=0.3)

        result = self.mono(
            "softmax([0, 1, 2]) = [0.09, 0.24, 0.67]",
            font_size=24, color=C_YELLOW,
        ).next_to(trick, DOWN, buff=0.3)

        check = self.en("✓", font_size=40, color=C_GREEN).next_to(result, RIGHT, buff=0.3)

        with self.voiceover(
            text="實際寫代碼嗰陣，數值穩定性好重要。"
            "如果 logits 好大，例如 1000，直接計算 e 嘅 1000 次方就會溢出。"
            "解決方法好簡單，就係所有 logits 減去最大值。"
            "數學上可以證明，減去任何常數都唔會改變 softmax 嘅結果。"
            "所以 softmax of 1000 comma 1001 comma 1002 同 softmax of 0 comma 1 comma 2 係一樣嘅。"
            "呢個技巧喺 PyTorch 入面已經自動做咗，但你要知道背後嘅原理。"
        ):
            self.play(Write(heading), run_time=0.6)
            self.play(Write(problem_title), run_time=0.8)
            self.play(Write(naive), run_time=0.8)
            self.play(FadeIn(crash, shift=UP * 0.1), FadeIn(cross, scale=1.5), run_time=0.6)
            self.play(FadeIn(solution_title, shift=UP * 0.2), run_time=0.5)
            self.play(Write(trick), run_time=1)
            self.play(Write(result), FadeIn(check, scale=1.5), run_time=1)

        self.wait(0.3)
        self.clear()

    # ── Scene 10 — Tiny Forward Pass ─────────────────────────────────────

    def scene_forward_pass(self):
        heading = self.make_heading("LLM 嘅一次前向傳播")

        boxes_data = [
            ("Token\nEmbedding", C_GREEN),
            ("Linear\n矩陣乘法", C_ORANGE),
            ("Softmax", C_PURPLE),
            ("Cross-Entropy\nLoss", C_RED),
        ]

        boxes = []
        for label, color in boxes_data:
            rect = RoundedRectangle(
                corner_radius=0.15, width=2.2, height=1.0,
                fill_color=color, fill_opacity=0.25, stroke_color=color,
            )
            txt = self.en(label, font_size=16, color=color).move_to(rect)
            boxes.append(VGroup(rect, txt))

        row = VGroup(*boxes).arrange(RIGHT, buff=0.6).shift(UP * 0.5)

        arrows = []
        for i in range(len(boxes) - 1):
            a = Arrow(
                boxes[i].get_right(), boxes[i + 1].get_left(),
                buff=0.1, color=C_WHITE, stroke_width=3,
            )
            arrows.append(a)

        embed_val = self.mono(
            "[-0.14, -0.17, 0.70]",
            font_size=16, color=C_GREEN,
        ).next_to(boxes[0], DOWN, buff=0.4)

        logits_val = self.mono(
            "[0.20, -0.21, 0.26, ...]",
            font_size=16, color=C_ORANGE,
        ).next_to(boxes[1], DOWN, buff=0.4)

        probs_val = self.mono(
            "[0.17, 0.12, 0.18, ...]",
            font_size=16, color=C_PURPLE,
        ).next_to(boxes[2], DOWN, buff=0.4)

        loss_val = VGroup(
            self.mono("Loss = ", font_size=20, color=C_RED),
            self.mono("2.40", font_size=24, color=C_YELLOW),
        ).arrange(RIGHT, buff=0.05).next_to(boxes[3], DOWN, buff=0.4)

        back_arrow = Arrow(
            boxes[3].get_bottom() + DOWN * 1.2,
            boxes[0].get_bottom() + DOWN * 1.2,
            buff=0, color=C_PINK, stroke_width=3,
        ).shift(DOWN * 0.3)
        back_label = self.zh(
            "反向傳播：更新所有權重", font_size=20, color=C_PINK,
        ).next_to(back_arrow, DOWN, buff=0.15)

        with self.voiceover(
            text="將所有嘢組合埋一齊。LLM 嘅一次前向傳播係咁樣嘅。"
            "首先，token 變成 embedding 向量。"
            "然後經過矩陣乘法，即係 linear layer，得到 logits。"
            "Softmax 將 logits 變成概率。"
            "最後，cross entropy 計算 loss。"
            "Loss 通過反向傳播傳返去更新所有權重。"
            "呢個過程重複幾十億次，模型就會越嚟越叻。"
        ):
            self.play(Write(heading), run_time=0.6)
            for i, box in enumerate(boxes):
                self.play(FadeIn(box, shift=RIGHT * 0.2), run_time=0.5)
                if i < len(arrows):
                    self.play(GrowArrow(arrows[i]), run_time=0.3)

            self.play(FadeIn(embed_val, shift=UP * 0.1), run_time=0.4)
            self.play(FadeIn(logits_val, shift=UP * 0.1), run_time=0.4)
            self.play(FadeIn(probs_val, shift=UP * 0.1), run_time=0.4)
            self.play(Write(loss_val), run_time=0.5)
            self.play(GrowArrow(back_arrow), FadeIn(back_label), run_time=1)

        self.wait(0.3)
        self.clear()

    # ── Scene 11 — Summary ───────────────────────────────────────────────

    def scene_summary(self):
        heading = self.make_heading("總結", font_size=44)

        bullets = VGroup(
            self.zh("•  點積衡量兩個向量嘅相似度", font_size=26, color=C_GREEN),
            self.zh("•  矩陣乘法係每層 Neural Network 嘅核心", font_size=26, color=C_ORANGE),
            self.zh("•  梯度話俾模型知點樣改進", font_size=26, color=C_PINK),
            self.zh("•  Cross-Entropy 衡量模型嘅預測有幾準", font_size=26, color=C_RED),
            self.zh("•  Softmax 將分數變成概率", font_size=26, color=C_PURPLE),
            self.zh("•  Log-Sum-Exp 確保數值唔會溢出", font_size=26, color=C_CYAN),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.35).move_to(ORIGIN)

        with self.voiceover(
            text="總結一下今日學咗嘅嘢。"
            "點積衡量兩個向量嘅相似度，Attention 機制靠佢運作。"
            "矩陣乘法係每層 neural network 嘅核心運算。"
            "梯度同反向傳播話俾模型知點樣一步步改進。"
            "Cross entropy 衡量模型嘅預測有幾準確，越低越好。"
            "Softmax 將原始分數變成概率分佈。"
            "Log sum exp 技巧確保計算唔會溢出。"
            "掌握咗呢六個概念，你就有足夠嘅數學基礎去理解 LLM 嘅運作原理喇。"
        ):
            self.play(Write(heading), run_time=0.6)
            for b in bullets:
                self.play(FadeIn(b, shift=RIGHT * 0.3), run_time=0.5)

        self.wait(0.5)
        self.clear()

        thanks = self.zh("多謝收睇！", font_size=52, color=C_YELLOW)

        with self.voiceover(
            text="多謝收睇！下一課我哋會學 Python 同 deep learning 嘅工具。記得繼續跟住學喇！"
        ):
            self.play(Write(thanks), run_time=1)

        self.wait(1)
        self.play(FadeOut(thanks), run_time=0.8)
