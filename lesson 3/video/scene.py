"""
Lesson 3 – Neural Network Building Blocks
Manim CE + Cantonese voiceover (Edge TTS)

Render (1080p):
    cd "lesson 3/video"
    manim render -qh scene.py NNBuildingBlocks
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


class NNBuildingBlocks(VoiceoverScene):
    """Ten scenes explaining neural network building blocks in Cantonese."""

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
        self.scene_overview()
        self.scene_linear()
        self.scene_activation()
        self.scene_layernorm()
        self.scene_dropout()
        self.scene_residual()
        self.scene_initialization()
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

    def make_heading(self, text, color=C_BLUE, font_size=40):
        return self.zh(text, font_size=font_size, color=color).to_edge(UP, buff=0.5)

    # ── Scene 1 — Title / Intro ──────────────────────────────────────────

    def scene_intro(self):
        title = self.zh("神經網路積木", font_size=56, color=C_BLUE)
        sub_en = self.en(
            "Neural Network Building Blocks", font_size=24, color=C_WHITE
        ).next_to(title, DOWN, buff=0.4)
        sub_zh = self.zh(
            "第三課", font_size=36, color=C_ORANGE
        ).next_to(sub_en, DOWN, buff=0.3)
        producer = self.zh(
            "制片人：Peter", font_size=24, color=C_DIM
        ).next_to(sub_zh, DOWN, buff=0.7)

        with self.voiceover(
            text="大家好！歡迎嚟到第三課。"
            "上一課我哋學咗 PyTorch 嘅基本工具，"
            "今日我哋嚟學神經網路嘅積木，"
            "即係 transformer 入面每一個組件。"
            "學完呢課之後，你建造 transformer 嘅時候，"
            "每一個部分你都會好清楚佢做緊咩。"
        ):
            self.play(Write(title), run_time=1.5)
            self.play(FadeIn(sub_en, shift=UP * 0.2), run_time=0.8)
            self.play(FadeIn(sub_zh, shift=UP * 0.2), run_time=0.8)
            self.play(FadeIn(producer, shift=UP * 0.2), run_time=0.5)

        self.wait(0.3)
        self.clear()

    # ── Scene 2 — Overview (6 building blocks) ───────────────────────────

    def scene_overview(self):
        heading = self.make_heading("六大積木")

        blocks_data = [
            ("Linear Layers", C_GREEN),
            ("Activation Functions", C_ORANGE),
            ("Layer Normalization", C_PINK),
            ("Dropout", C_PURPLE),
            ("Residual Connections", C_CYAN),
            ("Weight Initialization", C_YELLOW),
        ]

        boxes = VGroup()
        for label, color in blocks_data:
            boxes.add(self.make_box(label, color, width=3.5, height=0.8, font_size=22))
        boxes.arrange_in_grid(rows=2, cols=3, buff=0.4).move_to(ORIGIN)

        note = VGroup(
            RoundedRectangle(
                corner_radius=0.15, width=10, height=0.7,
                fill_color=C_BLUE, fill_opacity=0.12, stroke_color=C_BLUE,
            ),
            self.zh(
                "呢六個組件組合埋一齊，就係一個完整嘅 Transformer",
                font_size=22, color=C_BLUE,
            ),
        )
        note[1].move_to(note[0])
        note.to_edge(DOWN, buff=0.4)

        with self.voiceover(
            text="今日我哋會學六個重要嘅組件。"
            "第一，Linear Layers，即係線性層，做矩陣乘法。"
            "第二，Activation Functions，激活函數，加入非線性。"
            "第三，Layer Normalization，層正規化，穩定訓練。"
            "第四，Dropout，隨機丟棄，防止過擬合。"
            "第五，Residual Connections，殘差連接，幫助梯度流動。"
            "第六，Weight Initialization，權重初始化，確保訓練開頭穩定。"
            "呢六個組件組合埋一齊，就可以砌出一個完整嘅 Transformer。"
        ):
            self.play(Write(heading), run_time=0.6)
            for box in boxes:
                self.play(FadeIn(box, scale=0.85), run_time=0.5)
            self.play(FadeIn(note, shift=UP * 0.2), run_time=0.6)

        self.wait(0.3)
        self.clear()

    # ── Scene 3 — Linear Layers ──────────────────────────────────────────

    def scene_linear(self):
        heading = self.make_heading("線性層 Linear Layer")

        # Input vector
        in_label = self.mono("Input", font_size=20, color=C_GREEN)
        in_shape = self.mono("(batch, 768)", font_size=16, color=C_DIM)
        in_box = RoundedRectangle(
            corner_radius=0.1, width=2.0, height=1.4,
            fill_color=C_GREEN, fill_opacity=0.2, stroke_color=C_GREEN,
        )
        in_group = VGroup(in_box, in_label, in_shape)
        in_label.move_to(in_box.get_center() + UP * 0.2)
        in_shape.move_to(in_box.get_center() + DOWN * 0.2)

        # Weight matrix
        w_label = self.mono("W", font_size=28, color=C_ORANGE)
        w_shape = self.mono("(768, 3072)", font_size=16, color=C_DIM)
        w_box = RoundedRectangle(
            corner_radius=0.1, width=2.2, height=1.4,
            fill_color=C_ORANGE, fill_opacity=0.2, stroke_color=C_ORANGE,
        )
        w_group = VGroup(w_box, w_label, w_shape)
        w_label.move_to(w_box.get_center() + UP * 0.2)
        w_shape.move_to(w_box.get_center() + DOWN * 0.2)

        # Bias
        b_label = self.mono("+ b", font_size=24, color=C_PINK)

        # Output
        out_label = self.mono("Output", font_size=20, color=C_YELLOW)
        out_shape = self.mono("(batch, 3072)", font_size=16, color=C_DIM)
        out_box = RoundedRectangle(
            corner_radius=0.1, width=2.0, height=1.4,
            fill_color=C_YELLOW, fill_opacity=0.2, stroke_color=C_YELLOW,
        )
        out_group = VGroup(out_box, out_label, out_shape)
        out_label.move_to(out_box.get_center() + UP * 0.2)
        out_shape.move_to(out_box.get_center() + DOWN * 0.2)

        flow = VGroup(in_group, w_group, b_label, out_group).arrange(RIGHT, buff=0.6)
        flow.move_to(ORIGIN + UP * 0.2)

        arr1 = Arrow(
            in_group.get_right(), w_group.get_left(),
            buff=0.1, color=C_WHITE, stroke_width=3,
        )
        arr2 = Arrow(
            w_group.get_right(), b_label.get_left(),
            buff=0.1, color=C_WHITE, stroke_width=3,
        )
        arr3 = Arrow(
            b_label.get_right(), out_group.get_left(),
            buff=0.1, color=C_WHITE, stroke_width=3,
        )

        formula = self.mono("y = x @ W.T + b", font_size=28, color=C_CYAN)
        formula.to_edge(DOWN, buff=0.8)

        uses_title = self.zh("Transformer 入面用到嘅 Linear:", font_size=20, color=C_DIM)
        uses_items = VGroup(
            self.mono("Q, K, V projections", font_size=16, color=C_GREEN),
            self.mono("FFN up / down", font_size=16, color=C_ORANGE),
            self.mono("LM head (vocab)", font_size=16, color=C_PINK),
        ).arrange(RIGHT, buff=0.6)
        uses = VGroup(uses_title, uses_items).arrange(DOWN, buff=0.15)
        uses.to_edge(DOWN, buff=0.2)

        with self.voiceover(
            text="第一個積木係 Linear Layer，線性層。"
            "佢做嘅嘢好簡單，就係矩陣乘法加偏差。"
            "公式就係 y 等於 x 乘以 W 轉置加 b。"
            "例如輸入係 768 維，經過一個 768 乘 3072 嘅權重矩陣，"
            "再加上偏差，就會輸出 3072 維。"
            "喺 Transformer 入面，Linear Layer 到處都係。"
            "Q K V 嘅投影、FFN 嘅上下投影、同埋最後嘅 LM head，全部都係 Linear Layer。"
        ):
            self.play(Write(heading), run_time=0.6)
            self.play(FadeIn(in_group, shift=RIGHT * 0.2), run_time=0.6)
            self.play(GrowArrow(arr1), run_time=0.3)
            self.play(FadeIn(w_group, shift=RIGHT * 0.2), run_time=0.6)
            self.play(GrowArrow(arr2), run_time=0.3)
            self.play(FadeIn(b_label, scale=1.2), run_time=0.4)
            self.play(GrowArrow(arr3), run_time=0.3)
            self.play(FadeIn(out_group, shift=RIGHT * 0.2), run_time=0.6)
            self.play(FadeIn(formula, shift=UP * 0.2), run_time=0.6)
            self.play(
                FadeOut(formula),
                FadeIn(uses, shift=UP * 0.2),
                run_time=0.6,
            )

        self.wait(0.3)
        self.clear()

    # ── Scene 4 — Activation Functions ────────────────────────────────────

    def scene_activation(self):
        heading = self.make_heading("激活函數 Activation Functions")

        axes = Axes(
            x_range=[-4, 4, 1],
            y_range=[-2, 4, 1],
            x_length=8,
            y_length=4,
            axis_config={"color": C_DIM, "include_numbers": False},
        ).shift(DOWN * 0.3)

        x_label = self.mono("x", font_size=18, color=C_DIM).next_to(axes.x_axis, RIGHT, buff=0.15)
        y_label = self.mono("y", font_size=18, color=C_DIM).next_to(axes.y_axis, UP, buff=0.15)

        relu_curve = axes.plot(
            lambda x: max(0, x), x_range=[-4, 4, 0.01], color=C_ORANGE,
        )
        relu_label = self.mono("ReLU", font_size=20, color=C_ORANGE).move_to(
            axes.c2p(3.2, 3.5)
        )

        import math as _m
        def gelu_fn(x):
            return 0.5 * x * (1.0 + _m.tanh(_m.sqrt(2.0 / _m.pi) * (x + 0.044715 * x ** 3)))

        gelu_curve = axes.plot(gelu_fn, x_range=[-4, 4, 0.01], color=C_CYAN)
        gelu_label = self.mono("GELU", font_size=20, color=C_CYAN).move_to(
            axes.c2p(-3.0, 1.5)
        )

        diff_note = VGroup(
            RoundedRectangle(
                corner_radius=0.15, width=10, height=0.7,
                fill_color=C_CYAN, fill_opacity=0.12, stroke_color=C_CYAN,
            ),
            self.zh(
                "GELU 比 ReLU 更加平滑，唔會完全切斷負數  — GPT / BERT 都用 GELU",
                font_size=20, color=C_CYAN,
            ),
        )
        diff_note[1].move_to(diff_note[0])
        diff_note.to_edge(DOWN, buff=0.35)

        with self.voiceover(
            text="第二個積木係激活函數。"
            "如果冇激活函數，幾多層 Linear Layer 疊埋一齊都係等於一層。"
            "激活函數加入非線性，令網路可以學到複雜嘅 pattern。"
            "ReLU 係最經典嘅激活函數，負數全部變零，正數唔變。"
            "但係 ReLU 有個問題，負數嘅梯度完全係零，"
            "嗰啲 neuron 就會變成死嘅，永遠學唔到嘢。"
            "GELU 就唔同，佢比較平滑，"
            "負數唔會完全變零，而係畀少少值通過。"
            "所以 GPT 同 BERT 全部都係用 GELU。"
        ):
            self.play(Write(heading), run_time=0.6)
            self.play(Create(axes), FadeIn(x_label), FadeIn(y_label), run_time=0.8)
            self.play(Create(relu_curve), FadeIn(relu_label), run_time=1.2)
            self.play(Create(gelu_curve), FadeIn(gelu_label), run_time=1.2)
            self.play(FadeIn(diff_note, shift=UP * 0.2), run_time=0.6)

        self.wait(0.3)
        self.clear()

    # ── Scene 5 — Layer Normalization ─────────────────────────────────────

    def scene_layernorm(self):
        heading = self.make_heading("層正規化 Layer Normalization")

        # Before normalization: scattered dots
        np.random.seed(42)
        before_dots = VGroup()
        before_positions = []
        for _ in range(20):
            x_pos = np.random.uniform(-2.5, 2.5)
            y_pos = np.random.uniform(-1.5, 2.5)
            before_positions.append((x_pos, y_pos))
            dot = Dot(
                point=LEFT * 2.5 + RIGHT * (x_pos + 2.5) / 5 * 4 + UP * y_pos * 0.5,
                color=C_ORANGE, radius=0.08,
            )
            dot.shift(UP * 0.3)
            before_dots.add(dot)

        before_label = self.zh("正規化之前", font_size=22, color=C_ORANGE)
        before_label.next_to(before_dots, UP, buff=0.4)

        # After normalization: tightly clustered around center line
        after_dots = VGroup()
        for i in range(20):
            x_pos = np.random.uniform(-2.5, 2.5)
            y_pos = np.random.uniform(-0.3, 0.3)
            dot = Dot(
                point=LEFT * 2.5 + RIGHT * (x_pos + 2.5) / 5 * 4 + UP * y_pos * 0.5,
                color=C_GREEN, radius=0.08,
            )
            dot.shift(DOWN * 2.0)
            after_dots.add(dot)

        after_label = self.zh("正規化之後", font_size=22, color=C_GREEN)
        after_label.next_to(after_dots, UP, buff=0.4)

        mean_line = DashedLine(
            LEFT * 2.8 + DOWN * 2.0, RIGHT * 2.2 + DOWN * 2.0,
            color=C_YELLOW, stroke_width=2,
        )
        mean_text = self.mono("mean=0, std=1", font_size=16, color=C_YELLOW)
        mean_text.next_to(mean_line, RIGHT, buff=0.2)

        arrow_down = Arrow(
            UP * 0.0 + LEFT * 0.0,
            DOWN * 1.2 + LEFT * 0.0,
            buff=0.15, color=C_WHITE, stroke_width=3,
        )
        ln_label = self.mono("LayerNorm", font_size=22, color=C_CYAN).next_to(
            arrow_down, RIGHT, buff=0.2
        )

        rms_note = VGroup(
            RoundedRectangle(
                corner_radius=0.15, width=10, height=0.7,
                fill_color=C_PURPLE, fill_opacity=0.12, stroke_color=C_PURPLE,
            ),
            self.zh(
                "RMSNorm（LLaMA 用）更加簡單：唔使減 mean，淨係做 scaling",
                font_size=20, color=C_PURPLE,
            ),
        )
        rms_note[1].move_to(rms_note[0])
        rms_note.to_edge(DOWN, buff=0.35)

        with self.voiceover(
            text="第三個積木係 Layer Normalization。"
            "當數據經過好多層之後，數值會越變越大或者越變越細，"
            "訓練就會變得唔穩定。"
            "Layer Normalization 做嘅就係將每一層嘅輸出"
            "正規化返去 mean 等於零、standard deviation 等於一。"
            "你睇，上面啲點好分散，"
            "經過 LayerNorm 之後就集中返曬喺中間。"
            "LLaMA 用嘅 RMSNorm 仲簡單，"
            "佢唔使減去 mean，淨係做 scaling，快百分之十幾。"
        ):
            self.play(Write(heading), run_time=0.6)
            self.play(FadeIn(before_label), run_time=0.3)
            self.play(*[FadeIn(d, scale=0.5) for d in before_dots], run_time=1.0)
            self.play(GrowArrow(arrow_down), FadeIn(ln_label), run_time=0.6)
            self.play(FadeIn(after_label), run_time=0.3)
            self.play(*[FadeIn(d, scale=0.5) for d in after_dots], run_time=1.0)
            self.play(Create(mean_line), FadeIn(mean_text), run_time=0.5)
            self.play(FadeIn(rms_note, shift=UP * 0.2), run_time=0.6)

        self.wait(0.3)
        self.clear()

    # ── Scene 6 — Dropout ────────────────────────────────────────────────

    def scene_dropout(self):
        heading = self.make_heading("隨機丟棄 Dropout")

        # Grid of neurons — 4x6
        rows, cols = 4, 6
        neurons_group = VGroup()
        neuron_circles = []
        for r in range(rows):
            row_circles = []
            for c in range(cols):
                circle = Circle(
                    radius=0.28,
                    fill_color=C_GREEN,
                    fill_opacity=0.35,
                    stroke_color=C_GREEN,
                    stroke_width=2,
                )
                circle.move_to(
                    LEFT * 2.0 + RIGHT * c * 0.85 + UP * 1.0 + DOWN * r * 0.85
                )
                neurons_group.add(circle)
                row_circles.append(circle)
            neuron_circles.append(row_circles)

        train_label = self.zh("訓練模式 (p=0.3)", font_size=22, color=C_ORANGE)
        train_label.next_to(neurons_group, UP, buff=0.4)

        # Randomly cross out ~30% of neurons
        np.random.seed(42)
        crosses = VGroup()
        dropped_indices = []
        for r in range(rows):
            for c in range(cols):
                if np.random.random() < 0.3:
                    dropped_indices.append((r, c))
                    circ = neuron_circles[r][c]
                    x1 = Line(
                        circ.get_center() + UP * 0.18 + LEFT * 0.18,
                        circ.get_center() + DOWN * 0.18 + RIGHT * 0.18,
                        color=C_RED, stroke_width=3,
                    )
                    x2 = Line(
                        circ.get_center() + UP * 0.18 + RIGHT * 0.18,
                        circ.get_center() + DOWN * 0.18 + LEFT * 0.18,
                        color=C_RED, stroke_width=3,
                    )
                    crosses.add(x1, x2)

        scale_note = self.mono(
            "survivors scaled by 1/(1-p)", font_size=18, color=C_YELLOW
        ).next_to(neurons_group, DOWN, buff=0.3)

        eval_note = VGroup(
            RoundedRectangle(
                corner_radius=0.15, width=10, height=0.8,
                fill_color=C_CYAN, fill_opacity=0.12, stroke_color=C_CYAN,
            ),
            VGroup(
                self.zh(
                    "推理模式：所有 neuron 都開著，Dropout 關閉",
                    font_size=20, color=C_CYAN,
                ),
                self.mono(
                    "model.eval()  # 記得 call！", font_size=18, color=C_YELLOW,
                ),
            ).arrange(DOWN, buff=0.1),
        )
        eval_note[1].move_to(eval_note[0])
        eval_note.to_edge(DOWN, buff=0.3)

        with self.voiceover(
            text="第四個積木係 Dropout。"
            "訓練嘅時候，Dropout 會隨機將一部分 neuron 關閉，"
            "即係將佢哋嘅輸出設為零。"
            "例如 p 等於零點三，就有百分之三十嘅機會被關閉。"
            "剩低嘅 neuron 會被放大，確保期望值唔變。"
            "噉樣做嘅目的係防止網路過分依賴某啲 neuron，"
            "強迫佢學到更加穩健嘅 features。"
            "但係推理嘅時候一定要記得 call model dot eval，"
            "關閉 Dropout，否則輸出會好嘈。"
        ):
            self.play(Write(heading), run_time=0.6)
            self.play(FadeIn(train_label), run_time=0.3)
            self.play(
                *[FadeIn(n, scale=0.7) for n in neurons_group],
                run_time=1.0,
            )
            self.play(
                *[Create(c) for c in crosses],
                *[neuron_circles[r][c].animate.set_fill(C_RED, opacity=0.15).set_stroke(C_RED)
                  for r, c in dropped_indices],
                run_time=1.0,
            )
            self.play(FadeIn(scale_note, shift=UP * 0.1), run_time=0.5)
            self.play(FadeIn(eval_note, shift=UP * 0.2), run_time=0.6)

        self.wait(0.3)
        self.clear()

    # ── Scene 7 — Residual Connections ────────────────────────────────────

    def scene_residual(self):
        heading = self.make_heading("殘差連接 Residual Connection")

        # The main layer block
        layer_box = RoundedRectangle(
            corner_radius=0.15, width=3.5, height=1.6,
            fill_color=C_ORANGE, fill_opacity=0.2, stroke_color=C_ORANGE,
        )
        layer_label = self.zh("Sublayer", font_size=24, color=C_ORANGE)
        layer_detail = self.mono("(Attention / FFN)", font_size=16, color=C_DIM)
        layer_inner = VGroup(layer_label, layer_detail).arrange(DOWN, buff=0.1)
        layer_inner.move_to(layer_box)
        layer_group = VGroup(layer_box, layer_inner).move_to(ORIGIN)

        # Input arrow (from below)
        in_pos = layer_group.get_bottom() + DOWN * 1.2
        in_arrow = Arrow(
            in_pos, layer_group.get_bottom(),
            buff=0.1, color=C_WHITE, stroke_width=3,
        )
        in_label = self.mono("x", font_size=28, color=C_GREEN).next_to(in_pos, DOWN, buff=0.15)

        # Output arrow (to above)
        add_circle = Circle(
            radius=0.3, fill_color=C_CYAN, fill_opacity=0.3, stroke_color=C_CYAN,
        ).next_to(layer_group, UP, buff=0.6)
        add_text = self.mono("+", font_size=28, color=C_CYAN).move_to(add_circle)
        add_group = VGroup(add_circle, add_text)

        out_arrow_layer = Arrow(
            layer_group.get_top(), add_group.get_bottom(),
            buff=0.1, color=C_WHITE, stroke_width=3,
        )
        f_label = self.mono("F(x)", font_size=20, color=C_ORANGE).next_to(
            out_arrow_layer, RIGHT, buff=0.15
        )

        out_pos = add_group.get_top() + UP * 0.8
        out_arrow = Arrow(
            add_group.get_top(), out_pos,
            buff=0.1, color=C_WHITE, stroke_width=3,
        )
        out_label = self.mono("x + F(x)", font_size=22, color=C_YELLOW).next_to(
            out_pos, UP, buff=0.15
        )

        # THE skip connection — curved arrow bypassing the layer
        skip_start = in_arrow.get_start() + LEFT * 0.3
        skip_end = add_group.get_left()
        skip_arrow = CurvedArrow(
            skip_start, skip_end,
            angle=-TAU / 4,
            color=C_CYAN, stroke_width=4,
        )
        skip_label = self.mono("x", font_size=22, color=C_CYAN).next_to(
            skip_arrow, LEFT, buff=0.15
        )

        formula_box = VGroup(
            RoundedRectangle(
                corner_radius=0.15, width=10, height=0.7,
                fill_color=C_YELLOW, fill_opacity=0.12, stroke_color=C_YELLOW,
            ),
            self.zh(
                "梯度可以直接通過加法回傳 — 唔使經過 Sublayer，所以唔會消失",
                font_size=20, color=C_YELLOW,
            ),
        )
        formula_box[1].move_to(formula_box[0])
        formula_box.to_edge(DOWN, buff=0.3)

        with self.voiceover(
            text="第五個積木係 Residual Connection，殘差連接，"
            "我覺得呢個係最重要嘅概念之一。"
            "普通嘅網路就係將輸入 x 傳入一層，得到 F of x。"
            "但殘差連接就唔同，佢將原本嘅 x 跳過呢一層，"
            "直接加返去輸出，變成 x 加 F of x。"
            "呢條彎曲嘅箭頭就係 skip connection。"
            "佢嘅好處係，梯度可以直接通過加法回傳，"
            "唔使經過 sublayer 嘅乘法，所以唔會消失。"
            "GPT 三有 96 層，冇殘差連接嘅話根本冇辦法訓練。"
        ):
            self.play(Write(heading), run_time=0.6)
            self.play(FadeIn(in_label), GrowArrow(in_arrow), run_time=0.6)
            self.play(FadeIn(layer_group, scale=0.9), run_time=0.6)
            self.play(GrowArrow(out_arrow_layer), FadeIn(f_label), run_time=0.5)
            self.play(FadeIn(add_group, scale=0.8), run_time=0.4)
            self.play(
                Create(skip_arrow),
                FadeIn(skip_label),
                run_time=1.2,
            )
            self.play(GrowArrow(out_arrow), FadeIn(out_label), run_time=0.5)
            self.play(FadeIn(formula_box, shift=UP * 0.2), run_time=0.6)

        self.wait(0.3)
        self.clear()

    # ── Scene 8 — Weight Initialization ───────────────────────────────────

    def scene_initialization(self):
        heading = self.make_heading("權重初始化 Weight Initialization")

        # Bad init: exploding bars
        bad_title = self.zh("差嘅初始化", font_size=22, color=C_RED)
        bad_bars = VGroup()
        bad_heights = [4.0, 0.01, 3.5, 0.005, 5.0, 0.001]
        for i, h in enumerate(bad_heights):
            bar_h = min(h, 2.5)
            bar = Rectangle(
                width=0.4, height=bar_h,
                fill_color=C_RED, fill_opacity=0.5, stroke_color=C_RED,
            )
            bar.move_to(LEFT * 4 + RIGHT * i * 0.55 + UP * (bar_h / 2 - 1.5))
            bad_bars.add(bar)
        bad_labels = VGroup()
        for i in range(6):
            lbl = self.mono(f"L{i+1}", font_size=12, color=C_DIM)
            lbl.move_to(LEFT * 4 + RIGHT * i * 0.55 + DOWN * 1.7)
            bad_labels.add(lbl)
        bad_title.move_to(LEFT * 3 + UP * 2.0)
        bad_note = self.zh("爆炸或消失", font_size=16, color=C_RED)
        bad_note.next_to(bad_bars, DOWN, buff=0.6)

        # Good init: stable bars
        good_title = self.zh("好嘅初始化", font_size=22, color=C_GREEN)
        good_bars = VGroup()
        good_heights = [1.2, 1.1, 1.15, 1.05, 1.1, 1.08]
        for i, h in enumerate(good_heights):
            bar = Rectangle(
                width=0.4, height=h,
                fill_color=C_GREEN, fill_opacity=0.5, stroke_color=C_GREEN,
            )
            bar.move_to(RIGHT * 1.5 + RIGHT * i * 0.55 + UP * (h / 2 - 1.5))
            good_bars.add(bar)
        good_labels = VGroup()
        for i in range(6):
            lbl = self.mono(f"L{i+1}", font_size=12, color=C_DIM)
            lbl.move_to(RIGHT * 1.5 + RIGHT * i * 0.55 + DOWN * 1.7)
            good_labels.add(lbl)
        good_title.move_to(RIGHT * 2.8 + UP * 2.0)
        good_note = self.zh("穩定", font_size=16, color=C_GREEN)
        good_note.next_to(good_bars, DOWN, buff=0.6)

        methods = VGroup(
            RoundedRectangle(
                corner_radius=0.15, width=10, height=0.8,
                fill_color=C_CYAN, fill_opacity=0.12, stroke_color=C_CYAN,
            ),
            VGroup(
                self.mono(
                    "Xavier: std = sqrt(2/(fan_in + fan_out))", font_size=17, color=C_ORANGE,
                ),
                self.mono(
                    "Kaiming: std = sqrt(2/fan_in)", font_size=17, color=C_PINK,
                ),
            ).arrange(DOWN, buff=0.08),
        )
        methods[1].move_to(methods[0])
        methods.to_edge(DOWN, buff=0.3)

        with self.voiceover(
            text="最後一個積木係 Weight Initialization，權重初始化。"
            "如果權重初始值太大，activation 會爆炸。"
            "如果太細，activation 會消失到接近零。"
            "你睇左邊，用差嘅初始化，每一層嘅激活值忽大忽細，好唔穩定。"
            "右邊用好嘅初始化，每一層嘅激活值都差唔多，好穩定。"
            "常用嘅方法有 Xavier 同 Kaiming。"
            "Xavier 平衡輸入同輸出維度，Kaiming 專門配合 ReLU。"
            "GPT 二仲會用一個特殊嘅 scaling，"
            "將殘差層嘅權重縮小 one over square root of 2N，"
            "確保訓練穩定。"
        ):
            self.play(Write(heading), run_time=0.6)
            self.play(FadeIn(bad_title), run_time=0.3)
            self.play(
                *[FadeIn(b, shift=UP * 0.2) for b in bad_bars],
                *[FadeIn(l) for l in bad_labels],
                run_time=0.8,
            )
            self.play(FadeIn(bad_note), run_time=0.3)
            self.play(FadeIn(good_title), run_time=0.3)
            self.play(
                *[FadeIn(b, shift=UP * 0.2) for b in good_bars],
                *[FadeIn(l) for l in good_labels],
                run_time=0.8,
            )
            self.play(FadeIn(good_note), run_time=0.3)
            self.play(FadeIn(methods, shift=UP * 0.2), run_time=0.6)

        self.wait(0.3)
        self.clear()

    # ── Scene 9 — Summary ────────────────────────────────────────────────

    def scene_summary(self):
        heading = self.make_heading("總結", font_size=44)

        bullets = VGroup(
            self.zh("•  Linear Layer 係最基本嘅運算：y = xW + b", font_size=24, color=C_GREEN),
            self.zh("•  GELU 激活函數加入非線性，比 ReLU 更平滑", font_size=24, color=C_ORANGE),
            self.zh("•  LayerNorm / RMSNorm 穩定每一層嘅數值", font_size=24, color=C_PINK),
            self.zh("•  Dropout 防止過擬合，推理時要關閉", font_size=24, color=C_PURPLE),
            self.zh("•  殘差連接令梯度直接流過，深層訓練嘅關鍵", font_size=24, color=C_CYAN),
            self.zh("•  好嘅初始化確保訓練一開始就穩定", font_size=24, color=C_YELLOW),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.35).move_to(ORIGIN)

        with self.voiceover(
            text="嚟到總結。"
            "Linear Layer 係最基本嘅運算，做矩陣乘法加偏差。"
            "GELU 激活函數加入非線性，比 ReLU 更加平滑，唔會殺死 neuron。"
            "LayerNorm 同 RMSNorm 穩定每一層嘅數值，防止爆炸或消失。"
            "Dropout 防止過擬合，但推理嘅時候一定要記得關閉。"
            "殘差連接係訓練深層網路嘅關鍵，令梯度可以直接流過。"
            "好嘅權重初始化確保訓練一開始就穩定。"
            "呢六個積木組合埋一齊，就係 Transformer 嘅基礎。"
        ):
            self.play(Write(heading), run_time=0.6)
            for b in bullets:
                self.play(FadeIn(b, shift=RIGHT * 0.3), run_time=0.5)

        self.wait(0.5)
        self.clear()

    # ── Scene 10 — Outro ─────────────────────────────────────────────────

    def scene_outro(self):
        thanks = self.zh("多謝收睇！", font_size=52, color=C_YELLOW)
        next_lesson = self.zh(
            "下一課：文字表示與分詞 Tokenization", font_size=26, color=C_CYAN
        ).next_to(thanks, DOWN, buff=0.6)

        with self.voiceover(
            text="多謝收睇！"
            "下一課我哋會學 Tokenization，即係點樣將文字轉換成數字。"
            "包括 BPE 算法、詞彙表設計、同 token embeddings。"
            "記得繼續跟住學喇！"
        ):
            self.play(Write(thanks), run_time=1)
            self.play(FadeIn(next_lesson, shift=UP * 0.2), run_time=0.8)

        self.wait(1)
        self.play(FadeOut(thanks), FadeOut(next_lesson), run_time=0.8)
