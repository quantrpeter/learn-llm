"""
Lesson 2 – Python & Deep Learning Tooling
Manim CE + Cantonese voiceover (Edge TTS)

Render (1080p):
    cd "lesson 2/video"
    manim render -qh scene.py PyTorchTooling
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


class PyTorchTooling(VoiceoverScene):
    """Single scene explaining PyTorch & DL tooling in Cantonese."""

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
        self.scene_why_pytorch()
        self.scene_tensors()
        self.scene_tensor_ops()
        self.scene_autograd()
        self.scene_nn_module()
        self.scene_training_loop()
        self.scene_optimizer()
        self.scene_practical_tools()
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
        title = self.zh("PyTorch 基礎工具", font_size=56, color=C_BLUE)
        sub_en = self.en(
            "Python & Deep Learning Tooling", font_size=24, color=C_WHITE
        ).next_to(title, DOWN, buff=0.4)
        sub_zh = self.zh(
            "第二課", font_size=36, color=C_ORANGE
        ).next_to(sub_en, DOWN, buff=0.3)
        producer = self.zh(
            "制片人：Peter", font_size=24, color=C_DIM
        ).next_to(sub_zh, DOWN, buff=0.7)

        with self.voiceover(
            text="大家好！歡迎嚟到第二課。"
            "上一課我哋學咗 LLM 嘅數學基礎，"
            "今日我哋嚟學 PyTorch，呢個係建造 LLM 最重要嘅工具。"
            "學完呢課之後，你就識得點樣用 PyTorch 寫 deep learning 嘅代碼喇。"
        ):
            self.play(Write(title), run_time=1.5)
            self.play(FadeIn(sub_en, shift=UP * 0.2), run_time=0.8)
            self.play(FadeIn(sub_zh, shift=UP * 0.2), run_time=0.8)
            self.play(FadeIn(producer, shift=UP * 0.2), run_time=0.5)

        self.wait(0.3)
        self.clear()

    # ── Scene 2 — Why PyTorch ────────────────────────────────────────────

    def scene_why_pytorch(self):
        heading = self.make_heading("點解用 PyTorch？")

        reasons = [
            ("1. GPU 加速計算", "用 GPU 快過 CPU 幾十倍", C_GREEN),
            ("2. 自動微分 (Autograd)", "自動計算所有梯度", C_ORANGE),
            ("3. 豐富嘅 nn 模組", "現成嘅 layers 同 loss functions", C_PINK),
            ("4. LLM 社區標準", "GPT、LLaMA、Mistral 全部用 PyTorch", C_PURPLE),
        ]

        items = VGroup()
        for label, detail, col in reasons:
            line = VGroup(
                self.zh(label, font_size=28, color=col),
                self.zh(f"— {detail}", font_size=22, color=C_DIM),
            ).arrange(RIGHT, buff=0.3)
            items.add(line)
        items.arrange(DOWN, aligned_edge=LEFT, buff=0.5).move_to(ORIGIN)

        with self.voiceover(
            text="點解我哋要用 PyTorch 呢？有四個原因。"
            "第一，GPU 加速。用 GPU 做矩陣運算，快過 CPU 幾十甚至幾百倍。"
            "第二，自動微分。PyTorch 嘅 autograd 系統會自動幫你計算所有梯度，唔使你自己寫反向傳播。"
            "第三，佢有好豐富嘅 nn 模組，包括各種 layers 同 loss functions，可以直接攞嚟用。"
            "第四，PyTorch 已經係 LLM 社區嘅標準。GPT、LLaMA、Mistral 全部都係用 PyTorch 寫嘅。"
        ):
            self.play(Write(heading), run_time=0.6)
            for item in items:
                self.play(FadeIn(item, shift=RIGHT * 0.3), run_time=0.6)

        self.wait(0.3)
        self.clear()

    # ── Scene 3 — Tensors ────────────────────────────────────────────────

    def scene_tensors(self):
        heading = self.make_heading("張量 Tensors")

        # --- 1D: vector (row of boxes) ---
        label_1d = self.zh("1D 向量", font_size=22, color=C_GREEN).shift(UP * 1.6 + LEFT * 4)
        boxes_1d = VGroup()
        vals_1d = ["3", "1", "4", "1", "5"]
        for v in vals_1d:
            rect = Rectangle(
                width=0.6, height=0.6,
                fill_color=C_GREEN, fill_opacity=0.25,
                stroke_color=C_GREEN, stroke_width=2,
            )
            txt = self.mono(v, font_size=20, color=C_GREEN).move_to(rect)
            boxes_1d.add(VGroup(rect, txt))
        boxes_1d.arrange(RIGHT, buff=0.08).next_to(label_1d, RIGHT, buff=0.4)
        shape_1d = self.mono("shape: (5,)", font_size=18, color=C_DIM).next_to(boxes_1d, RIGHT, buff=0.3)

        # --- 2D: matrix (grid of boxes) ---
        label_2d = self.zh("2D 矩陣", font_size=22, color=C_ORANGE).shift(UP * 0.3 + LEFT * 4)
        rows_2d = VGroup()
        vals_2d = [["2", "7", "1"], ["8", "3", "5"]]
        for row_vals in vals_2d:
            row_group = VGroup()
            for v in row_vals:
                rect = Rectangle(
                    width=0.6, height=0.6,
                    fill_color=C_ORANGE, fill_opacity=0.25,
                    stroke_color=C_ORANGE, stroke_width=2,
                )
                txt = self.mono(v, font_size=20, color=C_ORANGE).move_to(rect)
                row_group.add(VGroup(rect, txt))
            row_group.arrange(RIGHT, buff=0.08)
            rows_2d.add(row_group)
        rows_2d.arrange(DOWN, buff=0.08).next_to(label_2d, RIGHT, buff=0.4)
        shape_2d = self.mono("shape: (2,3)", font_size=18, color=C_DIM).next_to(rows_2d, RIGHT, buff=0.3)

        # --- 3D: batch of matrices (stacked grids) ---
        label_3d = self.zh("3D 批次", font_size=22, color=C_PURPLE).shift(DOWN * 1.5 + LEFT * 4)

        def make_mini_grid(color, opacity):
            grid = VGroup()
            for _ in range(2):
                row = VGroup()
                for _ in range(3):
                    rect = Rectangle(
                        width=0.5, height=0.5,
                        fill_color=color, fill_opacity=opacity,
                        stroke_color=color, stroke_width=1.5,
                    )
                    row.add(rect)
                row.arrange(RIGHT, buff=0.06)
                grid.add(row)
            grid.arrange(DOWN, buff=0.06)
            return grid

        g_back = make_mini_grid(C_PURPLE, 0.1).shift(DOWN * 1.3 + RIGHT * 0.35)
        g_mid = make_mini_grid(C_PURPLE, 0.18).shift(DOWN * 1.45 + RIGHT * 0.18)
        g_front = make_mini_grid(C_PURPLE, 0.3).shift(DOWN * 1.6)
        stack_3d = VGroup(g_back, g_mid, g_front).next_to(label_3d, RIGHT, buff=0.4)
        shape_3d = self.mono("shape: (3,2,3)", font_size=18, color=C_DIM).next_to(stack_3d, RIGHT, buff=0.3)

        gpu_note = VGroup(
            RoundedRectangle(
                corner_radius=0.15, width=8, height=0.7,
                fill_color=C_CYAN, fill_opacity=0.15, stroke_color=C_CYAN,
            ),
            self.zh("Tensor 就好似 NumPy array，但可以喺 GPU 上面跑", font_size=22, color=C_CYAN),
        )
        gpu_note[1].move_to(gpu_note[0])
        gpu_note.to_edge(DOWN, buff=0.5)

        with self.voiceover(
            text="Tensor，中文叫做張量，係 PyTorch 最基本嘅數據結構。"
            "你可以當佢係加強版嘅 NumPy array。"
            "一維嘅 tensor 就係一個向量，好似一排數字咁。"
            "二維嘅 tensor 就係一個矩陣，好似一個表格。"
            "三維嘅 tensor 就好似一疊矩陣，我哋叫做 batch。"
            "喺訓練嘅時候，數據通常都係三維或者更高維度嘅 tensor。"
            "最重要嘅係，tensor 可以喺 GPU 上面運算，速度快好多。"
        ):
            self.play(Write(heading), run_time=0.6)
            self.play(FadeIn(label_1d), run_time=0.3)
            self.play(*[FadeIn(b, shift=RIGHT * 0.1) for b in boxes_1d], run_time=0.8)
            self.play(FadeIn(shape_1d), run_time=0.3)

            self.play(FadeIn(label_2d), run_time=0.3)
            self.play(*[FadeIn(r, shift=RIGHT * 0.1) for r in rows_2d], run_time=0.8)
            self.play(FadeIn(shape_2d), run_time=0.3)

            self.play(FadeIn(label_3d), run_time=0.3)
            self.play(FadeIn(g_back), FadeIn(g_mid), FadeIn(g_front), run_time=0.8)
            self.play(FadeIn(shape_3d), run_time=0.3)

            self.play(FadeIn(gpu_note, shift=UP * 0.2), run_time=0.6)

        self.wait(0.3)
        self.clear()

    # ── Scene 4 — Tensor Operations ──────────────────────────────────────

    def scene_tensor_ops(self):
        heading = self.make_heading("張量運算")

        # --- Element-wise addition ---
        ew_label = self.zh("逐元素相加", font_size=24, color=C_GREEN).shift(UP * 1.8 + LEFT * 4)
        a_vals = self.mono("[1, 2, 3]", font_size=26, color=C_GREEN)
        plus = self.mono("+", font_size=26, color=C_WHITE)
        b_vals = self.mono("[4, 5, 6]", font_size=26, color=C_ORANGE)
        eq1 = self.mono("=", font_size=26, color=C_WHITE)
        c_vals = self.mono("[5, 7, 9]", font_size=26, color=C_YELLOW)
        ew_row = VGroup(a_vals, plus, b_vals, eq1, c_vals).arrange(RIGHT, buff=0.2)
        ew_row.next_to(ew_label, DOWN, buff=0.3).shift(RIGHT * 1)

        # --- Matrix multiply ---
        mm_label = self.zh("矩陣乘法", font_size=24, color=C_ORANGE).shift(DOWN * 0.1 + LEFT * 4)

        shape_a = self.mono("(2x3)", font_size=22, color=C_GREEN)
        at_sym = self.mono("@", font_size=26, color=C_WHITE)
        shape_b = self.mono("(3x2)", font_size=22, color=C_ORANGE)
        eq2 = self.mono("=", font_size=26, color=C_WHITE)
        shape_c = self.mono("(2x2)", font_size=22, color=C_YELLOW)
        mm_row = VGroup(shape_a, at_sym, shape_b, eq2, shape_c).arrange(RIGHT, buff=0.2)
        mm_row.next_to(mm_label, DOWN, buff=0.3).shift(RIGHT * 1)

        mm_rule = self.mono(
            "inner dims must match: 3 == 3", font_size=20, color=C_DIM
        ).next_to(mm_row, DOWN, buff=0.2)

        # --- Reduction ---
        red_label = self.zh("聚合運算", font_size=24, color=C_PINK).shift(DOWN * 1.6 + LEFT * 4)
        red_ex = VGroup(
            self.mono("x = [[1, 2], [3, 4]]", font_size=22, color=C_WHITE),
            self.mono("x.sum(dim=0) = [4, 6]", font_size=22, color=C_CYAN),
            self.mono("x.sum(dim=1) = [3, 7]", font_size=22, color=C_PURPLE),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        red_ex.next_to(red_label, DOWN, buff=0.3).shift(RIGHT * 1)

        with self.voiceover(
            text="有咗 tensor 之後，我哋可以做各種運算。"
            "第一種係逐元素運算，例如兩個向量逐個位相加。"
            "1 加 4 等於 5，2 加 5 等於 7，3 加 6 等於 9。"
            "呢啲同上一課學嘅一樣，只不過而家用 PyTorch 嚟做。"
            "第二種係矩陣乘法，用 at 符號表示。"
            "一個 2 乘 3 嘅矩陣乘一個 3 乘 2 嘅矩陣，結果就係 2 乘 2。"
            "要記住，中間嗰個維度一定要相同。"
            "第三種係聚合運算，例如 sum。你可以沿住唔同嘅維度做加總。"
        ):
            self.play(Write(heading), run_time=0.6)

            self.play(FadeIn(ew_label, shift=RIGHT * 0.2), run_time=0.4)
            self.play(Write(ew_row), run_time=1.0)

            self.play(FadeIn(mm_label, shift=RIGHT * 0.2), run_time=0.4)
            self.play(Write(mm_row), run_time=1.0)
            self.play(FadeIn(mm_rule), run_time=0.4)

            self.play(FadeIn(red_label, shift=RIGHT * 0.2), run_time=0.4)
            for line in red_ex:
                self.play(FadeIn(line, shift=RIGHT * 0.2), run_time=0.4)

        self.wait(0.3)
        self.clear()

    # ── Scene 5 — Autograd ───────────────────────────────────────────────

    def scene_autograd(self):
        heading = self.make_heading("自動微分 Autograd")

        # Forward-pass boxes
        box_x = self.make_box("x", C_GREEN, width=1.5, height=0.7)
        box_mul = self.make_box("x * w", C_ORANGE, width=1.8, height=0.7)
        box_add = self.make_box("+ b", C_PURPLE, width=1.5, height=0.7)
        box_loss = self.make_box("Loss", C_RED, width=1.5, height=0.7)

        forward_boxes = VGroup(box_x, box_mul, box_add, box_loss).arrange(RIGHT, buff=1.0)
        forward_boxes.shift(UP * 0.8)

        # Forward arrows
        fwd_arrows = VGroup()
        fwd_labels = []
        pairs = [(box_x, box_mul), (box_mul, box_add), (box_add, box_loss)]
        for left, right in pairs:
            a = Arrow(
                left.get_right(), right.get_left(),
                buff=0.1, color=C_BLUE, stroke_width=3,
            )
            fwd_arrows.add(a)

        fwd_tag = self.zh("前向傳播", font_size=22, color=C_BLUE).next_to(forward_boxes, UP, buff=0.4)

        # Backward arrows (pink, reversed direction)
        bwd_arrows = VGroup()
        for left, right in reversed(pairs):
            a = Arrow(
                right.get_bottom() + DOWN * 0.15,
                left.get_bottom() + DOWN * 0.15,
                buff=0.1, color=C_PINK, stroke_width=3,
                path_arc=-0.4,
            )
            bwd_arrows.add(a)
        bwd_arrows.shift(DOWN * 0.3)

        bwd_tag = self.zh("反向傳播  .backward()", font_size=22, color=C_PINK).shift(DOWN * 1.0)

        # Gradient annotations
        grad_w = self.mono("dL/dw", font_size=20, color=C_PINK).next_to(box_mul, DOWN, buff=0.9)
        grad_b = self.mono("dL/db", font_size=20, color=C_PINK).next_to(box_add, DOWN, buff=0.9)

        explain = VGroup(
            RoundedRectangle(
                corner_radius=0.15, width=10, height=0.7,
                fill_color=C_YELLOW, fill_opacity=0.12, stroke_color=C_YELLOW,
            ),
            self.zh(
                "PyTorch 自動記錄所有運算，一call .backward() 就計好曬所有梯度",
                font_size=20, color=C_YELLOW,
            ),
        )
        explain[1].move_to(explain[0])
        explain.to_edge(DOWN, buff=0.4)

        with self.voiceover(
            text="Autograd，即係自動微分，係 PyTorch 最強大嘅功能。"
            "喺前向傳播嘅時候，PyTorch 會自動記錄你做過嘅每一個運算。"
            "例如，輸入 x 乘以權重 w，加上偏差 b，最後計算 loss。"
            "當你 call loss dot backward 嘅時候，"
            "PyTorch 就會自動沿住反方向計算每一個參數嘅梯度。"
            "即係 d L d w 同 d L d b，完全唔使你自己寫。"
            "呢個就係 PyTorch 最方便嘅地方，"
            "你只需要寫前向傳播，反向傳播 PyTorch 幫你搞掂。"
        ):
            self.play(Write(heading), run_time=0.6)
            self.play(FadeIn(fwd_tag), run_time=0.3)
            for i, box in enumerate([box_x, box_mul, box_add, box_loss]):
                self.play(FadeIn(box, shift=RIGHT * 0.2), run_time=0.4)
                if i < len(fwd_arrows):
                    self.play(GrowArrow(fwd_arrows[i]), run_time=0.3)

            self.play(FadeIn(bwd_tag, shift=UP * 0.2), run_time=0.5)
            for a in bwd_arrows:
                self.play(GrowArrow(a), run_time=0.4)
            self.play(FadeIn(grad_w), FadeIn(grad_b), run_time=0.5)
            self.play(FadeIn(explain, shift=UP * 0.2), run_time=0.6)

        self.wait(0.3)
        self.clear()

    # ── Scene 6 — nn.Module ──────────────────────────────────────────────

    def scene_nn_module(self):
        heading = self.make_heading("nn.Module 神經網路")

        layer_data = [
            ("Linear(4, 8)", C_GREEN),
            ("GELU", C_ORANGE),
            ("Linear(8, 1)", C_PURPLE),
        ]

        layers = VGroup()
        for label, color in layer_data:
            rect = RoundedRectangle(
                corner_radius=0.15, width=4, height=0.8,
                fill_color=color, fill_opacity=0.25, stroke_color=color,
            )
            txt = self.mono(label, font_size=24, color=color).move_to(rect)
            layers.add(VGroup(rect, txt))
        layers.arrange(DOWN, buff=0.6).shift(LEFT * 2 + DOWN * 0.2)

        # Arrows between layers
        layer_arrows = VGroup()
        for i in range(len(layers) - 1):
            a = Arrow(
                layers[i].get_bottom(), layers[i + 1].get_top(),
                buff=0.1, color=C_WHITE, stroke_width=3,
            )
            layer_arrows.add(a)

        # Input / output labels
        input_label = self.mono("input: (batch, 4)", font_size=20, color=C_DIM).next_to(
            layers[0], UP, buff=0.3
        )
        output_label = self.mono("output: (batch, 1)", font_size=20, color=C_DIM).next_to(
            layers[-1], DOWN, buff=0.3
        )

        # Parameter count on the right
        param_title = self.zh("參數數量", font_size=24, color=C_YELLOW).shift(RIGHT * 3 + UP * 0.8)
        params = VGroup(
            self.mono("Linear(4,8):  4*8+8 = 40", font_size=18, color=C_GREEN),
            self.mono("GELU:         0", font_size=18, color=C_ORANGE),
            self.mono("Linear(8,1):  8*1+1 = 9", font_size=18, color=C_PURPLE),
            self.mono("─────────────────", font_size=18, color=C_DIM),
            self.mono("Total:        49", font_size=20, color=C_YELLOW),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2).next_to(param_title, DOWN, buff=0.3)

        with self.voiceover(
            text="nn dot Module 係 PyTorch 用嚟組裝神經網路嘅方式。"
            "你可以將唔同嘅 layer 疊埋一齊，組成一個模型。"
            "例如呢度有一個簡單嘅網路：第一層係 Linear 4 到 8，"
            "即係將 4 維嘅輸入變成 8 維。"
            "中間有一個 GELU 激活函數，加入非線性。"
            "最後一層 Linear 8 到 1，輸出一個數值。"
            "每一個 Linear layer 都有自己嘅參數。"
            "Linear 4 到 8 有 40 個參數，Linear 8 到 1 有 9 個，"
            "成個模型總共有 49 個參數。"
            "真正嘅 LLM 有幾十億個參數，但原理係一樣嘅。"
        ):
            self.play(Write(heading), run_time=0.6)
            self.play(FadeIn(input_label), run_time=0.3)
            for i, layer in enumerate(layers):
                self.play(FadeIn(layer, shift=DOWN * 0.2), run_time=0.5)
                if i < len(layer_arrows):
                    self.play(GrowArrow(layer_arrows[i]), run_time=0.3)
            self.play(FadeIn(output_label), run_time=0.3)

            self.play(FadeIn(param_title, shift=LEFT * 0.2), run_time=0.4)
            for p in params:
                self.play(FadeIn(p, shift=RIGHT * 0.2), run_time=0.35)

        self.wait(0.3)
        self.clear()

    # ── Scene 7 — Training Loop ──────────────────────────────────────────

    def scene_training_loop(self):
        heading = self.make_heading("訓練循環")

        step_data = [
            ("Forward", C_GREEN),
            ("Loss", C_ORANGE),
            ("Backward", C_PINK),
            ("Step", C_PURPLE),
            ("Zero Grad", C_CYAN),
        ]

        n = len(step_data)
        radius = 1.8
        center = DOWN * 0.3
        boxes = VGroup()
        positions = []
        for i, (label, color) in enumerate(step_data):
            angle = PI / 2 - i * (2 * PI / n)
            pos = center + radius * np.array([np.cos(angle), np.sin(angle), 0])
            positions.append(pos)
            box = self.make_box(label, color, width=2.0, height=0.7)
            box.move_to(pos)
            boxes.add(box)

        # Arrows connecting boxes in cycle
        cycle_arrows = VGroup()
        for i in range(n):
            start_box = boxes[i]
            end_box = boxes[(i + 1) % n]
            direction = end_box.get_center() - start_box.get_center()
            direction = direction / np.linalg.norm(direction)
            a = Arrow(
                start_box.get_center() + direction * 0.9,
                end_box.get_center() - direction * 0.9,
                buff=0.1, color=C_WHITE, stroke_width=2.5,
            )
            cycle_arrows.add(a)

        # Highlight ring for animation
        repeat_label = self.zh(
            "重複幾千至幾十億次", font_size=22, color=C_YELLOW
        ).next_to(boxes, DOWN, buff=0.6)

        with self.voiceover(
            text="訓練循環係 deep learning 最核心嘅流程，你一定要記住。"
            "佢有五個步驟，不斷循環。"
            "第一步，Forward，將數據傳入模型，得到預測結果。"
            "第二步，計算 Loss，衡量預測同正確答案嘅差距。"
            "第三步，Backward，call dot backward 計算所有梯度。"
            "第四步，optimizer dot step，根據梯度更新所有參數。"
            "第五步，zero grad，將梯度歸零，準備下一輪。"
            "呢五個步驟不斷重複，模型就會越嚟越準確。"
            "真正嘅 LLM 訓練會重複幾十億次。"
        ):
            self.play(Write(heading), run_time=0.6)
            for i in range(n):
                self.play(FadeIn(boxes[i], scale=0.8), run_time=0.4)
                self.play(GrowArrow(cycle_arrows[i]), run_time=0.3)

            self.play(FadeIn(repeat_label, shift=UP * 0.2), run_time=0.5)

            for _ in range(2):
                for i in range(n):
                    self.play(
                        boxes[i][0].animate.set_fill(opacity=0.6),
                        run_time=0.25,
                    )
                    self.play(
                        boxes[i][0].animate.set_fill(opacity=0.25),
                        run_time=0.25,
                    )

        self.wait(0.3)
        self.clear()

    # ── Scene 8 — Optimizer (AdamW) ──────────────────────────────────────

    def scene_optimizer(self):
        heading = self.make_heading("AdamW 優化器")

        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[-0.5, 9, 2],
            x_length=6,
            y_length=3.2,
            axis_config={"color": C_DIM},
        ).shift(DOWN * 0.2)

        curve = axes.plot(lambda x: x ** 2, x_range=[-3, 3], color=C_BLUE)
        min_label = self.mono("min", font_size=18, color=C_YELLOW).next_to(
            axes.c2p(0, 0), DOWN, buff=0.2
        )

        dot = Dot(axes.c2p(2.5, 2.5 ** 2), color=C_YELLOW, radius=0.12)

        x_positions = [2.5, 1.8, 1.2, 0.7, 0.35, 0.15, 0.05]

        explain_box = VGroup(
            RoundedRectangle(
                corner_radius=0.15, width=10, height=0.8,
                fill_color=C_PURPLE, fill_opacity=0.15, stroke_color=C_PURPLE,
            ),
            self.zh(
                "AdamW = Adam + Weight Decay，GPT / LLaMA / Mistral 全部用佢",
                font_size=20, color=C_PURPLE,
            ),
        )
        explain_box[1].move_to(explain_box[0])
        explain_box.to_edge(DOWN, buff=0.35)

        with self.voiceover(
            text="有咗梯度之後，我哋需要一個優化器嚟更新參數。"
            "最常用嘅優化器叫做 AdamW。"
            "你可以想像 loss function 好似一個山谷，"
            "而優化器嘅工作就係搵到最低點。"
            "每一步，佢根據梯度嘅方向同大小，"
            "慢慢噉將參數移向 loss 最低嘅位置。"
            "AdamW 仲會自動調整每個參數嘅學習速率，"
            "令訓練更加穩定。"
            "基本上所有大型 LLM，包括 GPT 同 LLaMA，都係用 AdamW。"
        ):
            self.play(Write(heading), run_time=0.6)
            self.play(Create(axes), Create(curve), run_time=1.0)
            self.play(FadeIn(min_label), run_time=0.3)
            self.play(FadeIn(dot, scale=1.5), run_time=0.5)

            for x_next in x_positions[1:]:
                new_pos = axes.c2p(x_next, x_next ** 2)
                self.play(dot.animate.move_to(new_pos), run_time=0.4)

            self.play(FadeIn(explain_box, shift=UP * 0.2), run_time=0.6)

        self.wait(0.3)
        self.clear()

    # ── Scene 9 — Practical Tools ────────────────────────────────────────

    def scene_practical_tools(self):
        heading = self.make_heading("實用工具")

        tools_data = [
            ("Mixed Precision\n(FP16 / BF16)", C_GREEN,
             "用半精度浮點數，慳一半記憶體"),
            ("Gradient Clipping", C_ORANGE,
             "防止梯度爆炸，穩定訓練"),
            ("Checkpointing", C_PINK,
             "定時儲存模型，中斷後可以繼續"),
            ("Reproducibility\n(Seed)", C_PURPLE,
             "設定隨機種子，確保結果可重現"),
        ]

        tool_groups = VGroup()
        for label, color, desc in tools_data:
            box = self.make_box(label, color, width=4.5, height=1.0)
            desc_text = self.zh(desc, font_size=18, color=C_DIM).next_to(box, RIGHT, buff=0.3)
            tool_groups.add(VGroup(box, desc_text))
        tool_groups.arrange(DOWN, aligned_edge=LEFT, buff=0.35).move_to(ORIGIN)

        with self.voiceover(
            text="除咗核心嘅訓練循環，仲有幾個實用工具你要識。"
            "第一，Mixed Precision。用半精度浮點數做計算，可以慳一半記憶體，仲快啲。"
            "第二，Gradient Clipping。當梯度太大嘅時候，將佢裁剪返細啲，防止訓練爆炸。"
            "第三，Checkpointing。定時儲存模型嘅狀態，萬一訓練中斷咗可以繼續。"
            "第四，設定隨機種子確保 Reproducibility，即係每次跑嘅結果都一樣。"
            "呢四個工具喺實際訓練 LLM 嘅時候好常用。"
        ):
            self.play(Write(heading), run_time=0.6)
            for group in tool_groups:
                self.play(FadeIn(group, shift=RIGHT * 0.3), run_time=0.6)

        self.wait(0.3)
        self.clear()

    # ── Scene 10 — Summary ───────────────────────────────────────────────

    def scene_summary(self):
        heading = self.make_heading("總結", font_size=44)

        bullets = VGroup(
            self.zh("•  Tensor 係 PyTorch 嘅基本數據結構", font_size=26, color=C_GREEN),
            self.zh("•  Autograd 自動計算梯度", font_size=26, color=C_ORANGE),
            self.zh("•  nn.Module 組裝神經網路", font_size=26, color=C_PINK),
            self.zh("•  訓練循環：Forward → Loss → Backward → Step", font_size=26, color=C_PURPLE),
            self.zh("•  AdamW 係 LLM 最常用嘅優化器", font_size=26, color=C_CYAN),
            self.zh("•  Mixed Precision 節省記憶體同加速", font_size=26, color=C_YELLOW),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.35).move_to(ORIGIN)

        with self.voiceover(
            text="總結一下今日學咗嘅嘢。"
            "Tensor 係 PyTorch 嘅基本數據結構，好似加強版嘅 NumPy array。"
            "Autograd 可以自動計算所有梯度，唔使你自己寫反向傳播。"
            "nn dot Module 幫你將 layers 組裝成一個完整嘅神經網路。"
            "訓練循環有五個步驟：Forward、Loss、Backward、Step、同 Zero Grad。"
            "AdamW 係而家所有 LLM 最常用嘅優化器。"
            "Mixed Precision 可以節省記憶體同加速訓練。"
            "掌握咗呢啲工具，你就準備好建造自己嘅 neural network 喇。"
        ):
            self.play(Write(heading), run_time=0.6)
            for b in bullets:
                self.play(FadeIn(b, shift=RIGHT * 0.3), run_time=0.5)

        self.wait(0.5)
        self.clear()

        thanks = self.zh("多謝收睇！", font_size=52, color=C_YELLOW)

        with self.voiceover(
            text="多謝收睇！下一課我哋會學 Neural Network Building Blocks，"
            "包括 activation functions、layer normalization 同 residual connections。"
            "記得繼續跟住學喇！"
        ):
            self.play(Write(thanks), run_time=1)

        self.wait(1)
        self.play(FadeOut(thanks), run_time=0.8)
