"""
Lesson 8 – Pre-Training Your LLM
Manim CE + Cantonese voiceover (Edge TTS)

Render (1080p):
    cd "lesson 8/video"
    manim render -qh scene.py PreTrainingExplainer
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


class PreTrainingExplainer(VoiceoverScene):
    """Single scene explaining LLM pre-training in Cantonese."""

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
        self.scene_objective()
        self.scene_loss_function()
        self.scene_optimizer()
        self.scene_lr_schedule()
        self.scene_training_loop()
        self.scene_checkpointing()
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
        title = self.zh("預訓練 LLM", font_size=56, color=C_BLUE)
        sub_en = self.en(
            "Pre-Training Your LLM", font_size=24, color=C_WHITE
        ).next_to(title, DOWN, buff=0.4)
        sub_zh = self.zh(
            "第八課", font_size=36, color=C_ORANGE
        ).next_to(sub_en, DOWN, buff=0.3)
        producer = self.zh(
            "制片人：Peter", font_size=24, color=C_DIM
        ).next_to(sub_zh, DOWN, buff=0.7)

        with self.voiceover(
            text="大家好！歡迎嚟到第八課。"
            "前幾課我哋搭建好咗 Transformer 架構同數據管線。"
            "今日我哋終於要訓練一個語言模型喇！"
            "我哋會學習訓練目標、損失函數、優化器、學習率調度、"
            "完整嘅訓練循環，同埋點樣儲存同恢復訓練進度。"
        ):
            self.play(Write(title), run_time=1.5)
            self.play(FadeIn(sub_en, shift=UP * 0.2), run_time=0.8)
            self.play(FadeIn(sub_zh, shift=UP * 0.2), run_time=0.8)
            self.play(FadeIn(producer, shift=UP * 0.2), run_time=0.5)

        self.wait(0.3)
        self.clear()

    # ── Scene 2 — Training Objective (Next-Token Prediction) ─────────────

    def scene_objective(self):
        heading = self.make_heading("訓練目標：預測下一個 Token")

        # Full sequence at the top
        tokens = ["The", "cat", "sat", "on", "the", "mat"]
        tok_colors = [C_GREEN, C_ORANGE, C_PINK, C_PURPLE, C_CYAN, C_YELLOW]

        full_boxes = VGroup()
        for word, col in zip(tokens, tok_colors):
            box = self.make_mono_box(word, col, width=1.3, height=0.65, font_size=20)
            full_boxes.add(box)
        full_boxes.arrange(RIGHT, buff=0.2).shift(UP * 1.5)

        seq_label = self.zh(
            "完整序列", font_size=18, color=C_DIM
        ).next_to(full_boxes, LEFT, buff=0.4)

        # Input row: first 5 tokens
        input_label = self.zh(
            "Input", font_size=20, color=C_BLUE
        ).shift(LEFT * 5.5 + DOWN * 0.1)

        input_boxes = VGroup()
        for i in range(5):
            box = self.make_mono_box(
                tokens[i], tok_colors[i], width=1.3, height=0.65, font_size=20
            )
            input_boxes.add(box)
        input_boxes.arrange(RIGHT, buff=0.2).next_to(input_label, RIGHT, buff=0.3)

        # Target row: last 5 tokens (shifted by 1)
        target_label = self.zh(
            "Target", font_size=20, color=C_RED
        ).shift(LEFT * 5.5 + DOWN * 1.5)

        target_boxes = VGroup()
        for i in range(1, 6):
            box = self.make_mono_box(
                tokens[i], tok_colors[i], width=1.3, height=0.65, font_size=20
            )
            target_boxes.add(box)
        target_boxes.arrange(RIGHT, buff=0.2).next_to(target_label, RIGHT, buff=0.3)

        # Arrows from each input position to its target
        shift_arrows = VGroup()
        for i in range(5):
            a = Arrow(
                input_boxes[i].get_bottom(),
                target_boxes[i].get_top(),
                buff=0.1, color=C_WHITE, stroke_width=2,
            )
            shift_arrows.add(a)

        # Bottom explanation
        explain = VGroup(
            RoundedRectangle(
                corner_radius=0.12, width=10, height=0.85,
                fill_color=C_CYAN, fill_opacity=0.12,
                stroke_color=C_CYAN, stroke_width=1.5,
            ),
            self.zh(
                "每個位置預測下一個 token → N 個 token 產生 N-1 個訓練樣本",
                font_size=20, color=C_CYAN,
            ),
        )
        explain[1].move_to(explain[0])
        explain.to_edge(DOWN, buff=0.35)

        with self.voiceover(
            text="LLM 嘅訓練目標非常簡單：預測下一個 token。"
            "假設我哋有一個句子 The cat sat on the mat。"
            "我哋將佢分成 input 同 target。"
            "Input 係頭五個 token：The cat sat on the。"
            "Target 係後五個 token：cat sat on the mat。"
            "即係將整個序列向右移一個位置。"
            "每個位置，模型都要預測下一個 token 係乜。"
            "位置零，The 預測 cat。位置一，cat 預測 sat。如此類推。"
            "呢個就係 causal language modeling，"
            "一個 N 個 token 嘅序列可以產生 N 減一個訓練樣本。"
        ):
            self.play(Write(heading), run_time=0.6)

            self.play(FadeIn(seq_label), run_time=0.3)
            for box in full_boxes:
                self.play(FadeIn(box, shift=RIGHT * 0.15), run_time=0.25)

            self.wait(0.3)

            self.play(FadeIn(input_label), run_time=0.3)
            self.play(
                *[FadeIn(b, shift=DOWN * 0.15) for b in input_boxes],
                run_time=0.6,
            )

            self.play(FadeIn(target_label), run_time=0.3)
            self.play(
                *[FadeIn(b, shift=DOWN * 0.15) for b in target_boxes],
                run_time=0.6,
            )

            for a in shift_arrows:
                self.play(GrowArrow(a), run_time=0.2)

            self.play(FadeIn(explain, shift=UP * 0.2), run_time=0.6)

        self.wait(0.3)
        self.clear()

    # ── Scene 3 — Loss Function (Cross-Entropy Bar Chart) ────────────────

    def scene_loss_function(self):
        heading = self.make_heading("損失函數：Cross-Entropy")

        # Bar chart: probability distribution over mini vocabulary
        bar_tokens = ["the", "cat", "sat", "on", "mat", "dog", "ran", ".."]
        bar_probs = [0.05, 0.08, 0.60, 0.03, 0.05, 0.07, 0.04, 0.08]
        bar_colors = [C_DIM, C_DIM, C_GREEN, C_DIM, C_DIM, C_DIM, C_DIM, C_DIM]
        correct_idx = 2

        axes_origin = DOWN * 0.6 + LEFT * 3

        # Bars
        bars = VGroup()
        bar_labels = VGroup()
        prob_labels = VGroup()
        max_height = 3.0
        bar_w = 0.55
        gap = 0.15

        for i, (tok, prob, col) in enumerate(zip(bar_tokens, bar_probs, bar_colors)):
            h = prob * max_height / 0.65
            rect = Rectangle(
                width=bar_w, height=h,
                fill_color=col if i != correct_idx else C_GREEN,
                fill_opacity=0.6 if i != correct_idx else 0.85,
                stroke_color=col if i != correct_idx else C_GREEN,
                stroke_width=2,
            )
            x_pos = axes_origin[0] + i * (bar_w + gap)
            rect.move_to(
                np.array([x_pos, axes_origin[1] + h / 2, 0])
            )
            bars.add(rect)

            tok_label = self.mono(
                tok, font_size=14, color=C_WHITE
            ).next_to(rect, DOWN, buff=0.15)
            bar_labels.add(tok_label)

            p_label = self.mono(
                f"{prob:.2f}", font_size=13, color=C_YELLOW
            ).next_to(rect, UP, buff=0.08)
            prob_labels.add(p_label)

        # Highlight the correct token
        correct_box = SurroundingRectangle(
            bars[correct_idx], color=C_YELLOW, buff=0.08, stroke_width=2.5,
        )
        correct_label = self.zh(
            "正確 token", font_size=16, color=C_YELLOW,
        ).next_to(correct_box, UP, buff=0.35)

        # Loss formula annotation on the right
        formula_box = VGroup(
            RoundedRectangle(
                corner_radius=0.12, width=4.0, height=2.2,
                fill_color=C_BLUE, fill_opacity=0.12,
                stroke_color=C_BLUE, stroke_width=1.5,
            ),
            self.mono("loss = -log(P)", font_size=18, color=C_CYAN),
            self.mono("     = -log(0.60)", font_size=18, color=C_CYAN),
            self.mono("     = 0.51", font_size=18, color=C_YELLOW),
            self.zh("PPL = exp(0.51) = 1.67", font_size=16, color=C_ORANGE),
        )
        formula_box[1:].arrange(DOWN, buff=0.15, aligned_edge=LEFT)
        VGroup(*formula_box[1:]).move_to(formula_box[0])
        formula_box.shift(RIGHT * 4.2 + UP * 0.3)

        # Bottom note
        bottom = VGroup(
            RoundedRectangle(
                corner_radius=0.12, width=10, height=0.65,
                fill_color=C_ORANGE, fill_opacity=0.12,
                stroke_color=C_ORANGE, stroke_width=1.5,
            ),
            self.zh(
                "模型越自信答啱 → P(正確) 越高 → loss 越低 → 訓練越好",
                font_size=18, color=C_ORANGE,
            ),
        )
        bottom[1].move_to(bottom[0])
        bottom.to_edge(DOWN, buff=0.3)

        with self.voiceover(
            text="損失函數用 cross-entropy。"
            "模型喺每個位置輸出一個概率分布，覆蓋整個詞彙表。"
            "呢度我哋用一個小例子。"
            "模型預測下一個 token，正確答案係 sat。"
            "你可以睇到每個 token 嘅預測概率。"
            "綠色嘅 bar 代表正確 token sat，概率係 0.6。"
            "Cross-entropy loss 等於負 log P，"
            "即係負 log 0.6，等於 0.51。"
            "Perplexity 就係 exp of loss，等於 1.67。"
            "意思係模型大約喺 1.67 個 token 入面猶豫。"
            "目標好簡單：令模型對正確 token 越嚟越有信心，"
            "噉 loss 就會越嚟越低。"
        ):
            self.play(Write(heading), run_time=0.6)

            for bar, lbl in zip(bars, bar_labels):
                self.play(GrowFromEdge(bar, DOWN), FadeIn(lbl), run_time=0.2)

            self.play(*[FadeIn(pl) for pl in prob_labels], run_time=0.5)
            self.play(Create(correct_box), FadeIn(correct_label), run_time=0.5)
            self.play(FadeIn(formula_box, shift=LEFT * 0.3), run_time=0.7)
            self.play(FadeIn(bottom, shift=UP * 0.2), run_time=0.5)

        self.wait(0.3)
        self.clear()

    # ── Scene 4 — Optimizer (Gradient Descent + AdamW) ───────────────────

    def scene_optimizer(self):
        heading = self.make_heading("優化器：AdamW")

        # Loss landscape curve
        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[0, 10, 2],
            x_length=7,
            y_length=4,
            axis_config={"color": C_DIM, "include_numbers": False},
        ).shift(LEFT * 1.5 + DOWN * 0.3)

        x_label = self.mono(
            "weight", font_size=16, color=C_DIM
        ).next_to(axes.x_axis, DOWN, buff=0.2)
        y_label = self.mono(
            "loss", font_size=16, color=C_DIM
        ).next_to(axes.y_axis, UP, buff=0.2)

        curve = axes.plot(
            lambda x: (x - 0.5) ** 2 + 0.3 * (x - 0.5) ** 4 + 0.5,
            x_range=[-2.5, 2.8],
            color=C_BLUE,
        )

        # SGD path: noisy, slow steps
        sgd_points = [-2.0, -1.5, -1.2, -0.8, -0.4, -0.1, 0.1, 0.3, 0.4]
        sgd_dots = VGroup()
        for xp in sgd_points:
            yp = (xp - 0.5) ** 2 + 0.3 * (xp - 0.5) ** 4 + 0.5
            dot = Dot(axes.c2p(xp, yp), color=C_RED, radius=0.06)
            sgd_dots.add(dot)

        sgd_label = self.mono("SGD", font_size=18, color=C_RED)
        sgd_label.next_to(sgd_dots[0], UP, buff=0.2)

        # AdamW path: smooth, faster
        adamw_points = [-2.0, -0.8, 0.0, 0.3, 0.45, 0.5]
        adamw_dots = VGroup()
        for xp in adamw_points:
            yp = (xp - 0.5) ** 2 + 0.3 * (xp - 0.5) ** 4 + 0.5
            dot = Dot(axes.c2p(xp, yp), color=C_GREEN, radius=0.06)
            adamw_dots.add(dot)

        adamw_label = self.mono("AdamW", font_size=18, color=C_GREEN)
        adamw_label.next_to(adamw_dots[0], UR, buff=0.15)

        # Minimum marker
        min_dot = Dot(axes.c2p(0.5, 0.5), color=C_YELLOW, radius=0.1)
        min_label = self.mono(
            "min", font_size=16, color=C_YELLOW
        ).next_to(min_dot, DOWN, buff=0.15)

        # AdamW components annotation on the right
        info_box = VGroup(
            RoundedRectangle(
                corner_radius=0.12, width=4.5, height=3.5,
                fill_color=C_GREEN, fill_opacity=0.08,
                stroke_color=C_GREEN, stroke_width=1.5,
            ),
            self.zh("AdamW 三大要素", font_size=20, color=C_GREEN),
            self.zh("1. 動量 (momentum)", font_size=17, color=C_ORANGE),
            self.zh("   平滑梯度噪音", font_size=15, color=C_DIM),
            self.zh("2. 自適應學習率", font_size=17, color=C_PINK),
            self.zh("   每個參數獨立調整", font_size=15, color=C_DIM),
            self.zh("3. 權重衰減 (decoupled)", font_size=17, color=C_CYAN),
            self.zh("   防止過擬合", font_size=15, color=C_DIM),
        )
        formula_items = VGroup(*info_box[1:])
        formula_items.arrange(DOWN, buff=0.1, aligned_edge=LEFT)
        formula_items.move_to(info_box[0])
        info_box.shift(RIGHT * 4.5 + DOWN * 0.2)

        with self.voiceover(
            text="有咗 loss 之後，我哋需要一個 optimizer 嚟更新模型嘅權重。"
            "呢度係一個 loss landscape，x 軸係權重嘅值，y 軸係 loss。"
            "我哋嘅目標係搵到 loss 最低嘅位置。"
            "最簡單嘅方法係 SGD，即 stochastic gradient descent。"
            "但係 SGD 嘅步伐小而且受噪音影響大，收斂好慢。"
            "AdamW 就快好多。佢有三個關鍵特性。"
            "第一，動量：平滑梯度噪音，令步伐更穩定。"
            "第二，自適應學習率：每個參數有自己嘅學習率。"
            "第三，decoupled 權重衰減：直接縮小權重，防止過擬合。"
            "GPT、LLaMA、所有大型語言模型都用 AdamW。"
        ):
            self.play(Write(heading), run_time=0.6)
            self.play(
                Create(axes), FadeIn(x_label), FadeIn(y_label),
                run_time=0.6,
            )
            self.play(Create(curve), run_time=1.0)
            self.play(FadeIn(min_dot, scale=1.5), FadeIn(min_label), run_time=0.4)

            # SGD path
            self.play(FadeIn(sgd_label), run_time=0.3)
            for dot in sgd_dots:
                self.play(FadeIn(dot, scale=1.3), run_time=0.2)

            # AdamW path
            self.play(FadeIn(adamw_label), run_time=0.3)
            for dot in adamw_dots:
                self.play(FadeIn(dot, scale=1.3), run_time=0.25)

            self.play(FadeIn(info_box, shift=LEFT * 0.3), run_time=0.7)

        self.wait(0.3)
        self.clear()

    # ── Scene 5 — Learning Rate Schedule ─────────────────────────────────

    def scene_lr_schedule(self):
        heading = self.make_heading("學習率調度 Learning Rate Schedule")

        import math as _math

        # Plot warmup + cosine decay
        axes = Axes(
            x_range=[0, 1000, 200],
            y_range=[0, 3.5e-4, 5e-5],
            x_length=9,
            y_length=4,
            axis_config={"color": C_DIM, "include_numbers": False},
        ).shift(DOWN * 0.3)

        x_label = self.mono(
            "Training Step", font_size=16, color=C_WHITE
        ).next_to(axes.x_axis, DOWN, buff=0.3)
        y_label = self.mono(
            "LR", font_size=16, color=C_WHITE
        ).next_to(axes.y_axis, UP, buff=0.2)

        warmup_steps = 100
        max_steps = 1000
        max_lr = 3e-4
        min_lr = 3e-5

        def lr_func(step):
            if step < warmup_steps:
                return max_lr * step / warmup_steps
            progress = (step - warmup_steps) / (max_steps - warmup_steps)
            return min_lr + 0.5 * (max_lr - min_lr) * (1 + _math.cos(_math.pi * progress))

        # Warmup phase curve
        warmup_curve = axes.plot(
            lr_func, x_range=[0, warmup_steps], color=C_ORANGE,
        )

        # Cosine decay phase curve
        cosine_curve = axes.plot(
            lr_func, x_range=[warmup_steps, max_steps], color=C_CYAN,
        )

        # Labels for phases
        warmup_label = self.zh(
            "Warmup", font_size=20, color=C_ORANGE,
        ).move_to(axes.c2p(50, max_lr * 0.75))

        cosine_label = self.zh(
            "Cosine Decay", font_size=20, color=C_CYAN,
        ).move_to(axes.c2p(550, max_lr * 0.75))

        # Peak LR marker
        peak_dot = Dot(axes.c2p(warmup_steps, max_lr), color=C_YELLOW, radius=0.08)
        peak_label = self.mono(
            f"peak = {max_lr:.1e}", font_size=16, color=C_YELLOW
        ).next_to(peak_dot, UR, buff=0.15)

        # Min LR marker
        min_dot = Dot(axes.c2p(max_steps, min_lr), color=C_DIM, radius=0.08)
        min_label_text = self.mono(
            f"min = {min_lr:.1e}", font_size=16, color=C_DIM
        ).next_to(min_dot, LEFT, buff=0.2)

        # Vertical dashed line at warmup boundary
        warmup_line = DashedLine(
            axes.c2p(warmup_steps, 0), axes.c2p(warmup_steps, max_lr),
            color=C_ORANGE, stroke_width=1.5,
        )

        # Step labels
        step_labels = VGroup()
        for s in [0, 200, 400, 600, 800, 1000]:
            lbl = self.mono(
                str(s), font_size=14, color=C_DIM
            ).next_to(axes.c2p(s, 0), DOWN, buff=0.1)
            step_labels.add(lbl)

        # Bottom explanation
        note = VGroup(
            RoundedRectangle(
                corner_radius=0.12, width=10, height=0.85,
                fill_color=C_PURPLE, fill_opacity=0.12,
                stroke_color=C_PURPLE, stroke_width=1.5,
            ),
            self.zh(
                "Warmup 穩定初期訓練 → Cosine Decay 平滑降低學習率",
                font_size=20, color=C_PURPLE,
            ),
        )
        note[1].move_to(note[0])
        note.to_edge(DOWN, buff=0.3)

        with self.voiceover(
            text="學習率唔可以由頭到尾都一樣。所有 LLM 都用一個兩階段嘅 schedule。"
            "第一階段係 warmup，喺頭一百步將學習率由零線性增加到最大值。"
            "點解要 warmup 呢？因為訓練初期，模型嘅權重係隨機嘅，"
            "梯度好大好唔穩定。如果一開始就用高學習率，訓練會爆炸。"
            "Warmup 令 optimizer 嘅 momentum 先穩定落嚟。"
            "第二階段係 cosine decay，將學習率由最大值平滑噉降低到最小值。"
            "Cosine 嘅好處係佢好平滑，冇突然嘅跳變。"
            "呢個就係 GPT-3 同 LLaMA 用嘅標準 schedule。"
        ):
            self.play(Write(heading), run_time=0.6)
            self.play(Create(axes), FadeIn(x_label), FadeIn(y_label), run_time=0.6)
            self.play(*[FadeIn(sl) for sl in step_labels], run_time=0.3)

            self.play(Create(warmup_curve), run_time=1.0)
            self.play(FadeIn(warmup_label), run_time=0.3)
            self.play(Create(warmup_line), run_time=0.3)
            self.play(FadeIn(peak_dot, scale=1.5), FadeIn(peak_label), run_time=0.4)

            self.play(Create(cosine_curve), run_time=1.5)
            self.play(FadeIn(cosine_label), run_time=0.3)
            self.play(FadeIn(min_dot), FadeIn(min_label_text), run_time=0.3)

            self.play(FadeIn(note, shift=UP * 0.2), run_time=0.6)

        self.wait(0.3)
        self.clear()

    # ── Scene 6 — Training Loop (Animated Cycle) ─────────────────────────

    def scene_training_loop(self):
        heading = self.make_heading("訓練循環 Training Loop")

        # 4 main steps arranged in a rectangle
        steps = [
            ("Forward Pass\n前向傳播", C_GREEN),
            ("Compute Loss\n計算損失", C_ORANGE),
            ("Backward Pass\n反向傳播", C_PINK),
            ("Optimizer Step\n更新權重", C_CYAN),
        ]

        positions = [
            LEFT * 2.5 + UP * 1.0,     # top-left
            RIGHT * 2.5 + UP * 1.0,    # top-right
            RIGHT * 2.5 + DOWN * 1.0,  # bottom-right
            LEFT * 2.5 + DOWN * 1.0,   # bottom-left
        ]

        step_boxes = VGroup()
        for (label, col), pos in zip(steps, positions):
            box = self.make_box(label, col, width=3.0, height=1.0, font_size=18)
            box.move_to(pos)
            step_boxes.add(box)

        # Arrows connecting the cycle
        cycle_arrows = VGroup()
        for i in range(4):
            start = step_boxes[i].get_right() if i % 2 == 0 else step_boxes[i].get_bottom()
            end = step_boxes[(i + 1) % 4].get_left() if i % 2 == 0 else step_boxes[(i + 1) % 4].get_top()
            if i == 0:
                start = step_boxes[0].get_right()
                end = step_boxes[1].get_left()
            elif i == 1:
                start = step_boxes[1].get_bottom()
                end = step_boxes[2].get_top()
            elif i == 2:
                start = step_boxes[2].get_left()
                end = step_boxes[3].get_right()
            elif i == 3:
                start = step_boxes[3].get_top()
                end = step_boxes[0].get_bottom()
            a = Arrow(
                start, end,
                buff=0.12, color=C_WHITE, stroke_width=3,
            )
            cycle_arrows.add(a)

        # Center label: "repeat"
        repeat_label = self.zh(
            "重複", font_size=28, color=C_YELLOW
        ).move_to(ORIGIN)
        repeat_circle = Circle(
            radius=0.6, color=C_YELLOW, stroke_width=2, fill_opacity=0.1,
            fill_color=C_YELLOW,
        ).move_to(ORIGIN)

        # Extra details below
        details = VGroup(
            self.mono("+ gradient clipping (max_norm=1.0)", font_size=16, color=C_DIM),
            self.mono("+ learning rate scheduling", font_size=16, color=C_DIM),
            self.mono("+ loss / perplexity / grad norm logging", font_size=16, color=C_DIM),
        ).arrange(DOWN, buff=0.12, aligned_edge=LEFT).to_edge(DOWN, buff=0.35)

        with self.voiceover(
            text="而家將所有嘢組合成一個完整嘅訓練循環。"
            "第一步，Forward Pass：將 token 輸入模型，得到 logits。"
            "第二步，Compute Loss：用 cross-entropy 計算損失。"
            "第三步，Backward Pass：反向傳播計算所有參數嘅梯度。"
            "第四步，Optimizer Step：AdamW 更新所有權重。"
            "然後清零梯度，更新學習率，重複。"
            "每一步都要做 gradient clipping，防止梯度爆炸。"
            "同時記錄 loss、perplexity 同 gradient norm，方便監測訓練進度。"
            "呢個循環執行幾十萬到幾百萬步，就可以訓練出一個語言模型。"
        ):
            self.play(Write(heading), run_time=0.6)

            for i, box in enumerate(step_boxes):
                self.play(FadeIn(box, shift=DOWN * 0.15), run_time=0.4)
                if i < len(cycle_arrows):
                    self.play(GrowArrow(cycle_arrows[i]), run_time=0.3)
            self.play(GrowArrow(cycle_arrows[-1]), run_time=0.3)

            self.play(
                FadeIn(repeat_label, scale=1.3),
                Create(repeat_circle),
                run_time=0.6,
            )

            # Animate a "pulse" going around the cycle
            pulse = Dot(
                step_boxes[0].get_center(), color=C_YELLOW, radius=0.12
            )
            self.play(FadeIn(pulse, scale=2), run_time=0.3)
            for i in range(4):
                self.play(
                    pulse.animate.move_to(step_boxes[(i + 1) % 4].get_center()),
                    run_time=0.4,
                )
            self.play(FadeOut(pulse), run_time=0.2)

            self.play(FadeIn(details, shift=UP * 0.15), run_time=0.5)

        self.wait(0.3)
        self.clear()

    # ── Scene 7 — Checkpointing (Save / Load Boxes) ─────────────────────

    def scene_checkpointing(self):
        heading = self.make_heading("Checkpointing 訓練儲存點")

        # Three data components
        comp_model = self.make_box(
            "模型權重\nModel Weights", C_GREEN, width=3.0, height=1.0, font_size=18
        )
        comp_optim = self.make_box(
            "優化器狀態\nOptimizer State", C_ORANGE, width=3.0, height=1.0, font_size=18
        )
        comp_step = self.make_box(
            "Step + Config\n+ RNG State", C_CYAN, width=3.0, height=1.0, font_size=18
        )

        components = VGroup(comp_model, comp_optim, comp_step).arrange(
            RIGHT, buff=0.4
        ).shift(UP * 1.3)

        # Disk icon (represented as a rounded rectangle)
        disk = VGroup(
            RoundedRectangle(
                corner_radius=0.2, width=4.0, height=1.2,
                fill_color=C_PURPLE, fill_opacity=0.25,
                stroke_color=C_PURPLE, stroke_width=2.5,
            ),
            self.mono("checkpoint.pt", font_size=22, color=C_PURPLE),
        )
        disk[1].move_to(disk[0])
        disk.shift(DOWN * 1.0)

        # Save arrows (down)
        save_arrows = VGroup()
        for comp in components:
            a = Arrow(
                comp.get_bottom(), disk[0].get_top(),
                buff=0.15, color=C_YELLOW, stroke_width=2.5,
            )
            save_arrows.add(a)

        save_label = self.zh(
            "SAVE  儲存", font_size=22, color=C_YELLOW
        ).next_to(save_arrows[1], LEFT, buff=0.3)

        # Why save optimizer state?
        why_box = VGroup(
            RoundedRectangle(
                corner_radius=0.12, width=10, height=1.2,
                fill_color=C_RED, fill_opacity=0.12,
                stroke_color=C_RED, stroke_width=1.5,
            ),
            self.zh("必須儲存 Optimizer State！", font_size=20, color=C_RED),
            self.zh(
                "如果唔儲存 → Adam 嘅 momentum 歸零 → 恢復訓練質素下降",
                font_size=17, color=C_DIM,
            ),
        )
        why_box[1:].arrange(DOWN, buff=0.1)
        VGroup(*why_box[1:]).move_to(why_box[0])
        why_box.to_edge(DOWN, buff=0.3)

        with self.voiceover(
            text="訓練一個大型語言模型需要好長時間，可能幾日甚至幾個星期。"
            "如果中途硬件壞咗或者 job 被取消，冇 checkpoint 就要由頭嚟過。"
            "所以我哋需要定期儲存訓練進度。"
            "一個完整嘅 checkpoint 包含三個部分。"
            "第一，模型嘅權重，即係 state dict。"
            "第二，optimizer 嘅狀態，包括 momentum 同 adaptive learning rate。"
            "第三，training step、模型配置同 random state。"
            "特別要注意，你一定要儲存 optimizer state。"
            "如果唔儲存，恢復訓練嗰時 Adam 嘅 momentum 會歸零，"
            "訓練質素會下降。"
        ):
            self.play(Write(heading), run_time=0.6)

            for comp in components:
                self.play(FadeIn(comp, shift=DOWN * 0.15), run_time=0.4)

            self.play(FadeIn(disk, shift=UP * 0.2), run_time=0.5)

            for a in save_arrows:
                self.play(GrowArrow(a), run_time=0.25)
            self.play(FadeIn(save_label), run_time=0.3)

            self.play(FadeIn(why_box, shift=UP * 0.2), run_time=0.6)

        self.wait(0.3)
        self.clear()

    # ── Scene 8 — Summary ───────────────────────────────────────────────

    def scene_summary(self):
        heading = self.make_heading("總結", font_size=44)

        bullets = VGroup(
            self.zh("•  訓練目標：預測下一個 token（Causal LM）", font_size=24, color=C_GREEN),
            self.zh("•  損失函數：Cross-Entropy → Perplexity = exp(loss)", font_size=24, color=C_ORANGE),
            self.zh("•  優化器：AdamW = 動量 + 自適應 LR + 權重衰減", font_size=24, color=C_PINK),
            self.zh("•  學習率：Warmup + Cosine Decay", font_size=24, color=C_CYAN),
            self.zh("•  訓練循環：Forward → Loss → Backward → Step", font_size=24, color=C_PURPLE),
            self.zh("•  Checkpointing：保存模型 + 優化器 + Step", font_size=24, color=C_YELLOW),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.35).move_to(ORIGIN)

        with self.voiceover(
            text="總結一下今日學咗嘅嘢。"
            "第一，LLM 嘅訓練目標係預測下一個 token，即 causal language modeling。"
            "第二，損失函數用 cross-entropy，perplexity 就係 exp of loss。"
            "第三，優化器用 AdamW，佢結合咗動量、自適應學習率同 decoupled 權重衰減。"
            "第四，學習率用 warmup 加 cosine decay，穩定訓練過程。"
            "第五，訓練循環係 forward、loss、backward、optimizer step，"
            "加上 gradient clipping 同 logging。"
            "第六，Checkpointing 要保存模型、optimizer state 同 step，"
            "先可以完美恢復訓練。"
            "掌握咗呢啲，你已經識得從零開始訓練一個語言模型喇！"
        ):
            self.play(Write(heading), run_time=0.6)
            for b in bullets:
                self.play(FadeIn(b, shift=RIGHT * 0.3), run_time=0.5)

        self.wait(0.5)
        self.clear()

    # ── Scene 9 — Outro ─────────────────────────────────────────────────

    def scene_outro(self):
        thanks = self.zh("多謝收睇！", font_size=52, color=C_YELLOW)
        next_lesson = self.zh(
            "下一課：Distributed Training（分佈式訓練）",
            font_size=28, color=C_CYAN,
        ).next_to(thanks, DOWN, buff=0.6)

        with self.voiceover(
            text="多謝收睇！下一課我哋會學 Distributed Training，"
            "即係點樣用多張 GPU 嚟訓練更大嘅模型。"
            "包括 Data Parallelism、FSDP、Mixed Precision、"
            "Flash Attention 同 Chinchilla scaling laws。"
            "記得繼續跟住學喇！"
        ):
            self.play(Write(thanks), run_time=1)
            self.play(FadeIn(next_lesson, shift=UP * 0.2), run_time=0.8)

        self.wait(1)
        self.play(FadeOut(thanks), FadeOut(next_lesson), run_time=0.8)
