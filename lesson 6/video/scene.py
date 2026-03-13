"""
Lesson 6 – The Transformer Architecture
Manim CE + Cantonese voiceover (Edge TTS)

Render (1080p):
    cd "lesson 6/video"
    manim render -qh scene.py TransformerExplainer
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


class TransformerExplainer(VoiceoverScene):
    """Single scene explaining the Transformer architecture in Cantonese."""

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
        self.scene_enc_vs_dec()
        self.scene_block()
        self.scene_positional()
        self.scene_rope()
        self.scene_full_model()
        self.scene_dimensions()
        self.scene_generation()
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
        title = self.zh("Transformer 架構", font_size=56, color=C_BLUE)
        sub_en = self.en(
            "The Transformer Architecture", font_size=24, color=C_WHITE
        ).next_to(title, DOWN, buff=0.4)
        sub_zh = self.zh(
            "第六課", font_size=36, color=C_ORANGE
        ).next_to(sub_en, DOWN, buff=0.3)
        producer = self.zh(
            "制片人：Peter", font_size=24, color=C_DIM
        ).next_to(sub_zh, DOWN, buff=0.7)

        with self.voiceover(
            text="大家好！歡迎嚟到第六課。"
            "前幾課我哋學咗 attention 機制、tokenization 同 neural network 嘅組件。"
            "今日我哋會將所有嘢組裝埋一齊，建造一個完整嘅 Transformer 模型。"
            "呢個就係 GPT、LLaMA 同所有現代 LLM 嘅核心架構。"
        ):
            self.play(Write(title), run_time=1.5)
            self.play(FadeIn(sub_en, shift=UP * 0.2), run_time=0.8)
            self.play(FadeIn(sub_zh, shift=UP * 0.2), run_time=0.8)
            self.play(FadeIn(producer, shift=UP * 0.2), run_time=0.5)

        self.wait(0.3)
        self.clear()

    # ── Scene 2 — Encoder-Decoder vs Decoder-Only ────────────────────────

    def scene_enc_vs_dec(self):
        heading = self.make_heading("兩種 Transformer 架構")

        # --- Left: Encoder-Decoder ---
        enc_title = self.zh(
            "Encoder-Decoder（2017）", font_size=22, color=C_ORANGE
        )
        enc_box = RoundedRectangle(
            corner_radius=0.15, width=4.5, height=4.0,
            fill_color=C_ORANGE, fill_opacity=0.08,
            stroke_color=C_ORANGE, stroke_width=2,
        )

        enc_inner = self.make_box("Encoder", C_GREEN, width=3.2, height=0.7)
        dec_inner = self.make_box("Decoder", C_PURPLE, width=3.2, height=0.7)
        cross_attn = self.make_box("Cross\nAttention", C_ORANGE, width=3.2, height=0.7)

        enc_stack = VGroup(enc_inner, cross_attn, dec_inner).arrange(
            DOWN, buff=0.3
        )
        enc_stack.move_to(enc_box)
        enc_title.next_to(enc_box, UP, buff=0.2)

        enc_arrow1 = Arrow(
            enc_inner.get_bottom(), cross_attn.get_top(),
            buff=0.08, color=C_WHITE, stroke_width=2,
        )
        enc_arrow2 = Arrow(
            cross_attn.get_bottom(), dec_inner.get_top(),
            buff=0.08, color=C_WHITE, stroke_width=2,
        )

        enc_group = VGroup(enc_box, enc_title, enc_stack, enc_arrow1, enc_arrow2)
        enc_group.shift(LEFT * 3.2)

        enc_use = self.zh(
            "翻譯、摘要", font_size=18, color=C_DIM
        ).next_to(enc_box, DOWN, buff=0.2)

        # --- Right: Decoder-Only ---
        dec_title = self.zh(
            "Decoder-Only（GPT 風格）", font_size=22, color=C_CYAN
        )
        dec_box = RoundedRectangle(
            corner_radius=0.15, width=4.5, height=4.0,
            fill_color=C_CYAN, fill_opacity=0.08,
            stroke_color=C_CYAN, stroke_width=2,
        )

        dec_block1 = self.make_box("Causal Attention", C_CYAN, width=3.2, height=0.7)
        dec_block2 = self.make_box("FFN", C_GREEN, width=3.2, height=0.7)
        dec_label = self.zh("x N 層", font_size=20, color=C_YELLOW)

        dec_stack = VGroup(dec_block1, dec_block2, dec_label).arrange(
            DOWN, buff=0.3
        )
        dec_stack.move_to(dec_box)
        dec_title.next_to(dec_box, UP, buff=0.2)

        dec_arrow = Arrow(
            dec_block1.get_bottom(), dec_block2.get_top(),
            buff=0.08, color=C_WHITE, stroke_width=2,
        )

        dec_group = VGroup(dec_box, dec_title, dec_stack, dec_arrow)
        dec_group.shift(RIGHT * 3.2)

        dec_use = self.zh(
            "GPT / LLaMA / Mistral", font_size=18, color=C_DIM
        ).next_to(dec_box, DOWN, buff=0.2)

        # Winner highlight
        winner = VGroup(
            RoundedRectangle(
                corner_radius=0.15, width=5, height=0.65,
                fill_color=C_YELLOW, fill_opacity=0.12, stroke_color=C_YELLOW,
            ),
            self.zh(
                "Decoder-Only 更簡單、更容易擴展", font_size=20, color=C_YELLOW,
            ),
        )
        winner[1].move_to(winner[0])
        winner.to_edge(DOWN, buff=0.3)

        with self.voiceover(
            text="原始嘅 Transformer 係 2017 年發明嘅，佢有兩個部分。"
            "Encoder 負責讀取輸入，Decoder 負責生成輸出。"
            "中間有一個 cross attention 連接佢哋。"
            "呢個設計主要用嚟做翻譯同摘要。"
            "但係 GPT 簡化咗呢個設計，淨係用 Decoder，"
            "加上 causal masking，令每個 token 只能睇到之前嘅 token。"
            "呢個 Decoder-Only 設計更加簡單，更加容易擴展。"
            "所以而家幾乎所有大型語言模型，包括 GPT、LLaMA、Mistral，"
            "全部都係用 Decoder-Only 架構。"
        ):
            self.play(Write(heading), run_time=0.6)
            self.play(
                FadeIn(enc_box), FadeIn(enc_title),
                run_time=0.5,
            )
            for item in [enc_inner, enc_arrow1, cross_attn, enc_arrow2, dec_inner]:
                self.play(FadeIn(item, shift=DOWN * 0.15), run_time=0.35)
            self.play(FadeIn(enc_use), run_time=0.3)

            self.play(
                FadeIn(dec_box), FadeIn(dec_title),
                run_time=0.5,
            )
            self.play(FadeIn(dec_block1, shift=DOWN * 0.15), run_time=0.35)
            self.play(GrowArrow(dec_arrow), run_time=0.25)
            self.play(FadeIn(dec_block2, shift=DOWN * 0.15), run_time=0.35)
            self.play(FadeIn(dec_label), run_time=0.3)
            self.play(FadeIn(dec_use), run_time=0.3)

            self.play(FadeIn(winner, shift=UP * 0.2), run_time=0.6)

        self.wait(0.3)
        self.clear()

    # ── Scene 3 — Transformer Block (the KEY diagram) ────────────────────

    def scene_block(self):
        heading = self.make_heading("Transformer Block")

        # Build the block diagram vertically (bottom to top)
        box_input = self.make_box("Input", C_WHITE, width=3.0, height=0.6)
        box_attn = self.make_box("Multi-Head\nAttention", C_CYAN, width=3.0, height=0.8)
        box_norm1 = self.make_box("Add & Norm", C_GREEN, width=3.0, height=0.6)
        box_ffn = self.make_box("Feed-Forward\nNetwork", C_ORANGE, width=3.0, height=0.8)
        box_norm2 = self.make_box("Add & Norm", C_GREEN, width=3.0, height=0.6)
        box_output = self.make_box("Output", C_WHITE, width=3.0, height=0.6)

        block_col = VGroup(
            box_input, box_attn, box_norm1, box_ffn, box_norm2, box_output
        ).arrange(UP, buff=0.28)
        block_col.move_to(LEFT * 1 + DOWN * 0.15)

        # Straight arrows between consecutive boxes
        straight_arrows = VGroup()
        pairs = [
            (box_input, box_attn),
            (box_attn, box_norm1),
            (box_norm1, box_ffn),
            (box_ffn, box_norm2),
            (box_norm2, box_output),
        ]
        for bot, top in pairs:
            a = Arrow(
                bot.get_top(), top.get_bottom(),
                buff=0.06, color=C_WHITE, stroke_width=2.5,
            )
            straight_arrows.add(a)

        # Residual connection 1: input → Add&Norm1 (curves around attention)
        res1_start = box_input.get_right() + RIGHT * 0.1
        res1_end = box_norm1.get_right() + RIGHT * 0.1
        res1 = CurvedArrow(
            res1_start, res1_end,
            angle=-TAU / 4,
            color=C_PINK, stroke_width=2.5,
        )
        res1_label = self.zh("殘差", font_size=16, color=C_PINK).next_to(
            res1.get_center(), RIGHT, buff=0.15
        )

        # Residual connection 2: norm1 → Add&Norm2 (curves around FFN)
        res2_start = box_norm1.get_right() + RIGHT * 0.1
        res2_end = box_norm2.get_right() + RIGHT * 0.1
        res2 = CurvedArrow(
            res2_start, res2_end,
            angle=-TAU / 4,
            color=C_PINK, stroke_width=2.5,
        )
        res2_label = self.zh("殘差", font_size=16, color=C_PINK).next_to(
            res2.get_center(), RIGHT, buff=0.15
        )

        # Pre-norm annotation on the left
        prenorm_note = VGroup(
            RoundedRectangle(
                corner_radius=0.12, width=3.2, height=1.6,
                fill_color=C_PURPLE, fill_opacity=0.12,
                stroke_color=C_PURPLE, stroke_width=1.5,
            ),
            self.zh("Pre-Norm 設計：", font_size=18, color=C_PURPLE),
            self.zh("先 Norm → 再 Attention", font_size=16, color=C_DIM),
            self.zh("先 Norm → 再 FFN", font_size=16, color=C_DIM),
            self.zh("訓練更穩定", font_size=16, color=C_YELLOW),
        )
        prenorm_note[1:].arrange(DOWN, buff=0.12, aligned_edge=LEFT)
        VGroup(*prenorm_note[1:]).move_to(prenorm_note[0])
        prenorm_note.shift(LEFT * 5.2 + DOWN * 0.2)

        with self.voiceover(
            text="呢個係今日最重要嘅圖。一個 Transformer block 嘅結構係咁嘅。"
            "由下至上，輸入首先經過 Multi-Head Attention。"
            "Attention 嘅輸出加返原本嘅輸入，呢個就係殘差連接。"
            "然後做 Layer Normalization。"
            "之後經過 Feed Forward Network，再加殘差連接同 Normalization。"
            "最後輸出。"
            "呢兩條彎曲嘅粉紅色箭頭就係殘差連接。"
            "佢哋令梯度可以直接通過，就算疊好多層都可以穩定訓練。"
            "現代嘅 LLM 用 Pre-Norm 設計，即係先做 Normalization 再做 Attention 同 FFN，"
            "呢樣令到訓練更加穩定。"
        ):
            self.play(Write(heading), run_time=0.6)

            self.play(FadeIn(box_input, shift=UP * 0.15), run_time=0.4)
            self.play(GrowArrow(straight_arrows[0]), run_time=0.25)
            self.play(FadeIn(box_attn, shift=UP * 0.15), run_time=0.5)
            self.play(GrowArrow(straight_arrows[1]), run_time=0.25)
            self.play(FadeIn(box_norm1, shift=UP * 0.15), run_time=0.4)

            self.play(Create(res1), FadeIn(res1_label), run_time=0.8)

            self.play(GrowArrow(straight_arrows[2]), run_time=0.25)
            self.play(FadeIn(box_ffn, shift=UP * 0.15), run_time=0.5)
            self.play(GrowArrow(straight_arrows[3]), run_time=0.25)
            self.play(FadeIn(box_norm2, shift=UP * 0.15), run_time=0.4)

            self.play(Create(res2), FadeIn(res2_label), run_time=0.8)

            self.play(GrowArrow(straight_arrows[4]), run_time=0.25)
            self.play(FadeIn(box_output, shift=UP * 0.15), run_time=0.4)

            self.play(FadeIn(prenorm_note, shift=RIGHT * 0.3), run_time=0.7)

        self.wait(0.3)
        self.clear()

    # ── Scene 4 — Positional Encoding ────────────────────────────────────

    def scene_positional(self):
        heading = self.make_heading("位置編碼 Positional Encoding")

        # --- Problem: attention is permutation-invariant ---
        tokens = ["我", "鍾意", "食", "壽司"]
        colors = [C_GREEN, C_ORANGE, C_PINK, C_PURPLE]

        # Row 1: tokens without position
        row1_label = self.zh("無位置信息：", font_size=20, color=C_DIM).shift(UP * 1.5 + LEFT * 5)
        token_boxes_1 = VGroup()
        for word, col in zip(tokens, colors):
            box = self.make_box(word, col, width=1.5, height=0.65)
            token_boxes_1.add(box)
        token_boxes_1.arrange(RIGHT, buff=0.3).next_to(row1_label, RIGHT, buff=0.3)

        no_pos_note = self.zh(
            "Attention 分唔到邊個先邊個後！",
            font_size=20, color=C_RED,
        ).next_to(token_boxes_1, DOWN, buff=0.25)

        # Row 2: tokens with PE (colored position tags)
        row2_label = self.zh("加位置編碼：", font_size=20, color=C_DIM).shift(DOWN * 0.7 + LEFT * 5)
        token_boxes_2 = VGroup()
        for i, (word, col) in enumerate(zip(tokens, colors)):
            box = self.make_box(word, col, width=1.5, height=0.65)
            tag = RoundedRectangle(
                corner_radius=0.08, width=0.5, height=0.35,
                fill_color=C_YELLOW, fill_opacity=0.5,
                stroke_color=C_YELLOW, stroke_width=1.5,
            )
            tag_txt = self.mono(str(i), font_size=16, color=C_YELLOW)
            tag_txt.move_to(tag)
            pos_tag = VGroup(tag, tag_txt).next_to(box, UR, buff=-0.15)
            token_boxes_2.add(VGroup(box, pos_tag))
        token_boxes_2.arrange(RIGHT, buff=0.4).next_to(row2_label, RIGHT, buff=0.3)

        with_pos_note = self.zh(
            "每個 token 都知道自己喺邊個位置",
            font_size=20, color=C_GREEN,
        ).next_to(token_boxes_2, DOWN, buff=0.25)

        # Sinusoidal formula
        formula_box = VGroup(
            RoundedRectangle(
                corner_radius=0.12, width=10, height=1.1,
                fill_color=C_BLUE, fill_opacity=0.12,
                stroke_color=C_BLUE, stroke_width=1.5,
            ),
            self.zh("Sinusoidal PE：", font_size=18, color=C_BLUE),
            self.mono(
                "PE(pos, 2i) = sin(pos / 10000^(2i/d))",
                font_size=16, color=C_CYAN,
            ),
            self.mono(
                "PE(pos, 2i+1) = cos(pos / 10000^(2i/d))",
                font_size=16, color=C_CYAN,
            ),
        )
        formula_box[1:].arrange(DOWN, buff=0.1, aligned_edge=LEFT)
        VGroup(*formula_box[1:]).move_to(formula_box[0])
        formula_box.to_edge(DOWN, buff=0.35)

        with self.voiceover(
            text="Attention 有一個問題。佢係 permutation invariant，"
            "即係話佢分唔到 token 嘅順序。"
            "對佢嚟講，「我鍾意食壽司」同「壽司食鍾意我」係一樣嘅。"
            "所以我哋需要位置編碼，即係 Positional Encoding。"
            "每個 token 加上一個位置標記，例如第 0 個、第 1 個，"
            "咁模型就知道邊個 token 喺前面、邊個喺後面。"
            "原始嘅 Transformer 用 sinusoidal 位置編碼，"
            "用 sin 同 cos 函數生成唔同頻率嘅信號，每個位置都有獨特嘅編碼。"
        ):
            self.play(Write(heading), run_time=0.6)

            self.play(FadeIn(row1_label), run_time=0.3)
            for box in token_boxes_1:
                self.play(FadeIn(box, shift=RIGHT * 0.15), run_time=0.3)
            self.play(FadeIn(no_pos_note, shift=UP * 0.1), run_time=0.5)

            self.play(FadeIn(row2_label), run_time=0.3)
            for group in token_boxes_2:
                self.play(FadeIn(group, shift=RIGHT * 0.15), run_time=0.35)
            self.play(FadeIn(with_pos_note, shift=UP * 0.1), run_time=0.5)

            self.play(FadeIn(formula_box, shift=UP * 0.2), run_time=0.7)

        self.wait(0.3)
        self.clear()

    # ── Scene 5 — RoPE ───────────────────────────────────────────────────

    def scene_rope(self):
        heading = self.make_heading("RoPE 旋轉位置編碼")

        # Concept: rotation
        concept_label = self.zh(
            "核心概念：用旋轉編碼位置", font_size=26, color=C_CYAN
        ).shift(UP * 1.5)

        # Show a simple 2D rotation visualization
        circle = Circle(radius=1.2, color=C_DIM, stroke_width=1.5).shift(DOWN * 0.5)

        dot1 = Dot(circle.point_at_angle(PI / 6), color=C_GREEN, radius=0.1)
        label1 = self.zh("位置 0", font_size=16, color=C_GREEN).next_to(dot1, UR, buff=0.12)
        vec1 = Arrow(
            circle.get_center(), dot1.get_center(),
            buff=0, color=C_GREEN, stroke_width=3,
        )

        dot2 = Dot(circle.point_at_angle(PI / 6 + PI / 4), color=C_ORANGE, radius=0.1)
        label2 = self.zh("位置 1", font_size=16, color=C_ORANGE).next_to(dot2, UL, buff=0.12)
        vec2 = Arrow(
            circle.get_center(), dot2.get_center(),
            buff=0, color=C_ORANGE, stroke_width=3,
        )

        dot3 = Dot(circle.point_at_angle(PI / 6 + PI / 2), color=C_PINK, radius=0.1)
        label3 = self.zh("位置 2", font_size=16, color=C_PINK).next_to(dot3, UL, buff=0.12)
        vec3 = Arrow(
            circle.get_center(), dot3.get_center(),
            buff=0, color=C_PINK, stroke_width=3,
        )

        rotation_group = VGroup(
            circle, vec1, dot1, label1, vec2, dot2, label2, vec3, dot3, label3
        ).shift(LEFT * 3)

        # Advantages on the right
        adv_title = self.zh("RoPE 嘅優勢", font_size=24, color=C_YELLOW).shift(
            RIGHT * 2.5 + UP * 0.5
        )
        advantages = VGroup(
            self.zh("1. 編碼相對位置", font_size=20, color=C_GREEN),
            self.zh("2. 無額外參數", font_size=20, color=C_ORANGE),
            self.zh("3. 更好嘅長度外推", font_size=20, color=C_PINK),
            self.zh("4. LLaMA / Mistral 都用", font_size=20, color=C_PURPLE),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3).next_to(adv_title, DOWN, buff=0.35)

        # Bottom note
        bottom_note = VGroup(
            RoundedRectangle(
                corner_radius=0.12, width=10, height=0.65,
                fill_color=C_CYAN, fill_opacity=0.12,
                stroke_color=C_CYAN, stroke_width=1.5,
            ),
            self.zh(
                "RoPE 將每對維度旋轉一個角度，角度取決於位置",
                font_size=20, color=C_CYAN,
            ),
        )
        bottom_note[1].move_to(bottom_note[0])
        bottom_note.to_edge(DOWN, buff=0.35)

        with self.voiceover(
            text="RoPE，全稱 Rotary Position Embeddings，係而家最流行嘅位置編碼方法。"
            "佢嘅核心概念係：用旋轉嚟編碼位置。"
            "想像一個二維平面，每個 token 嘅 query 同 key 向量，"
            "會根據佢嘅位置旋轉一定嘅角度。"
            "位置 0 旋轉少少，位置 1 旋轉多啲，位置 2 旋轉更多。"
            "咁樣做嘅好處係，兩個 token 嘅 attention score 會自然反映佢哋嘅相對距離。"
            "RoPE 唔使額外參數，而且可以更好噉處理長序列。"
            "所以 LLaMA 同 Mistral 全部都用 RoPE。"
        ):
            self.play(Write(heading), run_time=0.6)
            self.play(FadeIn(concept_label), run_time=0.5)

            self.play(Create(circle), run_time=0.5)
            self.play(GrowArrow(vec1), FadeIn(dot1), FadeIn(label1), run_time=0.6)
            self.play(GrowArrow(vec2), FadeIn(dot2), FadeIn(label2), run_time=0.6)
            self.play(GrowArrow(vec3), FadeIn(dot3), FadeIn(label3), run_time=0.6)

            self.play(FadeIn(adv_title), run_time=0.3)
            for adv in advantages:
                self.play(FadeIn(adv, shift=RIGHT * 0.2), run_time=0.4)

            self.play(FadeIn(bottom_note, shift=UP * 0.2), run_time=0.6)

        self.wait(0.3)
        self.clear()

    # ── Scene 6 — Full GPT Model ─────────────────────────────────────────

    def scene_full_model(self):
        heading = self.make_heading("完整 GPT 模型")

        # Build from bottom to top
        box_tokens = self.make_box("Token IDs", C_DIM, width=4.0, height=0.55)
        box_tok_emb = self.make_box("Token Embedding", C_GREEN, width=4.0, height=0.55)
        box_pos_emb = self.make_box("+ Position Embedding", C_CYAN, width=4.0, height=0.55)

        # Stacked transformer blocks
        block_colors = [C_ORANGE, C_ORANGE, C_ORANGE, C_ORANGE]
        block_boxes = VGroup()
        for i, col in enumerate(block_colors):
            rect = RoundedRectangle(
                corner_radius=0.12, width=4.0, height=0.5,
                fill_color=col, fill_opacity=0.15 + i * 0.05,
                stroke_color=col, stroke_width=2,
            )
            label = self.zh(
                f"Transformer Block {i+1}", font_size=18, color=col
            ).move_to(rect)
            block_boxes.add(VGroup(rect, label))

        box_ln = self.make_box("Final LayerNorm", C_PURPLE, width=4.0, height=0.55)
        box_lm = self.make_box("LM Head", C_RED, width=4.0, height=0.55)
        box_logits = self.make_box("Logits", C_YELLOW, width=4.0, height=0.55)

        all_layers = VGroup(
            box_tokens, box_tok_emb, box_pos_emb,
            *block_boxes,
            box_ln, box_lm, box_logits
        ).arrange(UP, buff=0.12)
        all_layers.scale(0.85).move_to(LEFT * 0.5 + DOWN * 0.1)

        # Arrows
        layer_list = [box_tokens, box_tok_emb, box_pos_emb] + \
                     list(block_boxes) + [box_ln, box_lm, box_logits]
        arrows = VGroup()
        for i in range(len(layer_list) - 1):
            a = Arrow(
                layer_list[i].get_top(), layer_list[i + 1].get_bottom(),
                buff=0.04, color=C_WHITE, stroke_width=2,
            )
            arrows.add(a)

        # Bracket around transformer blocks
        brace = Brace(
            VGroup(*block_boxes), RIGHT, color=C_YELLOW, buff=0.2
        )
        brace_label = self.zh(
            "x N 層\n(GPT-2: N=12)",
            font_size=18, color=C_YELLOW,
        ).next_to(brace, RIGHT, buff=0.15)

        # Data flow annotation on the left
        flow_note = VGroup(
            self.mono("(B, T)", font_size=14, color=C_DIM),
            self.mono("(B, T, d)", font_size=14, color=C_DIM),
            self.mono("(B, T, d)", font_size=14, color=C_DIM),
            self.mono("(B, T, d)", font_size=14, color=C_DIM),
            self.mono("(B, T, V)", font_size=14, color=C_DIM),
        )
        shape_targets = [box_tokens, box_tok_emb, block_boxes[0], box_ln, box_logits]
        for note, target in zip(flow_note, shape_targets):
            note.next_to(target, LEFT, buff=0.3)

        with self.voiceover(
            text="而家將所有嘢組裝埋一齊，睇下一個完整嘅 GPT 模型係點樣嘅。"
            "最底部係 token IDs，即係 tokenizer 輸出嘅數字序列。"
            "首先經過 token embedding，將每個 ID 變成一個向量。"
            "加上 position embedding，令模型知道順序。"
            "然後經過 N 個 Transformer Block。GPT-2 有 12 層。"
            "每一層都做 attention 同 FFN，令模型對文本嘅理解越嚟越深。"
            "最後經過 Layer Normalization 同 LM Head。"
            "LM Head 係一個 linear layer，將 d model 維嘅向量投射到 vocab size 維。"
            "輸出嘅 logits 代表每個 token 嘅分數，用 softmax 就得到概率。"
        ):
            self.play(Write(heading), run_time=0.6)

            for i, layer in enumerate(layer_list):
                self.play(FadeIn(layer, shift=UP * 0.1), run_time=0.3)
                if i < len(arrows):
                    self.play(GrowArrow(arrows[i]), run_time=0.15)

            self.play(
                GrowFromCenter(brace), FadeIn(brace_label),
                run_time=0.6,
            )

            for note in flow_note:
                self.play(FadeIn(note), run_time=0.2)

        # Animate data flowing upward
        data_dot = Dot(
            box_tokens.get_center(), color=C_YELLOW, radius=0.1
        )
        self.play(FadeIn(data_dot, scale=2), run_time=0.3)
        for layer in layer_list[1:]:
            self.play(
                data_dot.animate.move_to(layer.get_center()),
                run_time=0.2,
            )
        self.play(FadeOut(data_dot), run_time=0.3)

        self.wait(0.3)
        self.clear()

    # ── Scene 7 — GPT-2 Dimensions ──────────────────────────────────────

    def scene_dimensions(self):
        heading = self.make_heading("GPT-2 模型配置")

        # Config card
        card = RoundedRectangle(
            corner_radius=0.2, width=8, height=5,
            fill_color=C_BLUE, fill_opacity=0.08,
            stroke_color=C_BLUE, stroke_width=2,
        ).move_to(DOWN * 0.15)

        card_title = self.zh(
            "GPT-2 Small (124M)", font_size=30, color=C_BLUE,
        ).next_to(card, UP, buff=0.05)

        config_entries = [
            ("vocab_size", "50,257", C_GREEN),
            ("d_model", "768", C_ORANGE),
            ("n_heads", "12", C_PINK),
            ("n_layers", "12", C_PURPLE),
            ("d_ff", "3,072  (= 4 x 768)", C_CYAN),
            ("max_seq_len", "1,024", C_YELLOW),
            ("d_head", "64  (= 768 / 12)", C_DIM),
        ]

        rows = VGroup()
        for param, value, col in config_entries:
            row = VGroup(
                self.mono(param, font_size=22, color=col),
                self.mono(value, font_size=22, color=C_WHITE),
            ).arrange(RIGHT, buff=1.5)
            rows.add(row)
        rows.arrange(DOWN, aligned_edge=LEFT, buff=0.25).move_to(card)

        # Separator line
        sep = Line(
            card.get_left() + RIGHT * 0.5 + DOWN * 1.4,
            card.get_right() + LEFT * 0.5 + DOWN * 1.4,
            color=C_DIM, stroke_width=1,
        )

        total = self.zh(
            "總參數量：~124M", font_size=26, color=C_YELLOW,
        ).next_to(sep, DOWN, buff=0.2)

        # Right side: family comparison
        family_title = self.zh(
            "GPT-2 家族", font_size=22, color=C_DIM,
        ).to_edge(DOWN, buff=0.6).shift(LEFT * 1)

        with self.voiceover(
            text="GPT-2 Small 係最經典嘅 Transformer 配置，一共有 124M 個參數。"
            "vocab size 係 50257，即係 tokenizer 嘅詞彙量。"
            "d model 係 768，即係每個 token 嘅向量維度。"
            "有 12 個 attention heads，每個 head 嘅維度係 64。"
            "有 12 層 Transformer block。"
            "FFN 嘅隱藏層係 3072，等於 4 乘以 d model。呢個 4 倍係一個常見嘅慣例。"
            "最大序列長度係 1024 個 token。"
            "呢啲數字你要記住，因為之後我哋自己嘅模型都會參考呢個配置。"
        ):
            self.play(Write(heading), run_time=0.6)
            self.play(FadeIn(card), FadeIn(card_title), run_time=0.5)

            for row in rows:
                self.play(FadeIn(row, shift=RIGHT * 0.2), run_time=0.4)

            self.play(Create(sep), run_time=0.3)
            self.play(FadeIn(total, shift=UP * 0.1), run_time=0.5)

        self.wait(0.3)
        self.clear()

    # ── Scene 8 — Autoregressive Generation ──────────────────────────────

    def scene_generation(self):
        heading = self.make_heading("自回歸生成 Autoregressive")

        # Show tokens appearing one at a time
        prompt_words = ["The", "cat"]
        gen_words = ["sat", "on", "the", "mat"]

        all_words = prompt_words + gen_words
        word_colors = [C_GREEN] * len(prompt_words) + [C_ORANGE] * len(gen_words)

        prompt_label = self.zh("Prompt:", font_size=20, color=C_GREEN).shift(
            UP * 1.5 + LEFT * 5.5
        )
        gen_label = self.zh("生成:", font_size=20, color=C_ORANGE).next_to(
            prompt_label, DOWN, buff=1.5, aligned_edge=LEFT
        )

        # Build token boxes
        token_boxes = VGroup()
        for word, col in zip(all_words, word_colors):
            rect = RoundedRectangle(
                corner_radius=0.1, width=1.2, height=0.6,
                fill_color=col, fill_opacity=0.25,
                stroke_color=col, stroke_width=2,
            )
            txt = self.en(word, font_size=20, color=col).move_to(rect)
            token_boxes.add(VGroup(rect, txt))
        token_boxes.arrange(RIGHT, buff=0.15).shift(UP * 0.3)

        # Step-by-step explanation
        steps_data = [
            "[The, cat] → model → 'sat'",
            "[The, cat, sat] → model → 'on'",
            "[The, cat, sat, on] → model → 'the'",
            "[The, cat, sat, on, the] → model → 'mat'",
        ]
        steps = VGroup()
        for s in steps_data:
            step_text = self.mono(s, font_size=18, color=C_DIM)
            steps.add(step_text)
        steps.arrange(DOWN, aligned_edge=LEFT, buff=0.22).shift(DOWN * 1.5)

        # Bottom note about O(n²)
        note = VGroup(
            RoundedRectangle(
                corner_radius=0.12, width=10, height=0.65,
                fill_color=C_RED, fill_opacity=0.12,
                stroke_color=C_RED, stroke_width=1.5,
            ),
            self.zh(
                "每生成一個 token 都要做一次完整嘅 forward pass",
                font_size=20, color=C_RED,
            ),
        )
        note[1].move_to(note[0])
        note.to_edge(DOWN, buff=0.3)

        with self.voiceover(
            text="Transformer 係點樣生成文本呢？答案係自回歸生成。"
            "首先，你輸入一個 prompt，例如 The cat。"
            "模型做一次 forward pass，預測下一個 token，例如 sat。"
            "然後將 sat 加入序列，再做一次 forward pass，預測 on。"
            "再加入 on，預測 the。"
            "再加入 the，預測 mat。"
            "每次都係一個一個 token 噉生成。"
            "呢個過程叫做自回歸，因為模型嘅輸出變成下一步嘅輸入。"
            "要注意，每生成一個 token 都要做一次完整嘅 forward pass，"
            "所以生成速度係比較慢嘅。"
        ):
            self.play(Write(heading), run_time=0.6)
            self.play(FadeIn(prompt_label), run_time=0.3)

            # Show prompt tokens
            for i in range(len(prompt_words)):
                self.play(FadeIn(token_boxes[i], shift=RIGHT * 0.2), run_time=0.4)

            self.play(FadeIn(gen_label), run_time=0.3)

            # Generate tokens one by one with step annotations
            for i in range(len(gen_words)):
                gen_idx = len(prompt_words) + i
                self.play(
                    FadeIn(token_boxes[gen_idx], shift=DOWN * 0.3, scale=1.2),
                    run_time=0.5,
                )
                self.play(
                    token_boxes[gen_idx][0].animate.set_fill(opacity=0.25),
                    run_time=0.2,
                )
                if i < len(steps):
                    self.play(FadeIn(steps[i], shift=RIGHT * 0.2), run_time=0.35)

            self.play(FadeIn(note, shift=UP * 0.2), run_time=0.6)

        self.wait(0.3)
        self.clear()

    # ── Scene 9 — Summary ───────────────────────────────────────────────

    def scene_summary(self):
        heading = self.make_heading("總結", font_size=44)

        bullets = VGroup(
            self.zh("•  Decoder-Only 係現代 LLM 嘅主流架構", font_size=26, color=C_GREEN),
            self.zh("•  Transformer Block = Attention + FFN + 殘差 + Norm", font_size=26, color=C_ORANGE),
            self.zh("•  RoPE 用旋轉嚟編碼位置信息", font_size=26, color=C_CYAN),
            self.zh("•  GPT = Token Emb + N 個 Block + LM Head", font_size=26, color=C_PINK),
            self.zh("•  GPT-2 Small: d=768, 12 heads, 12 layers = 124M", font_size=26, color=C_PURPLE),
            self.zh("•  生成係自回歸：一個一個 token 噉產生", font_size=26, color=C_YELLOW),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.35).move_to(ORIGIN)

        with self.voiceover(
            text="總結一下今日學咗嘅嘢。"
            "第一，Decoder-Only 係而家所有大型語言模型嘅主流架構。"
            "第二，每個 Transformer Block 由 Attention、FFN、殘差連接同 Normalization 組成。"
            "第三，RoPE 用旋轉嘅方式編碼位置信息，係而家最常用嘅方法。"
            "第四，一個完整嘅 GPT 模型由 Token Embedding、N 個 Block 同 LM Head 組成。"
            "第五，GPT-2 Small 嘅配置係 d model 768、12 個 heads、12 層，總共 124M 參數。"
            "第六，文本生成係自回歸嘅，每次生成一個 token。"
            "掌握咗呢啲概念，你已經理解咗 LLM 嘅核心架構喇。"
        ):
            self.play(Write(heading), run_time=0.6)
            for b in bullets:
                self.play(FadeIn(b, shift=RIGHT * 0.3), run_time=0.5)

        self.wait(0.5)
        self.clear()

    # ── Scene 10 — Outro ────────────────────────────────────────────────

    def scene_outro(self):
        thanks = self.zh("多謝收睇！", font_size=52, color=C_YELLOW)
        next_lesson = self.zh(
            "下一課：Data Pipeline（數據管線）",
            font_size=28, color=C_CYAN,
        ).next_to(thanks, DOWN, buff=0.6)

        with self.voiceover(
            text="多謝收睇！下一課我哋會學 Data Pipeline，"
            "即係點樣準備大規模嘅文本數據嚟訓練你自己嘅 LLM。"
            "包括數據清洗、tokenization、DataLoader 設計同 sequence packing。"
            "記得繼續跟住學喇！"
        ):
            self.play(Write(thanks), run_time=1)
            self.play(FadeIn(next_lesson, shift=UP * 0.2), run_time=0.8)

        self.wait(1)
        self.play(FadeOut(thanks), FadeOut(next_lesson), run_time=0.8)
