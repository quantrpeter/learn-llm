"""
Lesson 5 – Attention Mechanism — The Core Innovation
Manim CE + Cantonese voiceover (Edge TTS)

Render (1080p):
    cd "lesson 5/video"
    manim render -qh scene.py AttentionExplainer

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


class AttentionExplainer(VoiceoverScene):
    """Full lesson explaining the attention mechanism in Cantonese."""

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
        self.scene_intuition()
        self.scene_scaled_dot_product()
        self.scene_multi_head()
        self.scene_causal_mask()
        self.scene_kv_cache()
        self.scene_complexity()
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

    # ══════════════════════════════════════════════════════════════════════
    # Scene 1 — Title / Intro
    # ══════════════════════════════════════════════════════════════════════

    def scene_intro(self):
        title = self.zh("注意力機制", font_size=56, color=C_BLUE)
        sub_en = self.en(
            "Attention Mechanism", font_size=24, color=C_WHITE
        ).next_to(title, DOWN, buff=0.4)
        sub_zh = self.zh(
            "第五課", font_size=36, color=C_ORANGE
        ).next_to(sub_en, DOWN, buff=0.3)
        producer = self.zh(
            "制片人：Peter", font_size=24, color=C_DIM
        ).next_to(sub_zh, DOWN, buff=0.7)

        with self.voiceover(
            text="大家好！歡迎嚟到第五課。"
            "今日我哋嚟學注意力機制，即係 Attention Mechanism。"
            "呢個係 Transformer 最核心嘅發明，"
            "亦都係 LLM 之所以咁強大嘅根本原因。"
            "學完呢課之後，你就會完全明白 attention 點樣運作。"
        ):
            self.play(Write(title), run_time=1.5)
            self.play(FadeIn(sub_en, shift=UP * 0.2), run_time=0.8)
            self.play(FadeIn(sub_zh, shift=UP * 0.2), run_time=0.8)
            self.play(FadeIn(producer, shift=UP * 0.2), run_time=0.5)

        self.wait(0.3)
        self.clear()

    # ══════════════════════════════════════════════════════════════════════
    # Scene 2 — Intuition: Q / K / V as soft dictionary lookup
    # ══════════════════════════════════════════════════════════════════════

    def scene_intuition(self):
        heading = self.make_heading("直覺：軟性字典查詢")

        # Query box on the left
        q_box = self.make_box("Query (問題)", C_BLUE, width=3.2, height=0.9)
        q_box.shift(LEFT * 4 + UP * 0.5)

        # Key boxes on the right (3 entries)
        key_labels = ["Key 1: 貓", "Key 2: 狗", "Key 3: 小貓"]
        val_labels = ["Value 1: 貓坐喺蓆上", "Value 2: 狗好忠誠", "Value 3: 小貓玩毛線"]
        key_colors = [C_GREEN, C_ORANGE, C_PURPLE]

        key_boxes = VGroup()
        val_boxes = VGroup()
        for i, (kl, vl, kc) in enumerate(zip(key_labels, val_labels, key_colors)):
            kb = self.make_box(kl, kc, width=2.8, height=0.7, font_size=20)
            vb = self.make_box(vl, kc, width=3.6, height=0.7, font_size=18)
            key_boxes.add(kb)
            val_boxes.add(vb)

        key_boxes.arrange(DOWN, buff=0.35).shift(RIGHT * 0.5 + UP * 0.5)
        for i, vb in enumerate(val_boxes):
            vb.next_to(key_boxes[i], RIGHT, buff=0.3)

        # Arrows from Query to each Key
        arrows_qk = VGroup()
        for kb in key_boxes:
            a = Arrow(
                q_box.get_right(), kb.get_left(),
                buff=0.1, color=C_BLUE, stroke_width=2.5,
            )
            arrows_qk.add(a)

        # Weight labels on arrows
        weight_labels = ["0.45", "0.10", "0.45"]
        w_texts = VGroup()
        for i, (a, w) in enumerate(zip(arrows_qk, weight_labels)):
            wt = self.mono(w, font_size=18, color=C_YELLOW).next_to(
                a, UP if i != 1 else DOWN, buff=0.05
            )
            w_texts.add(wt)

        explain = VGroup(
            RoundedRectangle(
                corner_radius=0.15, width=11, height=0.7,
                fill_color=C_CYAN, fill_opacity=0.15, stroke_color=C_CYAN,
            ),
            self.zh(
                "Query 同每個 Key 比較相似度 → softmax → 加權混合所有 Values",
                font_size=20, color=C_CYAN,
            ),
        )
        explain[1].move_to(explain[0])
        explain.to_edge(DOWN, buff=0.4)

        with self.voiceover(
            text="注意力機制可以用一個好簡單嘅比喻嚟理解，就係軟性字典查詢。"
            "想像你有一個問題，即係 Query。"
            "然後有三個條目，每個條目有一個 Key 同一個 Value。"
            "Key 就好似標題，Value 就係內容。"
            "Query 會同每一個 Key 做比較，計算相似度。"
            "相似度越高嘅 Key，佢嘅 Value 就會貢獻越多。"
            "例如呢度，Query 問關於貓嘅嘢，所以 Key 1 同 Key 3 嘅權重比較高。"
            "最後嘅輸出就係所有 Value 嘅加權混合。"
            "呢個就係 attention 嘅核心直覺。"
        ):
            self.play(Write(heading), run_time=0.6)
            self.play(FadeIn(q_box, shift=RIGHT * 0.3), run_time=0.6)
            for kb in key_boxes:
                self.play(FadeIn(kb, shift=LEFT * 0.2), run_time=0.4)
            for a in arrows_qk:
                self.play(GrowArrow(a), run_time=0.3)
            for wt in w_texts:
                self.play(FadeIn(wt, scale=1.2), run_time=0.3)
            for vb in val_boxes:
                self.play(FadeIn(vb, shift=LEFT * 0.2), run_time=0.4)
            self.play(FadeIn(explain, shift=UP * 0.2), run_time=0.6)

        self.wait(0.3)
        self.clear()

    # ══════════════════════════════════════════════════════════════════════
    # Scene 3 — Scaled Dot-Product Attention (step by step)
    # ══════════════════════════════════════════════════════════════════════

    def scene_scaled_dot_product(self):
        heading = self.make_heading("Scaled Dot-Product Attention")

        # Step-by-step vertical flow
        # Row 1: Q and K^T → multiply
        q_box = self.make_mono_box("Q", C_GREEN, width=1.6, height=0.7)
        times1 = self.mono("@", font_size=28, color=C_WHITE)
        kt_box = self.make_mono_box("K^T", C_ORANGE, width=1.6, height=0.7)

        q_shape = self.mono("(4, 8)", font_size=16, color=C_DIM)
        kt_shape = self.mono("(8, 4)", font_size=16, color=C_DIM)

        row1 = VGroup(q_box, times1, kt_box).arrange(RIGHT, buff=0.4)
        row1.shift(UP * 2.0 + LEFT * 2.5)
        q_shape.next_to(q_box, DOWN, buff=0.1)
        kt_shape.next_to(kt_box, DOWN, buff=0.1)

        # Arrow down to Scores
        scores_box = self.make_mono_box("Scores", C_YELLOW, width=2.2, height=0.7)
        scores_shape = self.mono("(4, 4)", font_size=16, color=C_DIM)
        scores_box.next_to(row1, DOWN, buff=0.6)
        scores_shape.next_to(scores_box, DOWN, buff=0.1)
        arrow1 = Arrow(
            row1.get_bottom(), scores_box.get_top(),
            buff=0.15, color=C_WHITE, stroke_width=2.5,
        )

        # Scale label
        scale_label = self.mono("/ sqrt(d_k)", font_size=18, color=C_PINK)
        scale_label.next_to(arrow1, RIGHT, buff=0.15)

        # Arrow down to Weights
        weights_box = self.make_mono_box("Weights", C_PINK, width=2.2, height=0.7)
        weights_shape = self.mono("(4, 4)", font_size=16, color=C_DIM)
        weights_box.next_to(scores_box, DOWN, buff=0.6)
        weights_shape.next_to(weights_box, DOWN, buff=0.1)
        arrow2 = Arrow(
            scores_box.get_bottom(), weights_box.get_top(),
            buff=0.15, color=C_WHITE, stroke_width=2.5,
        )
        softmax_label = self.mono("softmax", font_size=18, color=C_CYAN)
        softmax_label.next_to(arrow2, RIGHT, buff=0.15)

        # V box and multiply to output
        v_box = self.make_mono_box("V", C_PURPLE, width=1.6, height=0.7)
        v_shape = self.mono("(4, 8)", font_size=16, color=C_DIM)
        v_box.shift(RIGHT * 2.5 + DOWN * 0.5)
        v_shape.next_to(v_box, DOWN, buff=0.1)

        output_box = self.make_mono_box("Output", C_CYAN, width=2.2, height=0.7)
        output_shape = self.mono("(4, 8)", font_size=16, color=C_DIM)
        output_box.next_to(weights_box, DOWN, buff=0.6).shift(RIGHT * 1.5)
        output_shape.next_to(output_box, DOWN, buff=0.1)

        times2 = self.mono("@", font_size=28, color=C_WHITE)
        times2.move_to((weights_box.get_bottom() + output_box.get_top()) / 2).shift(LEFT * 0.5)

        arrow3_w = Arrow(
            weights_box.get_bottom(), output_box.get_left() + UP * 0.1,
            buff=0.15, color=C_WHITE, stroke_width=2.5,
        )
        arrow3_v = Arrow(
            v_box.get_bottom(), output_box.get_top() + RIGHT * 0.3,
            buff=0.15, color=C_WHITE, stroke_width=2.5,
        )

        # Row-sum note
        note = self.zh(
            "每行加起嚟等於 1", font_size=18, color=C_PINK
        ).next_to(weights_shape, LEFT, buff=0.5)

        with self.voiceover(
            text="而家嚟睇 Attention 嘅完整計算流程。"
            "首先，我哋有三個矩陣：Q、K 同 V。"
            "Q 代表 Query，形狀係 4 乘 8，即係 4 個 token，每個有 8 個維度。"
            "K 嘅轉置 K T 形狀係 8 乘 4。"
            "第一步，將 Q 乘以 K 嘅轉置，得到一個 4 乘 4 嘅分數矩陣。"
            "分數矩陣入面，scores i j 代表 token i 對 token j 嘅關注程度。"
            "第二步，除以根號 d k 做 scaling，防止數值太大。"
            "第三步，通過 softmax 將分數變成權重，每一行加起嚟等於一。"
            "最後，用權重乘以 V 矩陣，得到輸出。"
            "輸出嘅形狀同 Q 一樣，都係 4 乘 8。"
        ):
            self.play(Write(heading), run_time=0.6)

            self.play(FadeIn(q_box), FadeIn(times1), FadeIn(kt_box), run_time=0.6)
            self.play(FadeIn(q_shape), FadeIn(kt_shape), run_time=0.3)

            self.play(GrowArrow(arrow1), FadeIn(scale_label), run_time=0.5)
            self.play(FadeIn(scores_box, shift=DOWN * 0.2), run_time=0.5)
            self.play(FadeIn(scores_shape), run_time=0.3)

            self.play(GrowArrow(arrow2), FadeIn(softmax_label), run_time=0.5)
            self.play(FadeIn(weights_box, shift=DOWN * 0.2), run_time=0.5)
            self.play(FadeIn(weights_shape), FadeIn(note), run_time=0.3)

            self.play(FadeIn(v_box, shift=LEFT * 0.3), FadeIn(v_shape), run_time=0.5)
            self.play(GrowArrow(arrow3_w), GrowArrow(arrow3_v), run_time=0.5)
            self.play(FadeIn(output_box, shift=DOWN * 0.2), run_time=0.5)
            self.play(FadeIn(output_shape), run_time=0.3)

        self.wait(0.3)
        self.clear()

    # ══════════════════════════════════════════════════════════════════════
    # Scene 4 — Multi-Head Attention (parallel lanes)
    # ══════════════════════════════════════════════════════════════════════

    def scene_multi_head(self):
        heading = self.make_heading("Multi-Head Attention")

        # Input at top
        input_box = self.make_mono_box(
            "Input (d_model=32)", C_WHITE, width=5, height=0.7
        )
        input_box.shift(UP * 2.3)

        # Split label
        split_label = self.zh(
            "分割成 4 個 Head", font_size=20, color=C_DIM
        ).next_to(input_box, DOWN, buff=0.25)

        # 4 parallel head lanes
        head_colors = [C_GREEN, C_ORANGE, C_PINK, C_PURPLE]
        head_boxes = VGroup()
        for i, c in enumerate(head_colors):
            hb = self.make_mono_box(
                f"Head {i+1}\nd_k=8", c, width=2.0, height=1.0, font_size=18
            )
            head_boxes.add(hb)
        head_boxes.arrange(RIGHT, buff=0.4).shift(DOWN * 0.1)

        # Arrows from input to each head
        split_arrows = VGroup()
        for hb in head_boxes:
            a = Arrow(
                input_box.get_bottom(), hb.get_top(),
                buff=0.3, color=C_DIM, stroke_width=2,
            )
            split_arrows.add(a)

        # "Attention" sublabels inside each head
        attn_labels = VGroup()
        for i, hb in enumerate(head_boxes):
            al = self.mono("Attention", font_size=14, color=head_colors[i])
            al.next_to(hb, DOWN, buff=0.08)
            attn_labels.add(al)

        # Concat box
        concat_box = self.make_mono_box(
            "Concat", C_YELLOW, width=5, height=0.7
        )
        concat_box.shift(DOWN * 1.7)

        # Arrows from each head to concat
        merge_arrows = VGroup()
        for hb in head_boxes:
            a = Arrow(
                hb.get_bottom() + DOWN * 0.2, concat_box.get_top(),
                buff=0.15, color=C_DIM, stroke_width=2,
            )
            merge_arrows.add(a)

        # Final projection
        proj_box = self.make_mono_box(
            "Linear (W_o)", C_CYAN, width=5, height=0.7
        )
        proj_box.next_to(concat_box, DOWN, buff=0.5)

        arrow_proj = Arrow(
            concat_box.get_bottom(), proj_box.get_top(),
            buff=0.1, color=C_WHITE, stroke_width=2.5,
        )

        output_label = self.mono(
            "Output (d_model=32)", font_size=18, color=C_DIM
        ).next_to(proj_box, DOWN, buff=0.2)

        with self.voiceover(
            text="Multi-head attention 就係將一個大嘅 attention 分成好多個細嘅 head。"
            "例如，d model 等於 32，我哋可以分成 4 個 head，每個 head 嘅維度係 8。"
            "每個 head 獨立做 scaled dot-product attention。"
            "佢哋可以學到唔同嘅嘢。"
            "例如一個 head 可能關注語法結構，另一個可能關注語義相似性。"
            "做完之後，將所有 head 嘅輸出拼接返埋。"
            "最後通過一個 linear projection，投射返去 d model 嘅維度。"
            "呢個就係點解 multi-head attention 比 single-head 更強大。"
        ):
            self.play(Write(heading), run_time=0.6)
            self.play(FadeIn(input_box, shift=DOWN * 0.2), run_time=0.5)
            self.play(FadeIn(split_label), run_time=0.3)

            for i, (hb, sa) in enumerate(zip(head_boxes, split_arrows)):
                self.play(GrowArrow(sa), FadeIn(hb, shift=DOWN * 0.2), run_time=0.4)

            self.play(*[FadeIn(al) for al in attn_labels], run_time=0.4)

            for i, (hb, ma) in enumerate(zip(head_boxes, merge_arrows)):
                self.play(
                    hb[0].animate.set_fill(opacity=0.5),
                    run_time=0.2,
                )
                self.play(
                    hb[0].animate.set_fill(opacity=0.25),
                    GrowArrow(ma),
                    run_time=0.3,
                )

            self.play(FadeIn(concat_box, shift=DOWN * 0.2), run_time=0.5)
            self.play(GrowArrow(arrow_proj), run_time=0.3)
            self.play(FadeIn(proj_box, shift=DOWN * 0.2), run_time=0.5)
            self.play(FadeIn(output_label), run_time=0.3)

        self.wait(0.3)
        self.clear()

    # ══════════════════════════════════════════════════════════════════════
    # Scene 5 — Causal Mask (the animated grid)
    # ══════════════════════════════════════════════════════════════════════

    def scene_causal_mask(self):
        heading = self.make_heading("因果遮罩 Causal Mask")

        seq_len = 6
        cell_size = 0.58
        gap = 0.06
        token_labels = ["The", "cat", "sat", "on", "the", "mat"]

        # Build grid of squares
        cells = {}
        grid = VGroup()
        for i in range(seq_len):
            for j in range(seq_len):
                cell = Square(
                    side_length=cell_size,
                    fill_color=C_DIM,
                    fill_opacity=0.08,
                    stroke_color=C_DIM,
                    stroke_width=1,
                )
                x_pos = j * (cell_size + gap)
                y_pos = -i * (cell_size + gap)
                cell.move_to(np.array([x_pos, y_pos, 0]))
                cells[(i, j)] = cell
                grid.add(cell)

        grid.move_to(ORIGIN + DOWN * 0.3 + RIGHT * 0.3)

        # Row labels (Query tokens) — left side
        row_labels = VGroup()
        for i in range(seq_len):
            lbl = self.mono(
                token_labels[i], font_size=14, color=C_WHITE
            ).next_to(cells[(i, 0)], LEFT, buff=0.2)
            row_labels.add(lbl)

        # Column labels (Key tokens) — top
        col_labels = VGroup()
        for j in range(seq_len):
            lbl = self.mono(
                token_labels[j], font_size=14, color=C_WHITE
            ).next_to(cells[(0, j)], UP, buff=0.2)
            col_labels.add(lbl)

        # Axis titles
        q_axis = self.zh(
            "Query", font_size=18, color=C_BLUE
        ).next_to(row_labels, LEFT, buff=0.3)
        k_axis = self.zh(
            "Key", font_size=18, color=C_ORANGE
        ).next_to(col_labels, UP, buff=0.15)

        # Legend
        legend_allowed = VGroup(
            Square(side_length=0.3, fill_color=C_GREEN, fill_opacity=0.6, stroke_width=0),
            self.zh("可以注意", font_size=16, color=C_GREEN),
        ).arrange(RIGHT, buff=0.15)
        legend_blocked = VGroup(
            Square(side_length=0.3, fill_color=C_RED, fill_opacity=0.6, stroke_width=0),
            self.zh("被遮擋", font_size=16, color=C_RED),
        ).arrange(RIGHT, buff=0.15)
        legend = VGroup(legend_allowed, legend_blocked).arrange(RIGHT, buff=0.6)
        legend.to_edge(DOWN, buff=0.4)

        with self.voiceover(
            text="Causal masking 係自回歸語言模型嘅關鍵機制。"
            "喺生成文字嘅時候，每個 token 只能睇到自己同之前嘅 token，"
            "絕對唔可以偷睇未來嘅 token。"
            "我哋用一個格子圖嚟展示。"
            "行代表 Query，列代表 Key。"
            "綠色代表可以注意到嘅位置，紅色代表被遮擋嘅位置。"
            "第一行，The 只能睇到自己。"
            "第二行，cat 可以睇到 The 同自己。"
            "第三行，sat 可以睇到 The、cat 同自己。"
            "一直到最後一行，mat 可以睇到所有之前嘅 token。"
            "呢個下三角形嘅 mask 就確保咗模型唔會偷睇未來。"
        ):
            self.play(Write(heading), run_time=0.6)
            self.play(FadeIn(grid), run_time=0.5)
            self.play(
                FadeIn(row_labels), FadeIn(col_labels),
                FadeIn(q_axis), FadeIn(k_axis),
                run_time=0.5,
            )
            self.play(FadeIn(legend), run_time=0.4)

            # Animate row by row
            for i in range(seq_len):
                anims = []
                for j in range(seq_len):
                    if j <= i:
                        anims.append(
                            cells[(i, j)].animate.set_fill(C_GREEN, opacity=0.6)
                            .set_stroke(C_GREEN, width=2)
                        )
                    else:
                        anims.append(
                            cells[(i, j)].animate.set_fill(C_RED, opacity=0.5)
                            .set_stroke(C_RED, width=2)
                        )
                self.play(*anims, run_time=0.5)

        self.wait(0.5)
        self.clear()

    # ══════════════════════════════════════════════════════════════════════
    # Scene 6 — KV Cache (step-by-step inference)
    # ══════════════════════════════════════════════════════════════════════

    def scene_kv_cache(self):
        heading = self.make_heading("KV Cache 推理優化")

        # We show 3 steps of generation with growing cache
        steps_data = [
            {
                "step": 1,
                "new_token": "The",
                "cache_k": ["K1"],
                "cache_v": ["V1"],
                "desc": "計算 K1, V1 並儲存",
            },
            {
                "step": 2,
                "new_token": "cat",
                "cache_k": ["K1", "K2"],
                "cache_v": ["V1", "V2"],
                "desc": "重用 K1, V1 + 計算新 K2, V2",
            },
            {
                "step": 3,
                "new_token": "sat",
                "cache_k": ["K1", "K2", "K3"],
                "cache_v": ["V1", "V2", "V3"],
                "desc": "重用 K1..K2, V1..V2 + 計算新 K3, V3",
            },
        ]

        with self.voiceover(
            text="KV Cache 係一個好重要嘅推理優化技巧。"
            "喺生成文字嘅時候，每生成一個新 token，都需要計算 attention。"
            "如果每次都重新計算所有 token 嘅 K 同 V，就會好慢。"
            "KV Cache 嘅做法係將之前算過嘅 K 同 V 儲存起嚟。"
            "第一步，Token The 進入模型，計算 K1 同 V1，儲存到 cache。"
            "第二步，Token cat 進入模型，只需要計算新嘅 K2 同 V2。"
            "之前嘅 K1 同 V1 直接從 cache 攞返出嚟用，唔使重新計算。"
            "第三步，Token sat 進入，同樣只計算 K3 同 V3，"
            "之前嘅全部從 cache 攞。"
            "Cache 越嚟越大，但每步只計算一個新 token 嘅 K V，所以好快。"
        ):
            self.play(Write(heading), run_time=0.6)

            prev_group = None
            for sd in steps_data:
                if prev_group is not None:
                    self.play(FadeOut(prev_group), run_time=0.3)

                step_group = VGroup()

                step_title = self.zh(
                    f"Step {sd['step']}: 新 Token = \"{sd['new_token']}\"",
                    font_size=24, color=C_YELLOW,
                ).shift(UP * 1.2)
                step_group.add(step_title)

                # Cache visualization: K row and V row
                k_label = self.mono("K cache:", font_size=20, color=C_GREEN).shift(UP * 0.3 + LEFT * 4)
                v_label = self.mono("V cache:", font_size=20, color=C_ORANGE).shift(DOWN * 0.5 + LEFT * 4)
                step_group.add(k_label, v_label)

                k_items = VGroup()
                for idx, ki in enumerate(sd["cache_k"]):
                    is_new = idx == len(sd["cache_k"]) - 1
                    color = C_YELLOW if is_new else C_GREEN
                    box = self.make_mono_box(
                        ki, color, width=1.3, height=0.6, font_size=20
                    )
                    k_items.add(box)
                k_items.arrange(RIGHT, buff=0.15).next_to(k_label, RIGHT, buff=0.3)
                step_group.add(k_items)

                v_items = VGroup()
                for idx, vi in enumerate(sd["cache_v"]):
                    is_new = idx == len(sd["cache_v"]) - 1
                    color = C_YELLOW if is_new else C_ORANGE
                    box = self.make_mono_box(
                        vi, color, width=1.3, height=0.6, font_size=20
                    )
                    v_items.add(box)
                v_items.arrange(RIGHT, buff=0.15).next_to(v_label, RIGHT, buff=0.3)
                step_group.add(v_items)

                # Description
                desc_text = self.zh(
                    sd["desc"], font_size=22, color=C_CYAN
                ).shift(DOWN * 1.5)
                step_group.add(desc_text)

                # NEW vs CACHED legend
                new_tag = VGroup(
                    Square(side_length=0.25, fill_color=C_YELLOW, fill_opacity=0.6, stroke_width=0),
                    self.zh("新計算", font_size=16, color=C_YELLOW),
                ).arrange(RIGHT, buff=0.1)
                cached_tag = VGroup(
                    Square(side_length=0.25, fill_color=C_GREEN, fill_opacity=0.6, stroke_width=0),
                    self.zh("從 Cache", font_size=16, color=C_GREEN),
                ).arrange(RIGHT, buff=0.1)
                tag_row = VGroup(new_tag, cached_tag).arrange(RIGHT, buff=0.6)
                tag_row.to_edge(DOWN, buff=0.5)
                step_group.add(tag_row)

                self.play(
                    FadeIn(step_title, shift=DOWN * 0.2),
                    FadeIn(k_label), FadeIn(v_label),
                    run_time=0.5,
                )
                for ki in k_items:
                    self.play(FadeIn(ki, shift=RIGHT * 0.1), run_time=0.25)
                for vi in v_items:
                    self.play(FadeIn(vi, shift=RIGHT * 0.1), run_time=0.25)
                self.play(FadeIn(desc_text), FadeIn(tag_row), run_time=0.4)

                self.wait(0.3)
                prev_group = step_group

            if prev_group is not None:
                self.play(FadeOut(prev_group), run_time=0.3)

        self.wait(0.3)
        self.clear()

    # ══════════════════════════════════════════════════════════════════════
    # Scene 7 — Complexity O(n^2) curve
    # ══════════════════════════════════════════════════════════════════════

    def scene_complexity(self):
        heading = self.make_heading("計算複雜度 O(n^2)")

        axes = Axes(
            x_range=[0, 1200, 200],
            y_range=[0, 1.2, 0.2],
            x_length=8,
            y_length=4,
            axis_config={"color": C_DIM, "include_numbers": False},
        ).shift(DOWN * 0.3)

        x_label = self.mono(
            "Sequence Length (n)", font_size=18, color=C_WHITE
        ).next_to(axes.x_axis, DOWN, buff=0.3)
        y_label = self.mono(
            "Compute Time", font_size=18, color=C_WHITE
        ).next_to(axes.y_axis, UP, buff=0.2).shift(LEFT * 0.5)

        # Quadratic curve (normalized so peak = 1.0)
        curve = axes.plot(
            lambda x: (x / 1024) ** 2, x_range=[0, 1100], color=C_BLUE
        )

        # Mark specific sequence lengths
        points = [128, 256, 512, 1024]
        dots = VGroup()
        dot_labels = VGroup()
        dashed_lines = VGroup()
        for n in points:
            y_val = (n / 1024) ** 2
            dot = Dot(axes.c2p(n, y_val), color=C_YELLOW, radius=0.08)
            dots.add(dot)

            lbl = self.mono(str(n), font_size=16, color=C_YELLOW).next_to(
                axes.c2p(n, 0), DOWN, buff=0.2
            )
            dot_labels.add(lbl)

            dline = DashedLine(
                axes.c2p(n, 0), axes.c2p(n, y_val),
                color=C_DIM, stroke_width=1.5,
            )
            dashed_lines.add(dline)

        # Scaling note
        scale_note = VGroup(
            RoundedRectangle(
                corner_radius=0.15, width=8, height=0.7,
                fill_color=C_RED, fill_opacity=0.15, stroke_color=C_RED,
            ),
            self.zh(
                "序列長度加倍 → 計算量增加 4 倍",
                font_size=22, color=C_RED,
            ),
        )
        scale_note[1].move_to(scale_note[0])
        scale_note.to_edge(DOWN, buff=0.35)

        with self.voiceover(
            text="Attention 嘅計算複雜度係 O of n squared，"
            "即係同序列長度嘅平方成正比。"
            "呢條曲線展示咗呢個關係。"
            "當序列長度由 128 增加到 256，計算量增加 4 倍。"
            "由 256 到 512，又增加 4 倍。"
            "到 1024 嘅時候，計算量已經係 128 嘅 64 倍。"
            "呢個就係點解處理好長嘅文字會好慢。"
            "GPT-4 有 128K 嘅 context window，"
            "128K 嘅平方超過一百六十億個元素。"
            "所以 Flash Attention 同 sparse attention 呢啲優化技術先咁重要。"
        ):
            self.play(Write(heading), run_time=0.6)
            self.play(Create(axes), FadeIn(x_label), FadeIn(y_label), run_time=1.0)
            self.play(Create(curve), run_time=1.5)

            for d, dl, dline in zip(dots, dot_labels, dashed_lines):
                self.play(
                    Create(dline),
                    FadeIn(d, scale=1.5),
                    FadeIn(dl),
                    run_time=0.5,
                )

            self.play(FadeIn(scale_note, shift=UP * 0.2), run_time=0.6)

        self.wait(0.3)
        self.clear()

    # ══════════════════════════════════════════════════════════════════════
    # Scene 8 — Summary
    # ══════════════════════════════════════════════════════════════════════

    def scene_summary(self):
        heading = self.make_heading("總結", font_size=44)

        bullets = VGroup(
            self.zh("•  Attention = 軟性字典查詢 (Q, K, V)", font_size=24, color=C_GREEN),
            self.zh("•  Scaled Dot-Product: softmax(QK^T / sqrt(d_k)) V", font_size=24, color=C_ORANGE),
            self.zh("•  Multi-Head: 分成多個 head，學唔同嘅特徵", font_size=24, color=C_PINK),
            self.zh("•  Causal Mask: 下三角遮罩，防止偷睇未來", font_size=24, color=C_PURPLE),
            self.zh("•  KV Cache: 儲存 K, V 加速推理", font_size=24, color=C_CYAN),
            self.zh("•  複雜度 O(n^2): 長序列嘅根本瓶頸", font_size=24, color=C_RED),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.35).move_to(ORIGIN)

        with self.voiceover(
            text="總結一下今日學咗嘅嘢。"
            "Attention 嘅本質係軟性字典查詢，用 Query 搵最相關嘅 Key，"
            "然後混合佢哋嘅 Value。"
            "Scaled Dot-Product Attention 嘅公式係"
            "softmax of Q K transpose over square root d k，再乘以 V。"
            "Multi-Head Attention 將維度分成多個 head，"
            "每個 head 可以關注唔同嘅特徵。"
            "Causal Mask 用下三角遮罩確保模型唔會偷睇未來嘅 token。"
            "KV Cache 將之前計算過嘅 K 同 V 儲存起嚟，大幅加速推理。"
            "最後，Attention 嘅 O of n squared 複雜度"
            "係處理長序列嘅根本瓶頸。"
        ):
            self.play(Write(heading), run_time=0.6)
            for b in bullets:
                self.play(FadeIn(b, shift=RIGHT * 0.3), run_time=0.5)

        self.wait(0.5)
        self.clear()

    # ══════════════════════════════════════════════════════════════════════
    # Scene 9 — Outro
    # ══════════════════════════════════════════════════════════════════════

    def scene_outro(self):
        thanks = self.zh("多謝收睇！", font_size=52, color=C_YELLOW)
        next_text = self.zh(
            "下一課：Transformer Architecture", font_size=28, color=C_CYAN
        ).next_to(thanks, DOWN, buff=0.5)

        with self.voiceover(
            text="多謝收睇！"
            "下一課我哋會學完整嘅 Transformer Architecture，"
            "將 attention、feed-forward network、layer norm "
            "同 residual connections 全部組裝成一個完整嘅 GPT 模型。"
            "記得繼續跟住學喇！"
        ):
            self.play(Write(thanks), run_time=1)
            self.play(FadeIn(next_text, shift=UP * 0.2), run_time=0.8)

        self.wait(1)
        self.play(FadeOut(thanks), FadeOut(next_text), run_time=0.8)
