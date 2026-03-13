"""
Lesson 10 – Text Generation & Decoding
Manim CE + Cantonese voiceover (Edge TTS)

Render (1080p):
    cd "lesson 10/video"
    manim render -qh scene.py TextGenerationExplainer

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


class TextGenerationExplainer(VoiceoverScene):
    """Single scene explaining text generation & decoding in Cantonese."""

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
        self.scene_greedy()
        self.scene_temperature()
        self.scene_topk()
        self.scene_topp()
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

    def make_heading(self, text, color=C_BLUE, font_size=40):
        return self.zh(text, font_size=font_size, color=color).to_edge(UP, buff=0.5)

    # ── Scene 1 — Title / Intro ──────────────────────────────────────────

    def scene_intro(self):
        title = self.zh("文字生成", font_size=56, color=C_BLUE)
        sub_en = self.en(
            "Text Generation & Decoding", font_size=24, color=C_WHITE
        ).next_to(title, DOWN, buff=0.4)
        sub_zh = self.zh(
            "第十課", font_size=36, color=C_ORANGE
        ).next_to(sub_en, DOWN, buff=0.3)
        producer = self.zh(
            "制片人：Peter", font_size=24, color=C_DIM
        ).next_to(sub_zh, DOWN, buff=0.7)

        with self.voiceover(
            text="大家好！歡迎嚟到第十課。"
            "上幾課我哋學咗點樣訓練 LLM。"
            "但訓練完之後，點樣用佢生成文字呢？"
            "今日我哋會學六種文字生成策略。"
            "包括 greedy decoding、temperature sampling、"
            "top-k、top-p nucleus sampling、"
            "repetition penalty 同 KV cache。"
        ):
            self.play(Write(title), run_time=1.5)
            self.play(FadeIn(sub_en, shift=UP * 0.2), run_time=0.8)
            self.play(FadeIn(sub_zh, shift=UP * 0.2), run_time=0.8)
            self.play(FadeIn(producer, shift=UP * 0.2), run_time=0.5)

        self.wait(0.3)
        self.clear()

    # ── Scene 2 — Greedy Decoding (bar chart token selection) ────────────

    def scene_greedy(self):
        heading = self.make_heading("Greedy Decoding")

        tokens = ["the", "a", "cat", "is", "dog"]
        probs = [0.40, 0.25, 0.15, 0.12, 0.08]
        colors = [C_GREEN, C_BLUE, C_ORANGE, C_PINK, C_PURPLE]

        bars = VGroup()
        labels = VGroup()
        prob_labels = VGroup()
        max_bar_height = 3.5

        for i, (tok, p, col) in enumerate(zip(tokens, probs, colors)):
            bar = Rectangle(
                width=1.0, height=max_bar_height * p,
                fill_color=col, fill_opacity=0.5,
                stroke_color=col, stroke_width=2,
            )
            tok_lbl = self.mono(tok, font_size=18, color=col)
            p_lbl = self.mono(f"{p:.0%}", font_size=16, color=col)
            bars.add(bar)
            labels.add(tok_lbl)
            prob_labels.add(p_lbl)

        bars.arrange(RIGHT, buff=0.4, aligned_edge=DOWN).shift(DOWN * 0.5)
        for lbl, bar in zip(labels, bars):
            lbl.next_to(bar, DOWN, buff=0.15)
        for plbl, bar in zip(prob_labels, bars):
            plbl.next_to(bar, UP, buff=0.1)

        # Highlight arrow on the tallest bar
        arrow = Arrow(
            bars[0].get_top() + UP * 0.5,
            bars[0].get_top() + UP * 0.05,
            buff=0, color=C_YELLOW, stroke_width=3,
        )
        pick_label = self.zh("永遠揀最高", font_size=20, color=C_YELLOW)
        pick_label.next_to(arrow, UP, buff=0.1)

        # Problem note at bottom
        problem = VGroup(
            RoundedRectangle(
                corner_radius=0.12, width=10, height=0.65,
                fill_color=C_RED, fill_opacity=0.12,
                stroke_color=C_RED, stroke_width=1.5,
            ),
            self.zh(
                "問題：每次都揀 argmax → 輸出重複又沉悶",
                font_size=20, color=C_RED,
            ),
        )
        problem[1].move_to(problem[0])
        problem.to_edge(DOWN, buff=0.3)

        with self.voiceover(
            text="最簡單嘅生成方法係 greedy decoding。"
            "每一步都揀概率最高嘅 token。"
            "好似呢個例子，the 嘅概率係百分之四十，最高。"
            "Greedy 就會揀 the。"
            "呢個方法係確定性嘅，每次結果都一樣。"
            "但問題係佢好容易陷入重複循環。"
            "例如生成 the the the the。"
            "因為 locally optimal 唔等於 globally optimal。"
            "所以實際應用好少用純 greedy。"
        ):
            self.play(Write(heading), run_time=0.6)
            for bar, lbl, plbl in zip(bars, labels, prob_labels):
                self.play(
                    GrowFromEdge(bar, DOWN),
                    FadeIn(lbl), FadeIn(plbl),
                    run_time=0.35,
                )
            self.play(GrowArrow(arrow), FadeIn(pick_label), run_time=0.5)
            self.play(FadeIn(problem, shift=UP * 0.2), run_time=0.6)

        self.wait(0.3)
        self.clear()

    # ── Scene 3 — Temperature (3 side-by-side distributions) ─────────────

    def scene_temperature(self):
        heading = self.make_heading("Temperature Sampling")

        temps = [0.5, 1.0, 2.0]
        temp_colors = [C_CYAN, C_GREEN, C_ORANGE]
        temp_labels = ["T=0.5 (sharp)", "T=1.0 (original)", "T=2.0 (flat)"]

        raw_logits = [2.0, 1.0, 0.5, -0.5, -1.0]

        import math

        all_groups = VGroup()
        for ti, (T, tcol, tlbl) in enumerate(
            zip(temps, temp_colors, temp_labels)
        ):
            scaled = [l / T for l in raw_logits]
            max_s = max(scaled)
            exps = [math.exp(s - max_s) for s in scaled]
            total = sum(exps)
            probs = [e / total for e in exps]

            col_group = VGroup()
            title = self.mono(tlbl, font_size=16, color=tcol)

            bar_group = VGroup()
            max_h = 2.8
            for p in probs:
                bar = Rectangle(
                    width=0.4, height=max(max_h * p * 2.5, 0.04),
                    fill_color=tcol, fill_opacity=0.5,
                    stroke_color=tcol, stroke_width=1.5,
                )
                bar_group.add(bar)
            bar_group.arrange(RIGHT, buff=0.08, aligned_edge=DOWN)

            # Prob labels under each bar
            p_labels = VGroup()
            for bar, p in zip(bar_group, probs):
                pl = self.mono(f".{int(p*100):02d}", font_size=12, color=C_DIM)
                pl.next_to(bar, DOWN, buff=0.08)
                p_labels.add(pl)

            title.next_to(bar_group, UP, buff=0.25)
            col_group.add(title, bar_group, p_labels)
            all_groups.add(col_group)

        all_groups.arrange(RIGHT, buff=1.0).shift(DOWN * 0.3)

        # Formula
        formula = self.mono(
            "logits / T  ->  softmax  ->  sample",
            font_size=20, color=C_YELLOW,
        ).to_edge(DOWN, buff=0.6)

        with self.voiceover(
            text="Temperature 係控制生成隨機度嘅最重要參數。"
            "原理好簡單，將 logits 除以 temperature 再做 softmax。"
            "T 等於零點五嘅時候，分佈變得好尖。"
            "最高概率嘅 token 幾乎一定會被揀中。"
            "T 等於一就係原始分佈，冇任何改變。"
            "T 等於二嘅時候，分佈變得好平。"
            "所有 token 嘅機會都差唔多。"
            "ChatGPT 嘅 temperature 滑桿用嘅就係呢個原理。"
            "低 temperature 適合需要準確答案嘅場景。"
            "高 temperature 適合需要創意嘅場景。"
        ):
            self.play(Write(heading), run_time=0.6)
            for g in all_groups:
                self.play(FadeIn(g, shift=UP * 0.2), run_time=0.6)
            self.play(FadeIn(formula, shift=UP * 0.1), run_time=0.5)

        self.wait(0.3)
        self.clear()

    # ── Scene 4 — Top-k (highlight top tokens) ───────────────────────────

    def scene_topk(self):
        heading = self.make_heading("Top-k Sampling")

        tokens = ["the", "a", "cat", "is", "dog", "sat", "ran", "xyz"]
        probs = [0.30, 0.20, 0.15, 0.12, 0.10, 0.06, 0.04, 0.03]
        k = 4

        bars = VGroup()
        tok_labels = VGroup()
        prob_labels = VGroup()
        max_h = 3.2

        for i, (tok, p) in enumerate(zip(tokens, probs)):
            kept = i < k
            col = C_GREEN if kept else C_RED
            opacity = 0.5 if kept else 0.15

            bar = Rectangle(
                width=0.9, height=max_h * p,
                fill_color=col, fill_opacity=opacity,
                stroke_color=col, stroke_width=2 if kept else 1,
            )
            tlbl = self.mono(tok, font_size=16, color=col if kept else C_DIM)
            plbl = self.mono(
                f"{p:.0%}", font_size=14,
                color=col if kept else C_DIM,
            )
            bars.add(bar)
            tok_labels.add(tlbl)
            prob_labels.add(plbl)

        bars.arrange(RIGHT, buff=0.25, aligned_edge=DOWN).shift(DOWN * 0.3)
        for lbl, bar in zip(tok_labels, bars):
            lbl.next_to(bar, DOWN, buff=0.12)
        for plbl, bar in zip(prob_labels, bars):
            plbl.next_to(bar, UP, buff=0.08)

        # Dividing line after k-th bar
        divider_x = (bars[k - 1].get_right()[0] + bars[k].get_left()[0]) / 2
        divider = DashedLine(
            start=UP * 2.2 + RIGHT * divider_x,
            end=DOWN * 1.8 + RIGHT * divider_x,
            color=C_YELLOW, dash_length=0.12,
        )
        keep_lbl = self.zh(
            "保留 (k=4)", font_size=18, color=C_GREEN
        )
        keep_lbl.next_to(divider, LEFT, buff=0.3).shift(UP * 1.5)
        drop_lbl = self.zh(
            "丟棄 → -inf", font_size=18, color=C_RED
        )
        drop_lbl.next_to(divider, RIGHT, buff=0.3).shift(UP * 1.5)

        # Note at bottom
        note = VGroup(
            RoundedRectangle(
                corner_radius=0.12, width=10.5, height=0.65,
                fill_color=C_CYAN, fill_opacity=0.12,
                stroke_color=C_CYAN, stroke_width=1.5,
            ),
            self.zh(
                "GPT-2 用 k=40 — 簡單有效，但 k 係固定嘅",
                font_size=20, color=C_CYAN,
            ),
        )
        note[1].move_to(note[0])
        note.to_edge(DOWN, buff=0.3)

        with self.voiceover(
            text="Top-k sampling 只保留概率最高嘅 k 個 token。"
            "其餘嘅全部設為負無窮，即係概率歸零。"
            "好似呢個例子，k 等於四。"
            "the、a、cat 同 is 被保留。"
            "dog、sat、ran 同 xyz 被丟棄。"
            "咁就保證輸出一定嚟自最可能嘅候選詞。"
            "GPT-2 發佈嘅時候就係用 k 等於 40。"
            "但 top-k 有一個問題。"
            "k 係固定嘅，唔理模型幾有信心。"
            "如果模型好確定，一個 token 就有九成概率，"
            "k 等於 40 仲係會保留 39 個唔需要嘅 token。"
            "呢個問題就由 top-p 嚟解決。"
        ):
            self.play(Write(heading), run_time=0.6)
            for bar, tlbl, plbl in zip(bars, tok_labels, prob_labels):
                self.play(
                    GrowFromEdge(bar, DOWN),
                    FadeIn(tlbl), FadeIn(plbl),
                    run_time=0.25,
                )
            self.play(
                Create(divider),
                FadeIn(keep_lbl), FadeIn(drop_lbl),
                run_time=0.5,
            )
            self.play(FadeIn(note, shift=UP * 0.2), run_time=0.6)

        self.wait(0.3)
        self.clear()

    # ── Scene 5 — Top-p / Nucleus (cumulative cutoff) ────────────────────

    def scene_topp(self):
        heading = self.make_heading("Top-p (Nucleus) Sampling")

        tokens = ["the", "a", "cat", "is", "dog", "sat", "ran", "xyz"]
        probs = [0.35, 0.22, 0.16, 0.11, 0.07, 0.04, 0.03, 0.02]
        p_threshold = 0.9

        cumsum = []
        running = 0.0
        for prob in probs:
            running += prob
            cumsum.append(running)

        # Find nucleus boundary
        nucleus_size = 0
        for i, c in enumerate(cumsum):
            nucleus_size = i + 1
            if c >= p_threshold:
                break

        bars = VGroup()
        tok_labels = VGroup()
        cum_labels = VGroup()
        max_h = 2.8

        for i, (tok, prob, cs) in enumerate(zip(tokens, probs, cumsum)):
            kept = i < nucleus_size
            col = C_GREEN if kept else C_RED
            opacity = 0.5 if kept else 0.15

            bar = Rectangle(
                width=0.9, height=max_h * prob,
                fill_color=col, fill_opacity=opacity,
                stroke_color=col, stroke_width=2 if kept else 1,
            )
            tlbl = self.mono(tok, font_size=16, color=col if kept else C_DIM)
            clbl = self.mono(
                f"{cs:.0%}", font_size=13,
                color=C_YELLOW if kept else C_DIM,
            )
            bars.add(bar)
            tok_labels.add(tlbl)
            cum_labels.add(clbl)

        bars.arrange(RIGHT, buff=0.25, aligned_edge=DOWN).shift(DOWN * 0.2)
        for lbl, bar in zip(tok_labels, bars):
            lbl.next_to(bar, DOWN, buff=0.12)
        for clbl, bar in zip(cum_labels, bars):
            clbl.next_to(bar, UP, buff=0.08)

        # Cumulative sum label
        cum_title = self.zh(
            "累積概率", font_size=16, color=C_YELLOW
        ).next_to(cum_labels, UP, buff=0.15)

        # Threshold line
        threshold_line = DashedLine(
            start=LEFT * 5.5 + UP * 0.5,
            end=RIGHT * 5.5 + UP * 0.5,
            color=C_YELLOW, dash_length=0.15,
        ).shift(UP * 1.3)
        threshold_lbl = self.mono(
            "p = 0.9", font_size=20, color=C_YELLOW
        ).next_to(threshold_line, RIGHT, buff=0.2)

        # Adaptive note at bottom
        note = VGroup(
            RoundedRectangle(
                corner_radius=0.12, width=10.5, height=0.9,
                fill_color=C_PURPLE, fill_opacity=0.12,
                stroke_color=C_PURPLE, stroke_width=1.5,
            ),
            self.zh(
                "自適應：模型有信心 → 少啲 token",
                font_size=20, color=C_GREEN,
            ),
            self.zh(
                "模型唔確定 → 多啲 token",
                font_size=20, color=C_ORANGE,
            ),
        )
        note[1].move_to(note[0]).shift(UP * 0.15)
        note[2].move_to(note[0]).shift(DOWN * 0.15)
        note.to_edge(DOWN, buff=0.3)

        with self.voiceover(
            text="Top-p 又叫 nucleus sampling。"
            "佢唔係固定保留幾多個 token。"
            "而係搵最細嘅集合，使得累積概率超過 p。"
            "例如 p 等於零點九。"
            "the 有百分之三十五，累積三十五。"
            "加 a 百分之二十二，累積五十七。"
            "加 cat 百分之十六，累積七十三。"
            "加 is 百分之十一，累積八十四。"
            "加 dog 百分之七，累積九十一，超過百分之九十。"
            "所以呢五個 token 就係 nucleus。"
            "其餘嘅全部丟棄。"
            "呢個方法嘅好處係佢會自動適應。"
            "模型好有信心嗰陣，可能一兩個 token 就夠。"
            "模型唔確定嗰陣，就會保留多啲選擇。"
            "大部分現代 LLM API 都用 top-p 做預設。"
        ):
            self.play(Write(heading), run_time=0.6)
            for bar, tlbl, clbl in zip(bars, tok_labels, cum_labels):
                self.play(
                    GrowFromEdge(bar, DOWN),
                    FadeIn(tlbl), FadeIn(clbl),
                    run_time=0.3,
                )
            self.play(FadeIn(cum_title), run_time=0.3)
            self.play(
                Create(threshold_line), FadeIn(threshold_lbl),
                run_time=0.5,
            )
            self.play(FadeIn(note, shift=UP * 0.2), run_time=0.6)

        self.wait(0.3)
        self.clear()

    # ── Scene 6 — KV Cache (growing cache visualization) ─────────────────

    def scene_kv_cache(self):
        heading = self.make_heading("KV Cache 加速生成")

        # Two columns: without cache vs with cache
        # --- Without cache ---
        no_cache_title = self.zh("冇 Cache", font_size=22, color=C_RED)
        no_cache_box = RoundedRectangle(
            corner_radius=0.15, width=4.8, height=3.8,
            fill_color=C_RED, fill_opacity=0.08,
            stroke_color=C_RED, stroke_width=2,
        )

        no_cache_steps = VGroup(
            self.mono("Step 1: [T1]", font_size=15, color=C_WHITE),
            self.mono("Step 2: [T1 T2]", font_size=15, color=C_WHITE),
            self.mono("Step 3: [T1 T2 T3]", font_size=15, color=C_WHITE),
            self.mono("Step 4: [T1 T2 T3 T4]", font_size=15, color=C_WHITE),
            self.mono("...", font_size=15, color=C_DIM),
            self.mono("Step N: [T1 T2 ... TN]", font_size=15, color=C_RED),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.15)
        no_cache_steps.move_to(no_cache_box)
        no_cache_title.next_to(no_cache_box, UP, buff=0.15)

        no_cache_cost = self.mono("O(N^2)", font_size=22, color=C_RED)
        no_cache_cost.next_to(no_cache_box, DOWN, buff=0.2)

        no_cache_group = VGroup(
            no_cache_box, no_cache_title, no_cache_steps, no_cache_cost
        ).shift(LEFT * 3.2)

        # --- With cache ---
        cache_title = self.zh("有 KV Cache", font_size=22, color=C_GREEN)
        cache_box = RoundedRectangle(
            corner_radius=0.15, width=4.8, height=3.8,
            fill_color=C_GREEN, fill_opacity=0.08,
            stroke_color=C_GREEN, stroke_width=2,
        )

        cache_steps = VGroup(
            self.mono("Prefill: [T1 T2 T3]", font_size=15, color=C_CYAN),
            self.mono("  -> cache K,V", font_size=14, color=C_DIM),
            self.mono("Step 1: [T4] + cache", font_size=15, color=C_GREEN),
            self.mono("Step 2: [T5] + cache", font_size=15, color=C_GREEN),
            self.mono("Step 3: [T6] + cache", font_size=15, color=C_GREEN),
            self.mono("  cache 越嚟越大", font_size=14, color=C_YELLOW),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.15)
        cache_steps.move_to(cache_box)
        cache_title.next_to(cache_box, UP, buff=0.15)

        cache_cost = self.mono("O(N)", font_size=22, color=C_GREEN)
        cache_cost.next_to(cache_box, DOWN, buff=0.2)

        cache_group = VGroup(
            cache_box, cache_title, cache_steps, cache_cost
        ).shift(RIGHT * 3.2)

        # Center comparison
        vs_text = self.zh("vs", font_size=30, color=C_YELLOW)

        # Bottom note
        note = VGroup(
            RoundedRectangle(
                corner_radius=0.12, width=10.5, height=0.65,
                fill_color=C_CYAN, fill_opacity=0.12,
                stroke_color=C_CYAN, stroke_width=1.5,
            ),
            self.zh(
                "所有生產環境 (vLLM, TGI, llama.cpp) 都用 KV Cache",
                font_size=20, color=C_CYAN,
            ),
        )
        note[1].move_to(note[0])
        note.to_edge(DOWN, buff=0.3)

        with self.voiceover(
            text="KV Cache 係 LLM 推理嘅最重要優化。"
            "冇 cache 嘅話，生成每個新 token 都要重新計算成個序列嘅 attention。"
            "第一步計算一個 token，第二步計算兩個，第三步計算三個。"
            "到第 N 步就要計算 N 個 token，總共係 O of N squared。"
            "有 KV cache 嘅話，做法完全唔同。"
            "首先做一次 prefill，將 prompt 嘅 K 同 V 全部緩存。"
            "之後每一步只需要計算一個新 token。"
            "新 token 嘅 Q 同緩存嘅 K、V 做 attention。"
            "Cache 每步增長一個 token，但每步嘅計算量係 O of 1。"
            "總共只要 O of N，快好多倍。"
            "所有生產環境嘅 LLM 伺服器都用 KV Cache。"
        ):
            self.play(Write(heading), run_time=0.6)

            self.play(
                FadeIn(no_cache_box), FadeIn(no_cache_title), run_time=0.4
            )
            for item in no_cache_steps:
                self.play(FadeIn(item, shift=RIGHT * 0.1), run_time=0.2)
            self.play(FadeIn(no_cache_cost), run_time=0.3)

            self.play(FadeIn(vs_text), run_time=0.3)

            self.play(
                FadeIn(cache_box), FadeIn(cache_title), run_time=0.4
            )
            for item in cache_steps:
                self.play(FadeIn(item, shift=RIGHT * 0.1), run_time=0.2)
            self.play(FadeIn(cache_cost), run_time=0.3)

            self.play(FadeIn(note, shift=UP * 0.2), run_time=0.6)

        self.wait(0.3)
        self.clear()

    # ── Scene 7 — Summary (6 bullets) ────────────────────────────────────

    def scene_summary(self):
        heading = self.make_heading("總結", font_size=44)

        bullets = VGroup(
            self.zh("•  Greedy：永遠揀 argmax — 確定但重複", font_size=24, color=C_GREEN),
            self.zh("•  Temperature：T 控制隨機度 — 低=準確，高=創意", font_size=24, color=C_CYAN),
            self.zh("•  Top-k：只保留最高 k 個 token — 簡單有效", font_size=24, color=C_ORANGE),
            self.zh("•  Top-p：累積概率 >= p 嘅最細集合 — 自適應", font_size=24, color=C_PURPLE),
            self.zh("•  Repetition Penalty：減低已出現 token 嘅概率", font_size=24, color=C_PINK),
            self.zh("•  KV Cache：緩存 K/V — O(N) 取代 O(N^2)", font_size=24, color=C_YELLOW),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.35).move_to(ORIGIN)

        with self.voiceover(
            text="總結一下今日學咗嘅六個重點。"
            "第一，greedy decoding 永遠揀最高概率，確定但容易重複。"
            "第二，temperature 控制分佈嘅尖或平，低就準確，高就有創意。"
            "第三，top-k 只保留最高嘅 k 個 token，簡單有效。"
            "第四，top-p nucleus sampling 自動適應分佈嘅形狀。"
            "第五，repetition penalty 減低已經出現嘅 token 嘅概率，避免重複。"
            "第六，KV cache 將推理嘅複雜度從 N 平方降到 N，"
            "係所有生產環境嘅標準優化。"
            "實際應用通常會組合幾種策略一齊用。"
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
            "下一課：Evaluation（評估同基準測試）",
            font_size=28, color=C_CYAN,
        ).next_to(thanks, DOWN, buff=0.6)

        with self.voiceover(
            text="多謝收睇！下一課我哋會學 Evaluation 同 Benchmarking。"
            "即係點樣衡量你嘅模型到底有幾好。"
            "包括 perplexity、HellaSwag、MMLU 呢啲基準測試。"
            "記得繼續跟住學喇！"
        ):
            self.play(Write(thanks), run_time=1)
            self.play(FadeIn(next_lesson, shift=UP * 0.2), run_time=0.8)

        self.wait(1)
        self.play(FadeOut(thanks), FadeOut(next_lesson), run_time=0.8)
