"""
Lesson 4 – Text Representation & Tokenization
Manim CE + Cantonese voiceover (Edge TTS)

Render (1080p):
    cd "lesson 4/video"
    manim render -qh scene.py TokenizationExplainer
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


class TokenizationExplainer(VoiceoverScene):
    """Single scene explaining tokenization & BPE in Cantonese."""

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
        self.scene_problem()
        self.scene_levels()
        self.scene_bpe_algorithm()
        self.scene_encode_decode()
        self.scene_special_tokens()
        self.scene_vocab_size()
        self.scene_embeddings()
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

    def make_token_box(self, label, color, width=1.0, height=0.65, font_size=20):
        rect = RoundedRectangle(
            corner_radius=0.1,
            width=width,
            height=height,
            fill_color=color,
            fill_opacity=0.3,
            stroke_color=color,
            stroke_width=2,
        )
        txt = self.mono(label, font_size=font_size, color=color).move_to(rect)
        return VGroup(rect, txt)

    # ── Scene 1 — Title / Intro ──────────────────────────────────────────

    def scene_intro(self):
        title = self.zh("文字分詞器", font_size=56, color=C_BLUE)
        sub_en = self.en(
            "Text Tokenization", font_size=24, color=C_WHITE
        ).next_to(title, DOWN, buff=0.4)
        sub_zh = self.zh(
            "第四課", font_size=36, color=C_ORANGE
        ).next_to(sub_en, DOWN, buff=0.3)
        producer = self.zh(
            "制片人：Peter", font_size=24, color=C_DIM
        ).next_to(sub_zh, DOWN, buff=0.7)

        with self.voiceover(
            text="大家好！歡迎嚟到第四課。"
            "今日我哋嚟學 Tokenization，即係文字分詞。"
            "呢個係建造 LLM 嘅第一步，"
            "因為神經網絡只睇得明數字，唔識睇文字。"
            "所以我哋需要一個方法，將文字轉換做數字。"
        ):
            self.play(Write(title), run_time=1.5)
            self.play(FadeIn(sub_en, shift=UP * 0.2), run_time=0.8)
            self.play(FadeIn(sub_zh, shift=UP * 0.2), run_time=0.8)
            self.play(FadeIn(producer, shift=UP * 0.2), run_time=0.5)

        self.wait(0.3)
        self.clear()

    # ── Scene 2 — The Problem ────────────────────────────────────────────

    def scene_problem(self):
        heading = self.make_heading("核心問題")

        text_box = self.make_token_box("Hello World", C_GREEN, width=3.5, height=0.8, font_size=28)
        text_box.shift(LEFT * 3.5 + UP * 0.3)

        question = self.mono("???", font_size=40, color=C_RED)
        question.move_to(ORIGIN + UP * 0.3)

        num_boxes = VGroup()
        nums = ["15496", "995"]
        for n in nums:
            box = self.make_token_box(n, C_ORANGE, width=1.6, height=0.8, font_size=24)
            num_boxes.add(box)
        num_boxes.arrange(RIGHT, buff=0.2).shift(RIGHT * 3.5 + UP * 0.3)

        arrow1 = Arrow(
            text_box.get_right(), question.get_left(),
            buff=0.2, color=C_DIM, stroke_width=3,
        )
        arrow2 = Arrow(
            question.get_right(), num_boxes.get_left(),
            buff=0.2, color=C_DIM, stroke_width=3,
        )

        brain_label = self.zh(
            "神經網絡只識處理數字！", font_size=26, color=C_YELLOW
        ).shift(DOWN * 1.2)

        explain_box = VGroup(
            RoundedRectangle(
                corner_radius=0.15, width=10, height=0.7,
                fill_color=C_CYAN, fill_opacity=0.15, stroke_color=C_CYAN,
            ),
            self.zh(
                "Tokenization = 將文字轉換做數字嘅橋樑",
                font_size=22, color=C_CYAN,
            ),
        )
        explain_box[1].move_to(explain_box[0])
        explain_box.to_edge(DOWN, buff=0.5)

        with self.voiceover(
            text="首先，我哋要理解核心問題。"
            "你有一段文字，例如 Hello World。"
            "但係神經網絡唔識睇文字，佢只識處理數字。"
            "所以我哋要將 Hello World 轉換做一組數字，"
            "例如 15496 同 995。"
            "呢個轉換過程就叫做 Tokenization。"
            "佢係文字同數字之間嘅橋樑。"
        ):
            self.play(Write(heading), run_time=0.6)
            self.play(FadeIn(text_box, shift=RIGHT * 0.3), run_time=0.6)
            self.play(GrowArrow(arrow1), run_time=0.4)
            self.play(Write(question), run_time=0.5)
            self.play(GrowArrow(arrow2), run_time=0.4)
            self.play(FadeIn(num_boxes, shift=RIGHT * 0.3), run_time=0.6)
            self.play(FadeIn(brain_label, shift=UP * 0.2), run_time=0.5)
            self.play(FadeIn(explain_box, shift=UP * 0.2), run_time=0.6)

        self.wait(0.3)
        self.clear()

    # ── Scene 3 — Three Levels ───────────────────────────────────────────

    def scene_levels(self):
        heading = self.make_heading("三種分詞方法")

        col_colors = [C_GREEN, C_ORANGE, C_PURPLE]
        col_titles = ["字符級", "詞語級", "子詞級 (BPE)"]
        col_examples_raw = [
            ["h", "e", "l", "l", "o"],
            ["hello", "world"],
            ["hel", "lo", "wor", "ld"],
        ]
        col_vocab = ["~256", "170,000+", "32,000"]
        col_seqlen = ["好長", "好短", "適中"]

        columns = VGroup()
        for idx in range(3):
            col = VGroup()
            title = self.zh(col_titles[idx], font_size=26, color=col_colors[idx])
            col.add(title)

            tok_group = VGroup()
            for t in col_examples_raw[idx]:
                box = self.make_token_box(t, col_colors[idx], width=max(0.8, len(t) * 0.3 + 0.3), height=0.55, font_size=18)
                tok_group.add(box)
            tok_group.arrange(RIGHT, buff=0.1)
            col.add(tok_group)

            vocab_label = self.mono(
                f"Vocab: {col_vocab[idx]}", font_size=16, color=C_DIM
            )
            seq_label = VGroup(
                self.zh("序列長度: ", font_size=16, color=C_DIM),
                self.zh(col_seqlen[idx], font_size=16, color=col_colors[idx]),
            ).arrange(RIGHT, buff=0.1)
            col.add(vocab_label)
            col.add(seq_label)
            col.arrange(DOWN, buff=0.3, aligned_edge=LEFT)
            columns.add(col)

        columns.arrange(RIGHT, buff=1.0, aligned_edge=UP).shift(DOWN * 0.3)

        winner = VGroup(
            RoundedRectangle(
                corner_radius=0.15, width=8, height=0.7,
                fill_color=C_PURPLE, fill_opacity=0.15, stroke_color=C_PURPLE,
            ),
            self.zh(
                "GPT / LLaMA / Claude 全部用子詞分詞 (BPE)",
                font_size=22, color=C_PURPLE,
            ),
        )
        winner[1].move_to(winner[0])
        winner.to_edge(DOWN, buff=0.4)

        with self.voiceover(
            text="分詞有三種方法。"
            "第一種係字符級，將每個字母變成一個 token。"
            "好處係 vocab 好細，大概 256 個就夠。"
            "壞處係序列會好長，每個字母都係一個 token。"
            "第二種係詞語級，將每個詞語變成一個 token。"
            "序列好短，但 vocab 要 17 萬個以上。"
            "仲有一個問題，如果遇到未見過嘅詞就唔識處理。"
            "第三種係子詞級，即係 BPE。"
            "佢係最佳平衡點，vocab 大概 3 萬到 10 萬。"
            "常見嘅詞保持完整，罕見嘅詞拆開做子詞。"
            "所有現代 LLM 都用呢種方法。"
        ):
            self.play(Write(heading), run_time=0.6)
            for i, col in enumerate(columns):
                anims = [FadeIn(item, shift=DOWN * 0.2) for item in col]
                self.play(*anims, run_time=0.8)
            self.play(FadeIn(winner, shift=UP * 0.2), run_time=0.6)

        self.wait(0.3)
        self.clear()

    # ── Scene 4 — BPE Algorithm (Animated) ────────────────────────────────

    def scene_bpe_algorithm(self):
        heading = self.make_heading("BPE 算法演示")

        tokens_data = list("aabdaaabac")

        def make_tok_row(toks, color=C_GREEN):
            grp = VGroup()
            for t in toks:
                box = self.make_token_box(t, color, width=max(0.65, len(t) * 0.35 + 0.3), height=0.6, font_size=20)
                grp.add(box)
            grp.arrange(RIGHT, buff=0.08)
            return grp

        step_colors = [C_GREEN, C_ORANGE, C_PINK, C_PURPLE]

        current_tokens = list(tokens_data)

        steps = [
            (("a", "a"), "aa"),
            (("aa", "b"), "aab"),
            (("aab", "d"), "aabd"),
            (("aab", "a"), "aaba"),
        ]

        tok_row = make_tok_row(current_tokens, step_colors[0])
        tok_row.shift(UP * 0.8)

        step_label = self.zh(
            "初始：每個字符係一個 token",
            font_size=22, color=C_DIM,
        ).shift(UP * 2.0)

        with self.voiceover(
            text="而家我哋用一個簡單嘅例子嚟演示 BPE 算法。"
            "由文字 a a b d a a a b a c 開始。"
            "一開始每個字符都係獨立嘅 token。"
            "然後我哋數一數邊對相鄰嘅 token 出現得最多。"
        ):
            self.play(Write(heading), run_time=0.6)
            self.play(FadeIn(step_label), run_time=0.4)
            self.play(FadeIn(tok_row, shift=DOWN * 0.2), run_time=0.8)

        self.wait(0.3)

        for step_i, (pair, merged) in enumerate(steps):
            new_tokens = []
            i = 0
            while i < len(current_tokens):
                if i < len(current_tokens) - 1 and current_tokens[i] == pair[0] and current_tokens[i + 1] == pair[1]:
                    new_tokens.append(merged)
                    i += 2
                else:
                    new_tokens.append(current_tokens[i])
                    i += 1

            col = step_colors[min(step_i + 1, len(step_colors) - 1)]
            new_row = make_tok_row(new_tokens, col)
            new_row.shift(UP * 0.8)

            merge_text = self.zh(
                f"第{step_i + 1}步：合併 '{pair[0]}' + '{pair[1]}' = '{merged}'",
                font_size=22, color=col,
            ).shift(DOWN * 0.5 + LEFT * 0.5)

            with self.voiceover(
                text=f"第{['一', '二', '三', '四'][step_i]}步，"
                f"合併 {pair[0]} 同 {pair[1]}，變成 {merged}。"
                + ("之後再搵下一對最常見嘅組合。" if step_i < 3 else
                   "經過四步合併之後，token 數量大幅減少咗。")
            ):
                self.play(FadeOut(tok_row), FadeOut(step_label), run_time=0.3)
                tok_row = new_row
                step_label = merge_text
                self.play(FadeIn(step_label), run_time=0.3)
                self.play(FadeIn(tok_row, shift=DOWN * 0.2), run_time=0.6)

            current_tokens = new_tokens
            self.wait(0.2)

        result_note = VGroup(
            RoundedRectangle(
                corner_radius=0.15, width=10, height=0.7,
                fill_color=C_YELLOW, fill_opacity=0.12, stroke_color=C_YELLOW,
            ),
            self.zh(
                f"10 個字符 → {len(current_tokens)} 個 token！合併規則就係 tokenizer",
                font_size=22, color=C_YELLOW,
            ),
        )
        result_note[1].move_to(result_note[0])
        result_note.to_edge(DOWN, buff=0.5)

        with self.voiceover(
            text=f"原本 10 個字符，而家只係 {len(current_tokens)} 個 token。"
            "呢啲合併規則嘅順序，就係成個 tokenizer 嘅核心。"
        ):
            self.play(FadeIn(result_note, shift=UP * 0.2), run_time=0.6)

        self.wait(0.3)
        self.clear()

    # ── Scene 5 — Encode / Decode ─────────────────────────────────────────

    def scene_encode_decode(self):
        heading = self.make_heading("編碼同解碼")

        text_label = self.zh("文字", font_size=22, color=C_DIM).shift(UP * 1.8 + LEFT * 4)
        text_box = self.make_token_box("the cat sat", C_GREEN, width=3.5, height=0.7, font_size=24)
        text_box.next_to(text_label, DOWN, buff=0.3)

        encode_arrow = Arrow(
            text_box.get_right() + RIGHT * 0.1,
            text_box.get_right() + RIGHT * 2.0,
            buff=0, color=C_BLUE, stroke_width=3,
        )
        encode_label = self.zh("encode", font_size=20, color=C_BLUE).next_to(encode_arrow, UP, buff=0.1)

        ids_label = self.zh("Token IDs", font_size=22, color=C_DIM).shift(UP * 1.8 + RIGHT * 2)
        id_boxes = VGroup()
        ids = [4, 15, 8, 22, 8, 19]
        for tid in ids:
            box = self.make_token_box(str(tid), C_ORANGE, width=0.7, height=0.6, font_size=18)
            id_boxes.add(box)
        id_boxes.arrange(RIGHT, buff=0.08).next_to(ids_label, DOWN, buff=0.3)

        decode_arrow = Arrow(
            id_boxes.get_bottom() + DOWN * 0.3,
            id_boxes.get_bottom() + DOWN * 1.5 + LEFT * 3,
            buff=0, color=C_PINK, stroke_width=3,
        )
        decode_label = self.zh("decode", font_size=20, color=C_PINK).next_to(decode_arrow, RIGHT, buff=0.1)

        result_box = self.make_token_box("the cat sat", C_GREEN, width=3.5, height=0.7, font_size=24)
        result_box.shift(DOWN * 2.3 + LEFT * 2)

        roundtrip = VGroup(
            RoundedRectangle(
                corner_radius=0.15, width=8, height=0.7,
                fill_color=C_YELLOW, fill_opacity=0.12, stroke_color=C_YELLOW,
            ),
            self.zh(
                "decode(encode(text)) == text  必須無損！",
                font_size=22, color=C_YELLOW,
            ),
        )
        roundtrip[1].move_to(roundtrip[0])
        roundtrip.to_edge(DOWN, buff=0.35)

        with self.voiceover(
            text="Tokenizer 要有兩個功能。"
            "第一個係 encode，將文字轉換做 token ID。"
            "例如 the cat sat 變成一組數字。"
            "第二個係 decode，將 token ID 轉返做文字。"
            "最重要嘅係，decode encode 之後一定要同原文一樣。"
            "呢個叫做無損轉換，roundtrip 必須完美。"
        ):
            self.play(Write(heading), run_time=0.6)
            self.play(FadeIn(text_label), FadeIn(text_box, shift=RIGHT * 0.2), run_time=0.5)
            self.play(GrowArrow(encode_arrow), FadeIn(encode_label), run_time=0.5)
            self.play(FadeIn(ids_label), run_time=0.3)
            for box in id_boxes:
                self.play(FadeIn(box, shift=DOWN * 0.1), run_time=0.15)
            self.play(GrowArrow(decode_arrow), FadeIn(decode_label), run_time=0.5)
            self.play(FadeIn(result_box, shift=UP * 0.2), run_time=0.5)
            self.play(FadeIn(roundtrip, shift=UP * 0.2), run_time=0.6)

        self.wait(0.3)
        self.clear()

    # ── Scene 6 — Special Tokens ─────────────────────────────────────────

    def scene_special_tokens(self):
        heading = self.make_heading("特殊 Tokens")

        specials = [
            ("<bos>", "開始", C_GREEN, "標記序列開始"),
            ("<eos>", "結束", C_RED, "標記序列結束"),
            ("<pad>", "填充", C_CYAN, "將短序列填滿"),
            ("<unk>", "未知", C_ORANGE, "處理未見過嘅詞"),
        ]

        token_sequence = VGroup()
        bos_box = self.make_token_box("<bos>", C_GREEN, width=1.2, height=0.7, font_size=18)
        w1 = self.make_token_box("the", C_WHITE, width=0.9, height=0.7, font_size=18)
        w2 = self.make_token_box("cat", C_WHITE, width=0.9, height=0.7, font_size=18)
        w3 = self.make_token_box("sat", C_WHITE, width=0.9, height=0.7, font_size=18)
        eos_box = self.make_token_box("<eos>", C_RED, width=1.2, height=0.7, font_size=18)
        pad1 = self.make_token_box("<pad>", C_CYAN, width=1.2, height=0.7, font_size=18)
        pad2 = self.make_token_box("<pad>", C_CYAN, width=1.2, height=0.7, font_size=18)

        token_sequence = VGroup(bos_box, w1, w2, w3, eos_box, pad1, pad2)
        token_sequence.arrange(RIGHT, buff=0.12).shift(UP * 0.5)

        labels = VGroup()
        for tok_name, zh_name, col, desc in specials:
            row = VGroup(
                self.mono(tok_name, font_size=22, color=col),
                self.zh(f"  {zh_name}：{desc}", font_size=20, color=C_DIM),
            ).arrange(RIGHT, buff=0.2)
            labels.add(row)
        labels.arrange(DOWN, aligned_edge=LEFT, buff=0.25).shift(DOWN * 1.5)

        with self.voiceover(
            text="除咗普通嘅文字 token 之外，仲有幾個特殊 token。"
            "B O S，即係 beginning of sequence，標記序列嘅開始。"
            "E O S，即係 end of sequence，標記序列嘅結束。"
            "模型生成到 E O S 就會停止。"
            "PAD 係填充 token，將唔同長度嘅序列填滿到同一個長度。"
            "因為 GPU 需要 batch 入面每條序列一樣長。"
            "UNK 係未知 token，遇到 vocab 入面無嘅詞就用佢。"
            "不過用 BPE 基本上唔會遇到 UNK，因為可以退回到字符級。"
        ):
            self.play(Write(heading), run_time=0.6)
            for box in token_sequence:
                self.play(FadeIn(box, shift=DOWN * 0.1), run_time=0.2)
            for label in labels:
                self.play(FadeIn(label, shift=RIGHT * 0.2), run_time=0.4)

        self.wait(0.3)
        self.clear()

    # ── Scene 7 — Vocabulary Size (Bar Chart) ─────────────────────────────

    def scene_vocab_size(self):
        heading = self.make_heading("Vocab 大小比較")

        data = [
            ("字符級\n256", 256, C_GREEN, 0.4),
            ("BERT\n30K", 30000, C_ORANGE, 2.0),
            ("LLaMA\n32K", 32000, C_CYAN, 2.1),
            ("GPT-2\n50K", 50257, C_PINK, 2.8),
            ("GPT-4\n100K", 100000, C_PURPLE, 4.0),
            ("LLaMA-3\n128K", 128256, C_YELLOW, 4.8),
        ]

        bars = VGroup()
        labels = VGroup()
        for name, vocab_size, col, bar_h in data:
            bar = Rectangle(
                width=1.0, height=bar_h,
                fill_color=col, fill_opacity=0.4,
                stroke_color=col, stroke_width=2,
            )
            label = self.zh(name, font_size=14, color=col)
            bars.add(bar)
            labels.add(label)

        bar_group = VGroup()
        for i, (bar, label) in enumerate(zip(bars, labels)):
            bar.move_to(LEFT * 4 + RIGHT * i * 1.6)
            bar.align_to(DOWN * 1.2, DOWN)
            label.next_to(bar, DOWN, buff=0.15)
            bar_group.add(VGroup(bar, label))

        bar_group.move_to(ORIGIN + DOWN * 0.2)

        tradeoff = VGroup(
            RoundedRectangle(
                corner_radius=0.15, width=10, height=0.7,
                fill_color=C_YELLOW, fill_opacity=0.12, stroke_color=C_YELLOW,
            ),
            self.zh(
                "Vocab 越大 = 序列越短，但 embedding 表越大",
                font_size=22, color=C_YELLOW,
            ),
        )
        tradeoff[1].move_to(tradeoff[0])
        tradeoff.to_edge(DOWN, buff=0.35)

        with self.voiceover(
            text="Vocab 嘅大小對模型有好大影響。"
            "字符級只需要 256 個 token，但序列會好長。"
            "BERT 用大概 3 萬個 token。"
            "LLaMA 用 3 萬 2 千。"
            "GPT-2 用 5 萬。"
            "GPT-4 增加到 10 萬。"
            "而最新嘅 LLaMA 3 用到 12 萬 8 千個 token。"
            "Vocab 越大，序列越短，速度越快。"
            "但代價係 embedding 表變大，佔用更多記憶體。"
        ):
            self.play(Write(heading), run_time=0.6)
            for bg in bar_group:
                self.play(FadeIn(bg, shift=UP * 0.3), run_time=0.4)
            self.play(FadeIn(tradeoff, shift=UP * 0.2), run_time=0.6)

        self.wait(0.3)
        self.clear()

    # ── Scene 8 — Token Embeddings ────────────────────────────────────────

    def scene_embeddings(self):
        heading = self.make_heading("Token Embeddings")

        tok_box = self.make_token_box("cat", C_GREEN, width=1.2, height=0.7, font_size=24)
        tok_box.shift(LEFT * 4.5 + UP * 0.3)

        tok_id_label = self.mono("ID: 42", font_size=18, color=C_DIM).next_to(tok_box, DOWN, buff=0.15)

        arrow1 = Arrow(
            tok_box.get_right(), tok_box.get_right() + RIGHT * 1.5,
            buff=0.1, color=C_BLUE, stroke_width=3,
        )
        lookup_label = self.zh("查表", font_size=18, color=C_BLUE).next_to(arrow1, UP, buff=0.1)

        table_rect = RoundedRectangle(
            corner_radius=0.15,
            width=3.0, height=2.5,
            fill_color=C_ORANGE, fill_opacity=0.15,
            stroke_color=C_ORANGE,
        )
        table_rect.move_to(ORIGIN + UP * 0.3)
        table_title = self.zh("Embedding 表", font_size=18, color=C_ORANGE).next_to(table_rect, UP, buff=0.1)

        table_rows = VGroup()
        row_data = [
            ("ID 40", "[0.12, -0.34, ...]", C_DIM),
            ("ID 41", "[0.56,  0.78, ...]", C_DIM),
            ("ID 42", "[0.91, -0.23, ...]", C_YELLOW),
            ("ID 43", "[-0.45, 0.67, ...]", C_DIM),
        ]
        for rid, vec, col in row_data:
            row = VGroup(
                self.mono(rid, font_size=14, color=col),
                self.mono(vec, font_size=14, color=col),
            ).arrange(RIGHT, buff=0.3)
            table_rows.add(row)
        table_rows.arrange(DOWN, aligned_edge=LEFT, buff=0.15).move_to(table_rect)

        highlight = SurroundingRectangle(
            table_rows[2], color=C_YELLOW, buff=0.08, stroke_width=2,
        )

        arrow2 = Arrow(
            table_rect.get_right(), table_rect.get_right() + RIGHT * 1.5,
            buff=0.1, color=C_PINK, stroke_width=3,
        )

        vec_boxes = VGroup()
        vec_vals = ["0.91", "-0.23", "0.45", "..."]
        for v in vec_vals:
            box = Rectangle(
                width=0.8, height=0.55,
                fill_color=C_PINK, fill_opacity=0.25,
                stroke_color=C_PINK, stroke_width=2,
            )
            txt = self.mono(v, font_size=14, color=C_PINK).move_to(box)
            vec_boxes.add(VGroup(box, txt))
        vec_boxes.arrange(RIGHT, buff=0.06).next_to(arrow2, RIGHT, buff=0.1)
        vec_label = self.zh("d_model 維向量", font_size=16, color=C_PINK).next_to(vec_boxes, DOWN, buff=0.15)

        tying_box = VGroup(
            RoundedRectangle(
                corner_radius=0.15, width=10, height=0.7,
                fill_color=C_PURPLE, fill_opacity=0.15, stroke_color=C_PURPLE,
            ),
            self.zh(
                "Weight Tying：輸入 embedding 同輸出 LM head 共用同一個表",
                font_size=20, color=C_PURPLE,
            ),
        )
        tying_box[1].move_to(tying_box[0])
        tying_box.to_edge(DOWN, buff=0.35)

        with self.voiceover(
            text="有咗 token ID 之後，我哋要將佢變成一個向量。"
            "呢個過程叫做 embedding。"
            "例如 cat 呢個 token 嘅 ID 係 42。"
            "我哋有一個好大嘅查找表，每一行對應一個 token。"
            "用 ID 42 去查找，就得到一個 d model 維嘅向量。"
            "呢個向量就係 cat 嘅數字表示。"
            "模型訓練嘅時候會學習呢啲向量。"
            "仲有一個叫 weight tying 嘅技巧，"
            "將輸入嘅 embedding 表同輸出嘅 LM head 共用同一個表，"
            "可以節省好多參數。"
        ):
            self.play(Write(heading), run_time=0.6)
            self.play(FadeIn(tok_box, shift=RIGHT * 0.2), FadeIn(tok_id_label), run_time=0.5)
            self.play(GrowArrow(arrow1), FadeIn(lookup_label), run_time=0.4)
            self.play(
                FadeIn(table_rect), FadeIn(table_title),
                *[FadeIn(r) for r in table_rows],
                run_time=0.8,
            )
            self.play(Create(highlight), run_time=0.4)
            self.play(GrowArrow(arrow2), run_time=0.4)
            self.play(
                *[FadeIn(b, shift=RIGHT * 0.1) for b in vec_boxes],
                FadeIn(vec_label),
                run_time=0.6,
            )
            self.play(FadeIn(tying_box, shift=UP * 0.2), run_time=0.6)

        self.wait(0.3)
        self.clear()

    # ── Scene 9 — Summary ────────────────────────────────────────────────

    def scene_summary(self):
        heading = self.make_heading("總結", font_size=44)

        bullets = VGroup(
            self.zh("1. Tokenization 將文字轉換做數字", font_size=26, color=C_GREEN),
            self.zh("2. BPE 合併最常見嘅字符對建立子詞", font_size=26, color=C_ORANGE),
            self.zh("3. encode 同 decode 必須無損往返", font_size=26, color=C_PINK),
            self.zh("4. 特殊 token：<bos> <eos> <pad> <unk>", font_size=26, color=C_CYAN),
            self.zh("5. Vocab 大小影響序列長度同模型大小", font_size=26, color=C_PURPLE),
            self.zh("6. Embedding 將 token ID 變成向量", font_size=26, color=C_YELLOW),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.35).move_to(ORIGIN)

        with self.voiceover(
            text="總結一下今日學咗嘅嘢。"
            "第一，Tokenization 係將文字轉換做數字嘅過程。"
            "第二，BPE 算法通過合併最常見嘅字符對嚟建立子詞詞匯。"
            "第三，encode 同 decode 必須係無損嘅往返轉換。"
            "第四，特殊 token 包括 BOS、EOS、PAD 同 UNK。"
            "第五，Vocab 大小直接影響序列長度同模型大小。"
            "第六，Embedding 將 token ID 變成 d model 維嘅向量。"
            "掌握咗呢啲概念，你就明白 LLM 點樣處理文字喇。"
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
            "下一課：注意力機制 Attention Mechanism",
            font_size=26, color=C_PURPLE,
        ).next_to(thanks, DOWN, buff=0.6)

        with self.voiceover(
            text="多謝收睇！下一課我哋會學注意力機制，"
            "即係 Attention Mechanism。"
            "呢個係 Transformer 最核心嘅創新，"
            "query、key、value 嘅概念會令你大開眼界。"
            "記得繼續跟住學喇！"
        ):
            self.play(Write(thanks), run_time=1)
            self.play(FadeIn(next_lesson, shift=UP * 0.2), run_time=0.8)

        self.wait(1)
        self.play(FadeOut(thanks), FadeOut(next_lesson), run_time=0.8)
