"""
Lesson 7 – Data Pipeline for Pre-Training
Manim CE + Cantonese voiceover (Edge TTS)

Render (1080p):
    cd "lesson 7/video"
    manim render -qh scene.py DataPipelineExplainer
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


class DataPipelineExplainer(VoiceoverScene):
    """Single scene explaining the data pipeline for LLM pre-training in Cantonese."""

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
        self.scene_data_sources()
        self.scene_cleaning()
        self.scene_formatting()
        self.scene_dataloader()
        self.scene_mixing()
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
        title = self.zh("數據管道", font_size=56, color=C_BLUE)
        sub_en = self.en(
            "Data Pipeline for Pre-Training", font_size=24, color=C_WHITE
        ).next_to(title, DOWN, buff=0.4)
        sub_zh = self.zh(
            "第七課", font_size=36, color=C_ORANGE
        ).next_to(sub_en, DOWN, buff=0.3)
        producer = self.zh(
            "制片人：Peter", font_size=24, color=C_DIM
        ).next_to(sub_zh, DOWN, buff=0.7)

        with self.voiceover(
            text="大家好！歡迎嚟到第七課。"
            "上一課我哋組裝咗一個完整嘅 Transformer 模型。"
            "今日我哋要解決一個同樣重要嘅問題：點樣準備數據嚟訓練呢個模型。"
            "一個好嘅數據管道係 LLM 成功嘅關鍵。"
        ):
            self.play(Write(title), run_time=1.5)
            self.play(FadeIn(sub_en, shift=UP * 0.2), run_time=0.8)
            self.play(FadeIn(sub_zh, shift=UP * 0.2), run_time=0.8)
            self.play(FadeIn(producer, shift=UP * 0.2), run_time=0.5)

        self.wait(0.3)
        self.clear()

    # ── Scene 2 — Data Sources ───────────────────────────────────────────

    def scene_data_sources(self):
        heading = self.make_heading("數據來源 Data Sources")

        sources = [
            ("Common Crawl", "網頁數據", C_ORANGE, 5.0),
            ("Wikipedia", "百科全書", C_GREEN, 1.5),
            ("Books", "書籍", C_CYAN, 1.5),
            ("Code", "程式碼", C_PINK, 1.5),
            ("FineWeb", "清洗後嘅網頁", C_PURPLE, 3.5),
        ]

        bars = VGroup()
        labels_left = VGroup()
        labels_right = VGroup()
        max_w = 6.0

        for name, desc, col, rel_size in sources:
            bar_w = rel_size / 5.0 * max_w
            bar = RoundedRectangle(
                corner_radius=0.1, width=bar_w, height=0.55,
                fill_color=col, fill_opacity=0.4,
                stroke_color=col, stroke_width=2,
            )
            bars.add(bar)

            left = self.en(name, font_size=18, color=col)
            labels_left.add(left)

            right = self.zh(desc, font_size=16, color=C_DIM)
            labels_right.add(right)

        all_rows = VGroup()
        for bar, ll, lr in zip(bars, labels_left, labels_right):
            row = VGroup(ll, bar, lr).arrange(RIGHT, buff=0.3)
            all_rows.add(row)

        all_rows.arrange(DOWN, aligned_edge=LEFT, buff=0.25).move_to(
            DOWN * 0.2
        )

        # Scale note
        scale_note = VGroup(
            RoundedRectangle(
                corner_radius=0.12, width=9, height=0.7,
                fill_color=C_YELLOW, fill_opacity=0.1,
                stroke_color=C_YELLOW, stroke_width=1.5,
            ),
            self.zh(
                "GPT-3: 300B tokens    LLaMA-2: 2T tokens    FineWeb: 15T tokens",
                font_size=18, color=C_YELLOW,
            ),
        )
        scale_note[1].move_to(scale_note[0])
        scale_note.to_edge(DOWN, buff=0.35)

        with self.voiceover(
            text="訓練 LLM 需要海量嘅文本數據。"
            "最大嘅來源係 Common Crawl，即係從互聯網爬取嘅網頁，佔大約百分之六十。"
            "Wikipedia 提供高質量嘅百科知識。"
            "Books 即書籍，提供長篇嘅連貫文本。"
            "Code 即程式碼，令模型識得寫 code。"
            "FineWeb 係 HuggingFace 清洗過嘅網頁數據，有 15 萬億個 token。"
            "GPT-3 訓練用咗 3000 億個 token，LLaMA-2 用咗 2 萬億個。"
            "數據嘅規模非常之大。"
        ):
            self.play(Write(heading), run_time=0.6)

            for row in all_rows:
                self.play(FadeIn(row, shift=RIGHT * 0.3), run_time=0.5)

            self.play(FadeIn(scale_note, shift=UP * 0.2), run_time=0.6)

        self.wait(0.3)
        self.clear()

    # ── Scene 3 — Data Cleaning (funnel) ─────────────────────────────────

    def scene_cleaning(self):
        heading = self.make_heading("數據清洗 Data Cleaning")

        # Funnel: wide top (raw data) → narrow bottom (clean data)
        funnel_top = RoundedRectangle(
            corner_radius=0.15, width=8, height=1.0,
            fill_color=C_RED, fill_opacity=0.2,
            stroke_color=C_RED, stroke_width=2,
        ).shift(UP * 1.6)
        funnel_top_label = self.zh(
            "原始數據 Raw Data", font_size=22, color=C_RED
        ).move_to(funnel_top)

        # Three filter stages
        filters = [
            ("去重 Deduplication", C_ORANGE, "移除重複嘅文檔"),
            ("長度過濾 Length Filter", C_YELLOW, "移除太短或太長嘅文檔"),
            ("質量過濾 Quality Filter", C_PINK, "移除低質量內容"),
        ]

        filter_boxes = VGroup()
        filter_notes = VGroup()
        widths = [6.5, 5.0, 3.5]

        for (label, col, note), w in zip(filters, widths):
            rect = RoundedRectangle(
                corner_radius=0.12, width=w, height=0.7,
                fill_color=col, fill_opacity=0.2,
                stroke_color=col, stroke_width=2,
            )
            txt = self.zh(label, font_size=18, color=col).move_to(rect)
            filter_boxes.add(VGroup(rect, txt))

            n = self.zh(note, font_size=14, color=C_DIM)
            filter_notes.add(n)

        filter_boxes.arrange(DOWN, buff=0.4).shift(DOWN * 0.3)

        for note, box in zip(filter_notes, filter_boxes):
            note.next_to(box, RIGHT, buff=0.3)

        # Arrows between stages
        arrow_top = Arrow(
            funnel_top.get_bottom(), filter_boxes[0].get_top(),
            buff=0.08, color=C_WHITE, stroke_width=2,
        )

        arrows = VGroup()
        for i in range(len(filter_boxes) - 1):
            a = Arrow(
                filter_boxes[i].get_bottom(), filter_boxes[i + 1].get_top(),
                buff=0.08, color=C_WHITE, stroke_width=2,
            )
            arrows.add(a)

        # Clean output
        clean_box = RoundedRectangle(
            corner_radius=0.15, width=2.5, height=0.7,
            fill_color=C_GREEN, fill_opacity=0.3,
            stroke_color=C_GREEN, stroke_width=2,
        ).next_to(filter_boxes[-1], DOWN, buff=0.5)
        clean_label = self.zh(
            "乾淨數據", font_size=20, color=C_GREEN
        ).move_to(clean_box)
        arrow_bottom = Arrow(
            filter_boxes[-1].get_bottom(), clean_box.get_top(),
            buff=0.08, color=C_WHITE, stroke_width=2,
        )

        pct_label = self.zh(
            "FineWeb 移除咗 ~90% 嘅 Common Crawl", font_size=18, color=C_YELLOW
        ).to_edge(DOWN, buff=0.35)

        with self.voiceover(
            text="原始嘅網頁數據非常之髒，所以清洗步驟好重要。"
            "第一步係去重，即 deduplication。用 hash 比對去除完全相同嘅文檔。"
            "Common Crawl 有大量重複嘅內容，例如 boilerplate 同鏡像網站。"
            "第二步係長度過濾。太短嘅文檔通常係導航欄或者錯誤頁面，"
            "太長嘅可能係自動生成嘅內容。"
            "第三步係質量過濾。檢查特殊字符比例、字母比例等等，"
            "移除垃圾內容同 SEO spam。"
            "經過呢三步之後，FineWeb 移除咗大約百分之九十嘅原始數據。"
            "記住：數據質量比數量更加重要。"
        ):
            self.play(Write(heading), run_time=0.6)

            self.play(FadeIn(funnel_top), FadeIn(funnel_top_label), run_time=0.5)
            self.play(GrowArrow(arrow_top), run_time=0.3)

            for i, (box, note) in enumerate(zip(filter_boxes, filter_notes)):
                self.play(FadeIn(box, shift=DOWN * 0.15), run_time=0.5)
                self.play(FadeIn(note, shift=RIGHT * 0.1), run_time=0.3)
                if i < len(arrows):
                    self.play(GrowArrow(arrows[i]), run_time=0.25)

            self.play(GrowArrow(arrow_bottom), run_time=0.3)
            self.play(
                FadeIn(clean_box), FadeIn(clean_label),
                run_time=0.5,
            )
            self.play(FadeIn(pct_label, shift=UP * 0.1), run_time=0.5)

        self.wait(0.3)
        self.clear()

    # ── Scene 4 — Formatting / Packing ───────────────────────────────────

    def scene_formatting(self):
        heading = self.make_heading("序列打包 Sequence Packing")

        # --- Padding example (top) ---
        pad_title = self.zh(
            "Padding（填充）", font_size=22, color=C_RED
        ).shift(UP * 1.5 + LEFT * 4.5)

        pad_colors = [C_GREEN, C_GREEN, C_DIM, C_DIM, C_DIM, C_DIM, C_DIM, C_DIM]
        pad_labels = ["the", "cat", "pad", "pad", "pad", "pad", "pad", "pad"]
        pad_row = VGroup()
        for lbl, col in zip(pad_labels, pad_colors):
            cell = VGroup(
                Square(side_length=0.55, fill_color=col, fill_opacity=0.3,
                       stroke_color=col, stroke_width=1.5),
                self.mono(lbl, font_size=11, color=col),
            )
            cell[1].move_to(cell[0])
            pad_row.add(cell)
        pad_row.arrange(RIGHT, buff=0.06).next_to(pad_title, DOWN, buff=0.25)

        waste_label = self.zh(
            "75% 嘥咗！", font_size=18, color=C_RED
        ).next_to(pad_row, RIGHT, buff=0.3)

        # --- Packing example (bottom) ---
        pack_title = self.zh(
            "Packing（打包）", font_size=22, color=C_GREEN
        ).shift(DOWN * 0.3 + LEFT * 4.5)

        pack_colors = [C_GREEN, C_GREEN, C_YELLOW, C_ORANGE, C_ORANGE,
                       C_ORANGE, C_YELLOW, C_CYAN]
        pack_labels = ["the", "cat", "sep", "a", "dog", "runs", "sep", "hi"]
        pack_row = VGroup()
        for lbl, col in zip(pack_labels, pack_colors):
            cell = VGroup(
                Square(side_length=0.55, fill_color=col, fill_opacity=0.3,
                       stroke_color=col, stroke_width=1.5),
                self.mono(lbl, font_size=11, color=col),
            )
            cell[1].move_to(cell[0])
            pack_row.add(cell)
        pack_row.arrange(RIGHT, buff=0.06).next_to(pack_title, DOWN, buff=0.25)

        no_waste = self.zh(
            "0% 浪費！", font_size=18, color=C_GREEN
        ).next_to(pack_row, RIGHT, buff=0.3)

        # Bottom explanation
        explain = VGroup(
            RoundedRectangle(
                corner_radius=0.12, width=10, height=1.0,
                fill_color=C_BLUE, fill_opacity=0.1,
                stroke_color=C_BLUE, stroke_width=1.5,
            ),
            self.zh(
                "Packing 將多個文檔連接，用 <sep> 分隔，然後切成固定長度",
                font_size=18, color=C_CYAN,
            ),
            self.zh(
                "所有 LLM 預訓練都用 Packing，唔會嘥 compute",
                font_size=18, color=C_YELLOW,
            ),
        )
        explain[1:].arrange(DOWN, buff=0.1)
        VGroup(*explain[1:]).move_to(explain[0])
        explain.to_edge(DOWN, buff=0.35)

        with self.voiceover(
            text="Transformer 需要固定長度嘅輸入序列，例如 2048 個 token。"
            "但係文檔嘅長度差異好大。有兩個做法。"
            "第一個係 Padding，即填充。"
            "短嘅文檔後面填滿 pad token，令佢變成固定長度。"
            "但係咁樣好嘥。例如 the cat 呢個短文檔，八個位入面有六個係 pad，"
            "即係百分之七十五嘅計算都嘥咗。"
            "第二個做法係 Packing，即打包。"
            "將所有文檔連接埋一齊，中間用 sep token 分隔。"
            "然後切成固定長度嘅 chunk。"
            "咁樣每一個位都係有用嘅 token，零浪費。"
            "所有 LLM 嘅預訓練都係用 Packing。"
        ):
            self.play(Write(heading), run_time=0.6)

            self.play(FadeIn(pad_title), run_time=0.3)
            for cell in pad_row:
                self.play(FadeIn(cell, shift=RIGHT * 0.1), run_time=0.15)
            self.play(FadeIn(waste_label), run_time=0.4)

            self.play(FadeIn(pack_title), run_time=0.3)
            for cell in pack_row:
                self.play(FadeIn(cell, shift=RIGHT * 0.1), run_time=0.15)
            self.play(FadeIn(no_waste), run_time=0.4)

            self.play(FadeIn(explain, shift=UP * 0.2), run_time=0.6)

        self.wait(0.3)
        self.clear()

    # ── Scene 5 — DataLoader / Batch Construction ────────────────────────

    def scene_dataloader(self):
        heading = self.make_heading("DataLoader 批次構建")

        # Show the pipeline: Sequences → Shuffle → Batch → GPU
        steps = [
            ("Packed\nSequences", C_CYAN),
            ("Shuffle", C_ORANGE),
            ("Batch", C_GREEN),
            ("GPU", C_PINK),
        ]

        step_boxes = VGroup()
        for label, col in steps:
            rect = RoundedRectangle(
                corner_radius=0.15, width=2.2, height=1.0,
                fill_color=col, fill_opacity=0.2,
                stroke_color=col, stroke_width=2,
            )
            txt = self.zh(label, font_size=18, color=col).move_to(rect)
            step_boxes.add(VGroup(rect, txt))

        step_boxes.arrange(RIGHT, buff=0.6).shift(UP * 1.0)

        step_arrows = VGroup()
        for i in range(len(step_boxes) - 1):
            a = Arrow(
                step_boxes[i].get_right(), step_boxes[i + 1].get_left(),
                buff=0.08, color=C_WHITE, stroke_width=2.5,
            )
            step_arrows.add(a)

        # Batch illustration below
        batch_title = self.zh(
            "一個 Batch 嘅形狀", font_size=20, color=C_YELLOW
        ).shift(DOWN * 0.5)

        grid_rows = 4
        grid_cols = 8
        grid = VGroup()
        for r in range(grid_rows):
            row = VGroup()
            for c in range(grid_cols):
                cell = Square(
                    side_length=0.4, fill_color=C_BLUE, fill_opacity=0.15 + r * 0.08,
                    stroke_color=C_BLUE, stroke_width=1,
                )
                row.add(cell)
            row.arrange(RIGHT, buff=0.04)
            grid.add(row)
        grid.arrange(DOWN, buff=0.04).next_to(batch_title, DOWN, buff=0.3)

        brace_b = Brace(grid, LEFT, color=C_ORANGE, buff=0.15)
        brace_b_label = self.zh(
            "batch_size", font_size=16, color=C_ORANGE
        ).next_to(brace_b, LEFT, buff=0.1)

        brace_s = Brace(grid, DOWN, color=C_CYAN, buff=0.15)
        brace_s_label = self.zh(
            "seq_len", font_size=16, color=C_CYAN
        ).next_to(brace_s, DOWN, buff=0.1)

        # Map vs Streaming note
        map_stream = VGroup(
            RoundedRectangle(
                corner_radius=0.12, width=10, height=0.7,
                fill_color=C_PURPLE, fill_opacity=0.1,
                stroke_color=C_PURPLE, stroke_width=1.5,
            ),
            self.zh(
                "Map-style: 隨機存取，容易 shuffle    Streaming: 處理 TB 級數據",
                font_size=17, color=C_PURPLE,
            ),
        )
        map_stream[1].move_to(map_stream[0])
        map_stream.to_edge(DOWN, buff=0.3)

        with self.voiceover(
            text="DataLoader 係數據同 GPU 之間嘅橋樑。"
            "佢將 packed 好嘅序列 shuffle，然後組成 batch，送去 GPU 訓練。"
            "一個 batch 嘅形狀係 batch size 乘以 seq len。"
            "例如 batch size 等於 4，seq len 等於 2048，"
            "咁每次送去 GPU 嘅就係四條長度 2048 嘅 token 序列。"
            "PyTorch 有兩種 Dataset。"
            "Map-style 支援隨機存取，適合中小型數據。"
            "Streaming 即流式讀取，適合 TB 級嘅超大數據集。"
        ):
            self.play(Write(heading), run_time=0.6)

            for i, box in enumerate(step_boxes):
                self.play(FadeIn(box, shift=RIGHT * 0.2), run_time=0.4)
                if i < len(step_arrows):
                    self.play(GrowArrow(step_arrows[i]), run_time=0.25)

            self.play(FadeIn(batch_title), run_time=0.3)
            self.play(FadeIn(grid), run_time=0.6)
            self.play(
                GrowFromCenter(brace_b), FadeIn(brace_b_label),
                GrowFromCenter(brace_s), FadeIn(brace_s_label),
                run_time=0.6,
            )

            self.play(FadeIn(map_stream, shift=UP * 0.2), run_time=0.6)

        self.wait(0.3)
        self.clear()

    # ── Scene 6 — Data Mixing (pie chart) ────────────────────────────────

    def scene_mixing(self):
        heading = self.make_heading("數據混合 Data Mixing")

        slices = [
            ("Web", 0.60, C_ORANGE),
            ("Wikipedia", 0.10, C_GREEN),
            ("Books", 0.10, C_CYAN),
            ("Code", 0.10, C_PINK),
            ("Academic", 0.05, C_PURPLE),
            ("Other", 0.05, C_DIM),
        ]

        pie = VGroup()
        labels = VGroup()
        start_angle = 0

        for name, pct, col in slices:
            angle = pct * TAU
            sector = AnnularSector(
                inner_radius=0,
                outer_radius=1.8,
                angle=angle,
                start_angle=start_angle,
                fill_color=col,
                fill_opacity=0.5,
                stroke_color=col,
                stroke_width=2,
            )
            pie.add(sector)

            mid_angle = start_angle + angle / 2
            label_pos = 2.4 * np.array([
                np.cos(mid_angle), np.sin(mid_angle), 0
            ])
            label = VGroup(
                self.en(name, font_size=16, color=col),
                self.zh(f"{int(pct*100)}%", font_size=14, color=C_WHITE),
            ).arrange(DOWN, buff=0.05).move_to(label_pos)
            labels.add(label)

            start_angle += angle

        chart_group = VGroup(pie, labels).shift(LEFT * 2.5 + DOWN * 0.2)

        # Right side: explanation
        mix_explain = VGroup(
            self.zh("混合比例好重要：", font_size=22, color=C_YELLOW),
            self.zh("太多 web → 生成 SEO 風格", font_size=18, color=C_ORANGE),
            self.zh("太多 code → 亂出 syntax", font_size=18, color=C_PINK),
            self.zh("太少 wiki → 缺少事實知識", font_size=18, color=C_GREEN),
            self.zh("太少 books → 長文唔連貫", font_size=18, color=C_CYAN),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3).shift(RIGHT * 3 + DOWN * 0.2)

        bottom_note = VGroup(
            RoundedRectangle(
                corner_radius=0.12, width=10, height=0.65,
                fill_color=C_YELLOW, fill_opacity=0.1,
                stroke_color=C_YELLOW, stroke_width=1.5,
            ),
            self.zh(
                "混合比例係一個 hyperparameter，需要根據下游任務調整",
                font_size=18, color=C_YELLOW,
            ),
        )
        bottom_note[1].move_to(bottom_note[0])
        bottom_note.to_edge(DOWN, buff=0.3)

        with self.voiceover(
            text="LLM 唔係淨係用一種數據訓練，而係用多種來源嘅混合。"
            "一般嚟講，web 數據佔最多，大約百分之六十。"
            "Wikipedia、Books 同 Code 各佔大約百分之十。"
            "學術論文同其他來源各佔百分之五。"
            "混合比例對模型能力有直接影響。"
            "太多 web 數據，模型會生成低質量嘅 SEO 風格文字。"
            "太多 code，模型會喺對話入面亂出程式語法。"
            "太少 Wikipedia，模型會缺乏事實知識。"
            "太少 Books，模型嘅長篇文章會唔連貫。"
            "所以混合比例係一個重要嘅 hyperparameter，需要根據下游任務嚟調整。"
        ):
            self.play(Write(heading), run_time=0.6)

            for sector, label in zip(pie, labels):
                self.play(
                    FadeIn(sector, shift=OUT * 0.1),
                    FadeIn(label),
                    run_time=0.4,
                )

            for item in mix_explain:
                self.play(FadeIn(item, shift=RIGHT * 0.2), run_time=0.35)

            self.play(FadeIn(bottom_note, shift=UP * 0.2), run_time=0.5)

        self.wait(0.3)
        self.clear()

    # ── Scene 7 — Summary ───────────────────────────────────────────────

    def scene_summary(self):
        heading = self.make_heading("總結", font_size=44)

        bullets = VGroup(
            self.zh("•  數據質量比數量更重要", font_size=26, color=C_GREEN),
            self.zh("•  清洗：去重 + 長度過濾 + 質量過濾", font_size=26, color=C_ORANGE),
            self.zh("•  Packing 消除 padding 浪費", font_size=26, color=C_CYAN),
            self.zh("•  DataLoader 負責 shuffle + batch + 送去 GPU", font_size=26, color=C_PINK),
            self.zh("•  數據混合比例直接影響模型能力", font_size=26, color=C_YELLOW),
            self.zh("•  完整管道：raw → clean → tokenize → pack → train", font_size=26, color=C_PURPLE),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.35).move_to(ORIGIN)

        with self.voiceover(
            text="總結一下今日學咗嘅嘢。"
            "第一，數據質量比數量更加重要，FineWeb 清除咗九成嘅原始數據。"
            "第二，清洗包括去重、長度過濾同質量過濾三個步驟。"
            "第三，Packing 將文檔連接再切割，消除 padding 浪費。"
            "第四，DataLoader 負責 shuffle、組成 batch，然後送去 GPU。"
            "第五，數據混合比例係一個重要嘅 hyperparameter。"
            "第六，完整嘅管道係：原始數據、清洗、tokenize、打包、訓練。"
            "掌握咗呢啲概念，你就識得點樣準備數據嚟訓練自己嘅 LLM 喇。"
        ):
            self.play(Write(heading), run_time=0.6)
            for b in bullets:
                self.play(FadeIn(b, shift=RIGHT * 0.3), run_time=0.5)

        self.wait(0.5)
        self.clear()

    # ── Scene 8 — Outro ─────────────────────────────────────────────────

    def scene_outro(self):
        thanks = self.zh("多謝收睇！", font_size=52, color=C_YELLOW)
        next_lesson = self.zh(
            "下一課：Pre-Training（預訓練）",
            font_size=28, color=C_CYAN,
        ).next_to(thanks, DOWN, buff=0.6)

        with self.voiceover(
            text="多謝收睇！下一課我哋會正式開始預訓練。"
            "包括 next-token prediction、AdamW 優化器、learning rate schedule、"
            "loss curve 分析同 checkpointing。"
            "我哋會真正訓練一個小型嘅語言模型。"
            "記得繼續跟住學喇！"
        ):
            self.play(Write(thanks), run_time=1)
            self.play(FadeIn(next_lesson, shift=UP * 0.2), run_time=0.8)

        self.wait(1)
        self.play(FadeOut(thanks), FadeOut(next_lesson), run_time=0.8)
