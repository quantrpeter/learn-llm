"""
Lesson 16 – Retrieval-Augmented Generation (RAG)
Manim CE + Cantonese voiceover (Edge TTS)

Render (1080p):
    cd "lesson 16/video"
    manim render -qh scene.py RAGExplainerV2

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


class RAGExplainerV2(VoiceoverScene):
    """Eight scenes explaining Retrieval-Augmented Generation in Cantonese."""

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
        self.scene_why_rag()
        self.scene_embeddings()
        self.scene_vector_db()
        self.scene_chunking()
        self.scene_retrieval_pipeline()
        self.scene_summary()
        self.scene_final_outro()

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
        title = self.zh("檢索增強生成", font_size=56, color=C_BLUE)
        sub_en = self.en(
            "Retrieval-Augmented Generation (RAG)", font_size=24, color=C_WHITE
        ).next_to(title, DOWN, buff=0.4)
        sub_zh = self.zh(
            "第十六課", font_size=36, color=C_ORANGE
        ).next_to(sub_en, DOWN, buff=0.3)
        producer = self.zh(
            "制片人：Peter", font_size=24, color=C_DIM
        ).next_to(sub_zh, DOWN, buff=0.7)

        with self.voiceover(
            text="大家好！歡迎嚟到第十六課，亦都係呢個課程嘅最後一課！"
            "今日我哋會學檢索增強生成，英文叫做 RAG。"
            "RAG 係目前業界最常用嘅技術之一。"
            "佢可以令你嘅 LLM 擁有最新嘅知識，"
            "唔再需要靠記憶嚟回答問題。"
            "我哋會學點樣將文字變成向量、"
            "點樣建立向量資料庫、"
            "點樣將文件切割成細塊、"
            "同埋點樣建立完整嘅檢索管道。"
        ):
            self.play(Write(title), run_time=1.5)
            self.play(FadeIn(sub_en, shift=UP * 0.2), run_time=0.8)
            self.play(FadeIn(sub_zh, shift=UP * 0.2), run_time=0.8)
            self.play(FadeIn(producer, shift=UP * 0.2), run_time=0.5)

        self.wait(0.3)
        self.clear()

    # ── Scene 2 — Why RAG (hallucination problem) ────────────────────────

    def scene_why_rag(self):
        heading = self.make_heading("點解需要 RAG？")

        # Without RAG — hallucination example (left)
        no_rag_box = RoundedRectangle(
            corner_radius=0.15, width=5.0, height=3.2,
            fill_color=C_RED, fill_opacity=0.10,
            stroke_color=C_RED, stroke_width=2,
        )
        no_rag_title = self.zh(
            "冇 RAG", font_size=22, color=C_RED
        ).next_to(no_rag_box, UP, buff=0.1)
        no_rag_q = self.zh(
            "問：2025年邊間公司市值最高？", font_size=16, color=C_WHITE
        ).move_to(no_rag_box).shift(UP * 0.8)
        no_rag_a = self.zh(
            "答：根據我嘅訓練數據...\n"
            "   （知識截止於2023年）\n"
            "   可能係 Apple...\n"
            "   [自信但過時嘅答案]",
            font_size=14, color=C_RED,
        ).move_to(no_rag_box).shift(DOWN * 0.3)
        no_rag_label = self.zh(
            "幻覺 Hallucination！", font_size=18, color=C_YELLOW
        ).move_to(no_rag_box).shift(DOWN * 1.2)
        no_rag_group = VGroup(no_rag_box, no_rag_title, no_rag_q, no_rag_a, no_rag_label)

        # With RAG — grounded answer (right)
        rag_box = RoundedRectangle(
            corner_radius=0.15, width=5.0, height=3.2,
            fill_color=C_GREEN, fill_opacity=0.10,
            stroke_color=C_GREEN, stroke_width=2,
        )
        rag_title = self.zh(
            "有 RAG", font_size=22, color=C_GREEN
        ).next_to(rag_box, UP, buff=0.1)
        rag_q = self.zh(
            "問：2025年邊間公司市值最高？", font_size=16, color=C_WHITE
        ).move_to(rag_box).shift(UP * 0.8)
        rag_a = self.zh(
            "1) 檢索最新財經數據\n"
            "2) 找到相關文件\n"
            "3) 基於真實數據回答\n"
            "答：根據檢索到嘅資料...",
            font_size=14, color=C_GREEN,
        ).move_to(rag_box).shift(DOWN * 0.3)
        rag_label = self.zh(
            "有根據嘅答案！", font_size=18, color=C_CYAN
        ).move_to(rag_box).shift(DOWN * 1.2)
        rag_group = VGroup(rag_box, rag_title, rag_q, rag_a, rag_label)

        examples = VGroup(no_rag_group, rag_group).arrange(RIGHT, buff=0.6)
        examples.shift(DOWN * 0.2)

        # Bottom: three problems
        problems = VGroup(
            self.make_box("知識截止", C_RED, width=3.0, height=0.6, font_size=20),
            self.make_box("幻覺生成", C_ORANGE, width=3.0, height=0.6, font_size=20),
            self.make_box("缺乏私有數據", C_PINK, width=3.0, height=0.6, font_size=20),
        ).arrange(RIGHT, buff=0.4).to_edge(DOWN, buff=0.4)

        with self.voiceover(
            text="LLM 有三個大問題。"
            "第一，知識截止。模型嘅知識停留喺訓練嗰個時間點。"
            "問佢最新嘅事情，佢唔知道。"
            "第二，幻覺生成。當模型唔確定嘅時候，"
            "佢唔會話唔知，而係好自信噉作一個答案出嚟。"
            "呢個叫做 hallucination。"
            "第三，缺乏私有數據。你公司嘅內部文件，"
            "根本唔喺任何訓練數據入面。"
            "RAG 可以解決晒呢三個問題。"
            "佢先檢索相關文件，再基於真實數據回答。"
            "噉就唔需要靠記憶，亦唔會亂作。"
        ):
            self.play(Write(heading), run_time=0.6)
            self.play(
                FadeIn(no_rag_box), FadeIn(no_rag_title),
                run_time=0.4,
            )
            self.play(
                FadeIn(no_rag_q, shift=DOWN * 0.1),
                FadeIn(no_rag_a, shift=DOWN * 0.1),
                run_time=0.6,
            )
            self.play(FadeIn(no_rag_label), run_time=0.4)
            self.play(
                FadeIn(rag_box), FadeIn(rag_title),
                run_time=0.4,
            )
            self.play(
                FadeIn(rag_q, shift=DOWN * 0.1),
                FadeIn(rag_a, shift=DOWN * 0.1),
                run_time=0.6,
            )
            self.play(FadeIn(rag_label), run_time=0.4)
            for p in problems:
                self.play(FadeIn(p, shift=UP * 0.2), run_time=0.4)

        self.wait(0.3)
        self.clear()

    # ── Scene 3 — Embeddings (text → vector visualization) ───────────────

    def scene_embeddings(self):
        heading = self.make_heading("文字嵌入 Embeddings")

        # Left: text sentences
        sentences = [
            "Python 係一種程式語言",
            "PyTorch 用嚟做深度學習",
            "貓咪坐喺地氈上面",
        ]
        sent_labels = VGroup()
        for i, s in enumerate(sentences):
            lbl = self.zh(s, font_size=18, color=[C_BLUE, C_GREEN, C_ORANGE][i])
            sent_labels.add(lbl)
        sent_labels.arrange(DOWN, buff=0.5, aligned_edge=LEFT)
        sent_labels.shift(LEFT * 3.5 + UP * 0.5)

        # Middle: arrow showing transformation
        arrow_embed = Arrow(LEFT * 1.2, RIGHT * 0.3, color=C_YELLOW, stroke_width=2.5)
        arrow_embed.shift(UP * 0.5)
        embed_label = self.zh(
            "嵌入模型", font_size=16, color=C_YELLOW
        ).next_to(arrow_embed, UP, buff=0.15)

        # Right: vector visualization (bars)
        vec_groups = VGroup()
        bar_colors = [C_BLUE, C_GREEN, C_ORANGE]
        vec_vals = [
            [0.8, 0.3, 0.9, 0.1, 0.6, 0.4, 0.7, 0.2],
            [0.7, 0.4, 0.8, 0.2, 0.5, 0.5, 0.6, 0.3],
            [0.1, 0.9, 0.2, 0.8, 0.1, 0.3, 0.1, 0.7],
        ]
        for vi, (vals, col) in enumerate(zip(vec_vals, bar_colors)):
            bars = VGroup()
            for j, v in enumerate(vals):
                bar = Rectangle(
                    width=0.2, height=v * 1.2,
                    fill_color=col, fill_opacity=0.6,
                    stroke_color=col, stroke_width=1,
                )
                bar.move_to(RIGHT * (2.0 + j * 0.3) + UP * (v * 0.6))
                bars.add(bar)
            vec_label = self.mono(
                f"[{', '.join(f'{v:.1f}' for v in vals[:4])}...]",
                font_size=12, color=col,
            ).next_to(bars, DOWN, buff=0.15)
            group = VGroup(bars, vec_label)
            vec_groups.add(group)

        vec_groups.arrange(DOWN, buff=0.3, aligned_edge=LEFT)
        vec_groups.shift(RIGHT * 2.5 + UP * 0.5)

        # Similarity annotation
        sim_high = self.mono(
            "cos = 0.95", font_size=16, color=C_CYAN
        ).shift(RIGHT * 5.0 + UP * 1.5)
        sim_low = self.mono(
            "cos = 0.23", font_size=16, color=C_RED
        ).shift(RIGHT * 5.0 + DOWN * 0.5)
        brace_high = Brace(VGroup(vec_groups[0], vec_groups[1]), RIGHT, color=C_CYAN)
        brace_low = Brace(VGroup(vec_groups[1], vec_groups[2]), RIGHT, color=C_RED)

        # Bottom explanation
        explain = self.zh(
            "意思相近 → 向量相近 → cosine similarity 高",
            font_size=20, color=C_YELLOW,
        ).to_edge(DOWN, buff=0.5)

        with self.voiceover(
            text="RAG 嘅第一步係將文字變成向量。"
            "呢個過程叫做 embedding，嵌入。"
            "一個嵌入模型會將任何一段文字，"
            "轉換成一串固定長度嘅數字。"
            "例如呢三句句子。"
            "Python 係一種程式語言，同 PyTorch 用嚟做深度學習，"
            "呢兩句意思相關，所以佢哋嘅向量會好接近。"
            "cosine similarity 接近 0.95。"
            "但係「貓咪坐喺地氈上面」同程式語言完全無關，"
            "所以 cosine similarity 只有 0.23。"
            "呢個就係 RAG 搜索嘅基礎："
            "將問題變成向量，搵出最接近嘅文件向量。"
        ):
            self.play(Write(heading), run_time=0.6)
            for lbl in sent_labels:
                self.play(FadeIn(lbl, shift=RIGHT * 0.2), run_time=0.4)
            self.play(GrowArrow(arrow_embed), FadeIn(embed_label), run_time=0.5)
            for vg in vec_groups:
                bars_in_group = vg[0]
                for bar in bars_in_group:
                    self.play(GrowFromEdge(bar, DOWN), run_time=0.05)
                self.play(FadeIn(vg[1]), run_time=0.2)
            self.play(
                GrowFromCenter(brace_high), FadeIn(sim_high),
                run_time=0.5,
            )
            self.play(
                GrowFromCenter(brace_low), FadeIn(sim_low),
                run_time=0.5,
            )
            self.play(FadeIn(explain, shift=UP * 0.1), run_time=0.5)

        self.wait(0.3)
        self.clear()

    # ── Scene 4 — Vector DB (similarity search with nearest neighbors) ───

    def scene_vector_db(self):
        heading = self.make_heading("向量資料庫 Vector Database")

        # Grid of document dots
        import random
        random.seed(42)
        dots = VGroup()
        dot_positions = []
        labels_group = VGroup()
        doc_names = [
            "RAG 教學", "PyTorch 入門", "向量搜索",
            "食譜大全", "貓咪護理", "旅行指南",
            "深度學習", "NLP 基礎", "數據庫設計",
        ]
        for i in range(9):
            x = random.uniform(-4.5, 4.5)
            y = random.uniform(-1.5, 1.5)
            dot = Dot(point=[x, y, 0], radius=0.12, color=C_DIM)
            dot_label = self.zh(
                doc_names[i], font_size=12, color=C_DIM
            ).next_to(dot, DOWN, buff=0.1)
            dots.add(dot)
            labels_group.add(dot_label)
            dot_positions.append((x, y))

        # Query dot
        query_dot = Dot(point=[0, 0.5, 0], radius=0.15, color=C_YELLOW)
        query_label = self.zh(
            "查詢", font_size=16, color=C_YELLOW
        ).next_to(query_dot, UP, buff=0.15)

        # Nearest neighbors (indices 0, 2, 6 are RAG-related)
        nn_indices = [0, 2, 6]
        nn_circles = VGroup()
        nn_lines = VGroup()
        for idx in nn_indices:
            x, y = dot_positions[idx]
            circle = Circle(radius=0.25, color=C_GREEN, stroke_width=2.5)
            circle.move_to([x, y, 0])
            nn_circles.add(circle)
            line = DashedLine(
                [0, 0.5, 0], [x, y, 0],
                color=C_GREEN, stroke_width=1.5, dash_length=0.1,
            )
            nn_lines.add(line)

        # Search radius circle
        radius_circle = Circle(
            radius=2.5, color=C_CYAN, stroke_width=1.5, stroke_opacity=0.4,
        ).move_to([0, 0.5, 0])

        # Bottom box
        result_box = RoundedRectangle(
            corner_radius=0.12, width=10, height=0.8,
            fill_color=C_GREEN, fill_opacity=0.12,
            stroke_color=C_GREEN, stroke_width=1.5,
        ).to_edge(DOWN, buff=0.35)
        result_text = self.zh(
            "搜索結果：RAG 教學 (0.95)  向量搜索 (0.91)  深度學習 (0.87)",
            font_size=18, color=C_GREEN,
        ).move_to(result_box)

        with self.voiceover(
            text="有咗向量之後，我哋需要一個地方儲存同搜索佢哋。"
            "呢個就係向量資料庫。"
            "想像每個文件都係空間中嘅一個點。"
            "意思相近嘅文件，喺空間中會比較接近。"
            "當用戶輸入一個查詢，"
            "我哋將佢變成向量，"
            "然後搵出距離最近嘅幾個點。"
            "呢個就係 nearest neighbor 搜索。"
            "生產環境會用 FAISS 或者 ChromaDB。"
            "佢哋用 HNSW 算法，"
            "可以喺幾十億個向量入面，"
            "用 log n 嘅時間搵到最近嘅結果。"
        ):
            self.play(Write(heading), run_time=0.6)
            self.play(
                *[FadeIn(d, scale=0.5) for d in dots],
                *[FadeIn(l) for l in labels_group],
                run_time=0.8,
            )
            self.play(
                FadeIn(query_dot, scale=0.5),
                FadeIn(query_label),
                run_time=0.5,
            )
            self.play(Create(radius_circle), run_time=0.6)
            for circle, line in zip(nn_circles, nn_lines):
                self.play(Create(line), Create(circle), run_time=0.4)
            # Highlight nearest docs
            for idx in nn_indices:
                dots[idx].set_color(C_GREEN)
                labels_group[idx].set_color(C_GREEN)
            self.play(
                *[dots[i].animate.set_color(C_GREEN) for i in nn_indices],
                run_time=0.3,
            )
            self.play(FadeIn(result_box), FadeIn(result_text), run_time=0.5)

        self.wait(0.3)
        self.clear()

    # ── Scene 5 — Chunking (document splitting animation) ────────────────

    def scene_chunking(self):
        heading = self.make_heading("文件切割 Chunking")

        # Original document (long rectangle)
        doc_rect = RoundedRectangle(
            corner_radius=0.1, width=10, height=1.0,
            fill_color=C_BLUE, fill_opacity=0.15,
            stroke_color=C_BLUE, stroke_width=2,
        ).shift(UP * 2.0)
        doc_label = self.zh(
            "原始文件（太長，唔適合直接 embed）",
            font_size=18, color=C_BLUE,
        ).move_to(doc_rect)

        # Fixed-size chunks
        fixed_label = self.zh(
            "固定大小切割", font_size=18, color=C_ORANGE
        ).shift(UP * 0.6 + LEFT * 4.5)
        fixed_chunks = VGroup()
        for i in range(5):
            chunk = RoundedRectangle(
                corner_radius=0.08, width=1.8, height=0.5,
                fill_color=C_ORANGE, fill_opacity=0.2,
                stroke_color=C_ORANGE, stroke_width=1.5,
            )
            chunk_txt = self.mono(f"chunk {i}", font_size=12, color=C_ORANGE).move_to(chunk)
            fixed_chunks.add(VGroup(chunk, chunk_txt))
        fixed_chunks.arrange(RIGHT, buff=0.15)
        fixed_chunks.next_to(fixed_label, DOWN, buff=0.2)

        # Sentence-based chunks
        sent_label = self.zh(
            "句子切割", font_size=18, color=C_GREEN
        ).shift(DOWN * 0.7 + LEFT * 4.5)
        sent_chunks = VGroup()
        sent_widths = [2.5, 1.8, 3.2, 2.0]
        for i, w in enumerate(sent_widths):
            chunk = RoundedRectangle(
                corner_radius=0.08, width=w, height=0.5,
                fill_color=C_GREEN, fill_opacity=0.2,
                stroke_color=C_GREEN, stroke_width=1.5,
            )
            chunk_txt = self.mono(f"sent {i}", font_size=12, color=C_GREEN).move_to(chunk)
            sent_chunks.add(VGroup(chunk, chunk_txt))
        sent_chunks.arrange(RIGHT, buff=0.15)
        sent_chunks.next_to(sent_label, DOWN, buff=0.2)

        # Recursive chunks
        rec_label = self.zh(
            "遞歸切割（最佳）", font_size=18, color=C_CYAN
        ).shift(DOWN * 2.0 + LEFT * 4.5)
        rec_chunks = VGroup()
        rec_widths = [3.0, 2.5, 2.8, 1.5]
        for i, w in enumerate(rec_widths):
            chunk = RoundedRectangle(
                corner_radius=0.08, width=w, height=0.5,
                fill_color=C_CYAN, fill_opacity=0.2,
                stroke_color=C_CYAN, stroke_width=1.5,
            )
            chunk_txt = self.mono(f"para {i}", font_size=12, color=C_CYAN).move_to(chunk)
            rec_chunks.add(VGroup(chunk, chunk_txt))
        rec_chunks.arrange(RIGHT, buff=0.15)
        rec_chunks.next_to(rec_label, DOWN, buff=0.2)

        # Scissors icon (text-based)
        scissors = self.zh("✂", font_size=28, color=C_YELLOW)

        with self.voiceover(
            text="真實嘅文件通常好長，唔可以直接變成一個向量。"
            "我哋需要將文件切割成細塊，英文叫 chunking。"
            "有三種主要嘅切割方法。"
            "第一種係固定大小切割。"
            "每隔一定數量嘅字符就切一刀。"
            "好處係簡單，壞處係可能切斷句子。"
            "第二種係句子切割。"
            "喺句號嘅位置切割，保持每個句子完整。"
            "但每塊嘅大小會唔均勻。"
            "第三種係遞歸切割，呢個係最好嘅方法。"
            "先嘗試喺段落邊界切，"
            "如果太長就喺句子邊界切，"
            "最後先喺字詞邊界切。"
            "LangChain 用嘅就係呢個方法。"
        ):
            self.play(Write(heading), run_time=0.6)
            self.play(FadeIn(doc_rect), FadeIn(doc_label), run_time=0.5)

            scissors.move_to(doc_rect.get_left() + RIGHT * 2)
            self.play(FadeIn(scissors), run_time=0.2)
            self.play(
                scissors.animate.move_to(doc_rect.get_right() + LEFT * 0.5),
                run_time=0.8,
            )
            self.play(FadeOut(scissors), run_time=0.2)

            self.play(FadeIn(fixed_label), run_time=0.3)
            for chunk in fixed_chunks:
                self.play(FadeIn(chunk, shift=DOWN * 0.1), run_time=0.15)

            self.play(FadeIn(sent_label), run_time=0.3)
            for chunk in sent_chunks:
                self.play(FadeIn(chunk, shift=DOWN * 0.1), run_time=0.15)

            self.play(FadeIn(rec_label), run_time=0.3)
            for chunk in rec_chunks:
                self.play(FadeIn(chunk, shift=DOWN * 0.1), run_time=0.15)

        self.wait(0.3)
        self.clear()

    # ── Scene 6 — Retrieval Pipeline (full flow with boxes + arrows) ─────

    def scene_retrieval_pipeline(self):
        heading = self.make_heading("完整檢索管道")

        # Stage boxes arranged top-to-bottom as a flow
        query_box = self.make_box("用戶查詢", C_YELLOW, width=2.8, height=0.7, font_size=20)
        embed_box = self.make_box("嵌入模型", C_BLUE, width=2.8, height=0.7, font_size=20)
        search_box = self.make_box("向量搜索", C_GREEN, width=2.8, height=0.7, font_size=20)
        rerank_box = self.make_box("重新排序", C_ORANGE, width=2.8, height=0.7, font_size=20)
        prompt_box = self.make_box("構建提示", C_PINK, width=2.8, height=0.7, font_size=20)
        llm_box = self.make_box("LLM 生成", C_CYAN, width=2.8, height=0.7, font_size=20)

        pipeline_stages = VGroup(
            query_box, embed_box, search_box, rerank_box, prompt_box, llm_box
        ).arrange(DOWN, buff=0.35)
        pipeline_stages.shift(LEFT * 2.5 + DOWN * 0.1)

        # Arrows between stages
        arrows = VGroup()
        stage_list = [query_box, embed_box, search_box, rerank_box, prompt_box, llm_box]
        for i in range(len(stage_list) - 1):
            arr = Arrow(
                stage_list[i][0].get_bottom(),
                stage_list[i + 1][0].get_top(),
                buff=0.1, color=C_DIM, stroke_width=2,
            )
            arrows.add(arr)

        # Right side: vector DB feeding into search
        vdb_box = self.make_box("向量資料庫", C_PURPLE, width=2.8, height=0.7, font_size=20)
        vdb_box.move_to(search_box).shift(RIGHT * 5)
        vdb_arrow = Arrow(
            vdb_box[0].get_left(), search_box[0].get_right(),
            buff=0.15, color=C_PURPLE, stroke_width=2,
        )

        # Right side: cross-encoder feeding into rerank
        ce_box = self.make_mono_box("Cross-Encoder", C_ORANGE, width=2.8, height=0.7, font_size=18)
        ce_box.move_to(rerank_box).shift(RIGHT * 5)
        ce_arrow = Arrow(
            ce_box[0].get_left(), rerank_box[0].get_right(),
            buff=0.15, color=C_ORANGE, stroke_width=2,
        )

        # Right side: context docs feeding into prompt
        docs_box = self.make_box("檢索到嘅文件", C_PINK, width=2.8, height=0.7, font_size=18)
        docs_box.move_to(prompt_box).shift(RIGHT * 5)
        docs_arrow = Arrow(
            docs_box[0].get_left(), prompt_box[0].get_right(),
            buff=0.15, color=C_PINK, stroke_width=2,
        )

        # Bottom: output
        output_box = RoundedRectangle(
            corner_radius=0.12, width=10, height=0.7,
            fill_color=C_CYAN, fill_opacity=0.12,
            stroke_color=C_CYAN, stroke_width=1.5,
        ).to_edge(DOWN, buff=0.3)
        output_text = self.zh(
            "輸出：基於真實文件嘅準確答案",
            font_size=20, color=C_CYAN,
        ).move_to(output_box)

        with self.voiceover(
            text="而家我哋將所有嘢組合成一個完整嘅管道。"
            "第一步，用戶輸入一個查詢。"
            "第二步，用嵌入模型將查詢變成向量。"
            "第三步，用呢個向量喺向量資料庫入面搜索。"
            "搵出最相似嘅文件。呢個用 bi-encoder 做，好快。"
            "第四步係重新排序。"
            "用 cross-encoder 對每對查詢同文件重新評分。"
            "Cross-encoder 可以同時睇到查詢同文件嘅所有 token，"
            "所以準確度比 bi-encoder 高好多。"
            "第五步，將最相關嘅文件放入提示入面。"
            "構建一個包含上下文嘅完整提示。"
            "最後一步，將提示送入 LLM 生成答案。"
            "呢個答案係基於真實文件嘅，唔係靠記憶。"
        ):
            self.play(Write(heading), run_time=0.6)
            for i, stage in enumerate(stage_list):
                self.play(FadeIn(stage, shift=DOWN * 0.15), run_time=0.3)
                if i < len(arrows):
                    self.play(GrowArrow(arrows[i]), run_time=0.2)

            self.play(FadeIn(vdb_box, shift=LEFT * 0.2), run_time=0.3)
            self.play(GrowArrow(vdb_arrow), run_time=0.3)

            self.play(FadeIn(ce_box, shift=LEFT * 0.2), run_time=0.3)
            self.play(GrowArrow(ce_arrow), run_time=0.3)

            self.play(FadeIn(docs_box, shift=LEFT * 0.2), run_time=0.3)
            self.play(GrowArrow(docs_arrow), run_time=0.3)

            self.play(FadeIn(output_box), FadeIn(output_text), run_time=0.5)

        self.wait(0.3)
        self.clear()

    # ── Scene 7 — Summary ────────────────────────────────────────────────

    def scene_summary(self):
        heading = self.make_heading("總結", font_size=44)

        bullets = VGroup(
            self.zh("•  RAG 解決幻覺：檢索真實文件，唔靠記憶", font_size=24, color=C_GREEN),
            self.zh("•  嵌入模型：將文字變成向量，cosine 度量相似度", font_size=24, color=C_BLUE),
            self.zh("•  向量資料庫：儲存同搜索高維向量（FAISS/ChromaDB）", font_size=24, color=C_PURPLE),
            self.zh("•  切割策略：遞歸切割最好，保持語義完整性", font_size=24, color=C_ORANGE),
            self.zh("•  重新排序：Cross-encoder 提高 top-k 精準度", font_size=24, color=C_PINK),
            self.zh("•  上下文管理：控制 token 預算，注意「中間遺失」", font_size=24, color=C_CYAN),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.35).move_to(ORIGIN)

        with self.voiceover(
            text="總結一下今日學咗嘅重點。"
            "第一，RAG 通過檢索真實文件嚟解決幻覺問題。"
            "第二，嵌入模型將文字變成向量，用 cosine similarity 度量相似度。"
            "第三，向量資料庫好似 FAISS 同 ChromaDB，可以高效噉搜索向量。"
            "第四，文件切割策略好重要，遞歸切割效果最好。"
            "第五，重新排序用 cross-encoder 可以大幅提高搜索精準度。"
            "第六，上下文管理要注意 token 預算，"
            "同埋「中間遺失」嘅問題。"
            "掌握咗呢啲，你就可以建立自己嘅 RAG 系統喇！"
        ):
            self.play(Write(heading), run_time=0.6)
            for b in bullets:
                self.play(FadeIn(b, shift=RIGHT * 0.3), run_time=0.5)

        self.wait(0.5)
        self.clear()

    # ── Scene 8 — Final Outro (Course Complete!) ─────────────────────────

    def scene_final_outro(self):
        congrats = self.zh("課程完結！", font_size=56, color=C_YELLOW)
        thanks = self.zh(
            "多謝收睇！", font_size=48, color=C_CYAN
        ).next_to(congrats, DOWN, buff=0.5)

        # Course journey summary
        journey_items = VGroup(
            self.zh("數學基礎 → PyTorch → 神經網絡", font_size=20, color=C_DIM),
            self.zh("Tokenization → Attention → Transformer", font_size=20, color=C_DIM),
            self.zh("預訓練 → 微調 → 對齊 → 量化 → 部署 → RAG", font_size=20, color=C_DIM),
        ).arrange(DOWN, buff=0.25).next_to(thanks, DOWN, buff=0.6)

        final_msg = self.zh(
            "你已經學識從零開始建立 LLM 嘅所有技術！",
            font_size=24, color=C_GREEN,
        ).next_to(journey_items, DOWN, buff=0.6)

        with self.voiceover(
            text="恭喜你！你已經完成咗成個課程！"
            "由數學基礎開始，"
            "到 PyTorch、神經網絡、"
            "Tokenization、Attention、Transformer 架構。"
            "再到預訓練、微調、對齊、量化、部署，"
            "最後到今日嘅 RAG。"
            "你已經學識咗從零開始建立大型語言模型嘅所有核心技術。"
            "多謝你一路跟住學！"
            "希望呢個課程對你有幫助。"
            "祝你喺 AI 嘅旅程上一帆風順！"
        ):
            self.play(Write(congrats), run_time=1.2)
            self.play(Write(thanks), run_time=1.0)
            for item in journey_items:
                self.play(FadeIn(item, shift=RIGHT * 0.2), run_time=0.5)
            self.play(FadeIn(final_msg, shift=UP * 0.2), run_time=0.8)

        self.wait(1.5)
        self.play(
            *[FadeOut(m) for m in [congrats, thanks, journey_items, final_msg]],
            run_time=1.0,
        )
