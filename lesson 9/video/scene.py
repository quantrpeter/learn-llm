"""
Lesson 9 – Distributed Training & Scaling
Manim CE + Cantonese voiceover (Edge TTS)

Render (1080p):
    cd "lesson 9/video"
    manim render -qh scene.py DistributedTrainingExplainer

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


class DistributedTrainingExplainer(VoiceoverScene):
    """Single scene explaining distributed training in Cantonese."""

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
        self.scene_ddp()
        self.scene_fsdp()
        self.scene_tensor_parallel()
        self.scene_mixed_precision()
        self.scene_flash_attention()
        self.scene_scaling_laws()
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
        title = self.zh("分佈式訓練", font_size=56, color=C_BLUE)
        sub_en = self.en(
            "Distributed Training & Scaling", font_size=24, color=C_WHITE
        ).next_to(title, DOWN, buff=0.4)
        sub_zh = self.zh(
            "第九課", font_size=36, color=C_ORANGE
        ).next_to(sub_en, DOWN, buff=0.3)
        producer = self.zh(
            "制片人：Peter", font_size=24, color=C_DIM
        ).next_to(sub_zh, DOWN, buff=0.7)

        with self.voiceover(
            text="大家好！歡迎嚟到第九課。"
            "上幾課我哋學咗點樣建造同訓練一個 LLM。"
            "但係當模型太大，一張 GPU 放唔落嘅時候，點算呢？"
            "今日我哋會學分佈式訓練同 scaling 嘅技術。"
            "包括 DDP、FSDP、Tensor Parallelism、Mixed Precision、"
            "Flash Attention 同 Chinchilla Scaling Laws。"
        ):
            self.play(Write(title), run_time=1.5)
            self.play(FadeIn(sub_en, shift=UP * 0.2), run_time=0.8)
            self.play(FadeIn(sub_zh, shift=UP * 0.2), run_time=0.8)
            self.play(FadeIn(producer, shift=UP * 0.2), run_time=0.5)

        self.wait(0.3)
        self.clear()

    # ── Scene 2 — DDP (4 GPU boxes + all-reduce arrows) ──────────────────

    def scene_ddp(self):
        heading = self.make_heading("Data Parallelism (DDP)")

        gpu_colors = [C_GREEN, C_ORANGE, C_PINK, C_PURPLE]
        gpu_boxes = VGroup()
        model_labels = VGroup()
        for i in range(4):
            box = RoundedRectangle(
                corner_radius=0.15, width=2.2, height=1.8,
                fill_color=gpu_colors[i], fill_opacity=0.12,
                stroke_color=gpu_colors[i], stroke_width=2,
            )
            title = self.mono(
                f"GPU {i}", font_size=20, color=gpu_colors[i]
            ).next_to(box, UP, buff=0.1)
            model_lbl = self.zh(
                "完整模型", font_size=16, color=C_WHITE
            ).move_to(box).shift(UP * 0.3)
            data_lbl = self.zh(
                f"數據 {i+1}/4", font_size=16, color=C_DIM
            ).move_to(box).shift(DOWN * 0.3)
            gpu_boxes.add(VGroup(box, title, model_lbl, data_lbl))
            model_labels.add(model_lbl)

        gpu_boxes.arrange(RIGHT, buff=0.5).shift(UP * 0.2)

        # All-reduce arrows (ring pattern between adjacent GPUs)
        ar_arrows = VGroup()
        for i in range(3):
            left_box = gpu_boxes[i][0]
            right_box = gpu_boxes[i + 1][0]
            fwd = Arrow(
                left_box.get_right() + UP * 0.15,
                right_box.get_left() + UP * 0.15,
                buff=0.1, color=C_YELLOW, stroke_width=2.5,
            )
            bwd = Arrow(
                right_box.get_left() + DOWN * 0.15,
                left_box.get_right() + DOWN * 0.15,
                buff=0.1, color=C_YELLOW, stroke_width=2.5,
            )
            ar_arrows.add(fwd, bwd)

        ar_label = self.zh(
            "All-Reduce: 同步梯度", font_size=20, color=C_YELLOW
        ).next_to(ar_arrows, DOWN, buff=0.35)

        # Steps at bottom
        steps = VGroup(
            self.zh("1. 複製模型到每張 GPU", font_size=20, color=C_GREEN),
            self.zh("2. 分割數據 — 每張 GPU 處理 1/4", font_size=20, color=C_ORANGE),
            self.zh("3. 各自計算梯度 → All-Reduce 平均", font_size=20, color=C_YELLOW),
            self.zh("4. 所有 GPU 用相同梯度更新", font_size=20, color=C_CYAN),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.18).to_edge(DOWN, buff=0.4)

        with self.voiceover(
            text="DDP 係最簡單嘅多 GPU 訓練方法。"
            "每張 GPU 都有一份完整嘅模型副本。"
            "訓練數據就分成四份，每張 GPU 處理其中一份。"
            "每張 GPU 獨立做 forward 同 backward。"
            "然後通過 All-Reduce 操作，"
            "將所有 GPU 嘅梯度平均。"
            "呢個過程用 ring all-reduce，效率好高。"
            "平均完之後，每張 GPU 都用同一組梯度更新權重。"
            "咁所有 GPU 嘅模型就永遠保持一致。"
            "DDP 嘅前提係：模型要放得落一張 GPU 嘅記憶體。"
        ):
            self.play(Write(heading), run_time=0.6)
            for gb in gpu_boxes:
                self.play(FadeIn(gb, shift=DOWN * 0.2), run_time=0.4)
            for a in ar_arrows:
                self.play(GrowArrow(a), run_time=0.2)
            self.play(FadeIn(ar_label, shift=UP * 0.1), run_time=0.4)
            for s in steps:
                self.play(FadeIn(s, shift=RIGHT * 0.2), run_time=0.35)

        self.wait(0.3)
        self.clear()

    # ── Scene 3 — FSDP (sharded memory bars) ─────────────────────────────

    def scene_fsdp(self):
        heading = self.make_heading("FSDP 全分片數據並行")

        bar_width = 1.4
        section_colors = [C_GREEN, C_ORANGE, C_RED]
        section_labels = ["參數", "梯度", "優化器"]
        section_heights = [1.0, 1.0, 2.0]

        # --- Left: DDP (full model on each GPU) ---
        ddp_title = self.zh("DDP (每張 GPU)", font_size=20, color=C_DIM)
        ddp_bars = VGroup()
        for i in range(4):
            sections = VGroup()
            y_offset = 0
            for h, c, lbl in zip(section_heights, section_colors, section_labels):
                rect = Rectangle(
                    width=bar_width, height=h,
                    fill_color=c, fill_opacity=0.4,
                    stroke_color=c, stroke_width=1.5,
                )
                txt = self.zh(lbl, font_size=12, color=c).move_to(rect)
                sections.add(VGroup(rect, txt))
            sections.arrange(UP, buff=0.04)
            gpu_label = self.mono(f"GPU{i}", font_size=14, color=C_DIM)
            gpu_label.next_to(sections, DOWN, buff=0.1)
            ddp_bars.add(VGroup(sections, gpu_label))
        ddp_bars.arrange(RIGHT, buff=0.15)
        ddp_title.next_to(ddp_bars, UP, buff=0.2)
        ddp_group = VGroup(ddp_title, ddp_bars).shift(LEFT * 3.8)

        ddp_mem = self.zh("112 GB/GPU", font_size=18, color=C_RED)
        ddp_mem.next_to(ddp_bars, DOWN, buff=0.3)

        # --- Right: FSDP (sharded across GPUs) ---
        fsdp_title = self.zh("FSDP (分片)", font_size=20, color=C_CYAN)
        fsdp_bars = VGroup()
        for i in range(4):
            sections = VGroup()
            for h, c, lbl in zip(section_heights, section_colors, section_labels):
                rect = Rectangle(
                    width=bar_width, height=h / 4,
                    fill_color=c, fill_opacity=0.4,
                    stroke_color=c, stroke_width=1.5,
                )
                txt = self.zh(f"{lbl}\n1/4", font_size=10, color=c).move_to(rect)
                sections.add(VGroup(rect, txt))
            sections.arrange(UP, buff=0.04)
            gpu_label = self.mono(f"GPU{i}", font_size=14, color=C_DIM)
            gpu_label.next_to(sections, DOWN, buff=0.1)
            fsdp_bars.add(VGroup(sections, gpu_label))
        fsdp_bars.arrange(RIGHT, buff=0.15)
        fsdp_title.next_to(fsdp_bars, UP, buff=0.2)
        fsdp_group = VGroup(fsdp_title, fsdp_bars).shift(RIGHT * 3.8)

        fsdp_mem = self.zh("28 GB/GPU", font_size=18, color=C_GREEN)
        fsdp_mem.next_to(fsdp_bars, DOWN, buff=0.3)

        # Center arrow
        center_arrow = Arrow(
            LEFT * 1.0, RIGHT * 1.0,
            buff=0, color=C_YELLOW, stroke_width=3,
        ).shift(DOWN * 0.5)
        savings = self.zh("4x 節省", font_size=22, color=C_YELLOW)
        savings.next_to(center_arrow, UP, buff=0.15)

        # Bottom note
        note = VGroup(
            RoundedRectangle(
                corner_radius=0.12, width=10, height=0.65,
                fill_color=C_PURPLE, fill_opacity=0.12,
                stroke_color=C_PURPLE, stroke_width=1.5,
            ),
            self.zh(
                "7B 模型 DDP 要 112 GB → FSDP 只要 28 GB/GPU",
                font_size=20, color=C_PURPLE,
            ),
        )
        note[1].move_to(note[0])
        note.to_edge(DOWN, buff=0.3)

        with self.voiceover(
            text="當模型大到一張 GPU 放唔落，就需要 FSDP。"
            "FSDP 嘅全稱係 Fully Sharded Data Parallelism。"
            "先睇 DDP 嘅問題。每張 GPU 都儲存完整嘅參數、梯度同優化器狀態。"
            "一個 7B 模型，用 FP32 嘅話，"
            "參數要 28 GB，梯度要 28 GB，"
            "Adam 優化器要 56 GB。加埋每張 GPU 要 112 GB，"
            "遠超一張 A100 嘅 80 GB。"
            "FSDP 嘅做法係將呢三樣嘢全部分片。"
            "4 張 GPU，每張只儲存四分之一。"
            "需要某層嘅參數嗰陣，先從其他 GPU 收集返嚟。"
            "咁每張 GPU 只需要 28 GB，輕鬆放落 80 GB 嘅 A100。"
        ):
            self.play(Write(heading), run_time=0.6)
            self.play(FadeIn(ddp_title), run_time=0.3)
            for bar in ddp_bars:
                self.play(FadeIn(bar, shift=UP * 0.15), run_time=0.3)
            self.play(FadeIn(ddp_mem), run_time=0.3)

            self.play(
                GrowArrow(center_arrow), FadeIn(savings),
                run_time=0.5,
            )

            self.play(FadeIn(fsdp_title), run_time=0.3)
            for bar in fsdp_bars:
                self.play(FadeIn(bar, shift=UP * 0.15), run_time=0.3)
            self.play(FadeIn(fsdp_mem), run_time=0.3)

            self.play(FadeIn(note, shift=UP * 0.2), run_time=0.6)

        self.wait(0.3)
        self.clear()

    # ── Scene 4 — Tensor Parallelism (split layer) ───────────────────────

    def scene_tensor_parallel(self):
        heading = self.make_heading("Tensor Parallelism")

        # Full layer at top
        full_rect = RoundedRectangle(
            corner_radius=0.15, width=8, height=1.2,
            fill_color=C_BLUE, fill_opacity=0.15,
            stroke_color=C_BLUE, stroke_width=2,
        ).shift(UP * 1.2)
        full_label = self.mono(
            "Linear(4096, 4096)", font_size=22, color=C_BLUE
        ).move_to(full_rect).shift(UP * 0.15)
        full_shape = self.mono(
            "Weight: (4096 x 4096) = 64 MB", font_size=16, color=C_DIM
        ).move_to(full_rect).shift(DOWN * 0.2)

        # Split into 2 halves
        split_label = self.zh(
            "Column Parallel: 分割輸出維度", font_size=20, color=C_YELLOW
        ).shift(DOWN * 0.1)

        gpu0_rect = RoundedRectangle(
            corner_radius=0.15, width=3.8, height=1.2,
            fill_color=C_GREEN, fill_opacity=0.15,
            stroke_color=C_GREEN, stroke_width=2,
        )
        gpu0_label = self.mono(
            "GPU 0", font_size=18, color=C_GREEN
        ).next_to(gpu0_rect, UP, buff=0.1)
        gpu0_shape = self.mono(
            "Linear(4096, 2048)\n32 MB", font_size=16, color=C_GREEN
        ).move_to(gpu0_rect)

        gpu1_rect = RoundedRectangle(
            corner_radius=0.15, width=3.8, height=1.2,
            fill_color=C_ORANGE, fill_opacity=0.15,
            stroke_color=C_ORANGE, stroke_width=2,
        )
        gpu1_label = self.mono(
            "GPU 1", font_size=18, color=C_ORANGE
        ).next_to(gpu1_rect, UP, buff=0.1)
        gpu1_shape = self.mono(
            "Linear(4096, 2048)\n32 MB", font_size=16, color=C_ORANGE
        ).move_to(gpu1_rect)

        gpu0_group = VGroup(gpu0_rect, gpu0_label, gpu0_shape)
        gpu1_group = VGroup(gpu1_rect, gpu1_label, gpu1_shape)
        gpu_row = VGroup(gpu0_group, gpu1_group).arrange(RIGHT, buff=0.4)
        gpu_row.shift(DOWN * 1.5)

        # Split arrows
        arrow_l = Arrow(
            full_rect.get_bottom() + LEFT * 1.5,
            gpu0_rect.get_top(),
            buff=0.15, color=C_DIM, stroke_width=2,
        )
        arrow_r = Arrow(
            full_rect.get_bottom() + RIGHT * 1.5,
            gpu1_rect.get_top(),
            buff=0.15, color=C_DIM, stroke_width=2,
        )

        # Bottom note
        note = VGroup(
            RoundedRectangle(
                corner_radius=0.12, width=10.5, height=0.65,
                fill_color=C_CYAN, fill_opacity=0.12,
                stroke_color=C_CYAN, stroke_width=1.5,
            ),
            self.zh(
                "結果拼接 (concat) 就等同完整層嘅輸出",
                font_size=20, color=C_CYAN,
            ),
        )
        note[1].move_to(note[0])
        note.to_edge(DOWN, buff=0.3)

        with self.voiceover(
            text="Tensor Parallelism 係將單獨嘅層分割到唔同嘅 GPU。"
            "例如一個 Linear 4096 乘 4096 嘅層，"
            "權重矩陣有 64 MB。"
            "用 column parallel 嘅方式，"
            "我哋將輸出維度分成兩半。"
            "GPU 0 負責前 2048 個輸出維度，"
            "GPU 1 負責後 2048 個輸出維度。"
            "每張 GPU 只需要儲存一半嘅權重，即係 32 MB。"
            "最後將兩張 GPU 嘅輸出拼接埋一齊，"
            "結果同完整層嘅輸出完全一樣。"
            "Tensor Parallelism 通常喺同一個節點入面用，"
            "因為 GPU 之間嘅 NVLink 連接好快。"
        ):
            self.play(Write(heading), run_time=0.6)
            self.play(
                FadeIn(full_rect), FadeIn(full_label), FadeIn(full_shape),
                run_time=0.6,
            )
            self.play(FadeIn(split_label), run_time=0.4)
            self.play(GrowArrow(arrow_l), GrowArrow(arrow_r), run_time=0.5)
            self.play(
                FadeIn(gpu0_group, shift=DOWN * 0.2),
                FadeIn(gpu1_group, shift=DOWN * 0.2),
                run_time=0.6,
            )
            self.play(FadeIn(note, shift=UP * 0.2), run_time=0.6)

        self.wait(0.3)
        self.clear()

    # ── Scene 5 — Mixed Precision (FP32 vs FP16 bars) ────────────────────

    def scene_mixed_precision(self):
        heading = self.make_heading("混合精度訓練 Mixed Precision")

        # FP32 tall bar
        fp32_bar = Rectangle(
            width=2.5, height=4.0,
            fill_color=C_RED, fill_opacity=0.3,
            stroke_color=C_RED, stroke_width=2,
        )
        fp32_label = self.mono("FP32", font_size=28, color=C_RED)
        fp32_label.next_to(fp32_bar, UP, buff=0.15)
        fp32_bytes = self.mono("4 bytes", font_size=18, color=C_RED)
        fp32_bytes.move_to(fp32_bar).shift(UP * 0.5)
        fp32_mem = self.mono("28 GB", font_size=20, color=C_RED)
        fp32_mem.move_to(fp32_bar).shift(DOWN * 0.3)
        fp32_note = self.zh("(7B 模型)", font_size=16, color=C_DIM)
        fp32_note.next_to(fp32_bar, DOWN, buff=0.15)
        fp32_group = VGroup(fp32_bar, fp32_label, fp32_bytes, fp32_mem, fp32_note)

        # BF16 half bar
        bf16_bar = Rectangle(
            width=2.5, height=2.0,
            fill_color=C_GREEN, fill_opacity=0.3,
            stroke_color=C_GREEN, stroke_width=2,
        )
        bf16_label = self.mono("BF16", font_size=28, color=C_GREEN)
        bf16_label.next_to(bf16_bar, UP, buff=0.15)
        bf16_bytes = self.mono("2 bytes", font_size=18, color=C_GREEN)
        bf16_bytes.move_to(bf16_bar).shift(UP * 0.2)
        bf16_mem = self.mono("14 GB", font_size=20, color=C_GREEN)
        bf16_mem.move_to(bf16_bar).shift(DOWN * 0.2)
        bf16_note = self.zh("(7B 模型)", font_size=16, color=C_DIM)
        bf16_note.next_to(bf16_bar, DOWN, buff=0.15)
        bf16_group = VGroup(bf16_bar, bf16_label, bf16_bytes, bf16_mem, bf16_note)

        bars = VGroup(fp32_group, bf16_group).arrange(RIGHT, buff=2.5, aligned_edge=DOWN)
        bars.shift(DOWN * 0.2)

        # Savings arrow
        savings_arrow = Arrow(
            fp32_bar.get_right() + RIGHT * 0.2,
            bf16_bar.get_left() + LEFT * 0.2,
            buff=0.1, color=C_YELLOW, stroke_width=3,
        ).shift(UP * 0.5)
        savings_text = self.zh("2x 節省", font_size=24, color=C_YELLOW)
        savings_text.next_to(savings_arrow, UP, buff=0.15)

        # Advantages
        advantages = VGroup(
            self.zh("BF16 優勢：", font_size=20, color=C_CYAN),
            self.zh("• 記憶體減半", font_size=18, color=C_GREEN),
            self.zh("• GPU 計算速度 2x", font_size=18, color=C_GREEN),
            self.zh("• 同 FP32 一樣嘅數值範圍", font_size=18, color=C_GREEN),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.15)
        advantages.to_edge(DOWN, buff=0.5).shift(RIGHT * 2)

        with self.voiceover(
            text="混合精度訓練係 LLM 訓練嘅標準做法。"
            "FP32 用 4 個 bytes，一個 7B 模型嘅參數要 28 GB。"
            "BF16 只用 2 個 bytes，同一個模型只要 14 GB。"
            "記憶體直接減半，GPU 嘅 Tensor Core 計算速度仲快兩倍。"
            "BF16 嘅特點係佢同 FP32 有一樣嘅數值範圍，"
            "唔會出現 overflow 嘅問題。"
            "所以幾乎所有現代 LLM 訓練都用 BF16。"
            "PyTorch 嘅 torch.amp.autocast 會自動幫你揀"
            "邊啲操作用 BF16，邊啲保持 FP32。"
            "矩陣乘法用 BF16 加速，歸一化同 loss 計算就保持 FP32 精度。"
        ):
            self.play(Write(heading), run_time=0.6)
            self.play(
                FadeIn(fp32_bar), FadeIn(fp32_label),
                FadeIn(fp32_bytes), FadeIn(fp32_mem), FadeIn(fp32_note),
                run_time=0.6,
            )
            self.play(
                FadeIn(bf16_bar), FadeIn(bf16_label),
                FadeIn(bf16_bytes), FadeIn(bf16_mem), FadeIn(bf16_note),
                run_time=0.6,
            )
            self.play(
                GrowArrow(savings_arrow), FadeIn(savings_text),
                run_time=0.5,
            )
            for adv in advantages:
                self.play(FadeIn(adv, shift=RIGHT * 0.2), run_time=0.3)

        self.wait(0.3)
        self.clear()

    # ── Scene 6 — Flash Attention (memory comparison) ─────────────────────

    def scene_flash_attention(self):
        heading = self.make_heading("Flash Attention")

        # Two columns: Standard vs Flash
        # Standard Attention
        std_title = self.zh("標準 Attention", font_size=22, color=C_RED)
        std_box = RoundedRectangle(
            corner_radius=0.15, width=4.8, height=3.5,
            fill_color=C_RED, fill_opacity=0.08,
            stroke_color=C_RED, stroke_width=2,
        )

        std_items = VGroup(
            self.mono("1. Q @ K^T", font_size=18, color=C_WHITE),
            self.mono("   -> (n x n) 矩陣", font_size=16, color=C_DIM),
            self.mono("2. softmax", font_size=18, color=C_WHITE),
            self.mono("   -> (n x n) 矩陣", font_size=16, color=C_DIM),
            self.mono("3. weights @ V", font_size=18, color=C_WHITE),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.15)
        std_items.move_to(std_box)
        std_title.next_to(std_box, UP, buff=0.15)

        std_mem = self.mono("Memory: O(n^2)", font_size=20, color=C_RED)
        std_mem.next_to(std_box, DOWN, buff=0.2)

        std_group = VGroup(std_box, std_title, std_items, std_mem)
        std_group.shift(LEFT * 3.2)

        # Flash Attention
        flash_title = self.zh("Flash Attention", font_size=22, color=C_GREEN)
        flash_box = RoundedRectangle(
            corner_radius=0.15, width=4.8, height=3.5,
            fill_color=C_GREEN, fill_opacity=0.08,
            stroke_color=C_GREEN, stroke_width=2,
        )

        flash_items = VGroup(
            self.mono("1. 分割 Q,K,V 成 tiles", font_size=18, color=C_WHITE),
            self.mono("2. 喺 SRAM 入面計算", font_size=18, color=C_WHITE),
            self.mono("   (唔使寫入 HBM)", font_size=16, color=C_DIM),
            self.mono("3. Online softmax", font_size=18, color=C_WHITE),
            self.mono("   累積結果", font_size=16, color=C_DIM),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.15)
        flash_items.move_to(flash_box)
        flash_title.next_to(flash_box, UP, buff=0.15)

        flash_mem = self.mono("Memory: O(n)", font_size=20, color=C_GREEN)
        flash_mem.next_to(flash_box, DOWN, buff=0.2)

        flash_group = VGroup(flash_box, flash_title, flash_items, flash_mem)
        flash_group.shift(RIGHT * 3.2)

        # Center comparison
        vs_text = self.zh("vs", font_size=30, color=C_YELLOW)

        # Bottom example
        example = VGroup(
            RoundedRectangle(
                corner_radius=0.12, width=10, height=0.7,
                fill_color=C_CYAN, fill_opacity=0.12,
                stroke_color=C_CYAN, stroke_width=1.5,
            ),
            self.zh(
                "n=8192: 標準要 128 MB → Flash 只要 3 MB (43x 節省)",
                font_size=20, color=C_CYAN,
            ),
        )
        example[1].move_to(example[0])
        example.to_edge(DOWN, buff=0.35)

        with self.voiceover(
            text="Flash Attention 係 2022 年嘅重大突破。"
            "標準嘅 attention 需要計算同儲存完整嘅 n 乘 n 注意力矩陣。"
            "首先計算 Q 乘以 K 轉置，得到 n 乘 n 嘅分數矩陣。"
            "然後做 softmax，又係 n 乘 n。"
            "呢兩個矩陣都要寫入 HBM，即係 GPU 嘅主記憶體。"
            "記憶體係 O of n squared。"
            "Flash Attention 嘅做法完全唔同。"
            "佢將 Q、K、V 分成細小嘅 tiles。"
            "每對 tiles 喺 SRAM 入面計算，SRAM 係 GPU 嘅快速緩存，比 HBM 快好多倍。"
            "用 online softmax 嘅技巧，唔使儲存完整嘅 n 乘 n 矩陣。"
            "記憶體降到 O of n，"
            "速度仲快兩到四倍。"
            "例如序列長度 8192，標準要 128 MB，Flash 只要 3 MB。"
        ):
            self.play(Write(heading), run_time=0.6)

            self.play(FadeIn(std_box), FadeIn(std_title), run_time=0.4)
            for item in std_items:
                self.play(FadeIn(item, shift=RIGHT * 0.1), run_time=0.25)
            self.play(FadeIn(std_mem), run_time=0.3)

            self.play(FadeIn(vs_text), run_time=0.3)

            self.play(FadeIn(flash_box), FadeIn(flash_title), run_time=0.4)
            for item in flash_items:
                self.play(FadeIn(item, shift=RIGHT * 0.1), run_time=0.25)
            self.play(FadeIn(flash_mem), run_time=0.3)

            self.play(FadeIn(example, shift=UP * 0.2), run_time=0.6)

        self.wait(0.3)
        self.clear()

    # ── Scene 7 — Scaling Laws (Chinchilla curve) ─────────────────────────

    def scene_scaling_laws(self):
        heading = self.make_heading("Chinchilla Scaling Laws")

        axes = Axes(
            x_range=[17, 25, 1],
            y_range=[1.5, 3.5, 0.5],
            x_length=8,
            y_length=4,
            axis_config={"color": C_DIM, "include_numbers": False},
        ).shift(DOWN * 0.2 + LEFT * 0.5)

        x_label = self.mono(
            "log10(Compute FLOPs)", font_size=16, color=C_WHITE
        ).next_to(axes.x_axis, DOWN, buff=0.3)
        y_label = self.mono(
            "Loss", font_size=16, color=C_WHITE
        ).next_to(axes.y_axis, UP, buff=0.2).shift(LEFT * 0.3)

        # Chinchilla curve (optimal frontier)
        curve = axes.plot(
            lambda x: 1.69 + 3.0 * (10 ** (-(x - 17) * 0.12)),
            x_range=[17, 24.5],
            color=C_CYAN,
        )
        curve_label = self.zh(
            "最優前沿", font_size=16, color=C_CYAN
        ).next_to(axes.c2p(24, 1.85), RIGHT, buff=0.15)

        # Mark real models
        models = [
            ("GPT-3", 23.6, 2.15, C_RED),
            ("Chinchilla", 23.8, 1.95, C_GREEN),
            ("LLaMA", 23.6, 1.90, C_ORANGE),
        ]

        model_dots = VGroup()
        model_labels = VGroup()
        for name, x, y, color in models:
            dot = Dot(axes.c2p(x, y), color=color, radius=0.1)
            lbl = self.mono(name, font_size=16, color=color).next_to(
                dot, UR, buff=0.1
            )
            model_dots.add(dot)
            model_labels.add(lbl)

        # Key finding box
        finding = VGroup(
            RoundedRectangle(
                corner_radius=0.12, width=5.5, height=1.4,
                fill_color=C_YELLOW, fill_opacity=0.12,
                stroke_color=C_YELLOW, stroke_width=1.5,
            ),
            self.zh("Chinchilla 法則：", font_size=20, color=C_YELLOW),
            self.mono("Tokens = 20 x Params", font_size=20, color=C_GREEN),
            self.mono("C = 6 x N x D", font_size=18, color=C_DIM),
        )
        finding[1:].arrange(DOWN, buff=0.12)
        VGroup(*finding[1:]).move_to(finding[0])
        finding.to_edge(RIGHT, buff=0.3).shift(DOWN * 2.0)

        with self.voiceover(
            text="Chinchilla Scaling Laws 回答咗一個根本問題。"
            "俾你一定嘅計算預算，應該訓練幾大嘅模型？"
            "用幾多數據？"
            "2022 年，Chinchilla 論文發現之前嘅模型全部訓練不足。"
            "GPT-3 有 1750 億參數，但只訓練咗 3000 億 token。"
            "呢條曲線顯示嘅係最優前沿。"
            "喺同樣嘅計算預算下，Chinchilla 用 700 億參數訓練 1.4 萬億 token，"
            "效果同 GPT-3 一樣好，但計算量少四倍。"
            "核心法則好簡單：最優嘅 token 數量大約係參數量嘅 20 倍。"
            "呢個發現改變咗成個 LLM 領域嘅訓練策略。"
        ):
            self.play(Write(heading), run_time=0.6)
            self.play(Create(axes), FadeIn(x_label), FadeIn(y_label), run_time=0.8)
            self.play(Create(curve), FadeIn(curve_label), run_time=1.5)

            for dot, lbl in zip(model_dots, model_labels):
                self.play(
                    FadeIn(dot, scale=1.5), FadeIn(lbl),
                    run_time=0.5,
                )

            self.play(FadeIn(finding, shift=LEFT * 0.3), run_time=0.7)

        self.wait(0.3)
        self.clear()

    # ── Scene 8 — Summary ────────────────────────────────────────────────

    def scene_summary(self):
        heading = self.make_heading("總結", font_size=44)

        bullets = VGroup(
            self.zh("•  DDP：複製模型、分割數據、All-Reduce 同步梯度", font_size=24, color=C_GREEN),
            self.zh("•  FSDP：分片參數、梯度、優化器 — 記憶體減 N 倍", font_size=24, color=C_ORANGE),
            self.zh("•  Tensor Parallelism：分割單獨嘅層到唔同 GPU", font_size=24, color=C_PINK),
            self.zh("•  BF16 混合精度：2x 記憶體節省 + 2x 計算加速", font_size=24, color=C_CYAN),
            self.zh("•  Flash Attention：O(n) 記憶體取代 O(n^2)", font_size=24, color=C_PURPLE),
            self.zh("•  Chinchilla：Tokens = 20 x Params 最優法則", font_size=24, color=C_YELLOW),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.35).move_to(ORIGIN)

        with self.voiceover(
            text="總結一下今日學咗嘅六個重點。"
            "第一，DDP 係最簡單嘅多 GPU 方法，複製模型同步梯度。"
            "第二，FSDP 將所有嘢分片，記憶體減 N 倍。"
            "第三，Tensor Parallelism 分割個別嘅層到唔同 GPU。"
            "第四，BF16 混合精度減半記憶體同時加速計算。"
            "第五，Flash Attention 將注意力嘅記憶體從 n 平方降到 n。"
            "第六，Chinchilla 法則話俾我哋知最優嘅 token 數等於參數量嘅 20 倍。"
            "掌握咗呢啲技術，你就可以訓練真正大規模嘅 LLM 喇。"
        ):
            self.play(Write(heading), run_time=0.6)
            for b in bullets:
                self.play(FadeIn(b, shift=RIGHT * 0.3), run_time=0.5)

        self.wait(0.5)
        self.clear()

    # ── Scene 9 — Outro ──────────────────────────────────────────────────

    def scene_outro(self):
        thanks = self.zh("多謝收睇！", font_size=52, color=C_YELLOW)
        next_lesson = self.zh(
            "下一課：Text Generation（文本生成）",
            font_size=28, color=C_CYAN,
        ).next_to(thanks, DOWN, buff=0.6)

        with self.voiceover(
            text="多謝收睇！下一課我哋會學 Text Generation，"
            "即係點樣用你嘅模型生成文本。"
            "包括 greedy decoding、temperature sampling、"
            "top-k、top-p 同 KV cache。"
            "記得繼續跟住學喇！"
        ):
            self.play(Write(thanks), run_time=1)
            self.play(FadeIn(next_lesson, shift=UP * 0.2), run_time=0.8)

        self.wait(1)
        self.play(FadeOut(thanks), FadeOut(next_lesson), run_time=0.8)
