"""
Lesson 13 – Preference Alignment (RLHF / DPO)
Manim CE + Cantonese voiceover (Edge TTS)

Render (1080p):
    cd "lesson 13/video"
    manim render -qh scene.py AlignmentExplainer

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


class AlignmentExplainer(VoiceoverScene):
    """Eight scenes explaining preference alignment in Cantonese."""

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
        self.scene_why_alignment()
        self.scene_rlhf_pipeline()
        self.scene_reward_model()
        self.scene_dpo()
        self.scene_kl_penalty()
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

    # ── Scene 1 — Intro ──────────────────────────────────────────────────

    def scene_intro(self):
        title = self.zh("偏好對齊", font_size=56, color=C_BLUE)
        sub_en = self.en(
            "Preference Alignment (RLHF / DPO)", font_size=24, color=C_WHITE
        ).next_to(title, DOWN, buff=0.4)
        sub_zh = self.zh(
            "第十三課", font_size=36, color=C_ORANGE
        ).next_to(sub_en, DOWN, buff=0.3)
        producer = self.zh(
            "制片人：Peter", font_size=24, color=C_DIM
        ).next_to(sub_zh, DOWN, buff=0.7)

        with self.voiceover(
            text="大家好！歡迎嚟到第十三課。"
            "上一課我哋學咗 Supervised Fine-Tuning，"
            "將 base model 變成識得跟指令嘅助手。"
            "但係 SFT 模型仲有一個大問題。"
            "佢唔識分邊個回答好，邊個回答唔好。"
            "今日我哋會學偏好對齊。"
            "包括 RLHF 嘅完整流程、Reward Model、"
            "DPO 嘅從零實現、同埋 KL penalty。"
        ):
            self.play(Write(title), run_time=1.5)
            self.play(FadeIn(sub_en, shift=UP * 0.2), run_time=0.8)
            self.play(FadeIn(sub_zh, shift=UP * 0.2), run_time=0.8)
            self.play(FadeIn(producer, shift=UP * 0.2), run_time=0.5)

        self.wait(0.3)
        self.clear()

    # ── Scene 2 — Why Alignment ──────────────────────────────────────────

    def scene_why_alignment(self):
        heading = self.make_heading("點解需要對齊？")

        # Helpful example (left)
        helpful_box = RoundedRectangle(
            corner_radius=0.15, width=5.0, height=3.0,
            fill_color=C_GREEN, fill_opacity=0.10,
            stroke_color=C_GREEN, stroke_width=2,
        )
        helpful_title = self.zh(
            "有用嘅回答", font_size=22, color=C_GREEN
        ).next_to(helpful_box, UP, buff=0.1)
        helpful_prompt = self.zh(
            "問：點樣改善簡歷？", font_size=18, color=C_WHITE
        ).move_to(helpful_box).shift(UP * 0.7)
        helpful_resp = self.zh(
            "答：1) 針對職位調整內容\n"
            "     2) 量化你嘅成就\n"
            "     3) 用專業格式排版",
            font_size=16, color=C_GREEN,
        ).move_to(helpful_box).shift(DOWN * 0.3)
        helpful_group = VGroup(helpful_box, helpful_title, helpful_prompt, helpful_resp)

        # Harmful example (right)
        harmful_box = RoundedRectangle(
            corner_radius=0.15, width=5.0, height=3.0,
            fill_color=C_RED, fill_opacity=0.10,
            stroke_color=C_RED, stroke_width=2,
        )
        harmful_title = self.zh(
            "有害嘅回答", font_size=22, color=C_RED
        ).next_to(harmful_box, UP, buff=0.1)
        harmful_prompt = self.zh(
            "問：寫一篇文話地球係平嘅",
            font_size=18, color=C_WHITE,
        ).move_to(harmful_box).shift(UP * 0.7)
        harmful_resp = self.zh(
            "答：地球確實係平嘅...\n"
            "     [令人信服嘅假資訊]",
            font_size=16, color=C_RED,
        ).move_to(harmful_box).shift(DOWN * 0.2)
        harmful_label = self.zh(
            "SFT 模型會照做！", font_size=18, color=C_YELLOW
        ).move_to(harmful_box).shift(DOWN * 1.0)
        harmful_group = VGroup(harmful_box, harmful_title, harmful_prompt, harmful_resp, harmful_label)

        examples = VGroup(helpful_group, harmful_group).arrange(RIGHT, buff=0.6)
        examples.shift(DOWN * 0.3)

        # Bottom pillars
        pillars = VGroup(
            self.make_box("有用 Helpful", C_GREEN, width=3.0, height=0.6, font_size=20),
            self.make_box("無害 Harmless", C_ORANGE, width=3.0, height=0.6, font_size=20),
            self.make_box("誠實 Honest", C_CYAN, width=3.0, height=0.6, font_size=20),
        ).arrange(RIGHT, buff=0.4).to_edge(DOWN, buff=0.4)

        with self.voiceover(
            text="SFT 模型學識咗跟指令，但佢唔識分好壞。"
            "例如你問點樣改善簡歷，佢會俾有用嘅建議。"
            "但如果你叫佢寫假資訊，佢一樣會照做。"
            "因為 SFT 訓練數據只有好嘅示範，"
            "冇教過模型乜嘢係唔應該做嘅。"
            "偏好對齊就係要教模型三個原則。"
            "第一，有用：提供準確有幫助嘅資訊。"
            "第二，無害：拒絕危險嘅請求。"
            "第三，誠實：唔知就話唔知，唔好作故仔。"
            "呢三個原則就係 Anthropic 提出嘅 HHH 框架。"
        ):
            self.play(Write(heading), run_time=0.6)
            self.play(
                FadeIn(helpful_box), FadeIn(helpful_title),
                run_time=0.4,
            )
            self.play(
                FadeIn(helpful_prompt, shift=DOWN * 0.1),
                FadeIn(helpful_resp, shift=DOWN * 0.1),
                run_time=0.6,
            )
            self.play(
                FadeIn(harmful_box), FadeIn(harmful_title),
                run_time=0.4,
            )
            self.play(
                FadeIn(harmful_prompt, shift=DOWN * 0.1),
                FadeIn(harmful_resp, shift=DOWN * 0.1),
                run_time=0.6,
            )
            self.play(FadeIn(harmful_label), run_time=0.4)
            for p in pillars:
                self.play(FadeIn(p, shift=UP * 0.2), run_time=0.4)

        self.wait(0.3)
        self.clear()

    # ── Scene 3 — RLHF Pipeline (3-stage flow) ───────────────────────────

    def scene_rlhf_pipeline(self):
        heading = self.make_heading("RLHF 三階段流程")

        # Stage 1: SFT
        sft_box = self.make_box("Stage 1: SFT", C_GREEN, width=3.2, height=1.2, font_size=20)
        sft_detail = self.zh(
            "指令數據\n→ 微調 Base Model", font_size=14, color=C_DIM
        ).next_to(sft_box, DOWN, buff=0.1)
        stage1 = VGroup(sft_box, sft_detail)

        # Stage 2: Reward Model
        rm_box = self.make_box("Stage 2: Reward Model", C_ORANGE, width=3.2, height=1.2, font_size=20)
        rm_detail = self.zh(
            "人類偏好數據\n→ 訓練評分模型", font_size=14, color=C_DIM
        ).next_to(rm_box, DOWN, buff=0.1)
        stage2 = VGroup(rm_box, rm_detail)

        # Stage 3: PPO
        ppo_box = self.make_box("Stage 3: PPO", C_PINK, width=3.2, height=1.2, font_size=20)
        ppo_detail = self.zh(
            "用 Reward 信號\n→ RL 優化策略", font_size=14, color=C_DIM
        ).next_to(ppo_box, DOWN, buff=0.1)
        stage3 = VGroup(ppo_box, ppo_detail)

        stages = VGroup(stage1, stage2, stage3).arrange(RIGHT, buff=0.8)
        stages.shift(UP * 0.3)

        # Arrows between stages
        arrow1 = Arrow(
            sft_box[0].get_right(), rm_box[0].get_left(),
            buff=0.15, color=C_YELLOW, stroke_width=2.5,
        )
        arrow2 = Arrow(
            rm_box[0].get_right(), ppo_box[0].get_left(),
            buff=0.15, color=C_YELLOW, stroke_width=2.5,
        )

        # Output box
        output_box = RoundedRectangle(
            corner_radius=0.15, width=10, height=0.8,
            fill_color=C_CYAN, fill_opacity=0.12,
            stroke_color=C_CYAN, stroke_width=1.5,
        ).to_edge(DOWN, buff=0.5)
        output_label = self.zh(
            "輸出：對齊嘅模型 — 最大化 Reward 同時保持接近 SFT",
            font_size=20, color=C_CYAN,
        ).move_to(output_box)

        # DPO shortcut arrow
        dpo_arrow = CurvedArrow(
            sft_box[0].get_bottom() + DOWN * 0.6,
            ppo_box[0].get_bottom() + DOWN * 0.6,
            angle=-TAU / 6,
            color=C_PURPLE, stroke_width=2.5,
        )
        dpo_label = self.zh(
            "DPO: 跳過 Stage 2+3！", font_size=18, color=C_PURPLE
        ).next_to(dpo_arrow, DOWN, buff=0.15)

        with self.voiceover(
            text="RLHF 有三個階段。"
            "第一階段係 SFT，我哋上一課已經學咗。"
            "用指令數據微調 base model。"
            "第二階段係訓練一個 Reward Model。"
            "收集人類偏好數據：同一個問題嘅兩個回答，邊個好啲？"
            "用呢啲數據訓練一個評分模型。"
            "第三階段係 PPO，用強化學習嘅方法。"
            "將 Reward Model 嘅評分作為獎勵信號，"
            "優化 SFT 模型去生成更高分嘅回答。"
            "但係 RLHF 好複雜，要同時運行四個模型。"
            "所以有人發明咗 DPO。"
            "DPO 直接跳過第二同第三階段，"
            "用一個簡單嘅 loss function 就搞掂。"
        ):
            self.play(Write(heading), run_time=0.6)
            self.play(FadeIn(stage1, shift=DOWN * 0.2), run_time=0.5)
            self.play(GrowArrow(arrow1), run_time=0.4)
            self.play(FadeIn(stage2, shift=DOWN * 0.2), run_time=0.5)
            self.play(GrowArrow(arrow2), run_time=0.4)
            self.play(FadeIn(stage3, shift=DOWN * 0.2), run_time=0.5)
            self.play(
                FadeIn(output_box), FadeIn(output_label),
                run_time=0.5,
            )
            self.play(
                Create(dpo_arrow), FadeIn(dpo_label),
                run_time=0.8,
            )

        self.wait(0.3)
        self.clear()

    # ── Scene 4 — Reward Model (scoring responses) ───────────────────────

    def scene_reward_model(self):
        heading = self.make_heading("Reward Model 評分模型")

        # Prompt at top
        prompt_box = RoundedRectangle(
            corner_radius=0.15, width=8, height=0.8,
            fill_color=C_BLUE, fill_opacity=0.15,
            stroke_color=C_BLUE, stroke_width=2,
        ).shift(UP * 1.8)
        prompt_label = self.zh(
            "問題：點樣向小朋友解釋量子計算？",
            font_size=20, color=C_BLUE,
        ).move_to(prompt_box)

        # Response A (good) — left
        resp_a_box = RoundedRectangle(
            corner_radius=0.15, width=4.8, height=2.0,
            fill_color=C_GREEN, fill_opacity=0.10,
            stroke_color=C_GREEN, stroke_width=2,
        )
        resp_a_title = self.zh("回答 A", font_size=20, color=C_GREEN)
        resp_a_title.next_to(resp_a_box, UP, buff=0.1)
        resp_a_text = self.zh(
            "想像一個神奇硬幣，\n可以同時係公同字！\n普通電腦用普通硬幣...",
            font_size=16, color=C_WHITE,
        ).move_to(resp_a_box)
        resp_a_group = VGroup(resp_a_box, resp_a_title, resp_a_text)

        # Response B (bad) — right
        resp_b_box = RoundedRectangle(
            corner_radius=0.15, width=4.8, height=2.0,
            fill_color=C_RED, fill_opacity=0.10,
            stroke_color=C_RED, stroke_width=2,
        )
        resp_b_title = self.zh("回答 B", font_size=20, color=C_RED)
        resp_b_title.next_to(resp_b_box, UP, buff=0.1)
        resp_b_text = self.zh(
            "量子計算利用\nHilbert 空間中嘅\n量子態疊加同糾纏...",
            font_size=16, color=C_WHITE,
        ).move_to(resp_b_box)
        resp_b_group = VGroup(resp_b_box, resp_b_title, resp_b_text)

        responses = VGroup(resp_a_group, resp_b_group).arrange(RIGHT, buff=0.5)
        responses.shift(DOWN * 0.2)

        # Reward model scoring
        rm_box = self.make_box("Reward Model", C_ORANGE, width=3.0, height=0.7, font_size=20)
        rm_box.shift(DOWN * 2.2)

        score_a = self.mono(
            "Score: 4.2", font_size=22, color=C_GREEN
        ).next_to(rm_box, LEFT, buff=1.2)
        score_b = self.mono(
            "Score: 1.8", font_size=22, color=C_RED
        ).next_to(rm_box, RIGHT, buff=1.2)

        # Arrows from responses to RM
        arrow_a = Arrow(
            resp_a_box.get_bottom(), rm_box[0].get_top() + LEFT * 1.5,
            buff=0.15, color=C_DIM, stroke_width=2,
        )
        arrow_b = Arrow(
            resp_b_box.get_bottom(), rm_box[0].get_top() + RIGHT * 1.5,
            buff=0.15, color=C_DIM, stroke_width=2,
        )

        # Loss formula
        loss_label = self.mono(
            "Loss = -log sigmoid(r_A - r_B)",
            font_size=18, color=C_YELLOW,
        ).to_edge(DOWN, buff=0.3)

        with self.voiceover(
            text="Reward Model 係一個評分模型。"
            "佢嘅輸入係一個問題加一個回答，"
            "輸出係一個分數，分數越高代表回答越好。"
            "例如呢個問題：點樣向小朋友解釋量子計算？"
            "回答 A 用咗簡單嘅比喻，小朋友容易明白。"
            "回答 B 用咗好多專業術語，小朋友完全聽唔明。"
            "Reward Model 會俾回答 A 高分，回答 B 低分。"
            "訓練嘅 loss function 係 Bradley-Terry model。"
            "即係負 log sigmoid of r A 減 r B。"
            "呢個 loss 會推高好回答嘅分數，壓低差回答嘅分數。"
        ):
            self.play(Write(heading), run_time=0.6)
            self.play(FadeIn(prompt_box), FadeIn(prompt_label), run_time=0.5)
            self.play(
                FadeIn(resp_a_group, shift=DOWN * 0.2),
                FadeIn(resp_b_group, shift=DOWN * 0.2),
                run_time=0.6,
            )
            self.play(GrowArrow(arrow_a), GrowArrow(arrow_b), run_time=0.5)
            self.play(FadeIn(rm_box, shift=UP * 0.2), run_time=0.4)
            self.play(FadeIn(score_a), FadeIn(score_b), run_time=0.5)
            self.play(FadeIn(loss_label, shift=UP * 0.1), run_time=0.5)

        self.wait(0.3)
        self.clear()

    # ── Scene 5 — DPO (chosen vs rejected with arrows) ───────────────────

    def scene_dpo(self):
        heading = self.make_heading("DPO 直接偏好優化")

        # Policy model in center
        policy_box = self.make_box("Policy Model", C_BLUE, width=3.2, height=0.9, font_size=22)
        ref_box = self.make_box("Reference Model", C_DIM, width=3.2, height=0.9, font_size=22)

        models = VGroup(policy_box, ref_box).arrange(RIGHT, buff=1.5)
        models.shift(UP * 1.6)

        frozen_label = self.zh("(凍結)", font_size=16, color=C_RED)
        frozen_label.next_to(ref_box, DOWN, buff=0.08)

        # Chosen and Rejected
        chosen_box = RoundedRectangle(
            corner_radius=0.15, width=4.5, height=1.5,
            fill_color=C_GREEN, fill_opacity=0.12,
            stroke_color=C_GREEN, stroke_width=2,
        )
        chosen_title = self.zh("Chosen (好回答)", font_size=20, color=C_GREEN)
        chosen_title.next_to(chosen_box, UP, buff=0.08)
        chosen_formula = self.mono(
            "log_ratio_w =\nlog pi(w|x) - log ref(w|x)",
            font_size=14, color=C_GREEN,
        ).move_to(chosen_box)
        chosen_group = VGroup(chosen_box, chosen_title, chosen_formula)

        rejected_box = RoundedRectangle(
            corner_radius=0.15, width=4.5, height=1.5,
            fill_color=C_RED, fill_opacity=0.12,
            stroke_color=C_RED, stroke_width=2,
        )
        rejected_title = self.zh("Rejected (差回答)", font_size=20, color=C_RED)
        rejected_title.next_to(rejected_box, UP, buff=0.08)
        rejected_formula = self.mono(
            "log_ratio_l =\nlog pi(l|x) - log ref(l|x)",
            font_size=14, color=C_RED,
        ).move_to(rejected_box)
        rejected_group = VGroup(rejected_box, rejected_title, rejected_formula)

        pairs = VGroup(chosen_group, rejected_group).arrange(RIGHT, buff=0.5)
        pairs.shift(DOWN * 0.5)

        # Arrows from models to chosen/rejected
        arrow_pc = Arrow(
            policy_box[0].get_bottom(), chosen_box.get_top() + LEFT * 0.5,
            buff=0.25, color=C_BLUE, stroke_width=2,
        )
        arrow_pr = Arrow(
            policy_box[0].get_bottom(), rejected_box.get_top() + LEFT * 0.5,
            buff=0.25, color=C_BLUE, stroke_width=2,
        )
        arrow_rc = Arrow(
            ref_box[0].get_bottom(), chosen_box.get_top() + RIGHT * 0.5,
            buff=0.25, color=C_DIM, stroke_width=2,
        )
        arrow_rr = Arrow(
            ref_box[0].get_bottom(), rejected_box.get_top() + RIGHT * 0.5,
            buff=0.25, color=C_DIM, stroke_width=2,
        )

        # DPO loss at bottom
        loss_box = RoundedRectangle(
            corner_radius=0.15, width=10, height=0.9,
            fill_color=C_YELLOW, fill_opacity=0.15,
            stroke_color=C_YELLOW, stroke_width=2,
        ).to_edge(DOWN, buff=0.4)
        loss_formula = self.mono(
            "DPO Loss = -log sigmoid( beta * (log_ratio_w - log_ratio_l) )",
            font_size=18, color=C_YELLOW,
        ).move_to(loss_box)

        # Arrows from pairs to loss
        arrow_cl = Arrow(
            chosen_box.get_bottom(), loss_box.get_top() + LEFT * 2,
            buff=0.15, color=C_GREEN, stroke_width=2,
        )
        arrow_rl = Arrow(
            rejected_box.get_bottom(), loss_box.get_top() + RIGHT * 2,
            buff=0.15, color=C_RED, stroke_width=2,
        )

        with self.voiceover(
            text="DPO 係 2023 年嘅重大突破。"
            "佢唔需要 Reward Model，唔需要 PPO。"
            "只需要一個簡單嘅 loss function。"
            "首先我哋有兩個模型。"
            "Policy model 係我哋要訓練嘅模型。"
            "Reference model 係凍結咗嘅 SFT 模型。"
            "對於每對偏好數據，我哋計算兩個 log ratio。"
            "Chosen 嘅 log ratio 等於 policy 嘅 log prob 減去 reference 嘅 log prob。"
            "Rejected 嘅 log ratio 都係一樣嘅計算。"
            "DPO 嘅 loss 就係負 log sigmoid of beta 乘以兩個 log ratio 嘅差。"
            "呢個 loss 會推高 chosen 嘅概率，壓低 rejected 嘅概率。"
            "Beta 控制對齊嘅力度。Beta 越大，改變越大。"
        ):
            self.play(Write(heading), run_time=0.6)
            self.play(
                FadeIn(policy_box, shift=DOWN * 0.2),
                FadeIn(ref_box, shift=DOWN * 0.2),
                FadeIn(frozen_label),
                run_time=0.6,
            )
            self.play(
                GrowArrow(arrow_pc), GrowArrow(arrow_pr),
                GrowArrow(arrow_rc), GrowArrow(arrow_rr),
                run_time=0.5,
            )
            self.play(
                FadeIn(chosen_group, shift=DOWN * 0.2),
                FadeIn(rejected_group, shift=DOWN * 0.2),
                run_time=0.6,
            )
            self.play(GrowArrow(arrow_cl), GrowArrow(arrow_rl), run_time=0.4)
            self.play(FadeIn(loss_box), FadeIn(loss_formula), run_time=0.6)

        self.wait(0.3)
        self.clear()

    # ── Scene 6 — KL Penalty (divergence visualization) ──────────────────

    def scene_kl_penalty(self):
        heading = self.make_heading("KL Penalty 防止模型偏離")

        # Two distribution curves represented as bar charts
        bar_width = 0.4
        num_bars = 8
        x_positions = [i * (bar_width + 0.15) for i in range(num_bars)]
        center_offset = sum(x_positions) / num_bars / 2

        # Reference distribution (blue, symmetric)
        ref_heights = [0.3, 0.8, 1.5, 2.5, 2.3, 1.4, 0.7, 0.2]
        ref_bars = VGroup()
        for i, h in enumerate(ref_heights):
            bar = Rectangle(
                width=bar_width, height=h,
                fill_color=C_BLUE, fill_opacity=0.4,
                stroke_color=C_BLUE, stroke_width=1.5,
            )
            bar.move_to(LEFT * 4 + RIGHT * x_positions[i] + UP * h / 2)
            ref_bars.add(bar)
        ref_label = self.zh(
            "Reference 分佈", font_size=18, color=C_BLUE
        ).next_to(ref_bars, UP, buff=0.2)
        ref_group = VGroup(ref_bars, ref_label).shift(UP * 0.3)

        # Low-KL policy (green, similar shape)
        low_kl_heights = [0.4, 0.9, 1.6, 2.8, 2.1, 1.3, 0.6, 0.3]
        low_kl_bars = VGroup()
        for i, h in enumerate(low_kl_heights):
            bar = Rectangle(
                width=bar_width, height=h,
                fill_color=C_GREEN, fill_opacity=0.4,
                stroke_color=C_GREEN, stroke_width=1.5,
            )
            bar.move_to(RIGHT * 0.5 + RIGHT * x_positions[i] + UP * h / 2)
            low_kl_bars.add(bar)
        low_kl_label = self.zh(
            "低 KL (好)", font_size=18, color=C_GREEN
        ).next_to(low_kl_bars, UP, buff=0.2)
        low_kl_value = self.mono(
            "KL = 0.3", font_size=16, color=C_GREEN
        ).next_to(low_kl_bars, DOWN, buff=0.2)
        low_kl_group = VGroup(low_kl_bars, low_kl_label, low_kl_value).shift(UP * 0.3)

        # High-KL policy (red, very different shape)
        high_kl_heights = [2.8, 0.2, 0.1, 0.3, 0.1, 0.2, 0.1, 2.5]
        high_kl_bars = VGroup()
        for i, h in enumerate(high_kl_heights):
            bar = Rectangle(
                width=bar_width, height=h,
                fill_color=C_RED, fill_opacity=0.4,
                stroke_color=C_RED, stroke_width=1.5,
            )
            bar.move_to(RIGHT * 5.0 + RIGHT * x_positions[i] + UP * h / 2)
            high_kl_bars.add(bar)
        high_kl_label = self.zh(
            "高 KL (危險)", font_size=18, color=C_RED
        ).next_to(high_kl_bars, UP, buff=0.2)
        high_kl_value = self.mono(
            "KL = 12.5", font_size=16, color=C_RED
        ).next_to(high_kl_bars, DOWN, buff=0.2)
        high_kl_group = VGroup(high_kl_bars, high_kl_label, high_kl_value).shift(UP * 0.3)

        # RLHF objective at bottom
        obj_box = RoundedRectangle(
            corner_radius=0.12, width=10.5, height=0.8,
            fill_color=C_YELLOW, fill_opacity=0.12,
            stroke_color=C_YELLOW, stroke_width=1.5,
        ).to_edge(DOWN, buff=0.35)
        obj_formula = self.mono(
            "J = E[ Reward(y|x) ] - beta * KL( pi || pi_ref )",
            font_size=20, color=C_YELLOW,
        ).move_to(obj_box)

        with self.voiceover(
            text="KL penalty 係偏好對齊入面好重要嘅概念。"
            "KL divergence 衡量兩個分佈之間嘅距離。"
            "呢度藍色係 reference model 嘅分佈。"
            "綠色係一個低 KL 嘅 policy，形狀同 reference 好似。"
            "KL 只有 0.3，代表模型改變唔大，呢個係好嘅。"
            "但紅色呢個 policy，分佈完全唔同。"
            "KL 去到 12.5，代表模型已經偏離太遠。"
            "模型可能已經忘記咗點樣講流暢嘅句子，"
            "或者學咗啲奇怪嘅捷徑去攞高分。"
            "所以 RLHF 嘅訓練目標係："
            "最大化 reward 同時要減去 beta 乘以 KL。"
            "DPO 入面嘅 beta 參數就係扮演呢個角色。"
            "Beta 越大，KL penalty 越重，模型越保守。"
        ):
            self.play(Write(heading), run_time=0.6)
            for bar in ref_bars:
                self.play(GrowFromEdge(bar, DOWN), run_time=0.08)
            self.play(FadeIn(ref_label), run_time=0.3)

            for bar in low_kl_bars:
                self.play(GrowFromEdge(bar, DOWN), run_time=0.08)
            self.play(FadeIn(low_kl_label), FadeIn(low_kl_value), run_time=0.3)

            for bar in high_kl_bars:
                self.play(GrowFromEdge(bar, DOWN), run_time=0.08)
            self.play(FadeIn(high_kl_label), FadeIn(high_kl_value), run_time=0.3)

            self.play(FadeIn(obj_box), FadeIn(obj_formula), run_time=0.6)

        self.wait(0.3)
        self.clear()

    # ── Scene 7 — Summary ────────────────────────────────────────────────

    def scene_summary(self):
        heading = self.make_heading("總結", font_size=44)

        bullets = VGroup(
            self.zh("•  SFT 唔夠：模型需要偏好信號分辨好壞", font_size=24, color=C_GREEN),
            self.zh("•  RLHF 三階段：SFT → Reward Model → PPO", font_size=24, color=C_ORANGE),
            self.zh("•  Reward Model：用 Bradley-Terry loss 學評分", font_size=24, color=C_PINK),
            self.zh("•  DPO：跳過 RM 同 PPO，一個 loss 搞掂對齊", font_size=24, color=C_CYAN),
            self.zh("•  KL Penalty：防止模型偏離太遠", font_size=24, color=C_PURPLE),
            self.zh("•  Beta：控制對齊力度，通常 0.1 到 0.5", font_size=24, color=C_YELLOW),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.35).move_to(ORIGIN)

        with self.voiceover(
            text="總結一下今日學咗嘅重點。"
            "第一，SFT 模型唔識分好壞，需要偏好信號嚟對齊。"
            "第二，RLHF 有三個階段，由 SFT 到 Reward Model 到 PPO。"
            "第三，Reward Model 用 Bradley-Terry loss 學識俾回答評分。"
            "第四，DPO 係一個突破，跳過 Reward Model 同 PPO，"
            "直接用一個簡單嘅 loss function 做偏好對齊。"
            "第五，KL penalty 防止模型偏離太遠，避免災難性遺忘。"
            "第六，beta 參數控制對齊嘅力度，通常設 0.1 到 0.5。"
            "掌握咗呢啲，你就可以對齊自己嘅 LLM 喇！"
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
            "下一課：Quantization（量化同推理優化）",
            font_size=28, color=C_CYAN,
        ).next_to(thanks, DOWN, buff=0.6)

        with self.voiceover(
            text="多謝收睇！下一課我哋會學 Quantization。"
            "即係點樣將模型壓縮到更細，推理更快。"
            "包括 INT8、INT4 量化、"
            "GPTQ、AWQ 同 GGUF 格式。"
            "記得繼續跟住學喇！"
        ):
            self.play(Write(thanks), run_time=1)
            self.play(FadeIn(next_lesson, shift=UP * 0.2), run_time=0.8)

        self.wait(1)
        self.play(FadeOut(thanks), FadeOut(next_lesson), run_time=0.8)
