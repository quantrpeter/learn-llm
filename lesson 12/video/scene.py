"""
Lesson 12 – Supervised Fine-Tuning (SFT)
Manim CE + Cantonese voiceover (Edge TTS)

Render (1080p):
    cd "lesson 12/video"
    manim render -qh scene.py SFTExplainer

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


class SFTExplainer(VoiceoverScene):
    """Eight scenes explaining Supervised Fine-Tuning in Cantonese."""

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
        self.scene_base_vs_chat()
        self.scene_chat_template()
        self.scene_sft_training()
        self.scene_lora()
        self.scene_qlora()
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
        title = self.zh("監督微調", font_size=56, color=C_BLUE)
        sub_en = self.en(
            "Supervised Fine-Tuning (SFT)", font_size=24, color=C_WHITE
        ).next_to(title, DOWN, buff=0.4)
        sub_zh = self.zh(
            "第十二課", font_size=36, color=C_ORANGE
        ).next_to(sub_en, DOWN, buff=0.3)
        producer = self.zh(
            "制片人：Peter", font_size=24, color=C_DIM
        ).next_to(sub_zh, DOWN, buff=0.7)

        with self.voiceover(
            text="大家好！歡迎嚟到第十二課。"
            "上幾課我哋學咗點樣預訓練、生成文本、同評估模型。"
            "但係一個 base model 唔識聽指令，唔識對話。"
            "今日我哋會學 Supervised Fine-Tuning，"
            "即係點樣將一個 base model 變成 ChatGPT 咁嘅對話助手。"
            "我哋會講 chat template、loss masking、"
            "LoRA 同 QLoRA 呢啲核心技術。"
        ):
            self.play(Write(title), run_time=1.5)
            self.play(FadeIn(sub_en, shift=UP * 0.2), run_time=0.8)
            self.play(FadeIn(sub_zh, shift=UP * 0.2), run_time=0.8)
            self.play(FadeIn(producer, shift=UP * 0.2), run_time=0.5)

        self.wait(0.3)
        self.clear()

    # ── Scene 2 — Base vs Chat (side-by-side) ────────────────────────────

    def scene_base_vs_chat(self):
        heading = self.make_heading("Base Model vs Chat Model")

        # Prompt at top center
        prompt_box = RoundedRectangle(
            corner_radius=0.15, width=9, height=0.8,
            fill_color=C_YELLOW, fill_opacity=0.12,
            stroke_color=C_YELLOW, stroke_width=2,
        ).shift(UP * 1.8)
        prompt_txt = self.mono(
            'Prompt: "What is the capital of France?"',
            font_size=18, color=C_YELLOW,
        ).move_to(prompt_box)
        prompt = VGroup(prompt_box, prompt_txt)

        # Left: Base model response
        base_rect = RoundedRectangle(
            corner_radius=0.15, width=5.2, height=3.0,
            fill_color=C_RED, fill_opacity=0.08,
            stroke_color=C_RED, stroke_width=2,
        )
        base_title = self.zh(
            "Base Model", font_size=24, color=C_RED
        ).next_to(base_rect, UP, buff=0.15)
        base_lines = VGroup(
            self.mono("What is the capital of", font_size=14, color=C_DIM),
            self.mono("Germany? Berlin.", font_size=14, color=C_WHITE),
            self.mono("What is the capital of", font_size=14, color=C_DIM),
            self.mono("Japan? Tokyo.", font_size=14, color=C_WHITE),
            self.mono("What is the largest...", font_size=14, color=C_DIM),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.12).move_to(base_rect)
        base_tag = self.zh(
            "只係繼續文本", font_size=16, color=C_RED
        ).next_to(base_rect, DOWN, buff=0.15)
        base_group = VGroup(base_rect, base_title, base_lines, base_tag)

        # Right: Chat model response
        chat_rect = RoundedRectangle(
            corner_radius=0.15, width=5.2, height=3.0,
            fill_color=C_GREEN, fill_opacity=0.08,
            stroke_color=C_GREEN, stroke_width=2,
        )
        chat_title = self.zh(
            "Chat Model (SFT)", font_size=24, color=C_GREEN
        ).next_to(chat_rect, UP, buff=0.15)
        chat_lines = VGroup(
            self.mono("The capital of France", font_size=14, color=C_GREEN),
            self.mono("is Paris. It is the", font_size=14, color=C_GREEN),
            self.mono("largest city and has", font_size=14, color=C_GREEN),
            self.mono("been the capital since", font_size=14, color=C_GREEN),
            self.mono("the 10th century.", font_size=14, color=C_GREEN),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.12).move_to(chat_rect)
        chat_tag = self.zh(
            "直接回答問題", font_size=16, color=C_GREEN
        ).next_to(chat_rect, DOWN, buff=0.15)
        chat_group = VGroup(chat_rect, chat_title, chat_lines, chat_tag)

        both = VGroup(base_group, chat_group).arrange(RIGHT, buff=0.6)
        both.shift(DOWN * 0.4)

        # Arrow from base to chat
        arrow = Arrow(
            base_rect.get_right(), chat_rect.get_left(),
            buff=0.15, color=C_YELLOW, stroke_width=3,
        )
        arrow_label = self.zh(
            "SFT", font_size=22, color=C_YELLOW
        ).next_to(arrow, UP, buff=0.1)

        with self.voiceover(
            text="首先睇下 base model 同 chat model 嘅分別。"
            "同一個 prompt，問法國嘅首都係邊度。"
            "Base model 唔會直接答你，佢只係繼續生成文本。"
            "佢可能會生成更多嘅地理問題，因為訓練數據入面有好多呢類文本。"
            "但係經過 SFT 之後嘅 chat model，"
            "會直接回答你：法國嘅首都係巴黎。"
            "呢個就係 SFT 嘅作用。"
            "佢教識模型：收到指令之後，要回答，唔係繼續寫文章。"
        ):
            self.play(Write(heading), run_time=0.6)
            self.play(FadeIn(prompt, shift=DOWN * 0.2), run_time=0.5)
            self.play(FadeIn(base_rect), FadeIn(base_title), run_time=0.4)
            for line in base_lines:
                self.play(FadeIn(line, shift=RIGHT * 0.1), run_time=0.25)
            self.play(FadeIn(base_tag), run_time=0.3)
            self.play(
                GrowArrow(arrow), FadeIn(arrow_label), run_time=0.5,
            )
            self.play(FadeIn(chat_rect), FadeIn(chat_title), run_time=0.4)
            for line in chat_lines:
                self.play(FadeIn(line, shift=RIGHT * 0.1), run_time=0.25)
            self.play(FadeIn(chat_tag), run_time=0.3)

        self.wait(0.3)
        self.clear()

    # ── Scene 3 — Chat Template (message boxes with roles) ───────────────

    def scene_chat_template(self):
        heading = self.make_heading("Chat Template 對話模板")

        # Message boxes stacked vertically
        def msg_box(role, content, role_color, width=9.0):
            rect = RoundedRectangle(
                corner_radius=0.15, width=width, height=0.75,
                fill_color=role_color, fill_opacity=0.12,
                stroke_color=role_color, stroke_width=2,
            )
            role_lbl = self.mono(
                role, font_size=16, color=role_color
            ).move_to(rect).align_to(rect, LEFT).shift(RIGHT * 0.25)
            cont = self.mono(
                content, font_size=14, color=C_WHITE
            ).move_to(rect).shift(RIGHT * 0.6)
            return VGroup(rect, role_lbl, cont)

        sys_box = msg_box("system   ", "You are a helpful assistant.", C_PURPLE)
        usr_box = msg_box("user     ", "What is 2 + 2?", C_BLUE)
        ast_box = msg_box("assistant", "2 + 2 = 4.", C_GREEN)

        messages = VGroup(sys_box, usr_box, ast_box).arrange(
            DOWN, buff=0.25
        ).shift(UP * 0.3)

        # Template tokens on the right
        token_col = VGroup(
            self.mono("<|im_start|>system", font_size=12, color=C_PURPLE),
            self.mono("You are a helpful...", font_size=12, color=C_DIM),
            self.mono("<|im_end|>", font_size=12, color=C_DIM),
            self.mono("<|im_start|>user", font_size=12, color=C_BLUE),
            self.mono("What is 2 + 2?", font_size=12, color=C_DIM),
            self.mono("<|im_end|>", font_size=12, color=C_DIM),
            self.mono("<|im_start|>assistant", font_size=12, color=C_GREEN),
            self.mono("2 + 2 = 4.", font_size=12, color=C_GREEN),
            self.mono("<|im_end|>", font_size=12, color=C_DIM),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.08)

        token_title = self.zh(
            "ChatML Tokens", font_size=18, color=C_YELLOW
        )

        token_group = VGroup(token_title, token_col).arrange(DOWN, buff=0.2)
        token_group.to_edge(RIGHT, buff=0.4).shift(DOWN * 0.3)

        # Shrink messages to make room
        messages.shift(LEFT * 1.8).scale(0.85)

        # Connecting arrow
        conn_arrow = Arrow(
            messages.get_right() + RIGHT * 0.1,
            token_group.get_left() + LEFT * 0.1,
            buff=0.1, color=C_YELLOW, stroke_width=2,
        )

        # Bottom note
        note_rect = RoundedRectangle(
            corner_radius=0.12, width=11, height=0.6,
            fill_color=C_ORANGE, fill_opacity=0.12,
            stroke_color=C_ORANGE, stroke_width=1.5,
        )
        note_txt = self.zh(
            "模板錯配 = 模型唔知邊度係指令、邊度係回答 → 輸出混亂",
            font_size=18, color=C_ORANGE,
        ).move_to(note_rect)
        note = VGroup(note_rect, note_txt).to_edge(DOWN, buff=0.35)

        with self.voiceover(
            text="Chat template 定義咗點樣將對話編碼成 token 序列。"
            "每段對話有三個角色。"
            "System 定義助手嘅行為。"
            "User 係用戶嘅輸入。"
            "Assistant 係模型要學嘅回應。"
            "呢啲角色會用特殊 token 包裝成 ChatML 格式。"
            "例如 im_start system、im_end 呢啲標記。"
            "模型本身見到嘅只係 token，"
            "靠 template 先識分邊啲係指令、邊啲係回答。"
            "所以推理嘅時候一定要用同訓練時一樣嘅 template。"
            "Template 錯配係 fine-tune 之後模型「唔 work」嘅最常見原因。"
        ):
            self.play(Write(heading), run_time=0.6)
            self.play(FadeIn(sys_box, shift=RIGHT * 0.2), run_time=0.5)
            self.play(FadeIn(usr_box, shift=RIGHT * 0.2), run_time=0.5)
            self.play(FadeIn(ast_box, shift=RIGHT * 0.2), run_time=0.5)
            self.play(GrowArrow(conn_arrow), run_time=0.4)
            self.play(FadeIn(token_group, shift=LEFT * 0.2), run_time=0.8)
            self.play(FadeIn(note, shift=UP * 0.2), run_time=0.6)

        self.wait(0.3)
        self.clear()

    # ── Scene 4 — SFT Training (loss masking visualization) ──────────────

    def scene_sft_training(self):
        heading = self.make_heading("SFT Loss Masking")

        # Token strip
        tok_labels = [
            "SYS", "You", "are", "...",
            "USR", "What", "is", "2+2",
            "AST", "The", "ans", "is", "4", "."
        ]
        tok_colors = (
            [C_PURPLE] * 4 + [C_BLUE] * 4 + [C_GREEN] * 6
        )
        is_train = (
            [False] * 4 + [False] * 4 + [True] * 6
        )

        token_boxes = VGroup()
        for i, (label, color) in enumerate(zip(tok_labels, tok_colors)):
            rect = RoundedRectangle(
                corner_radius=0.08, width=0.78, height=0.55,
                fill_color=color, fill_opacity=0.3 if is_train[i] else 0.08,
                stroke_color=color,
                stroke_width=2.5 if is_train[i] else 1.0,
            )
            txt = self.mono(label, font_size=11, color=color).move_to(rect)
            token_boxes.add(VGroup(rect, txt))
        token_boxes.arrange(RIGHT, buff=0.06).shift(UP * 1.0)

        # Loss row below tokens
        loss_labels = VGroup()
        for i, train in enumerate(is_train):
            if train:
                lbl = self.mono("CE", font_size=10, color=C_GREEN)
            else:
                lbl = self.mono("-100", font_size=10, color=C_DIM)
            lbl.next_to(token_boxes[i], DOWN, buff=0.15)
            loss_labels.add(lbl)

        loss_title = self.zh(
            "Loss:", font_size=18, color=C_WHITE
        ).next_to(loss_labels, LEFT, buff=0.3)

        # Gradient arrows on assistant tokens only
        grad_arrows = VGroup()
        for i, train in enumerate(is_train):
            if train:
                arr = Arrow(
                    token_boxes[i].get_bottom() + DOWN * 0.6,
                    token_boxes[i].get_bottom() + DOWN * 0.1,
                    buff=0, color=C_YELLOW, stroke_width=2,
                    max_tip_length_to_length_ratio=0.3,
                )
                grad_arrows.add(arr)

        grad_label = self.zh(
            "梯度只流過 assistant tokens",
            font_size=18, color=C_YELLOW,
        ).next_to(grad_arrows, DOWN, buff=0.2)

        # Legend at bottom
        legend = VGroup(
            VGroup(
                RoundedRectangle(
                    corner_radius=0.05, width=0.4, height=0.3,
                    fill_color=C_PURPLE, fill_opacity=0.3,
                    stroke_color=C_PURPLE,
                ),
                self.zh("System", font_size=14, color=C_PURPLE),
            ).arrange(RIGHT, buff=0.1),
            VGroup(
                RoundedRectangle(
                    corner_radius=0.05, width=0.4, height=0.3,
                    fill_color=C_BLUE, fill_opacity=0.3,
                    stroke_color=C_BLUE,
                ),
                self.zh("User", font_size=14, color=C_BLUE),
            ).arrange(RIGHT, buff=0.1),
            VGroup(
                RoundedRectangle(
                    corner_radius=0.05, width=0.4, height=0.3,
                    fill_color=C_GREEN, fill_opacity=0.3,
                    stroke_color=C_GREEN,
                ),
                self.zh("Assistant (計算 loss)", font_size=14, color=C_GREEN),
            ).arrange(RIGHT, buff=0.1),
        ).arrange(RIGHT, buff=0.6).to_edge(DOWN, buff=0.4)

        with self.voiceover(
            text="SFT 用同 pre-training 一樣嘅 cross-entropy loss。"
            "但有一個關鍵分別：loss masking。"
            "呢一行就係一段對話嘅 token 序列。"
            "紫色係 system，藍色係 user，綠色係 assistant。"
            "我哋只計算 assistant token 嘅 loss。"
            "System 同 user 嘅 token 設為負 100，"
            "PyTorch 嘅 CrossEntropyLoss 會自動忽略佢哋。"
            "梯度只流過 assistant token。"
            "咁模型每次更新都係學緊點樣回答得更好，"
            "而唔係浪費容量去模仿用戶嘅問題。"
            "呢個係 SFT 最重要嘅實現細節。"
        ):
            self.play(Write(heading), run_time=0.6)
            for tb in token_boxes:
                self.play(FadeIn(tb, shift=DOWN * 0.1), run_time=0.12)
            self.play(FadeIn(loss_title), run_time=0.3)
            for ll in loss_labels:
                self.play(FadeIn(ll), run_time=0.08)
            for ga in grad_arrows:
                self.play(GrowArrow(ga), run_time=0.15)
            self.play(FadeIn(grad_label, shift=UP * 0.1), run_time=0.5)
            self.play(FadeIn(legend, shift=UP * 0.2), run_time=0.5)

        self.wait(0.3)
        self.clear()

    # ── Scene 5 — LoRA (W + AB decomposition) ────────────────────────────

    def scene_lora(self):
        heading = self.make_heading("LoRA: Low-Rank Adaptation")

        # Large W matrix
        w_rect = RoundedRectangle(
            corner_radius=0.15, width=3.0, height=3.0,
            fill_color=C_BLUE, fill_opacity=0.15,
            stroke_color=C_BLUE, stroke_width=2,
        )
        w_label = self.mono("W", font_size=36, color=C_BLUE).move_to(w_rect)
        w_dim = self.mono(
            "d x d", font_size=16, color=C_DIM
        ).next_to(w_rect, DOWN, buff=0.15)
        w_frozen = self.zh(
            "凍結", font_size=16, color=C_RED
        ).next_to(w_dim, DOWN, buff=0.1)
        w_group = VGroup(w_rect, w_label, w_dim, w_frozen)

        # Plus sign
        plus = self.mono("+", font_size=40, color=C_YELLOW)

        # B matrix (tall and thin)
        b_rect = RoundedRectangle(
            corner_radius=0.1, width=0.8, height=3.0,
            fill_color=C_GREEN, fill_opacity=0.2,
            stroke_color=C_GREEN, stroke_width=2,
        )
        b_label = self.mono("B", font_size=28, color=C_GREEN).move_to(b_rect)
        b_dim = self.mono(
            "d x r", font_size=14, color=C_DIM
        ).next_to(b_rect, DOWN, buff=0.15)
        b_init = self.zh(
            "初始=0", font_size=13, color=C_GREEN
        ).next_to(b_dim, DOWN, buff=0.08)
        b_group = VGroup(b_rect, b_label, b_dim, b_init)

        # Multiply sign
        times = self.mono("@", font_size=28, color=C_DIM)

        # A matrix (short and wide)
        a_rect = RoundedRectangle(
            corner_radius=0.1, width=3.0, height=0.8,
            fill_color=C_ORANGE, fill_opacity=0.2,
            stroke_color=C_ORANGE, stroke_width=2,
        )
        a_label = self.mono("A", font_size=28, color=C_ORANGE).move_to(a_rect)
        a_dim = self.mono(
            "r x d", font_size=14, color=C_DIM
        ).next_to(a_rect, DOWN, buff=0.15)
        a_init = self.zh(
            "初始=隨機", font_size=13, color=C_ORANGE
        ).next_to(a_dim, DOWN, buff=0.08)
        a_group = VGroup(a_rect, a_label, a_dim, a_init)

        # Layout: W + B @ A
        ba_group = VGroup(b_group, times, a_group).arrange(RIGHT, buff=0.2)
        equation = VGroup(w_group, plus, ba_group).arrange(RIGHT, buff=0.5)
        equation.shift(UP * 0.3)

        # Equals and result label
        eq_sign = self.mono("=  W'", font_size=32, color=C_YELLOW).next_to(
            equation, RIGHT, buff=0.4
        )

        # Parameter savings box at bottom
        savings_rect = RoundedRectangle(
            corner_radius=0.12, width=10.5, height=1.5,
            fill_color=C_CYAN, fill_opacity=0.08,
            stroke_color=C_CYAN, stroke_width=1.5,
        )
        savings_lines = VGroup(
            self.mono(
                "d=4096, r=16:", font_size=18, color=C_CYAN
            ),
            self.mono(
                "Original: 4096x4096 = 16.8M params",
                font_size=16, color=C_RED,
            ),
            self.mono(
                "LoRA:     4096x16 + 16x4096 = 131K params (0.8%)",
                font_size=16, color=C_GREEN,
            ),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.12).move_to(savings_rect)
        savings = VGroup(savings_rect, savings_lines).to_edge(DOWN, buff=0.35)

        with self.voiceover(
            text="LoRA 係最受歡迎嘅高效微調方法。"
            "核心諗法好簡單。"
            "原始權重矩陣 W 係 d 乘 d，非常大。"
            "我哋凍結 W，唔更新佢。"
            "然後加兩個細嘅矩陣 B 同 A。"
            "B 係 d 乘 r，A 係 r 乘 d。"
            "r 叫做 rank，通常只有 8 到 32。"
            "新嘅權重 W prime 等於 W 加 B 乘 A。"
            "初始化嗰陣，B 係全零，A 係隨機。"
            "因為 B 係零，所以 B 乘 A 等於零。"
            "訓練開始嗰陣，W prime 同 W 完全一樣，唔會破壞原有嘅能力。"
            "以 d 等於 4096 為例，原始有 1680 萬參數。"
            "LoRA rank 16 只有 13 萬參數，只係原來嘅百分之零點八。"
            "記憶體大幅節省，效果仲好接近 full fine-tune。"
        ):
            self.play(Write(heading), run_time=0.6)
            self.play(FadeIn(w_group, shift=RIGHT * 0.2), run_time=0.6)
            self.play(FadeIn(plus), run_time=0.3)
            self.play(FadeIn(b_group, shift=DOWN * 0.2), run_time=0.5)
            self.play(FadeIn(times), run_time=0.2)
            self.play(FadeIn(a_group, shift=UP * 0.2), run_time=0.5)
            self.play(FadeIn(eq_sign, shift=LEFT * 0.2), run_time=0.4)
            self.play(FadeIn(savings, shift=UP * 0.2), run_time=0.7)

        self.wait(0.3)
        self.clear()

    # ── Scene 6 — QLoRA (quantized + LoRA stack) ─────────────────────────

    def scene_qlora(self):
        heading = self.make_heading("QLoRA: 4-bit 底座 + LoRA")

        # Stack: three approaches as bars
        bar_w = 3.0

        # Full FT bar (tallest)
        full_bar = Rectangle(
            width=bar_w, height=4.5,
            fill_color=C_RED, fill_opacity=0.25,
            stroke_color=C_RED, stroke_width=2,
        )
        full_label = self.mono("Full FT", font_size=20, color=C_RED)
        full_label.next_to(full_bar, UP, buff=0.1)
        full_mem = self.mono("112 GB", font_size=22, color=C_RED).move_to(full_bar)
        full_note = self.zh(
            "8x A100", font_size=14, color=C_DIM
        ).next_to(full_bar, DOWN, buff=0.1)

        # LoRA bar (medium)
        lora_bar = Rectangle(
            width=bar_w, height=1.8,
            fill_color=C_ORANGE, fill_opacity=0.25,
            stroke_color=C_ORANGE, stroke_width=2,
        )
        lora_label = self.mono("LoRA", font_size=20, color=C_ORANGE)
        lora_label.next_to(lora_bar, UP, buff=0.1)
        lora_mem = self.mono("16 GB", font_size=22, color=C_ORANGE).move_to(lora_bar)
        lora_note = self.zh(
            "1x A100", font_size=14, color=C_DIM
        ).next_to(lora_bar, DOWN, buff=0.1)

        # QLoRA bar (smallest) — split into two sections
        qlora_bar_base = Rectangle(
            width=bar_w, height=0.55,
            fill_color=C_PURPLE, fill_opacity=0.3,
            stroke_color=C_PURPLE, stroke_width=2,
        )
        qlora_bar_lora = Rectangle(
            width=bar_w, height=0.25,
            fill_color=C_GREEN, fill_opacity=0.3,
            stroke_color=C_GREEN, stroke_width=2,
        )
        qlora_stack = VGroup(qlora_bar_base, qlora_bar_lora).arrange(UP, buff=0)
        qlora_label = self.mono("QLoRA", font_size=20, color=C_GREEN)
        qlora_label.next_to(qlora_stack, UP, buff=0.1)
        qlora_mem = self.mono("6 GB", font_size=22, color=C_GREEN).move_to(qlora_stack)
        base_tag = self.mono(
            "4-bit", font_size=12, color=C_PURPLE
        ).move_to(qlora_bar_base)
        lora_tag = self.mono(
            "FP16", font_size=10, color=C_GREEN
        ).move_to(qlora_bar_lora)
        qlora_note = self.zh(
            "消費級 GPU", font_size=14, color=C_DIM
        ).next_to(qlora_stack, DOWN, buff=0.1)

        full_group = VGroup(full_bar, full_label, full_mem, full_note)
        lora_group = VGroup(lora_bar, lora_label, lora_mem, lora_note)
        qlora_group = VGroup(
            qlora_stack, qlora_label, qlora_mem, base_tag, lora_tag, qlora_note
        )

        bars = VGroup(full_group, lora_group, qlora_group).arrange(
            RIGHT, buff=1.2, aligned_edge=DOWN
        ).shift(DOWN * 0.3)

        # Savings arrows
        arrow1 = Arrow(
            full_bar.get_right() + RIGHT * 0.1,
            lora_bar.get_left() + LEFT * 0.1,
            buff=0.05, color=C_YELLOW, stroke_width=2.5,
        ).shift(UP * 0.3)
        lbl1 = self.mono(
            "7x", font_size=20, color=C_YELLOW
        ).next_to(arrow1, UP, buff=0.1)

        arrow2 = Arrow(
            lora_bar.get_right() + RIGHT * 0.1,
            qlora_stack.get_left() + LEFT * 0.1,
            buff=0.05, color=C_YELLOW, stroke_width=2.5,
        ).shift(UP * 0.1)
        lbl2 = self.mono(
            "2.7x", font_size=20, color=C_YELLOW
        ).next_to(arrow2, UP, buff=0.1)

        # Bottom key innovations
        innovations = VGroup(
            self.zh("NF4：專為正態分佈設計嘅 4-bit 格式", font_size=18, color=C_PURPLE),
            self.zh("Double Quant：量化常數都量化", font_size=18, color=C_CYAN),
            self.zh("Paged Optimizer：OOM 時自動用 CPU", font_size=18, color=C_ORANGE),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.15).to_edge(DOWN, buff=0.35)

        with self.voiceover(
            text="QLoRA 將 LoRA 再推進一步。"
            "Full fine-tune 一個 7B 模型要 112 GB，需要 8 張 A100。"
            "LoRA 減到 16 GB，一張 A100 搞掂。"
            "QLoRA 再減到 6 GB。"
            "點做到嘅呢？"
            "既然 base weights 係凍結嘅，我哋可以量化佢。"
            "用 NF4 格式壓縮到 4-bit，記憶體減四倍。"
            "LoRA adapters 繼續用 FP16 訓練，保持精度。"
            "NF4 專為正態分佈設計，因為神經網絡嘅權重近似正態分佈。"
            "Double quantization 連量化常數都壓縮。"
            "Paged optimizer 喺 GPU 記憶體不足嗰陣，自動用 CPU 幫手。"
            "QLoRA 將成本降低咗大約 50 倍，令到普通人都可以 fine-tune 大模型。"
        ):
            self.play(Write(heading), run_time=0.6)
            self.play(
                FadeIn(full_bar), FadeIn(full_label),
                FadeIn(full_mem), FadeIn(full_note),
                run_time=0.6,
            )
            self.play(
                GrowArrow(arrow1), FadeIn(lbl1), run_time=0.4,
            )
            self.play(
                FadeIn(lora_bar), FadeIn(lora_label),
                FadeIn(lora_mem), FadeIn(lora_note),
                run_time=0.6,
            )
            self.play(
                GrowArrow(arrow2), FadeIn(lbl2), run_time=0.4,
            )
            self.play(
                FadeIn(qlora_stack), FadeIn(qlora_label),
                FadeIn(qlora_mem), FadeIn(base_tag), FadeIn(lora_tag),
                FadeIn(qlora_note),
                run_time=0.6,
            )
            for inn in innovations:
                self.play(FadeIn(inn, shift=RIGHT * 0.2), run_time=0.35)

        self.wait(0.3)
        self.clear()

    # ── Scene 7 — Summary ────────────────────────────────────────────────

    def scene_summary(self):
        heading = self.make_heading("總結", font_size=44)

        bullets = VGroup(
            self.zh("•  Base model 只識補完文本，唔識聽指令", font_size=24, color=C_RED),
            self.zh("•  Chat template 將角色資訊編碼成 token 序列", font_size=24, color=C_PURPLE),
            self.zh("•  SFT loss 只計算 assistant token — 最重要嘅細節", font_size=24, color=C_GREEN),
            self.zh("•  LoRA：W' = W + BA，訓練 <1% 參數", font_size=24, color=C_ORANGE),
            self.zh("•  QLoRA：4-bit 底座 + LoRA，6 GB 微調 7B 模型", font_size=24, color=C_CYAN),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.4).move_to(ORIGIN)

        with self.voiceover(
            text="總結一下今日學咗嘅五個重點。"
            "第一，base model 只識補完文本，唔識回答問題。"
            "第二，chat template 定義咗角色邊界，推理時一定要同訓練時一致。"
            "第三，SFT loss 只計算 assistant token，"
            "呢個係最重要嘅實現細節。"
            "第四，LoRA 用低秩分解，只訓練不到百分之一嘅參數。"
            "第五，QLoRA 將 base weights 量化到 4-bit，"
            "6 GB 就可以微調一個 7B 模型。"
            "掌握咗呢啲技術，你就可以將任何 base model 變成你自己嘅助手。"
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
            "下一課：Alignment（偏好對齊）",
            font_size=28, color=C_CYAN,
        ).next_to(thanks, DOWN, buff=0.6)

        with self.voiceover(
            text="多謝收睇！下一課我哋會學 Alignment，"
            "即係偏好對齊。"
            "包括 RLHF、Reward Model、"
            "同更簡單嘅 DPO 方法。"
            "目標係令你嘅助手更有用、更安全。"
            "記得繼續跟住學喇！"
        ):
            self.play(Write(thanks), run_time=1)
            self.play(FadeIn(next_lesson, shift=UP * 0.2), run_time=0.8)

        self.wait(1)
        self.play(FadeOut(thanks), FadeOut(next_lesson), run_time=0.8)
