#!/usr/bin/env python3
"""Build ICAITech 2026 deck on top of the official template backgrounds."""
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

NAVY = RGBColor(0x14, 0x3B, 0x65)
BLACK = RGBColor(0x1A, 0x1A, 0x1A)
GRAY = RGBColor(0x59, 0x59, 0x59)
ORANGE = RGBColor(0xC0, 0x55, 0x00)
LIGHT = RGBColor(0xED, 0xF2, 0xF7)
BAND = RGBColor(0xF5, 0xF7, 0xFA)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
GOLD = RGBColor(0xB7, 0x86, 0x0B)

BASE = '/Users/wedjaw/project/kws-pi5-benchmark/deck-icaitech'
BG_TITLE = f'{BASE}/bg-title.jpg'
BG_CONTENT = f'{BASE}/bg-content.jpg'

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]
TOTAL = 14


def new_slide(bg):
    s = prs.slides.add_slide(BLANK)
    s.shapes.add_picture(bg, Inches(0), Inches(0),
                         Inches(13.333), Inches(7.5))
    return s


def add_number(s, n):
    box = s.shapes.add_textbox(Inches(12.15), Inches(6.18),
                               Inches(0.85), Inches(0.3))
    tf = box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.RIGHT
    r = p.add_run()
    r.text = f'{n} / {TOTAL}'
    r.font.size = Pt(12)
    r.font.color.rgb = GRAY
    r.font.name = 'Calibri'


def add_title(s, text, size=38):
    box = s.shapes.add_textbox(Inches(1.55), Inches(1.62),
                               Inches(10.9), Inches(0.95))
    tf = box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    r = p.add_run()
    r.text = text
    r.font.size = Pt(size)
    r.font.bold = True
    r.font.color.rgb = NAVY
    r.font.name = 'Calibri'


def add_bullets(s, items, left=1.55, top=2.72, width=10.9, size=22,
                gap=6, color=BLACK, bold_prefix=False):
    box = s.shapes.add_textbox(Inches(left), Inches(top),
                               Inches(width), Inches(3.4))
    tf = box.text_frame
    tf.word_wrap = True
    first = True
    for it in items:
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.space_after = Pt(gap)
        p.space_before = Pt(0)
        if isinstance(it, tuple):
            head, rest = it
            rh = p.add_run()
            rh.text = head
            rh.font.size = Pt(size)
            rh.font.bold = True
            rh.font.color.rgb = NAVY
            rh.font.name = 'Calibri'
            rr = p.add_run()
            rr.text = rest
            rr.font.size = Pt(size)
            rr.font.color.rgb = color
            rr.font.name = 'Calibri'
        else:
            r = p.add_run()
            r.text = '\u2022  ' + it
            r.font.size = Pt(size)
            r.font.color.rgb = color
            r.font.name = 'Calibri'
    return box


def add_table(s, headers, rows, left, top, width, col_widths=None,
              size=16, header_size=None):
    nrows = 1 + len(rows)
    ncols = len(headers)
    gf = s.shapes.add_table(nrows, ncols, Inches(left), Inches(top),
                            Inches(width), Inches(0.5))
    tbl = gf.table
    if col_widths:
        for i, w in enumerate(col_widths):
            tbl.columns[i].width = Inches(w)
    hs = header_size or size
    for j, h in enumerate(headers):
        c = tbl.cell(0, j)
        c.fill.solid()
        c.fill.fore_color.rgb = NAVY
        c.vertical_anchor = 1  # middle
        p = c.text_frame.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        r = p.add_run()
        r.text = h
        r.font.size = Pt(hs)
        r.font.bold = True
        r.font.color.rgb = WHITE
        r.font.name = 'Calibri'
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            c = tbl.cell(i + 1, j)
            if i % 2 == 1:
                c.fill.solid()
                c.fill.fore_color.rgb = BAND
            c.vertical_anchor = 1
            p = c.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.CENTER if j > 0 else PP_ALIGN.LEFT
            r = p.add_run()
            r.text = str(val)
            r.font.size = Pt(size)
            r.font.color.rgb = BLACK
            r.font.name = 'Calibri'
    return gf


def add_note(s, text, top=5.30, size=19):
    box = s.shapes.add_textbox(Inches(1.55), Inches(top),
                               Inches(10.9), Inches(0.6))
    tf = box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    r = p.add_run()
    r.text = text
    r.font.size = Pt(size)
    r.font.bold = True
    r.font.color.rgb = ORANGE
    r.font.name = 'Calibri'


# ============ 1. TITLE ============
s = new_slide(BG_TITLE)
box = s.shapes.add_textbox(Inches(1.2), Inches(1.55), Inches(10.93), Inches(1.9))
tf = box.text_frame
tf.word_wrap = True
for i, (t, sz) in enumerate([
        ('Benchmarking Keyword Spotting Architectures', 40),
        ('across TensorFlow Lite and ONNX Runtime on a Raspberry Pi 5', 32)]):
    p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = t
    r.font.size = Pt(sz)
    r.font.bold = True
    r.font.color.rgb = NAVY
    r.font.name = 'Calibri'

box = s.shapes.add_textbox(Inches(1.2), Inches(3.55), Inches(10.93), Inches(0.7))
tf = box.text_frame
tf.word_wrap = True
p = tf.paragraphs[0]
p.alignment = PP_ALIGN.CENTER
for txt, bold, ul in [('Fatih Jawwad Al Mumtaz', True, True),
                      ('  \u00b7  Amirul Al Aziz', False, False)]:
    r = p.add_run()
    r.text = txt
    r.font.size = Pt(26)
    r.font.bold = bold
    r.font.underline = ul
    r.font.color.rgb = BLACK
    r.font.name = 'Calibri'

box = s.shapes.add_textbox(Inches(1.2), Inches(4.25), Inches(10.93), Inches(0.5))
tf = box.text_frame
tf.word_wrap = True
p = tf.paragraphs[0]
p.alignment = PP_ALIGN.CENTER
r = p.add_run()
r.text = 'Faculty of Science and Technology, Universitas Darussalam Gontor, Indonesia'
r.font.size = Pt(20)
r.font.color.rgb = BLACK
r.font.name = 'Calibri'

add_number(s, 1)

# ============ 2. OUTLINE ============
s = new_slide(BG_CONTENT)
add_title(s, 'Outline')
add_bullets(s, [
    ('1  Background & motivation  ', '\u2014 why runtime choice matters for always-on voice'),
    ('2  Models & method  ', '\u2014 four architectures, one frontend, rigorous protocol'),
    ('3  Results  ', '\u2014 accuracy, latency, threads, energy, and the trade-off'),
    ('4  Conclusion  ', '\u2014 what to pick, and why habit is not evidence'),
], size=24, gap=14)
add_number(s, 2)

# ============ 3. MOTIVATION ============
s = new_slide(BG_CONTENT)
add_title(s, 'Why benchmark KWS on the Pi 5?')
add_bullets(s, [
    'Keyword spotting listens all day for a word or two \u2014 on a small board that means sub-millisecond latency and milliwatt power, with no internet to fall back on.',
    ('Everyone assumes TensorFlow Lite is fastest on Arm. ',
     'That assumption is rarely tested head-to-head on 64-bit Arm, across CNNs, dense nets, and RNNs.'),
    'On-device voice also matters for care settings: no cloud means usable in low-connectivity clinics and no health-speech privacy risk (SDG 3).',
], size=22, gap=12)
add_note(s, 'Question: does the default runtime deserve its default status?')
add_number(s, 3)

# ============ 4. APPROACH ============
s = new_slide(BG_CONTENT)
add_title(s, 'What we test')
add_bullets(s, [
    ('Four models, three configs, one board: ',
     'DS-CNN (22.6k) \u00b7 DNN (88.4k) \u00b7 LSTM (20.0k) \u00b7 CRNN (56.8k) \u00d7 TFLite fp32 / TFLite int8 / ONNX fp32, all on a Raspberry Pi 5.'),
    ('Same everything else: ',
     'identical 49\u00d710 MFCC frontend, same 12-way head, same clips, same protocol.'),
    ('Three contributions: ',
     '(1) full training + export pipeline incl. fp32 reconstruction; (2) accuracy + latency on both tasks, plus thread sweep and PMIC energy; (3) proof the runtime gap does not generalize.'),
], size=20, gap=10)
add_number(s, 4)

# ============ 5. MODELS ============
s = new_slide(BG_CONTENT)
add_title(s, 'Four models, one frontend')
add_table(s,
          ['Model', 'Params', 'Structure'],
          [['DS-CNN (MLPerf ref.)', '22,604',
            '10\u00d74 conv (stride 2) + 4\u00d7 depthwise/pointwise blocks + GAP + dense'],
           ['DNN baseline', '88,396', '490\u2013128\u2013128\u201364\u201312 dense, dropout 0.1 (lower-bound anchor)'],
           ['LSTM', '19,980', 'LSTM-64 unrolled over 49 steps + dense'],
           ['CRNN', '56,844', '2\u00d7 3\u00d73 conv + 2\u00d7 maxpool + GRU-64 + dense']],
          1.55, 2.72, 10.9, col_widths=[2.6, 1.4, 6.9], size=17)
add_bullets(s, [
    'Same 49\u00d710 MFCC in, same 12-way softmax out. DNN is deliberately the anchor \u2014 dense-only is not competitive at this budget, its low score is expected, not a bug.',
], top=4.95, size=18)
add_number(s, 5)

# ============ 6. RECONSTRUCTION ============
s = new_slide(BG_CONTENT)
add_title(s, 'The reference artifact was not fp32')
add_bullets(s, [
    ('Found: ', 'the official MLPerf kws_ref_model artifact is mixed-precision \u2014 int8 conv kernels, fp32 rest. Benchmarking it as \u201cfp32\u201d would poison the comparison.'),
    ('Rebuilt: ', 'enumerated every tensor via the LiteRT API, dequantized kernels with \u0175 = (q \u2212 z)\u00b7s, reassembled 9 fused conv\u2013bias\u2013ReLU blocks + GAP + dense in Keras.'),
    ('Verified: ', 'max abs diff 7\u00d710\u207b\u00b3 on softmax, 100% argmax agreement over 1,000 frames \u2014 then exported clean fp32 TFLite + fp32 ONNX of identical structure.'),
], size=21, gap=12)
add_note(s, 'All three artifacts (original, rebuilt TFLite, rebuilt ONNX) are benchmarked.')
add_number(s, 6)

# ============ 7. QUANTIZATION ============
s = new_slide(BG_CONTENT)
add_title(s, 'Full-int8 quantization, done properly')
add_bullets(s, [
    'New full-int8 DS-CNN via post-training quantization, calibrated on 1,000 training clips (seed 42) \u2014 25% faster than the mixed-precision artifact at equal accuracy.',
    'Keras 3 emits dynamic tensor-list loops for RNNs that the TFLite converter cannot lower \u2014 exported with unroll=True, an identical-weight static graph both runtimes accept.',
    ('Cost is uneven: ', 'CNNs lose < 0.5 pt to int8; the LSTM loses 3.05 pts (subset) / 4.12 pts (full task) \u2014 recurrence wants quantization-aware training.'),
], size=21, gap=12)
add_number(s, 7)

# ============ 8. PROTOCOL ============
s = new_slide(BG_CONTENT)
add_title(s, 'How we measured')
add_bullets(s, [
    ('Board: ', 'Pi 5 (BCM2712, 4\u00d7 A76 @ 2.4 GHz), 8 GB RAM \u00b7 Debian 64-bit \u00b7 governor pinned to performance \u00b7 LiteRT 2.1.6 vs ONNX Runtime 1.28.'),
    ('Stats: ', '1,000 warmup + 1,000 timed passes, sequential, features precomputed \u00b7 IQR outlier filter \u00b7 10k bootstrap 95% CIs.'),
    ('Data: ', '4,074 clips (10 commands) + 12,005 clips (full 12-class) \u00b7 thread sweep at 1/2/4 threads \u00b7 PMIC energy (SoC only; DNN omitted, < 0.1 ms too fast to sample).'),
], size=21, gap=12)
add_number(s, 8)

# ============ 9. ACCURACY ============
s = new_slide(BG_CONTENT)
add_title(s, 'Accuracy holds up \u2014 except LSTM')
add_table(s,
          ['Model', 'fp32', 'int8', '\u0394', 'fp32', 'int8', '\u0394'],
          [['DS-CNN', '90.55%', '90.57%', '+0.02', '95.49%', '95.21%', '\u22120.28'],
           ['DNN', '79.63%', '79.65%', '+0.02', '72.65%', '71.90%', '\u22120.75'],
           ['LSTM', '90.48%', '87.43%', '\u22123.05', '88.84%', '84.72%', '\u22124.12'],
           ['CRNN', '92.56%', '92.17%', '\u22120.39', '92.08%', '91.60%', '\u22120.48']],
          1.55, 2.72, 10.9,
          col_widths=[2.2, 1.25, 1.25, 1.1, 1.25, 1.25, 1.1], size=17)
add_bullets(s, [
    'Left block: 10 commands (4,074 clips) \u00b7 right block: full 12-class (12,005 clips). TFLite fp32 and ONNX fp32 agree on all clips (diff < 3\u00d710\u207b\u2076) \u2014 runtimes are numerically interchangeable, speed is the whole story.',
], top=5.0, size=17)
add_number(s, 9)

# ============ 10. LATENCY ============
s = new_slide(BG_CONTENT)
add_title(s, 'Latency flips by architecture')
add_table(s,
          ['Model', 'TFLite fp32', 'TFLite int8', 'ONNX fp32', 'ONNX / TFLite'],
          [['DS-CNN', '0.211 ms', '0.127 ms', '0.170 ms', '0.81 \u2014 ONNX wins 19.2%'],
           ['DNN', '0.016 ms', '0.007 ms', '0.029 ms', '1.81\u00d7 \u2014 TFLite wins'],
           ['LSTM', '0.372 ms', '0.157 ms', '0.778 ms', '2.09\u00d7 \u2014 TFLite wins'],
           ['CRNN', '0.259 ms', '0.102 ms', '0.361 ms', '1.40\u00d7 \u2014 TFLite wins']],
          1.55, 2.72, 10.9,
          col_widths=[2.0, 1.9, 1.9, 1.9, 3.2], size=17)
add_note(s, 'int8 is 1.7\u20132.5\u00d7 faster almost for free \u00b7 ONNX wins only on depthwise convs.')
add_number(s, 10)

# ============ 11. THREADS + ENERGY ============
s = new_slide(BG_CONTENT)
add_title(s, 'Threads don\u2019t flip it \u2014 and faster means cooler')
add_table(s,
          ['Config', 'T=1', 'T=2', 'T=4'],
          [['DS-CNN TFLite', '0.212', '0.123', '0.078'],
           ['DS-CNN ONNX', '0.438', '0.259', '0.169'],
           ['CRNN TFLite', '0.260', '0.187', '0.220'],
           ['CRNN ONNX', '0.508', '0.419', '0.361']],
          1.55, 2.72, 5.2, size=15)
_t11b = add_table(s,
          ['Model', 'P\u0394 (W)', 't (ms)', 'E (mJ)'],
          [['DS-CNN TFLite', '1.59', '0.22', '0.34'],
           ['DS-CNN ONNX', '1.18', '0.45', '0.53'],
           ['CRNN TFLite', '1.85', '0.27', '0.51'],
           ['CRNN ONNX', '1.58', '0.49', '0.78']],
          7.05, 2.72, 5.4, size=15)
for _r in _t11b.table.rows:
    _r.height = Inches(0.30)
box = s.shapes.add_textbox(Inches(1.55), Inches(5.00), Inches(10.9), Inches(0.60))
tf = box.text_frame
tf.word_wrap = True
for i, t in enumerate([
    'Default ONNX quietly spawns 4 threads while TFLite runs 1 \u2014 pinned equally, TFLite still wins everywhere.',
    'Race-to-sleep wins: TFLite draws more power yet finishes ~2\u00d7 faster, so energy drops ~35% (0.34 vs 0.53 mJ).']):
    p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
    p.space_after = Pt(2)
    r = p.add_run()
    r.text = '\u2022  ' + t
    r.font.size = Pt(16)
    r.font.color.rgb = BLACK
    r.font.name = 'Calibri'
add_number(s, 11)

# ============ 12. TRADE-OFF ============
s = new_slide(BG_CONTENT)
add_title(s, 'So which model do you ship?')
s.shapes.add_picture(f'{BASE}/acc_params.png', Inches(1.55), Inches(2.72),
                     Inches(6.6), Inches(2.6))
add_bullets(s, [
    ('Sweet spot: ', 'DS-CNN \u2014 22.6k params, 95.5% full-task, 0.13 ms in int8.'),
    ('Most accurate: ', 'CRNN \u2014 92.6% subset at 56.8k params, 0.10 ms int8.'),
    'LSTM is small but int8-unfriendly; DNN is just the anchor \u2014 big and weak.',
    ('Rule: ', 'pick the runtime by operator mix, not by habit.'),
], left=8.45, top=2.72, width=4.3, size=19, gap=10)
add_number(s, 12)

# ============ 13. CONCLUSION ============
s = new_slide(BG_CONTENT)
add_title(s, 'Takeaways')
add_bullets(s, [
    ('Test both runtimes before you commit: ',
     'ONNX wins only on DS-CNN depthwise convs; TFLite wins 1.4\u20132.1\u00d7 on dense and recurrent.'),
    ('Pin your threads: ', 'defaults lie \u2014 ONNX spawns 4, TFLite runs 1.'),
    'Full int8 is nearly free for CNNs (1.7\u20132.5\u00d7 faster, < 0.5 pt drop); LSTM needs QAT.',
    'Latency saves battery: finishing 2\u00d7 faster cuts energy by about a third.',
], size=22, gap=14)
add_number(s, 13)

# ============ 14. THANKS ============
s = new_slide(BG_CONTENT)
box = s.shapes.add_textbox(Inches(1.2), Inches(2.0), Inches(10.93), Inches(1.0))
tf = box.text_frame
tf.word_wrap = True
p = tf.paragraphs[0]
p.alignment = PP_ALIGN.CENTER
r = p.add_run()
r.text = 'Thanks \u2014 questions welcome'
r.font.size = Pt(44)
r.font.bold = True
r.font.color.rgb = NAVY
r.font.name = 'Calibri'

box = s.shapes.add_textbox(Inches(1.2), Inches(3.2), Inches(10.93), Inches(1.2))
tf = box.text_frame
tf.word_wrap = True
for i, (t, sz, bold, col) in enumerate([
        ('Artifact, code, and data are open', 22, False, BLACK),
        ('github.com/jaweed3/kws-pi5-benchmark', 24, True, NAVY),
        ('DOI 10.5281/zenodo.23765521  \u00b7  AGPLv3', 20, False, GRAY)]):
    p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = t
    r.font.size = Pt(sz)
    r.font.bold = bold
    r.font.color.rgb = col
    r.font.name = 'Calibri'

box = s.shapes.add_textbox(Inches(1.2), Inches(4.9), Inches(10.93), Inches(0.9))
tf = box.text_frame
tf.word_wrap = True
p = tf.paragraphs[0]
p.alignment = PP_ALIGN.CENTER
r = p.add_run()
r.text = ('Fatih Jawwad Al Mumtaz  \u00b7  fatihalmumtaz76@student.cs.unida.gontor.ac.id\n'
          'Amirul Al Aziz  \u00b7  Faculty of Science and Technology, Universitas Darussalam Gontor')
r.font.size = Pt(16)
r.font.color.rgb = GRAY
r.font.name = 'Calibri'
add_number(s, 14)

prs.save(f'{BASE}/slides-icaitech-2026.pptx')
print('saved, slides:', len(prs.slides._sldIdLst))
