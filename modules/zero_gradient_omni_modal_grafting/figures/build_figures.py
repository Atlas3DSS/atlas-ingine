"""Builds the three paper figures as standalone SVG, then renders each to a 2x PNG with headless Chromium.

Every plotted value is read from a run report or from the size table in PAPER.md; none is typed here. Values the
figure text depends on are asserted against the report before anything is written. Figure 3 separates byte counts
recorded in run reports from the paper's size targets, which have not been measured.

    python3 build_figures.py [--packet-a REPORT.json] [--packet-b REPORT.json] [--strip REPORT.json] [--no-png]

Writes fig1/fig2/fig3 .svg and .png beside this file, plus sources.sha256.json.
"""
import argparse, glob, hashlib, json, math, os, re, subprocess
from PIL import ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = '/mnt/storage/omni_gate2_runs'
ap = argparse.ArgumentParser()
ap.add_argument('--packet-a', default=f'{RUNS}/h3_platos_key_phase0_24p_a8146fa7/h3_platos_key_phase0_report.json')
ap.add_argument('--packet-b', default=f'{RUNS}/vision_ridge_norm_dev_val192/vision_ridge_norm_dev_report.json')
ap.add_argument('--strip', default=f'{RUNS}/phase0_912_spinup/packetB2_ocr_towers_strip_verify_report.json')
ap.add_argument('--image-nvfp4', default=os.path.expanduser('~/dev_genius/models/minimax_h3_image_pruned_nvfp4.safetensors'))
ap.add_argument('--paper', default=os.path.join(HERE, '..', 'PAPER.md'))
ap.add_argument('--no-png', action='store_true')
ARGS = ap.parse_args()

BG = '#0A0B0C'; INK = '#F4F3EE'; BODY = '#D3D5D1'; SEC = '#A0A6A9'; DIM = '#5d656a'; RULE = '#2B3034'
G = '#82C232'; G2 = '#4E7A22'; AMBER = '#eab94e'; RED = '#e5645a'; GREY = '#8a9197'
SANS = "'Noto Sans','Liberation Sans',Arial,sans-serif"; SERIF = "'Noto Serif','DejaVu Serif',Georgia,serif"
W = 1240; M = 52; CW = W - 2 * M
MINUS = chr(0x2212); TIMES = chr(0xD7)


def sha(p): return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def _font(bold, serif, size):
    name = ('NotoSerif' if serif else 'NotoSans') + ('-Bold' if bold else '-Regular') + '.ttf'
    for d in (os.path.expanduser('~/.local/share/fonts/noto-core'), '/usr/share/fonts/truetype/noto'):
        if os.path.exists(os.path.join(d, name)): return ImageFont.truetype(os.path.join(d, name), size)
    raise SystemExit(f'{name} not found; text wrapping needs the Noto fonts')


def esc(s): return s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def t(x, y, s, size=13, fill=SEC, anchor='start', w=400, serif=False, ls=None):
    extra = f' letter-spacing="{ls}"' if ls else ''
    return f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" font-weight="{w}" font-family="{SERIF if serif else SANS}"{extra}>{esc(s)}</text>'


def wrap(s, size, width, bold=False, serif=False):
    f = _font(bold, serif, 64); lines, cur = [], ''
    for word in s.split():
        trial = (cur + ' ' + word).strip()
        if f.getlength(trial) * size / 64 <= width or not cur: cur = trial
        else: lines.append(cur); cur = word
    return lines + [cur]


def para(x, y, s, size=13, fill=SEC, width=CW, lh=1.5, w=400, serif=False):
    """Wrapped text block. Returns (svg, y of the last baseline)."""
    out = ''
    for i, line in enumerate(wrap(s, size, width, w >= 600, serif)): out += t(x, y + i * size * lh, line, size, fill, 'start', w, serif)
    return out, y + i * size * lh


def header(tag, tagcol, kicker, title, lede):
    tw = _font(True, False, 64).getlength(tag) * 11 / 64 + len(tag) * 0.88 + 24
    s = f'<rect x="{M}" y="56" width="{tw:.0f}" height="25" fill="none" stroke="{tagcol}" stroke-width="1.5"/>' + t(M + 12, 73, tag, 11, tagcol, 'start', 700, ls='0.08em')
    s += t(M + tw + 12, 73.5, kicker, 12.5, SEC, ls='0.05em')
    h, y = para(M, 128, title, 31, INK, CW - 30, 1.22, 700, True); s += h
    h, y = para(M, y + 36, lede, 15.5, BODY, CW - 90, 1.55); s += h
    return s, y + 44


def footer(y, note):
    s = f'<line x1="{M}" x2="{W - M}" y1="{y}" y2="{y}" stroke="{INK}" stroke-width="1"/>'
    h, y2 = para(M, y + 24, note, 12.5, SEC, CW - 250, 1.5); s += h
    s += t(W - M, y + 24, 'github.com/Atlas3DSS/atlas-ingine', 12.5, SEC, 'end')
    return s, y2 + 40


def doc(body, H): return f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H:.0f}" viewBox="0 0 {W} {H:.0f}"><rect width="{W}" height="{H:.0f}" fill="{BG}"/>{body}</svg>\n'


def n4(v): return (MINUS if v < 0 else '') + f'{abs(v):.4f}'
def s4(v): return (MINUS if v < 0 else '+') + f'{abs(v):.4f}'
def gb(b): return b / 1e9
def sci(v):
    m, e = f'{v:.2e}'.split('e'); return f'{m} {TIMES} 10' + ''.join({'-': chr(0x207B), '1': chr(0xB9), '2': chr(0xB2), '3': chr(0xB3)}.get(c, chr(0x2070 + int(c)) if c.isdigit() else c) for c in str(int(e)))


A = json.load(open(ARGS.packet_a)); B = json.load(open(ARGS.packet_b)); ST = json.load(open(ARGS.strip)); PAPER = open(ARGS.paper).read()
SOURCES = {os.path.basename(p): sha(p) for p in (ARGS.packet_a, ARGS.packet_b, ARGS.strip)}; SOURCES['PAPER.md (size targets, figure 3)'] = sha(ARGS.paper)
assert A['qualifier']['qualified_complete'] and B['qualifier']['qualified_complete'] and ST['all_verified']

# ------------------------------------------------------------------ figure 1: Packet A adherence and size
FE = A['functional_evidence']; SUM = FE['arm_summaries']; PP = FE['per_prompt_outcomes']; MF = A['measured_memory_footprint']
ARMS = [('teacher_reinject', 'Teacher encoder', 'Qwen3-VL-32B, layer 50', INK), ('inkling_platos_key_concat4', 'Four-tap map', '9,216 to 5,120 dims', G),
        ('inkling_platos_key_single_tap', 'Single-tap map', '2,304 to 5,120 dims', G2), ('b0_shuffled_single_tap', 'Shuffled control', 'map fitted on mismatched prompts', DIM)]
NP = len(PP); assert NP == 24 == A['split_audit']['functional_dit_eval_prompts_used'] and FE['source_group_audit']['n_source_groups'] == 16
for k, *_ in ARMS:  # per-prompt rows must reproduce the arm means
    assert abs(sum(p['arms'][k]['siglip2_text_image_cos'] for p in PP) / NP - SUM[k]['siglip2_text_image_cos_mean']) < 1e-9
    if k != 'teacher_reinject': assert abs(sum(p['arms'][k]['siglip2_image_vs_teacher_cos'] for p in PP) / NP - SUM[k]['siglip2_image_vs_teacher_cos_mean']) < 1e-9
C4 = FE['paired_vs_b0_and_teacher']['inkling_platos_key_concat4']; S1 = FE['paired_vs_b0_and_teacher']['inkling_platos_key_single_tap']
VT = C4['siglip2_adherence_vs_teacher']; VB = C4['siglip2_adherence_vs_b0']
assert VB['n_a_gt_b'] == 24 and VB['n_a_lt_b'] == 0 and S1['siglip2_adherence_vs_b0']['n_a_gt_b'] == 24 and VT['ci95_low'] < 0 < VT['ci95_high']
assert A['fixed_invariants']['zero_gradient_training'] and A['split_audit']['confirm_reserve_prompts_touched'] == 0 and FE['teacher_reinject_floor']['passed']
TB = MF['teacher_qwen3vl_32b_layer50_int8_convrot']['disk_bytes']; K4 = MF['platos_key_concat4_fp16']; K1 = MF['platos_key_single_tap_fp16']
assert TB == A['pinned_weights_verified']['teacher_qwen3vl_32b_layer50']['bytes'] and K4['disk_bytes'] == A['fitted_keys']['concat4']['saved']['bytes']
FIT = A['fitted_keys']['concat4']['fit_audit']; FITS = A['split_audit']['fit_prompts_used']; FITT = A['staged_handoffs']['stageB_elapsed_seconds']


def strip_panel(x0, x1, y0, rows, lo, hi, ticks, key, title, sub, fmt):
    """One row of per-prompt dots per arm, with the arm mean as a vertical bar."""
    sx = lambda v: x0 + (v - lo) / (hi - lo) * (x1 - x0); RH = 58
    s = t(x0, y0 - 34, title, 14.5, INK, 'start', 700) + t(x0, y0 - 16, sub, 12, SEC)
    for v in ticks:
        s += f'<line x1="{sx(v):.1f}" x2="{sx(v):.1f}" y1="{y0}" y2="{y0 + len(ARMS) * RH}" stroke="{SEC if v == 0 else RULE}" stroke-width="1"/>' + t(sx(v), y0 + len(ARMS) * RH + 18, fmt(v), 11.5, SEC, 'middle')
    for i, (k, _, _, col) in enumerate(ARMS):
        cy = y0 + i * RH + RH / 2
        if k not in rows:
            s += t(sx(hi), cy + 4, 'reference', 12, DIM, 'end'); continue
        vals = sorted(p['arms'][k][key] for p in PP)
        for j, v in enumerate(vals):  # deterministic vertical offsets so equal values stay visible
            s += f'<circle cx="{sx(v):.1f}" cy="{cy + ((j * 7) % 5 - 2) * 4.2:.1f}" r="4" fill="{col}" fill-opacity="0.78"/>'
        m = sum(vals) / NP
        s += f'<line x1="{sx(m):.1f}" x2="{sx(m):.1f}" y1="{cy - 19}" y2="{cy + 19}" stroke="{INK}" stroke-width="2.5"/>'
        right = sx(max(vals)) + 12; left = sx(min(vals)) - 12
        s += t(right, cy + 5, n4(m), 14, INK, 'start', 700) if right + 50 < x1 + 60 else t(left, cy + 5, n4(m), 14, INK, 'end', 700)
    return s


def fig1():
    ret = C4['siglip2_adherence_margin_retention_vs_teacher']
    s, y = header('MEASURED', G, 'INKLING-2-MINI TO MINIMAX-H3 IMAGE DIFFUSION ' + chr(183) + ' 24 DEVELOPMENT PROMPTS',
                  f'A {K4["mib"]:.1f} MiB linear map stands in for a {gb(TB):.2f} GB text encoder and keeps {ret * 100:.1f}% of its prompt adherence margin',
                  f'The image model normally takes its prompt from a {gb(TB):.2f} GB encoder. Here it takes Inkling-2-Mini hidden states, passed through a ridge map solved in closed form with no gradient steps. Each dot is one prompt; the white bar is the mean of {NP}.')
    y0 = y + 46; LAB = M; PA0, PA1 = M + 250, M + 640; PB0, PB1 = M + 760, M + 1070; RH = 58
    for i, (k, nm, sub, col) in enumerate(ARMS):
        cy = y0 + i * RH + RH / 2
        s += f'<line x1="{M}" x2="{W - M}" y1="{y0 + i * RH}" y2="{y0 + i * RH}" stroke="{RULE}"/>'
        s += f'<rect x="{LAB}" y="{cy - 12}" width="5" height="24" fill="{col}"/>' + t(LAB + 16, cy - 1, nm, 14, INK, 'start', 700) + t(LAB + 16, cy + 16, sub, 11.5, SEC)
    s += f'<line x1="{M}" x2="{W - M}" y1="{y0 + 4 * RH}" y2="{y0 + 4 * RH}" stroke="{RULE}"/>'
    s += strip_panel(PA0, PA1, y0, {a[0] for a in ARMS}, -0.2, 0.3, (-0.2, -0.1, 0, 0.1, 0.2, 0.3), 'siglip2_text_image_cos', 'Does the picture match the prompt?', 'SigLIP2 cosine, prompt text to image', lambda v: (MINUS if v < 0 else '') + f'{abs(v):.1f}')
    s += strip_panel(PB0, PB1, y0, {a[0] for a in ARMS[1:]}, 0.2, 1.0, (0.2, 0.4, 0.6, 0.8, 1.0), 'siglip2_image_vs_teacher_cos', "Does it match the teacher's picture?", 'SigLIP2 cosine, image to teacher image', lambda v: f'{v:.1f}')
    y = y0 + 4 * RH + 62
    # paired statistics for the four-tap map
    cols = [(f'{ret * 100:.1f}%', f"of the teacher's margin over the shuffled control is kept by the four-tap map. The single-tap map keeps {S1['siglip2_adherence_margin_retention_vs_teacher'] * 100:.1f}%."),
            (s4(VT['paired_delta']), f"paired difference to the teacher, 95% interval {s4(VT['ci95_low'])} to {s4(VT['ci95_high'])}. Sign test p = {VT['sign_exact_p']:.3f}: {VT['n_a_gt_b']} prompts above the teacher, {VT['n_a_lt_b']} below."),
            (f"{VB['n_a_gt_b']} of {NP}", f"prompts score above the shuffled control, for both maps. Sign test p = {sci(VB['sign_exact_p'])}.")]
    cw = (CW - 2 * 36) / 3; yb = y
    for i, (big, txt) in enumerate(cols):
        x = M + i * (cw + 36)
        s += f'<line x1="{x}" x2="{x + cw}" y1="{y}" y2="{y}" stroke="{RULE}"/>' + t(x, y + 38, big, 27, INK, 'start', 700, True)
        h, yy = para(x, y + 62, txt, 13, SEC, cw, 1.5); s += h; yb = max(yb, yy)
    y = yb + 50
    # size: areas in proportion to bytes on disk
    s += f'<line x1="{M}" x2="{W - M}" y1="{y}" y2="{y}" stroke="{RULE}"/>' + t(M, y + 34, 'Size on disk', 14.5, INK, 'start', 700) + t(M + 100, y + 34, 'areas drawn in proportion to bytes', 12, SEC)
    side = 190; ys = y + 56
    s += f'<rect x="{M}" y="{ys}" width="{side}" height="{side}" fill="none" stroke="{RED}" stroke-width="2"/>' + t(M + side / 2, ys + side / 2 - 2, f'{gb(TB):.2f} GB', 22, INK, 'middle', 700, True) + t(M + side / 2, ys + side / 2 + 20, 'teacher encoder', 12.5, SEC, 'middle')
    xk = M + side + 70
    for i, (kk, nm, col) in enumerate(((K4, 'four-tap map', G), (K1, 'single-tap map', G2))):
        sd = side * math.sqrt(kk['disk_bytes'] / TB); yy = ys + 34 + i * 74
        s += f'<rect x="{xk + 8 - sd / 2:.1f}" y="{yy - sd / 2:.1f}" width="{sd:.1f}" height="{sd:.1f}" fill="{col}"/>'
        s += t(xk + 34, yy + 1, f'{kk["mib"]:.1f} MiB', 20, INK, 'start', 700, True) + t(xk + 34, yy + 20, f'{nm}, {TB / kk["disk_bytes"]:,.0f}{TIMES} smaller than the teacher', 12.5, SEC)
    xr = M + 640
    h, yy = para(xr, ys + 20, f'The four-tap map was fitted on {FITS} prompts ({FIT["n_samples"]:,} aligned word spans) in {FITT:.0f} s on CPU. It reads four hidden-state taps from Inkling-2-Mini, so that model has to be loaded; the teacher encoder does not.', 13.5, BODY, CW - 640, 1.55); s += h
    h, yy = para(xr, yy + 32, f'The teacher re-injection check reproduced the native run exactly on 4 prompts before any map was scored.', 13.5, BODY, CW - 640, 1.55); s += h
    y = ys + side + 46
    f, H = footer(y, f'{NP} held-out development prompts from {FE["source_group_audit"]["n_source_groups"]} source groups. {A["fixed_invariants"]["dit_pdd_steps"]}-step sampling at 1024 by 1024, with the same seed and starting noise in every arm. The sealed 160-prompt confirmation set has not been run. Margin kept is the map minus control, divided by teacher minus control.')
    return doc(s + f, H)


# ------------------------------------------------------------------ figure 2: Packet B per-domain boundary
NC = B['native_critique_comparison']; RF = NC['arms']['ridge_frozen']; RN = NC['arms']['ridge_norm']
DOMS = [('counting_spatial', 'Counting and layout'), ('natural_photos', 'Natural photos'), ('text_typography_infographics', 'Text and typography'), ('charts_ui', 'Charts and interfaces')]
assert all(RF['by_domain'][d]['n_cases'] == 48 for d, _ in DOMS) and NC['n_cases'] == 192 and B['split_audit']['locked_test_cases_touched'] == 0
for arm in (RF, RN):
    for key in ('teacher_gold_correct', 'bridge_teacher_agree_count', 'bridge_gold_correct', 'b0_teacher_agree_count', 'b0_gold_correct'):
        assert sum(arm['by_domain'][d][key] for d, _ in DOMS) == arm['overall'][key]
assert all(NC['teacher_gold_overall']['by_domain'][d]['gold_correct'] == RF['by_domain'][d]['teacher_gold_correct'] for d, _ in DOMS)
CMP = NC['candidate_vs_ridge_frozen']['overall']; assert NC['predeclared_readouts']['dev_advancement_verdict'] == 'reject_no_gain_over_ridge'


def bars_panel(x0, x1, y0, title, sub, series):
    """Grouped vertical bars, one group per domain, counts out of 48."""
    Hh = 250; sy = lambda v: y0 + Hh - v / 48 * Hh; gw = (x1 - x0) / len(DOMS); bw = 27; gap = 5
    s = t(x0, y0 - 40, title, 14.5, INK, 'start', 700) + t(x0, y0 - 22, sub, 12, SEC)
    for v in (0, 12, 24, 36, 48): s += f'<line x1="{x0}" x2="{x1}" y1="{sy(v):.1f}" y2="{sy(v):.1f}" stroke="{SEC if v == 0 else RULE}"/>' + t(x0 - 8, sy(v) + 4, f'{v}', 11.5, SEC, 'end')
    for i, (d, lab) in enumerate(DOMS):
        tot = len(series) * bw + (len(series) - 1) * gap; bx = x0 + i * gw + (gw - tot) / 2
        for j, (_, col, fn) in enumerate(series):
            v = fn(d); x = bx + j * (bw + gap)
            s += f'<rect x="{x:.1f}" y="{sy(v):.1f}" width="{bw}" height="{max(v / 48 * Hh, 1.5):.1f}" fill="{col}"/>' + t(x + bw / 2, sy(v) - 6, f'{v}', 12.5, INK, 'middle', 700)
        for li, line in enumerate(wrap(lab, 12.5, gw - 8)): s += t(x0 + i * gw + gw / 2, y0 + Hh + 20 + li * 16, line, 12.5, BODY, 'middle')
    return s


def fig2():
    c, o = RF['by_domain']['counting_spatial'], RF['by_domain']['text_typography_infographics']
    s, y = header('MEASURED', G, 'CACHED VISION FEATURES TO INKLING-2-MINI ' + chr(183) + ' 192 DEVELOPMENT IMAGES, 48 PER DOMAIN',
                  f'The linear vision bridge carries counting and layout ({c["bridge_teacher_agree_count"]} of 48 match the teacher) and loses text ({o["bridge_teacher_agree_count"]} to {RN["by_domain"]["text_typography_infographics"]["bridge_teacher_agree_count"]} of 48)',
                  'A ridge map sends cached low-resolution vision features into Inkling-2-Mini in place of its own vision tokens. The teacher is the model answering from its native vision tokens. Rescaling the map to the teacher\'s variance did not help.')
    ser_gold = [('Teacher', INK, lambda d: RF['by_domain'][d]['teacher_gold_correct']), ('Ridge map', G, lambda d: RF['by_domain'][d]['bridge_gold_correct']),
                ('Ridge map, variance matched', AMBER, lambda d: RN['by_domain'][d]['bridge_gold_correct']), ('Shuffled control', DIM, lambda d: RF['by_domain'][d]['b0_gold_correct'])]
    ser_agree = [s_ for s_ in ser_gold[1:]]
    ser_agree = [('Ridge map', G, lambda d: RF['by_domain'][d]['bridge_teacher_agree_count']), ('Ridge map, variance matched', AMBER, lambda d: RN['by_domain'][d]['bridge_teacher_agree_count']), ('Shuffled control', DIM, lambda d: RF['by_domain'][d]['b0_teacher_agree_count'])]
    x = M
    for nm, col, _ in ser_gold:  # legend
        s += f'<rect x="{x}" y="{y - 11}" width="13" height="13" fill="{col}"/>' + t(x + 20, y, nm, 13, BODY); x += 20 + _font(False, False, 64).getlength(nm) * 13 / 64 + 30
    y0 = y + 84
    s += bars_panel(M + 26, M + 570, y0, 'Correct answers', 'against the gold answer, of 48 per domain', ser_gold)
    s += bars_panel(M + 660, M + 1136, y0, "Same answer as the teacher", 'of 48 per domain', ser_agree)
    y = y0 + 250 + 78
    ov = RF['overall']; on = RN['overall']
    cols = [(f"{c['bridge_gold_correct']} vs {c['teacher_gold_correct']}", f"correct on counting and layout, ridge map against teacher, of 48. The shuffled control gets {c['b0_gold_correct']}."),
            (f"{o['bridge_gold_correct']} vs {o['teacher_gold_correct']}", f"correct on text and typography, of 48. On charts and interfaces the teacher itself gets only {RF['by_domain']['charts_ui']['teacher_gold_correct']}, so that domain says little about the map."),
            (f"{on['bridge_gold_correct']} vs {ov['bridge_gold_correct']}", f"correct overall of {NC['n_cases']}, variance matched against plain ridge (teacher {ov['teacher_gold_correct']}). McNemar p = {CMP['gold_accuracy']['mcnemar_exact_p']:.3f}. Variance matching was rejected.")]
    cw = (CW - 2 * 36) / 3; yb = y
    for i, (big, txt) in enumerate(cols):
        x = M + i * (cw + 36)
        s += f'<line x1="{x}" x2="{x + cw}" y1="{y}" y2="{y}" stroke="{RULE}"/>' + t(x, y + 38, big, 27, INK, 'start', 700, True)
        h, yy = para(x, y + 62, txt, 13, SEC, cw, 1.5); s += h; yb = max(yb, yy)
    f, H = footer(yb + 50, f'{NC["n_cases"]} development images, {B["split_audit"]["fit_examples_used"]:,} images used to fit the map. The locked test set was not touched. The shuffled control is the same solver fitted on mismatched image pairs; the one shown is the control matched to the plain ridge map.')
    return doc(s + f, H)


# ------------------------------------------------------------------ figure 3: footprint, measured and targeted
PW = A['pinned_weights_verified']; INK_B = A['donor_and_exporter_pins_verified']['donor_checkpoint_files']['model-00001-of-00001.safetensors']['bytes']
DIT_B = PW['dit_h3_image']['bytes']; VAE_B = PW['vae_h3_image']['bytes']
assert DIT_B == MF['dit_h3_image_int8_convrot_plus_vae']['dit_disk_bytes'] and VAE_B == MF['dit_h3_image_int8_convrot_plus_vae']['vae_disk_bytes']
SEC6 = PAPER.split('## 6. System Footprint')[1].split('\n## ')[0]


SEC62 = SEC6.split('### 6.2')[1]


def trow(label): return next(line.split('|') for line in SEC62.splitlines() if line.startswith('|') and label in line.split('|')[1])


def target(label):
    """Low and high end, in GB, of the target cell for one row of the paper's projected-size table."""
    m = re.search(r'\*\*([\d.]+)(?:' + chr(0x2013) + r'([\d.]+))?\*\*', trow(label)[2]); return float(m.group(1)), float(m.group(2) or m.group(1))


TG = {k: target(lab) for k, lab in (('trunk', 'Trunk'), ('maps', 'Bridges'), ('sight', 'Sight'), ('ears', 'Ears'), ('mouth', 'Mouth'))}
CAP = float(re.search(r'<=([\d.]+) GB', trow('Trunk')[2]).group(1))  # largest trunk the paper allows on a 12 GB card
PACK = tuple(float(v) for v in re.search(r'~([\d.]+)' + chr(0x2013) + r'([\d.]+) GB', trow('Diffusion Image')[2]).groups())
# The quantized image model is a file on disk, so its size is read from the file, not from the table.
NV_B = os.path.getsize(ARGS.image_nvfp4); IMG = gb(NV_B + VAE_B); SOURCES[os.path.basename(ARGS.image_nvfp4) + ' (bytes)'] = NV_B
REST = [sum(TG[k][i] for k in ('maps', 'sight', 'ears', 'mouth')) for i in (0, 1)]
VST = [TG['trunk'][i] + REST[i] for i in (0, 1)]; VCAP = CAP + REST[1]; ALL = [VST[i] + IMG for i in (0, 1)]
assert tuple(round(v, 2) for v in VST) == target('Voice + Sight + Text') and abs(gb(ST['got_ocr2']['stripped_bytes']) - TG['sight'][0]) < 0.01
for nm, mine, theirs in (('image model with VAE', (IMG, IMG), target('Diffusion Image')), ('all four', tuple(ALL), target('Full 4-Modality'))):
    if any(abs(a_ - b_) > 0.05 for a_, b_ in zip(mine, theirs)): print(f'NOTE: {nm}: files give {mine[0]:.2f} to {mine[1]:.2f} GB, PAPER.md states {theirs[0]:.2f} to {theirs[1]:.2f}')


def rng(lo_hi, d=1): return f'{lo_hi[0]:.{d}f}' if lo_hi[0] == lo_hi[1] else f'{lo_hi[0]:.{d}f} to {lo_hi[1]:.{d}f}'


def fig3():
    stock = INK_B + TB + DIT_B + VAE_B; ours = INK_B + K4['disk_bytes'] + DIT_B + VAE_B
    s, y = header('MEASURED AND TARGET', AMBER, 'OMNI-INKLING-2-MINI ' + chr(183) + ' WEIGHTS ON DISK, DECIMAL GB',
                  f'The map removes {gb(stock - ours):.1f} GB of text encoder today. The 12 GB card is a target that has not been run.',
                  'The upper panel is byte counts recorded in the run reports. The lower panel is the sizes the paper draft aims for after stripping and quantizing each part. Most parts do not exist at those sizes yet, and no card in the ladder has been run.')
    LW = 300; x0 = M + LW; x1 = W - M - 96; sc = (x1 - x0) / 70; BH = 30
    s += f'<rect x="{M}" y="{y - 14}" width="84" height="22" fill="none" stroke="{G}" stroke-width="1.5"/>' + t(M + 10, y + 1.5, 'MEASURED', 10.5, G, 'start', 700, ls='0.08em') + t(M + 98, y + 2, 'Language model with image generation, bytes on disk from the run reports', 14.5, INK, 'start', 700)
    y += 40
    rows = [('With the stock text encoder', [('Inkling-2-Mini', INK_B, G2), ('teacher encoder', TB, RED), ('image model', DIT_B, DIM), ('VAE', VAE_B, GREY)], stock),
            (f'With the {K4["mib"]:.0f} MiB map', [('Inkling-2-Mini', INK_B, G2), ('map', K4['disk_bytes'], AMBER), ('image model', DIT_B, DIM), ('VAE', VAE_B, GREY)], ours)]
    for v in range(0, 71, 10): s += f'<line x1="{x0 + v * sc:.1f}" x2="{x0 + v * sc:.1f}" y1="{y - 6}" y2="{y + 2 * 62 - 14}" stroke="{RULE}"/>' + t(x0 + v * sc, y + 2 * 62 + 4, f'{v} GB' if v else '0', 11.5, SEC, 'middle')
    for i, (lab, segs, tot) in enumerate(rows):
        yy = y + i * 62; x = x0; s += t(M, yy + 20, lab, 14, INK, 'start', 700)
        for nm, b, col in segs:
            w = max(gb(b) * sc, 3); s += f'<rect x="{x:.1f}" y="{yy}" width="{w:.1f}" height="{BH}" fill="{col}"/>'
            if w > 96: s += t(x + 8, yy + 19.5, f'{nm} {gb(b):.2f}', 12, INK if col != GREY else BG, 'start', 700)
            elif nm == 'map': s += t(x + 1.5, yy + BH + 15, f'map {gb(b):.3f}', 11.5, AMBER, 'middle', 700)
            x += w + 1.5
        s += t(x + 8, yy + 20.5, f'{gb(tot):.1f} GB', 15, INK, 'start', 700)
    y += 2 * 62 + 34
    pk = MF['teacher_qwen3vl_32b_layer50_int8_convrot']['peak_card_vram_bytes'], MF['donor_inkling_2_mini_nvfp4']['cuda_mem_peak_used_bytes'], MF['dit_h3_image_int8_convrot_plus_vae']['peak_card_vram_bytes']
    h, y = para(M, y, f'VAE {gb(VAE_B):.2f} GB. Peak card memory measured with each stage loaded alone: teacher encoder {gb(pk[0]):.2f} GB, Inkling-2-Mini {gb(pk[1]):.2f} GB, image model with VAE {gb(pk[2]):.2f} GB. Each is above its file size, so weights on disk understate the memory a card needs.', 13, SEC, CW - 120, 1.5); s += h
    y += 46
    s += f'<line x1="{M}" x2="{W - M}" y1="{y}" y2="{y}" stroke="{RULE}"/>'; y += 40
    s += f'<rect x="{M}" y="{y - 14}" width="66" height="22" fill="none" stroke="{AMBER}" stroke-width="1.5"/>' + t(M + 10, y + 1.5, 'TARGET', 10.5, AMBER, 'start', 700, ls='0.08em') + t(M + 80, y + 2, 'After stripping and quantizing, weights only, from the paper draft', 14.5, INK, 'start', 700) + t(M + 80, y + 21, 'targets, not measurements; no card in this ladder has been run', 12, SEC)
    y += 66; sc2 = (x1 - x0) / 34
    s += f'<defs><pattern id="hatch" width="7" height="7" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><rect width="7" height="7" fill="{BG}"/><rect width="3" height="7" fill="{AMBER}" fill-opacity="0.75"/></pattern></defs>'
    trows = [('Voice, sight and text', 'trunk, four maps, sight, ears, mouth', VST), ('Image model alone', 'NVFP4 file on disk, with VAE', [IMG, IMG]), ('All four resident', 'voice, sight, text and image', ALL)]
    for v in range(0, 35, 4): s += f'<line x1="{x0 + v * sc2:.1f}" x2="{x0 + v * sc2:.1f}" y1="{y - 6}" y2="{y + 3 * 62 - 14}" stroke="{RULE}"/>' + t(x0 + v * sc2, y + 3 * 62 + 4, f'{v} GB' if v else '0', 11.5, SEC, 'middle')
    for v, card in ((12, 'RTX 3060, 12 GB'), (24, 'RTX 3090 or 4090, 24 GB'), (32, 'RTX 5090, 32 GB')):
        s += f'<line x1="{x0 + v * sc2:.1f}" x2="{x0 + v * sc2:.1f}" y1="{y - 22}" y2="{y + 3 * 62 - 14}" stroke="{INK}" stroke-width="1.5" stroke-dasharray="5 4"/>' + t(x0 + v * sc2 - 6, y - 26, card, 11.5, INK, 'end', 700)
    for i, (lab, sub, (lo, hi)) in enumerate(trows):
        yy = y + i * 62; s += t(M, yy + 14, lab, 14, INK, 'start', 700) + t(M, yy + 31, sub, 11.5, SEC)
        s += f'<rect x="{x0}" y="{yy}" width="{lo * sc2:.1f}" height="{BH}" fill="url(#hatch)" stroke="{AMBER}" stroke-width="1"/><rect x="{x0 + lo * sc2:.1f}" y="{yy}" width="{(hi - lo) * sc2:.1f}" height="{BH}" fill="none" stroke="{AMBER}" stroke-width="1" stroke-dasharray="3 3"/>'
        s += t(x0 + hi * sc2 + 10, yy + 20.5, f'{rng((lo, hi), 2)} GB', 15, INK, 'start', 700).replace('<text ', f'<text paint-order="stroke" stroke="{BG}" stroke-width="7" stroke-linejoin="round" ')
        if i == 0: s += f'<line x1="{x0 + VCAP * sc2:.1f}" x2="{x0 + VCAP * sc2:.1f}" y1="{yy - 5}" y2="{yy + BH + 5}" stroke="{INK}" stroke-width="2.5"/>' + t(x0 + VCAP * sc2, yy + BH + 19, f'{VCAP:.2f} with a trunk of {CAP:g} GB', 11.5, INK, 'end', 700)
    y += 3 * 62 + 40
    h, y = para(M, y, f'Hatched to the low end of each target, dashed outline to the high end. Parts, in GB: trunk {rng(TG["trunk"])}, four maps {rng(TG["maps"], 2)} (one measured file, three computed from their shapes), sight tower {rng(TG["sight"], 2)}, ears {rng(TG["ears"])}, mouth {rng(TG["mouth"])}. Totals are the sums of the parts. The image model at its quantized size on disk is above 12 GB by itself; {PACK[0]:g} to {PACK[1]:g} GB is a further packing target.', 13, SEC, CW - 120, 1.5); s += h
    f, H = footer(y + 46, f'Upper panel: file sizes of the pinned checkpoints and the fitted map, and peak card memory, as recorded in the Packet A run report. Lower panel: targets, except the quantized image model, the VAE and the sight tower, which are files on disk. Weights exclude working memory during generation, and the multi-card sweep that would measure them is pending.')
    return doc(s + f, H)


FIGS = {'fig1_packetA_adherence_and_size': fig1, 'fig2_packetB_ocr_boundary': fig2, 'fig3_omni_footprint_ladder': fig3}
chrome = sorted(glob.glob(os.path.expanduser('~/.cache/ms-playwright/chromium_headless_shell-*/chrome-headless-shell-linux64/chrome-headless-shell')))
for name, fn in FIGS.items():
    svg = fn(); assert chr(8212) not in svg and not re.search(r'job-\d|att-[0-9a-f]{6}|/mnt/|/home/', svg), 'banned text'
    path = os.path.join(HERE, name + '.svg'); open(path, 'w').write(svg); H = int(re.search(r'height="(\d+)"', svg).group(1))
    if not ARGS.no_png:
        assert chrome, 'headless Chromium not found; rerun with --no-png'
        subprocess.run([chrome[-1], '--no-sandbox', '--hide-scrollbars', '--force-device-scale-factor=2', f'--window-size={W},{H}', f'--screenshot={os.path.join(HERE, name + ".png")}', 'file://' + path], check=True, capture_output=True)
    print(name, W, H)
open(os.path.join(HERE, 'sources.sha256.json'), 'w').write(json.dumps(SOURCES, indent=1) + '\n')
