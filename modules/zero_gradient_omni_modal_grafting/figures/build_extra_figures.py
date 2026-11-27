"""Builds two additional publication figures (fig0 architecture + fig4 multi-donor & cross-encoder ladder)
in the visual language of build_figures.py. Quantitative series in fig4 and headline retention/cosine values in fig0
are read directly from the 7 pinned JSON reports (recorded in extra_sources.sha256.json); architectural annotations in
fig0 summarize PAPER.md v0.9.

    python3 build_extra_figures.py [--no-png]

Requires local pinned run JSONs, Noto fonts, and headless Chromium (omit with --no-png).
Writes fig0_architecture_pipeline.{svg,png}, fig4_multidonor_and_universal.{svg,png}, and extra_sources.sha256.json.
"""
import argparse, glob, hashlib, json, os, re, subprocess
from PIL import ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument('--packet-a-confirm', default='/mnt/storage/omni_gate2_runs/packetA_confirm160_full/h3_confirm160_report.json')
ap.add_argument('--qwen-confirm', default='/mnt/storage/omni_platos_keys_multimodel/qwen38/devbox_cu130_out/qwen38_h3_confirm160_report.json')
ap.add_argument('--blend3-confirm', default='/mnt/storage/omni_platos_keys_multimodel/blends/out_blend3_qwen_ink_ds41/blend3_qwen_ink_ds41_confirm160_postseal_report.json')
ap.add_argument('--nplus1-real', default='/mnt/storage/omni_platos_keys_multimodel/blends/out_nplus1_real_3way/nplus1_real_3way_nplus1_scaling_report.json')
ap.add_argument('--nplus1-stress', default='/mnt/storage/omni_platos_keys_multimodel/blends/out_nplus1_stress_2way/nplus1_2way_plus_stress_nplus1_scaling_report.json')
ap.add_argument('--flux2-dev', default='/mnt/storage/omni_platos_keys_images/wave_b/blackwell_out/flux2_dev/siglip2/flux2_dev/flux2_dev_singleton_siglip2_parity.json')
ap.add_argument('--waveb-part1', default='/mnt/storage/omni_platos_keys_images/wave_b/blackwell_out/parity_part1/wave_b_blackwell_parity_summary.json')
ap.add_argument('--no-png', action='store_true')
ARGS = ap.parse_args()

BG = '#0A0B0C'; PANEL = '#111417'; INK = '#F4F3EE'; BODY = '#D3D5D1'; SEC = '#A0A6A9'; DIM = '#5d656a'; RULE = '#2B3034'
G = '#82C232'; G2 = '#4E7A22'; AMBER = '#eab94e'; RED = '#e5645a'; BLUE = '#58a6ff'; GREY = '#8a9197'
SANS = "'Noto Sans','Liberation Sans',Arial,sans-serif"; SERIF = "'Noto Serif','DejaVu Serif',Georgia,serif"
MONO = "'DejaVu Sans Mono','Liberation Mono',monospace"
W = 1240; M = 52; CW = W - 2 * M

SOURCES = {}


def load_json(path):
    raw = open(path, 'rb').read()
    SOURCES[os.path.basename(path)] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw.decode('utf-8'))


def _font(bold, serif, size):
    name = ('NotoSerif' if serif else 'NotoSans') + ('-Bold' if bold else '-Regular') + '.ttf'
    for d in (os.path.expanduser('~/.local/share/fonts/noto-core'), '/usr/share/fonts/truetype/noto'):
        if os.path.exists(os.path.join(d, name)):
            return ImageFont.truetype(os.path.join(d, name), size)
    raise SystemExit(f'{name} not found')


def esc(s):
    return s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def t(x, y, s, size=13, fill=SEC, anchor='start', w=400, serif=False, mono=False, ls=None):
    fam = MONO if mono else (SERIF if serif else SANS)
    extra = f' letter-spacing="{ls}"' if ls else ''
    return f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" font-weight="{w}" font-family="{fam}"{extra}>{esc(s)}</text>'


def wrap(s, size, width, bold=False, serif=False):
    f = _font(bold, serif, 64); lines, cur = [], ''
    for word in s.split():
        trial = (cur + ' ' + word).strip()
        if f.getlength(trial) * size / 64 <= width or not cur:
            cur = trial
        else:
            lines.append(cur); cur = word
    return lines + [cur]


def para(x, y, s, size=13, fill=SEC, width=CW, lh=1.5, w=400, serif=False):
    out = ''
    for i, line in enumerate(wrap(s, size, width, w >= 600, serif)):
        out += t(x, y + i * size * lh, line, size, fill, 'start', w, serif)
    return out, y + i * size * lh


def header(tag, tagcol, kicker, title, lede):
    tw = _font(True, False, 64).getlength(tag) * 11 / 64 + len(tag) * 0.88 + 24
    s = f'<rect x="{M}" y="52" width="{tw:.0f}" height="25" fill="none" stroke="{tagcol}" stroke-width="1.5"/>' + t(M + 12, 69, tag, 11, tagcol, 'start', 700, ls='0.08em')
    s += t(M + tw + 12, 69.5, kicker, 12.5, SEC, ls='0.05em')
    h, y = para(M, 122, title, 30, INK, CW - 20, 1.22, 700, True); s += h
    h, y = para(M, y + 32, lede, 15, BODY, CW - 70, 1.52); s += h
    return s, y + 36


def footer(y, note):
    s = f'<line x1="{M}" x2="{W - M}" y1="{y}" y2="{y}" stroke="{INK}" stroke-width="1"/>'
    h, y2 = para(M, y + 22, note, 12, SEC, CW - 250, 1.48); s += h
    s += t(W - M, y + 22, 'github.com/Atlas3DSS/atlas-ingine', 12, SEC, 'end')
    return s, y2 + 34


def doc(body, H):
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H:.0f}" viewBox="0 0 {W} {H:.0f}"><rect width="{W}" height="{H:.0f}" fill="{BG}"/>{body}</svg>\n'


def box(x, y, w, h, stroke=RULE, fill=PANEL, sw=1.5, dash=None):
    da = f' stroke-dasharray="{dash}"' if dash else ''
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="4" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{da}/>'


def arrow(x1, y1, x2, y2, col=INK, sw=2.0, dash=None):
    da = f' stroke-dasharray="{dash}"' if dash else ''
    dx, dy = x2 - x1, y2 - y1
    L = (dx * dx + dy * dy) ** 0.5 or 1.0
    ux, uy = dx / L, dy / L
    px, py = -uy, ux
    ax, ay = x2 - ux * 9, y2 - uy * 9
    pts = f'{x2:.1f},{y2:.1f} {ax + px * 4.5:.1f},{ay + py * 4.5:.1f} {ax - px * 4.5:.1f},{ay - py * 4.5:.1f}'
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{ax:.1f}" y2="{ay:.1f}" stroke="{col}" stroke-width="{sw}"{da}/><polygon points="{pts}" fill="{col}"/>'


# Load verified sources
PA = load_json(ARGS.packet_a_confirm)
QW = load_json(ARGS.qwen_confirm)
B3 = load_json(ARGS.blend3_confirm)
NREAL = load_json(ARGS.nplus1_real)
NSTRESS = load_json(ARGS.nplus1_stress)
F2D = load_json(ARGS.flux2_dev)
WB1 = load_json(ARGS.waveb_part1)

pa_c4_mean = PA['separate_concat4_analysis_g16']['vs_teacher_reinject']['siglip2_text_image']['mean_a']
pa_t_mean = PA['separate_concat4_analysis_g16']['vs_teacher_reinject']['siglip2_text_image']['mean_b']
pa_b0_mean = PA['primary_single_tap_matched_b0_analysis_g16']['vs_matched_b0']['siglip2_text_image']['mean_b']
pa_concat4_ret = 100.0 * (pa_c4_mean - pa_b0_mean) / (pa_t_mean - pa_b0_mean)
assert abs(pa_concat4_ret - 97.50) < 0.05, pa_concat4_ret

qw_st_ret = 100.0 * QW['primary_single_tap_matched_b0_analysis_g16']['siglip2_margin_retention_vs_teacher_and_b0']
assert abs(qw_st_ret - 98.90) < 0.05, qw_st_ret

f2d_ret = 100.0 * F2D['verdict']['retention_vs_native']
assert abs(f2d_ret - 99.65) < 0.05, f2d_ret

wb_r = WB1['renders']
f2k9_ret = 100.0 * wb_r['flux2_klein_9b']['base']['retention_vs_native']
f1dev_ret = 100.0 * wb_r['flux1_schnell_dev']['base']['retention_vs_native']
f1sch_ret = 100.0 * wb_r['flux1_schnell_dev']['turbo']['retention_vs_native']
sdxl_b_ret = 100.0 * wb_r['sdxl_base_turbo']['base']['retention_vs_native']
sdxl_t_ret = 100.0 * wb_r['sdxl_base_turbo']['turbo']['retention_vs_native']

b3_qwen = B3['single_model_baselines_confirm160']['concat_all4']['qwen38']['cosine_mean']
b3_mk = B3['mode_b_stacked_ridge_confirm160']['stacked_multikernel_concat4']['confirm160_metrics']['cosine_mean']
b3_cvx = B3['mode_a_convex_blends_confirm160']['concat_all4']['confirm160_metrics']['cosine_mean']
b3_raw = B3['mode_b_stacked_ridge_confirm160']['stacked_raw_concat4']['confirm160_metrics']['cosine_mean']
b3_rms = B3['mode_b_stacked_ridge_confirm160']['stacked_rms_norm_concat4']['confirm160_metrics']['cosine_mean']
b3_eng = B3['mode_b_stacked_ridge_confirm160']['stacked_energy_norm_concat4']['confirm160_metrics']['cosine_mean']
assert abs(b3_mk - 0.963201) < 1e-5, b3_mk


def fig0():
    s, y = header(
        'ARCHITECTURE', G, 'ZERO-GRADIENT OMNI-MODAL GRAFTING · PACKET A EXAMPLE & SEAM OVERVIEW',
        'Frozen donor trunks drive media receivers via closed-form FP64 ridge maps without loading 27.14 GB or 44.7 GiB text encoders.',
        'Center shows the Packet A example (640 word-aligned prompts, 355 s CPU, frozen FP16 safetensors). No trunk weight changes. Right column separates sealed confirmation results, Wave B Dev-24 results, word-span blend scope, and measured failure boundaries.'
    )

    col1_x, col1_w = M, 310
    col2_x, col2_w = M + 356, 336
    col3_x, col3_w = M + 738, CW - 738
    top_y = y + 6

    # Column 1: Frozen Donor(s)
    s += box(col1_x, top_y, col1_w, 344, stroke=G, sw=2.0)
    s += t(col1_x + 16, top_y + 26, '1. FROZEN DONOR TRUNK(S)', 11.5, G, 'start', 700, ls='0.07em')
    s += t(col1_x + 16, top_y + 52, 'Inkling-2-Mini (Primary)', 16, INK, 'start', 700)
    s += t(col1_x + 16, top_y + 72, '24 layers looped x2 (48 visits), d=2304', 12, BODY)
    s += t(col1_x + 16, top_y + 90, '0 gradient steps · 0 weight edits', 12, G)
    s += f'<line x1="{col1_x+16}" x2="{col1_x+col1_w-16}" y1="{top_y+106}" y2="{top_y+106}" stroke="{RULE}"/>'
    s += t(col1_x + 16, top_y + 128, 'Packet A concat4 Taps (4 x 2304 = 9216)', 12.5, INK, 'start', 700)
    taps = [('visit_5', 'd=2304 (d_2027=0)'), ('visit_23', 'd=2304 (d_2027=0)'), ('visit_41', 'd=2304 (single tap)'), ('final_xf_normed', 'd=2304 (d_2027=0)')]
    for idx, (tn, td) in enumerate(taps):
        ty = top_y + 142 + idx * 27
        s += box(col1_x + 16, ty, col1_w - 32, 22, stroke=G2, fill='#151c12', sw=1.0)
        s += t(col1_x + 24, ty + 15, tn, 11.2, G, 'start', 700, mono=True)
        s += t(col1_x + col1_w - 24, ty + 15, td, 10.5, SEC, 'end')
    s += f'<line x1="{col1_x+16}" x2="{col1_x+col1_w-16}" y1="{top_y+258}" y2="{top_y+258}" stroke="{RULE}"/>'
    s += t(col1_x + 16, top_y + 278, 'Multi-Donor Word-Span Blend', 12.5, BLUE, 'start', 700)
    s += t(col1_x + 16, top_y + 297, '+ Qwen3.8-27B (layer_31 / concat4)', 11.5, BODY)
    s += t(col1_x + 16, top_y + 314, '+ DeepSeek-V4.1-Flash (layer_18 / c4)', 11.5, BODY)
    s += t(col1_x + 16, top_y + 332, f'3-way multi-kernel word cos {b3_mk:.4f}', 11, SEC)

    # Column 2: Closed-Form FP64 Bridge (Packet A Example)
    s += box(col2_x, top_y, col2_w, 344, stroke=AMBER, sw=2.0)
    s += t(col2_x + 16, top_y + 26, '2. CLOSED-FORM FP64 RIDGE (PACKET A)', 11.2, AMBER, 'start', 700, ls='0.06em')
    s += t(col2_x + 16, top_y + 52, 'Word-Span Aligned Linear Map', 16, INK, 'start', 700)
    s += box(col2_x + 16, top_y + 68, col2_w - 32, 54, stroke=AMBER, fill='#19150d', sw=1.0)
    s += t(col2_x + col2_w / 2, top_y + 90, 'R = (X_c^T X_c + lam I)^(-1) X_c^T Y_c', 12.2, AMBER, 'middle', 700, mono=True)
    s += t(col2_x + col2_w / 2, top_y + 111, 'y_hat = mu_Y + s * (x - mu_X) R', 12, INK, 'middle', 400, mono=True)

    bullets = [
        ('Packet A fit:', '640 prompts (13,872 word spans)'),
        ('Packet A time:', '355 s on CPU (0 GPU hours, s=1)'),
        ('Bridge sizes:', '22.5 to 271 MiB FP16 safetensors'),
        ('Replaced TEs:', '27.14 GB Qwen3-VL / 44.7 GiB Mistral'),
        ('V100 <-> Bw:', 'FP16 min_cos >= 0.99999992 (4 shown)'),
        ('Residual MLP:', '+0.00018 grp cos (p=1.0, null gain)'),
        ('Procrustes ST:', 'REJECTED (R^2 = -0.405 on 2304->5120)'),
    ]
    for i, (k, v) in enumerate(bullets):
        by = top_y + 146 + i * 27
        s += t(col2_x + 16, by, k, 11.5, INK, 'start', 700)
        s += t(col2_x + 112, by, v, 11.2, BODY if 'REJECTED' not in v else RED)

    s += arrow(col1_x + col1_w + 4, top_y + 170, col2_x - 4, top_y + 170, G, 2.5)
    s += t(col1_x + col1_w + 23, top_y + 158, 'x_i', 12, G, 'middle', 700, mono=True)

    # Column 3: Frozen Receivers
    rcards = [
        ('MiniMax-H3 (Confirm-160 + Dev-4 AV)', f'concat4 {pa_concat4_ret:.2f}% (90 MiB) · DINOv3 16/16 · 0 extra B to Dev-4 FL2VA', G),
        ('Wave B DiTs (Dev-24 img; C-160 word 4/6)', f'FLUX.2-dev {f2d_ret:.2f}% (44.7 GiB off; C-160 R^2=0.652) · FLUX.2-klein {f2k9_ret:.2f}% · FLUX.1 {f1dev_ret:.1f}/{f1sch_ret:.1f}%', G),
        ('Qwen3.8 L31 (C-160) & 3-Way Word-Span', f'L31/64 (50 MiB): {qw_st_ret:.2f}% cu130 · 3-way word-span cos {b3_mk:.4f} (16/16 vs Qwen)', BLUE),
        ('Measured Failure Boundaries', f'SDXL Dev-24 ({sdxl_b_ret:.1f}%/{sdxl_t_ret:.1f}%) · OCR Dev-192 (2/48) · Mouth Confirm-80 FAIL (4/6)', RED),
    ]
    ch = 78; gap = 11
    for i, (ctitle, cdesc, ccol) in enumerate(rcards):
        cy = top_y + i * (ch + gap)
        s += box(col3_x, cy, col3_w, ch, stroke=ccol, sw=1.5)
        s += t(col3_x + 14, cy + 24, ctitle, 13, ccol, 'start', 700)
        h, _ = para(col3_x + 14, cy + 43, cdesc, 11.2, BODY, col3_w - 28, 1.38)
        s += h
        s += arrow(col2_x + col2_w + 4, top_y + 170, col3_x - 4, cy + ch / 2, ccol, 1.8)

    y_bot = top_y + 362
    s += box(M, y_bot, CW, 66, stroke=RULE, fill='#0f1215', sw=1.2)
    s += t(M + 16, y_bot + 23, 'SEAM-SPECIFIC CAUSAL CONTROLS (SEE PAPER SECTION 2):', 11.2, INK, 'start', 700, ls='0.06em')
    s += t(M + 16, y_bot + 44, '(1) 0.0 RMSE Native Reinjection Floor  |  (2) Shuffled B0 (matched to primary single_tap; dev-only on blends)  |  (3) Sealed Confirm Splits (Packet A, Qwen L31, 3-Way word-span, Mouth)  |  (4) Wave B Dev-24', 11.2, BODY)

    f, H = footer(y_bot + 88, 'Summary architecture diagram combining JSON-loaded retention/cosine values (extra_sources.sha256.json) with Paper v0.9.3 seam annotations. Zero receiver text encoder weights are loaded for Plato Key or B0 inference arms.')
    return doc(s + f, H)


def fig4():
    s, y = header(
        'MULTI-DONOR & CROSS-ENCODER', G, 'MULTI-DONOR BLENDING BOUNDARY & WAVE B / CONFIRM-160 RETENTION',
        'Block-weighted multi-kernel ridge beats Qwen3.8 alone while naive stacking regresses; LLM/T5 receivers clear 91 to 99.6% while SDXL fails.',
        'Left: real N=1 to 3 donors (solid) and separate N=2 plus synthetic clone stress series branching from N=2 (dashed), plus Sealed Confirm-160 word-span cosines. Right: zero-text-encoder SigLIP2 retention across Confirm-160 and Wave B Dev-24.'
    )

    P_TOP = y + 10
    P_H = 386
    L_X, L_W = M, 550
    R_X, R_W = M + 586, CW - 586

    # --- LEFT PANEL: Separate Real N=1..3 (solid) and 2-Real + Stress Clones branching from N=2 (dashed) ---
    s += box(L_X, P_TOP, L_W, P_H, stroke=RULE, fill=PANEL, sw=1.2)
    s += t(L_X + 16, P_TOP + 24, 'A. REAL N=1..3 (SOLID) VS 2-REAL + CLONES FROM N=2 (DASHED)', 10.6, INK, 'start', 700, ls='0.04em')

    gx0, gx1 = L_X + 58, L_X + L_W - 22
    gy0, gy1 = P_TOP + 48, P_TOP + 220
    ymin, ymax = 0.935, 0.966

    def mx(n): return gx0 + (n - 1) * (gx1 - gx0) / 6.0
    def my(v): return gy1 - (max(ymin, min(ymax, v)) - ymin) / (ymax - ymin) * (gy1 - gy0)

    for v in (0.940, 0.945, 0.950, 0.955, 0.960, 0.965):
        yy = my(v)
        s += f'<line x1="{gx0:.1f}" x2="{gx1:.1f}" y1="{yy:.1f}" y2="{yy:.1f}" stroke="{RULE}"/>'
        s += t(gx0 - 8, yy + 4, f'{v:.3f}', 10.5, SEC, 'end', mono=True)
    for n in range(1, 8):
        xx = mx(n)
        s += f'<line x1="{xx:.1f}" x2="{xx:.1f}" y1="{gy0:.1f}" y2="{gy1:.1f}" stroke="{RULE}" stroke-dasharray="{"" if n <= 3 else "3 3"}"/>'
        lbl = f'N={n}' if n <= 2 else ('N=3 / 3*' if n == 3 else f'N={n}*')
        s += t(xx, gy1 + 16, lbl, 10.0, INK if n <= 3 else SEC, 'middle', 700 if n <= 3 else 400)
    s += f'<rect x="{mx(2)+3:.1f}" y="{gy0:.1f}" width="{gx1 - mx(2) - 3:.1f}" height="{gy1-gy0:.1f}" fill="#ffffff" fill-opacity="0.02"/>'
    s += t(mx(7) - 2, gy0 + 11, 'dashed: 2 real + clones from N=2', 9.2, SEC, 'end')

    rb_real = NREAL['regression_boundary_summary']
    rb_str = NSTRESS['regression_boundary_summary']
    series = [
        ('stacked_multikernel', 'Multi-Kernel', G, 2.8),
        ('convex_mse_opt', 'OOF Convex', BLUE, 2.4),
        ('stacked_rms_norm', 'RMS-Norm', AMBER, 1.8),
        ('stacked_energy_norm', 'Energy-Norm', '#d98a4e', 1.6),
        ('stacked_raw', 'Raw', RED, 1.8),
    ]
    for key, label, col, sw in series:
        real_vals = rb_real[key]['cosine_by_n']  # N=1, 2, 3 (3 real donors)
        str_vals = rb_str[key]['cosine_by_n']    # N=1, 2 (same 2 real) + N=3*, 4*, 5*, 6*, 7* (clones)
        # Draw stress continuation branching from shared N=2 (index 1 -> indices 2..6) FIRST
        pts_str = ' '.join(f'{mx(i+1):.1f},{my(v):.1f}' for i, v in enumerate(str_vals[1:], start=1))
        s += f'<polyline fill="none" stroke="{col}" stroke-width="{sw*0.78:.1f}" stroke-dasharray="4 3" stroke-opacity="0.80" points="{pts_str}"/>'
        # Draw real N=1..3 solid on top
        pts_real = ' '.join(f'{mx(i+1):.1f},{my(v):.1f}' for i, v in enumerate(real_vals))
        s += f'<polyline fill="none" stroke="{col}" stroke-width="{sw}" points="{pts_real}"/>'
        for i, v in enumerate(real_vals):
            s += f'<circle cx="{mx(i+1):.1f}" cy="{my(v):.1f}" r="3.8" fill="{col}"/>'

    # Explicit non-overlapping callouts distinguishing real N=3 (0.962435) vs flat stress (0.961592)
    s += t(mx(3) - 6, my(rb_real['stacked_multikernel']['cosine_by_n'][2]) - 8, 'real N=3: 0.96244', 9.4, G, 'end', 700)
    s += t(mx(7) - 2, my(rb_str['stacked_multikernel']['cosine_by_n'][4]) + 15, 'stress flat: 0.96159', 9.2, G, 'end')
    s += t(mx(4.15), my(0.9372), 'raw drops to 0.873 (off-scale)', 9.2, RED, 'end', 700)

    # Compact inline curve legend
    leg_items = [('Multi-Kernel', G), ('OOF Convex', BLUE), ('RMS-Norm', AMBER), ('Energy-Norm', '#d98a4e'), ('Raw', RED)]
    lx = L_X + 16
    for lname, lcol in leg_items:
        s += f'<line x1="{lx:.1f}" x2="{lx+14:.1f}" y1="{P_TOP+256}" y2="{P_TOP+256}" stroke="{lcol}" stroke-width="2.5"/>'
        s += t(lx + 18, P_TOP + 259.5, lname, 9.8, lcol, 'start', 700)
        lx += len(lname) * 6.4 + 34

    # Sealed Confirm-160 callout inside left panel
    ly = P_TOP + 276
    s += f'<line x1="{L_X+16}" x2="{L_X+L_W-16}" y1="{ly-8}" y2="{ly-8}" stroke="{RULE}"/>'
    s += t(L_X + 16, ly + 8, 'SEALED CONFIRM-160 (3-WAY WORD-SPAN, n=3,413, fit_calls=0):', 10.2, G, 'start', 700, ls='0.03em')
    c_rows = [
        (f'Multi-Kernel (Mode B): {b3_mk:.6f} (+0.00360 vs Qwen, 16/16, p=3.05e-5)', G),
        (f'OOF Convex (Mode A):   {b3_cvx:.6f} (+0.00313 vs Qwen, 16/16, p=3.05e-5)', BLUE),
        (f'Best Single (Qwen3.8): {b3_qwen:.6f} | vs 2-way (+0.000601 descriptive)', INK),
        (f'All 3 naive stacks < Qwen: raw {b3_raw:.5f} · rms {b3_rms:.5f} · eng {b3_eng:.5f}', RED),
    ]
    for i, (txt, col) in enumerate(c_rows):
        s += t(L_X + 16, ly + 27 + i * 19, txt, 10.6, col, 'start', 700 if i < 2 else 400, mono=True)

    # --- RIGHT PANEL: Two Explicitly Separated Sub-Axes (No Shared Formula Axis) ---
    s += box(R_X, P_TOP, R_W, P_H, stroke=RULE, fill=PANEL, sw=1.2)
    s += t(R_X + 16, P_TOP + 22, 'B. ZERO-TEXT-ENCODER SIGLIP2 RETENTION (SEPARATE AXES)', 10.6, INK, 'start', 700, ls='0.04em')

    bx0 = R_X + 236
    bx1 = R_X + R_W - 72
    bw = bx1 - bx0

    # Sub-panel B1: Sealed Confirm-160 (B0-anchored formula)
    b1_top = P_TOP + 34
    s += t(R_X + 16, b1_top + 11, 'B1. SEALED CONFIRM-160 · B0-anchored (cand-B0)/(teach-B0)', 9.6, G, 'start', 700)
    b1_bars = [
        ('MiniMax-H3 · Qwen3.8 L31 (C-160)', '50 MiB L31 (donor LLM resident)', qw_st_ret, G),
        ('MiniMax-H3 · Inkling c4 (C-160)', '90 MiB c4 (donor LLM resident)', pa_concat4_ret, G),
    ]
    b1_y0 = b1_top + 19
    b1_y1 = b1_y0 + len(b1_bars) * 35 + 4
    for pct, pcol in ((50, RULE), (80, RULE), (100, INK)):
        xx = bx0 + (pct / 100.0) * bw
        s += f'<line x1="{xx:.1f}" x2="{xx:.1f}" y1="{b1_y0}" y2="{b1_y1}" stroke="{pcol}" stroke-opacity="{"0.65" if pct==100 else "1"}" stroke-dasharray="{"4 3" if pct==100 else ""}"/>'
        s += t(xx, b1_y1 + 12, f'{pct}%', 9.2, pcol if pct == 100 else SEC, 'middle', mono=True)
    for i, (name, sub, val, col) in enumerate(b1_bars):
        by = b1_y0 + 3 + i * 35
        s += t(R_X + 16, by + 12, name, 10.8, INK, 'start', 700)
        s += t(R_X + 16, by + 24, sub, 9.4, SEC)
        wbar = max(4.0, (val / 100.0) * bw)
        s += f'<rect x="{bx0:.1f}" y="{by+2:.1f}" width="{wbar:.1f}" height="18" rx="2" fill="{col}"/>'
        lbl_x = bx1 + 8 if val >= 88.0 else bx0 + wbar + 8
        s += t(lbl_x, by + 15.5, f'{val:.2f}%', 11.0, col, 'start', 700, mono=True).replace('<text ', f'<text paint-order="stroke" stroke="{PANEL}" stroke-width="5" stroke-linejoin="round" ')

    # Divider between B1 and B2
    div_y = b1_y1 + 21
    s += f'<line x1="{R_X+16}" x2="{R_X+R_W-16}" y1="{div_y}" y2="{div_y}" stroke="{RULE}" stroke-width="1.2"/>'

    # Sub-panel B2: Unsealed Wave B Dev-24 (cand_margin / native_margin formula)
    b2_top = div_y + 15
    s += t(R_X + 16, b2_top, 'B2. UNSEALED WAVE B DEV-24 · cand/native (sep. B0 gate)', 9.6, BLUE, 'start', 700)
    b2_bars = [
        ('FLUX.2-dev (Dev-24)', '271 MiB key vs 44.7 GiB Mistral', f2d_ret, G),
        ('FLUX.2-klein-9B (Dev-24)', 'replaces Qwen3-8B encoder', f2k9_ret, G),
        ('FLUX.1-dev (Dev-24)', 'T5-XXL+CLIP-L (C-160 word R2=0.53)', f1dev_ret, BLUE),
        ('FLUX.1-schnell (Dev-24)', 'T5-XXL+CLIP-L (C-160 word R2=0.53)', f1sch_ret, BLUE),
        ('SDXL-base-1.0 (Dev-24 FAIL)', 'C-160 word R2=0.54 · pad hyp.', sdxl_b_ret, RED),
        ('SDXL-Turbo (Dev-24 FAIL)', 'C-160 word R2=0.54 · pad hyp.', sdxl_t_ret, RED),
    ]
    b2_y0 = b2_top + 8
    b2_y1 = b2_y0 + len(b2_bars) * 35 + 4
    for pct, pcol in ((50, RULE), (80, RULE), (100, INK)):
        xx = bx0 + (pct / 100.0) * bw
        s += f'<line x1="{xx:.1f}" x2="{xx:.1f}" y1="{b2_y0}" y2="{b2_y1}" stroke="{pcol}" stroke-opacity="{"0.65" if pct==100 else "1"}" stroke-dasharray="{"4 3" if pct==100 else ""}"/>'
        s += t(xx, b2_y1 + 12, f'{pct}%', 9.2, pcol if pct == 100 else SEC, 'middle', mono=True)
    for i, (name, sub, val, col) in enumerate(b2_bars):
        by = b2_y0 + 3 + i * 35
        s += t(R_X + 16, by + 12, name, 10.8, INK if col != RED else RED, 'start', 700)
        s += t(R_X + 16, by + 24, sub, 9.4, SEC)
        wbar = max(4.0, (val / 100.0) * bw)
        s += f'<rect x="{bx0:.1f}" y="{by+2:.1f}" width="{wbar:.1f}" height="18" rx="2" fill="{col}"/>'
        lbl_x = bx1 + 8 if val >= 88.0 else bx0 + wbar + 8
        s += t(lbl_x, by + 15.5, f'{val:.2f}%', 11.0, col, 'start', 700, mono=True).replace('<text ', f'<text paint-order="stroke" stroke="{PANEL}" stroke-width="5" stroke-linejoin="round" ')

    f, H = footer(P_TOP + P_H + 26, 'Left: real N=1..3 val_dev (solid) and separate 2-real + clone stress series from shared N=2 (dashed; convex_equal omitted at 0.622); Confirm-160 significance is vs Qwen3.8 alone (+0.000601 vs 2-way descriptive). Right: B1 (Sealed Confirm-160, B0-anchored formula) and B2 (Unsealed Wave B Dev-24, cand/native formula with separate B0 gate) are plotted on separate axes; Wave B Confirm-160 word bars 4/6 PASS (Confirm-160 image renders sealed).')
    return doc(s + f, H)


FIGS = {
    'fig0_architecture_pipeline': fig0,
    'fig4_multidonor_and_universal': fig4,
}
chrome = sorted(glob.glob(os.path.expanduser('~/.cache/ms-playwright/chromium_headless_shell-*/chrome-headless-shell-linux64/chrome-headless-shell')))
for name, fn in FIGS.items():
    svg = fn()
    assert chr(8212) not in svg and not re.search(r'job-\d|att-[0-9a-f]{6}|/mnt/|/home/', svg), f'banned text in {name}'
    path = os.path.join(HERE, name + '.svg')
    open(path, 'w').write(svg)
    H = int(re.search(r'height="(\d+)"', svg).group(1))
    if not ARGS.no_png:
        assert chrome, 'headless Chromium not found'
        subprocess.run([
            chrome[-1], '--no-sandbox', '--hide-scrollbars', '--force-device-scale-factor=2',
            f'--window-size={W},{H}', f'--screenshot={os.path.join(HERE, name + ".png")}',
            'file://' + path
        ], check=True, capture_output=True)
    print(name, W, H)

open(os.path.join(HERE, 'extra_sources.sha256.json'), 'w').write(json.dumps(SOURCES, indent=2, sort_keys=True) + '\n')
print('extra_sources.sha256.json written with', len(SOURCES), 'pins')
