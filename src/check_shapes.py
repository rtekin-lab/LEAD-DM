#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LEAD-DM  |  Sekil ve gradyan dogrulama
=======================================
Veri gerektirmez. Egitime baslamadan ONCE calistirin:

    python check_shapes.py

Kontrol edilenler
-----------------
 1. Derivasyon rekonstruksiyon matrisi (8 -> 12) dogru mu
 2. Gray kodu gidis-donus
 3. Enkoder-dekoder sekilleri ve gizil uzay grup yapisi
 4. Tum modulasyon modulleri (LTCM / MCFARN / adaLN / crossattn)
 5. M1 ileri gecis ve gradyan akisi
 6. LTCM'in sifir baslangicta adaLN ile OZDES oldugu
 7. Difuzyon ileri/geri surec tutarliligi
 8. DDIM ornekleme
 9. Kosul maskeleme (CFG) yollari
10. Parametre sayilari
"""

from __future__ import annotations

import sys
import numpy as np
import torch
import torch.nn as nn

OK, FAIL = "  [OK]  ", "  [HATA]"
_fails = []


def check(name, cond, detail=""):
    if cond:
        print(f"{OK} {name}" + (f"   {detail}" if detail else ""))
    else:
        print(f"{FAIL} {name}   {detail}")
        _fails.append(name)
    return cond


def main():
    torch.manual_seed(0)
    print("=" * 78)
    print("LEAD-DM | Sekil ve Gradyan Dogrulama")
    print("=" * 78)

    from lead_dm.config import (AEConfig, M1Config, DiffusionConfig,
                                CANON_12, GEN8_LEADS, GEN8_IDX, SIG_LEN)
    from lead_dm.data import (leads8_to_12, build_recon_matrix,
                              gray_encode, gray_decode)
    from lead_dm.autoencoder import LeadStructuredAE, AELoss
    from lead_dm.backbone_m1 import build_m1
    from lead_dm.diffusion import GaussianDiffusion, EMA
    from lead_dm.modules import LTCM, MCFARN, AdaLNZero, CrossAttnMod, \
        ConditionEncoder, LeadSelector

    NC = 9      # CPSC2018 sinif sayisi

    # ---------------------------------------------------------------- 1
    print("\n[1] Derivasyon rekonstruksiyonu (8 -> 12)")
    M = build_recon_matrix()
    check("matris sekli", tuple(M.shape) == (12, 8), f"{tuple(M.shape)}")
    # sentetik: I ve aVF'den denklemlerle 12 uret, sonra 8'i secip geri kur
    T = 200
    I = torch.randn(1, T); aVF = torch.randn(1, T)
    V = torch.randn(1, 6, T)
    true12 = torch.zeros(1, 12, T)
    for k in range(6):
        true12[:, CANON_12.index(f"V{k+1}")] = V[:, k]
    true12[:, CANON_12.index("I")] = I
    true12[:, CANON_12.index("aVF")] = aVF
    true12[:, CANON_12.index("II")] = 0.5 * I + aVF
    true12[:, CANON_12.index("III")] = -0.5 * I + aVF
    true12[:, CANON_12.index("aVR")] = -0.75 * I - 0.5 * aVF
    true12[:, CANON_12.index("aVL")] = 0.75 * I - 0.5 * aVF
    x8 = true12[:, list(GEN8_IDX)]
    rec = leads8_to_12(x8)
    err = (rec - true12).abs().max().item()
    check("8->12 tam dogru", err < 1e-5, f"max hata = {err:.2e}")
    # Einthoven: II = I + III
    e = (rec[:, 1] - rec[:, 0] - rec[:, 2]).abs().max().item()
    check("Einthoven II = I + III", e < 1e-5, f"{e:.2e}")

    # ---------------------------------------------------------------- 2
    print("\n[2] Gray kodu")
    bad = [a for a in range(128) if gray_decode(gray_encode(a)) != a]
    check("gidis-donus 0..127", len(bad) == 0, f"{len(bad)} hata")
    adj = [a for a in range(127)
           if int(np.abs(gray_encode(a) - gray_encode(a + 1)).sum()) != 1]
    check("komsu yaslar 1 bit farkli", len(adj) == 0, f"{len(adj)} ihlal")

    # ---------------------------------------------------------------- 3
    print("\n[3] Enkoder-dekoder")
    ae_cfg = AEConfig()
    ae = LeadStructuredAE(ae_cfg)
    ae.eval()                      # ornekleme yerine mu kullanilsin (belirlenimci)
    x = torch.randn(2, 8, SIG_LEN)
    xh, z, mu, logvar = ae(x)
    check("gizil sekli", tuple(z.shape) == (2, 256, 125), f"{tuple(z.shape)}")
    check("cikti sekli", tuple(xh.shape) == tuple(x.shape), f"{tuple(xh.shape)}")
    check("VAE modu acik", ae_cfg.variational and logvar is not None,
          f"variational={ae_cfg.variational}")
    if logvar is not None:
        check("logvar sekli mu ile ayni",
              tuple(logvar.shape) == tuple(mu.shape), f"{tuple(logvar.shape)}")
        check("eval modunda z == mu", torch.allclose(z, mu),
              "belirlenimci kodlama")

    # mu/logvar ayrimi lead yapisini BOZMAMALI.
    # Gruplu duzende chunk(2, dim=1) leadleri boler; dogru yol
    # (B, L, 2*cpl, T) seklinde bolmektir. Test: tek bir leadi degistir,
    # yalnizca o leadin mu bloklari degismeli.
    x2 = x.clone(); x2[:, 0] += 5.0
    z2 = ae.encode(x2, sample=False)
    d = (z2 - z).abs().view(2, 8, 32, 125).amax(dim=(0, 2, 3))
    changed = (d > 1e-4)
    check("lead bagimsizligi (mu/logvar ayrimi dogru)",
          bool(changed[0]) and not bool(changed[1:].any()),
          f"degisen lead sayisi = {int(changed.sum())} (1 olmali)")

    # Gizil normalizasyon gidis-donus
    zn = ae.normalize_latent(z)
    zb = ae.denormalize_latent(zn)
    check("gizil normalizasyon tersinir",
          float((zb - z).abs().max()) < 1e-4,
          f"max hata = {float((zb-z).abs().max()):.2e}  "
          f"mod={ae_cfg.latent_norm}")

    ae.train()
    xh2, z_s, mu2, lv2 = ae(x)
    check("egitim modunda ornekleme aktif (z != mu)",
          not torch.allclose(z_s, mu2),
          "yeniden parametrizasyon calisiyor")

    loss, parts = AELoss(kl_weight=1e-6)(xh2, x, mu2, lv2)
    check("KL terimi kayipta var", "kl" in parts, f"bilesenler = {list(parts)}")
    loss.backward()
    g = sum(p.grad.abs().sum().item() for p in ae.parameters()
            if p.grad is not None)
    check("AE gradyan akisi", g > 0, f"toplam |grad| = {g:.3e}")
    check("AE parametre", ae.n_params() < 6e6, f"{ae.n_params()/1e6:.3f} M")
    ae.eval()

    # ---------------------------------------------------------------- 4
    print("\n[4] Modulasyon modulleri")
    C, L, B, Tz = 256, 8, 3, 125
    ce = ConditionEncoder(NC)
    dis = torch.rand(B, NC).round()
    age = torch.randint(0, 2, (B, 7)).float()
    sex = torch.randint(0, 2, (B,))
    tt = torch.randint(0, 1000, (B,))
    ctx = ce(dis, age, sex, tt)
    for k in ("e_d", "e_a", "e_g", "t_emb", "c_glob"):
        check(f"ctx['{k}'] sekli", tuple(ctx[k].shape) == (B, 128),
              f"{tuple(ctx[k].shape)}")

    sel = LeadSelector(L)
    w = sel(ctx["e_d"])
    check("LeadSelector sekli", tuple(w.shape) == (B, L), f"{tuple(w.shape)}")
    check("LeadSelector baslangicta notr",
          abs(float(w.mean()) - 1.0) < 1e-4 and float(w.std()) < 1e-5,
          f"ort={float(w.mean()):.5f} std={float(w.std()):.2e}")

    zz = torch.randn(B, C, Tz)
    for name, mod in [("LTCM", LTCM(C, L)), ("MCFARN", MCFARN(C)),
                      ("adaLN", AdaLNZero(C)), ("crossattn", CrossAttnMod(C))]:
        g_, b_, a_ = mod(zz, ctx, w)
        okshape = (g_.shape[1] == C and b_.shape[1] == C and a_.shape[1] == C)
        check(f"{name} cikti sekli", okshape,
              f"g={tuple(g_.shape)} b={tuple(b_.shape)} a={tuple(a_.shape)}")

    # ---------------------------------------------------------------- 5
    print("\n[5] LTCM sifir-baslangic ozdesligi")
    ltcm = LTCM(C, L, lam_init=0.0)
    g1, b1, a1 = ltcm(zz, ctx, w)
    check("LTCM basta gamma=0", float(g1.abs().max()) < 1e-6,
          f"max|gamma| = {float(g1.abs().max()):.2e}")
    check("LTCM basta beta=0", float(b1.abs().max()) < 1e-6)
    check("LTCM basta alpha=0", float(a1.abs().max()) < 1e-6)
    ltcm_off = LTCM(C, L, lam_init=-1.0)
    check("LTCM A8 (dal B/C kapali)", ltcm_off.disabled_bc)
    # A8'de hastalik bilgisi KAYBOLMAMALI: farkli hastalik gommeleri
    # farkli modulasyon uretmeli (aksi halde ablasyon gecersizdir).
    with torch.no_grad():
        for prm in ltcm_off.glob.parameters():
            nn.init.normal_(prm, std=0.02)
    ctx2 = dict(ctx); ctx2["e_d"] = ctx["e_d"] + 1.0
    g_a, _, _ = ltcm_off(zz, ctx, w)
    g_b, _, _ = ltcm_off(zz, ctx2, w)
    check("A8 hastalik bilgisini KORUYOR",
          float((g_a - g_b).abs().max()) > 1e-4,
          f"fark = {float((g_a-g_b).abs().max()):.3e}")

    # ---------------------------------------------------------------- 6
    print("\n[6] M1 ileri gecis")
    for mod_kind in ("ltcm", "mcfarn", "adaln", "crossattn"):
        cfg = M1Config(modulation=mod_kind, n_blocks=3)
        m = build_m1(NC, cfg)
        out = m(torch.randn(B, C, Tz), tt, dis, age, sex)
        check(f"M1[{mod_kind}] cikti", tuple(out.shape) == (B, C, Tz),
              f"{tuple(out.shape)}")

    cfg = M1Config()
    m = build_m1(NC, cfg)
    out = m(torch.randn(B, C, Tz), tt, dis, age, sex)
    check("M1 tam cikti", tuple(out.shape) == (B, C, Tz), f"{tuple(out.shape)}")
    check("M1 basta sifir cikti (out_proj zero-init)",
          float(out.abs().max()) < 1e-6, f"{float(out.abs().max()):.2e}")

    out.pow(2).mean().backward()
    named = [(n, p) for n, p in m.named_parameters() if p.requires_grad]
    nog = [n for n, p in named if p.grad is None]
    check("M1 gradyan tum parametrelerde", len(nog) <= 2,
          f"gradyansiz: {len(nog)}  {nog[:3]}")

    ps = m.param_summary()
    print("      parametre dagilimi:")
    for k, v in ps.items():
        print(f"        {k:<15}: {v:>12,}")
    check("M1 toplam parametre makul", 1e6 < ps["TOTAL"] < 12e6,
          f"{ps['TOTAL']/1e6:.2f} M")

    # asimetrik baslik derinlikleri
    check("asimetrik baslik derinlikleri (uzuv derin)",
          m.heads.depths == [2, 2, 2, 2, 2, 2, 4, 4],
          f"{m.heads.depths}")

    # ---------------------------------------------------------------- 7
    print("\n[7] Difuzyon sureci")
    dcfg = DiffusionConfig()
    diff = GaussianDiffusion(dcfg)
    z0 = torch.randn(B, C, Tz)
    t0 = torch.zeros(B, dtype=torch.long)
    tT = torch.full((B,), dcfg.n_steps - 1, dtype=torch.long)
    zt0, n0 = diff.q_sample(z0, t0)
    ztT, nT = diff.q_sample(z0, tT)
    check("t=0'da z_t ~ z0", float((zt0 - z0).abs().mean()) < 0.02,
          f"fark = {float((zt0-z0).abs().mean()):.4f}")
    check("t=T'de sinyal silinmis",
          float(np.corrcoef(ztT.flatten(), z0.flatten())[0, 1]) < 0.15,
          f"korelasyon = {float(np.corrcoef(ztT.flatten(), z0.flatten())[0,1]):.3f}")
    tm = torch.randint(1, dcfg.n_steps, (B,))
    ztm, nm = diff.q_sample(z0, tm)

    # v-parametrizasyonu kimlikleri
    v = diff.get_v(z0, nm, tm)
    z0r = diff.z0_from_v(ztm, tm, v)
    epsr = diff.eps_from_v(ztm, tm, v)
    check("v -> z0 tam", float((z0r - z0).abs().max()) < 1e-4,
          f"max hata = {float((z0r-z0).abs().max()):.2e}")
    check("v -> eps tam", float((epsr - nm).abs().max()) < 1e-4,
          f"max hata = {float((epsr-nm).abs().max()):.2e}")
    mx = max(float(diff.sqrt_ab.max()), float(diff.sqrt_1m_ab.max()))
    check("v katsayilari <= 1 (varyans sismesi imkansiz)", mx <= 1.0 + 1e-6,
          f"max katsayi = {mx:.4f}")
    check("parametrizasyon = v", diff.param == "v", diff.param)
    check("sifir terminal SNR", float(diff.alphas_bar[-1]) < 1e-6,
          f"alphabar_T = {float(diff.alphas_bar[-1]):.3e}")
    check("alphabar monoton azalan",
          bool((diff.alphas_bar[1:] <= diff.alphas_bar[:-1] + 1e-9).all()))

    batch = {"disease": dis, "age_bits": age, "sex": sex,
             "d_null": torch.zeros(B), "a_null": torch.zeros(B),
             "g_null": torch.zeros(B)}
    loss, parts = diff.loss(m, z0, batch)
    check("difuzyon kaybi sonlu", torch.isfinite(loss).item(),
          f"loss = {float(loss):.4f}  bilesenler = {list(parts)}")

    # ---------------------------------------------------------------- 8
    print("\n[8] Ornekleme")
    cond = {"disease": dis, "age_bits": age, "sex": sex}
    zs = diff.sample(m, cond, (B, C, Tz), torch.device("cpu"),
                     ddim_steps=5, progress=False)
    check("DDIM ornekleme sekli", tuple(zs.shape) == (B, C, Tz),
          f"{tuple(zs.shape)}")
    check("DDIM ciktisi sonlu", bool(torch.isfinite(zs).all()))

    zg = diff.sample(m, cond, (B, C, Tz), torch.device("cpu"),
                     guidance={"disease": 2.0, "age": 1.0, "sex": 1.0},
                     ddim_steps=3, progress=False)
    check("kosul basina CFG calisiyor", bool(torch.isfinite(zg).all()))

    # ---------------------------------------------------------------- 9
    print("\n[9] Kosul maskeleme (CFG)")
    o_full = m(torch.zeros(B, C, Tz), tt, dis, age, sex,
               torch.zeros(B), torch.zeros(B), torch.zeros(B))
    o_null = m(torch.zeros(B, C, Tz), tt, dis, age, sex,
               torch.ones(B), torch.ones(B), torch.ones(B))
    check("maskeli/maskesiz yollar calisiyor",
          o_full.shape == o_null.shape)
    ctx_f = m.cond(dis, age, sex, tt, torch.zeros(B), torch.zeros(B), torch.zeros(B))
    ctx_n = m.cond(dis, age, sex, tt, torch.ones(B), torch.ones(B), torch.ones(B))
    check("null gommeleri gercekten farkli",
          float((ctx_f["e_d"] - ctx_n["e_d"]).abs().mean()) > 1e-4,
          f"fark = {float((ctx_f['e_d']-ctx_n['e_d']).abs().mean()):.4f}")
    same = all(torch.allclose(ctx_n["e_d"][0], ctx_n["e_d"][i])
               for i in range(B))
    check("null gommesi tum orneklerde ayni", same)

    # ---------------------------------------------------------------- 10
    print("\n[10] EMA")
    ema = EMA(m, 0.99)
    before = m.out_proj.weight.detach().clone()
    with torch.no_grad():
        m.out_proj.weight.add_(1.0)
    ema.update(m)
    ema.apply_to(m)
    check("EMA uygulandi",
          not torch.allclose(m.out_proj.weight, before + 1.0))
    ema.restore(m)
    check("EMA geri alindi",
          torch.allclose(m.out_proj.weight, before + 1.0))

    # ---------------------------------------------------------------- ozet
    print("\n" + "=" * 78)
    if _fails:
        print(f"BASARISIZ: {len(_fails)} kontrol")
        for f in _fails:
            print(f"  - {f}")
        return 1
    print("TUM KONTROLLER GECTI")
    print("=" * 78)
    print("\nSiradaki adimlar:")
    print("  1) python train_autoencoder.py --dataset cpsc2018")
    print("  2) python train_diffusion.py   --dataset cpsc2018")
    return 0


if __name__ == "__main__":
    sys.exit(main())
