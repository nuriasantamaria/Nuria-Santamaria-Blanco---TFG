
"""
Se usan tests no paramétricos pareados: Friedman como ómnibus y, si sale significativo, post-hoc de
Wilcoxon pareado contra una referencia, con corrección de Holm para las
comparaciones múltiples y tamaño del efecto por correlación
rank-biserial (Kerby, 2014).

Uso: python3 analisis.py  (depende solo de numpy y scipy.stats).
"""

import numpy as np
from scipy.stats import wilcoxon, friedmanchisquare, rankdata


# Formato de cada diccionario: {semilla: (tiempo_cruce_min, magnetopares_Am2, wfinal_degs)}

SEEDS = [1234, 1235, 1236, 1237, 1238, 1239, 1240, 1241, 1242, 1243, 1244, 1245, 1246, 1247, 1248]

ppo_k0 = {
1234:(104.8,4.4852,0.1937),1235:(28.4,4.3734,0.1911),1236:(62.5,4.2607,0.2009),
1237:(48.4,5.2876,0.3186),1238:(62.4,4.4632,0.1917),1239:(58.0,4.1236,0.1974),
1240:(22.2,4.4108,0.2067),1241:(39.2,4.4055,0.2203),1242:(52.7,4.3696,0.2343),
1243:(56.9,4.3982,0.1765),1244:(54.0,4.3655,0.2084),1245:(67.4,4.4310,0.2640),
1246:(13.2,4.2637,0.2091),1247:(105.3,4.3020,0.1766),1248:(54.0,4.3842,0.2027)}

ppo_k2 = {
1234:(61.7,1.7838,0.1964),1235:(34.9,1.5075,0.1823),1236:(59.0,1.5933,0.1664),
1237:(49.6,1.5126,0.2196),1238:(63.8,1.8250,0.1827),1239:(19.7,1.3720,0.1606),
1240:(99.5,1.6977,0.1945),1241:(17.9,1.4754,0.1587),1242:(23.2,1.5447,0.2060),
1243:(58.5,1.6573,0.1602),1244:(57.6,1.5880,0.2061),1245:(24.7,1.5100,0.1585),
1246:(53.6,1.3449,0.1598),1247:(56.0,1.6046,0.1729),1248:(56.0,1.6213,0.1745)}

ppo_k4 = {
1234:(97.7,6.5170,0.1988),1235:(107.2,6.4576,0.2229),1236:(44.8,6.5571,0.1812),
1237:(54.7,6.5480,0.1749),1238:(52.1,6.6464,0.1907),1239:(143.6,6.2776,0.2393),
1240:(149.6,6.4407,0.2402),1241:(42.8,6.6522,0.1840),1242:(30.0,6.6532,0.1832),
1243:(54.0,6.6594,0.1878),1244:(45.2,6.6008,0.2177),1245:(45.0,6.5323,0.2303),
1246:(16.8,5.7611,0.2376),1247:(58.0,6.6008,0.1756),1248:(47.4,6.6792,0.1674)}

ppo_k8 = {
1234:(102.8,9.0396,0.1530),1235:(22.1,8.3799,0.1582),1236:(54.8,6.9138,0.1732),
1237:(58.9,6.5672,0.1905),1238:(66.7,9.1290,0.2099),1239:(18.9,8.0585,0.1923),
1240:(45.2,8.9675,0.1987),1241:(51.6,6.6472,0.1695),1242:(55.2,8.4564,0.1643),
1243:(55.0,6.7593,0.1468),1244:(107.2,6.8421,0.1521),1245:(47.2,6.5231,0.2684),
1246:(22.5,7.1314,0.1727),1247:(66.9,6.7952,0.1958),1248:(29.4,7.0801,0.1586)}

sac_k0 = {
1234:(92.0,15.0807,0.2327),1235:(53.8,15.4725,0.2862),1236:(28.9,15.6546,0.3537),
1237:(103.8,14.8449,0.3203),1238:(140.6,15.5026,0.2673),1239:(57.6,15.4426,0.2061),
1240:(48.7,15.5965,0.2394),1241:(22.0,15.7033,0.3527),1242:(37.8,15.5820,0.1586),
1243:(183.7,15.5207,0.3447),1244:(57.9,15.3109,0.2578),1245:(136.7,15.5190,0.3476),
1246:(105.2,15.5572,0.2836),1247:(61.0,15.4142,0.1649),1248:(52.2,14.9011,0.1781)}

sac_k2 = {
1234:(186.9,8.2037,0.1713),1235:(148.3,7.7476,0.1811),1236:(15.4,7.6217,0.2241),
1237:(193.4,7.9982,0.1617),1238:(102.2,7.7478,0.2259),1239:(192.7,7.7374,0.1677),
1240:(18.9,7.6488,0.1953),1241:(13.2,7.7354,0.2048),1242:(27.4,7.7278,0.2218),
1243:(13.4,7.6259,0.1935),1244:(151.3,7.8746,0.2202),1245:(192.4,7.8648,0.1778),
1246:(148.0,7.6631,0.2636),1247:(147.4,7.6775,0.1659),1248:(149.7,7.8762,0.2972)}

sac_k4 = {
1234:(69.1,6.3394,0.3709),1235:(71.9,6.2529,0.1871),1236:(56.6,6.4792,0.1804),
1237:(49.9,6.3579,0.1751),1238:(68.4,6.5660,0.1928),1239:(53.2,6.1589,0.2694),
1240:(66.2,6.5422,0.2089),1241:(94.3,6.5361,0.3714),1242:(44.4,6.4422,0.3710),
1243:(60.0,6.5710,0.2613),1244:(57.6,6.3869,0.1774),1245:(98.0,6.2802,0.2196),
1246:(43.6,6.0942,0.2070),1247:(95.8,6.3805,0.3708),1248:(82.8,6.3818,0.1729)}

sac_k8 = {
1234:(680.0,14.4174,0.5298),1235:(97.8,14.2253,0.5447),1236:(680.0,13.8888,0.3815),
1237:(247.7,14.3194,0.2240),1238:(680.0,14.2500,0.3473),1239:(55.3,13.8425,0.3108),
1240:(680.0,14.3764,0.5091),1241:(680.0,13.8834,0.3808),1242:(14.3,13.7845,0.2138),
1243:(46.2,13.8432,0.3799),1244:(392.7,19.5714,0.1953),1245:(436.7,13.9163,0.4232),
1246:(680.0,13.9608,0.5382),1247:(680.0,14.2939,0.3701),1248:(246.9,14.3623,0.1843)}

# RecurrentPPO 
rppo = {
1234:(55.9,3.0246,0.1970),1235:(45.3,2.9403,0.1907),1236:(54.1,3.0112,0.2530),
1237:(55.2,2.7457,0.1676),1238:(99.8,3.0899,0.1655),1239:(61.1,2.9088,0.1957),
1240:(102.5,3.0466,0.1784),1241:(46.8,3.1340,0.1694),1242:(58.1,3.1396,0.1912),
1243:(62.8,3.2299,0.1974),1244:(73.5,3.1602,0.1882),1245:(57.6,3.0698,0.1899),
1246:(55.4,2.8823,0.1900),1247:(57.0,3.0127,0.2182),1248:(23.8,2.8926,0.2595)}

# Controlador Bdot clásico (k=1e6)
bdot = {
1234:(177.7,0.3241,0.1500),1235:(115.0,0.1947,0.1439),1236:(226.8,0.2714,0.1985),
1237:(190.7,0.1738,0.1627),1238:(213.6,0.3591,0.1955),1239:(190.9,0.1439,0.1820),
1240:(169.8,0.2760,0.1477),1241:(167.8,0.2766,0.1982),1242:(156.4,0.2736,0.1521),
1243:(277.8,0.2863,0.1799),1244:(277.2,0.3030,0.1707),1245:(105.8,0.2526,0.1793),
1246:(226.6,0.1727,0.1644),1247:(168.7,0.1928,0.1484),1248:(170.9,0.2602,0.1973)}

# SAC con hiperparámetros optimizados 
sac_opt = {
1234:(106.4,5.4377,0.2071),1235:(20.4,5.1078,0.1821),1236:(96.5,5.2201,0.1842),
1237:(55.8,5.1120,0.2157),1238:(116.2,5.4796,0.1789),1239:(15.3,5.0980,0.2262),
1240:(47.6,5.2816,0.2046),1241:(25.9,5.1793,0.2071),1242:(23.4,5.1946,0.1941),
1243:(99.0,5.1900,0.2069),1244:(60.1,5.1898,0.2043),1245:(25.3,5.1674,0.1800),
1246:(55.1,5.1067,0.2120),1247:(85.9,5.1981,0.2027),1248:(51.3,5.1393,0.2283)}

METRIC_NAMES = ["Tiempo cruce [min]", "Magnetopares [A·m²]", "wfinal [°/s]"]


def to_array(d, idx):
    """Vector de la métrica `idx` (0,1,2) ordenado por SEEDS, para que las
    configuraciones queden alineadas semilla a semilla (tests pareados)."""
    return np.array([d[s][idx] for s in SEEDS], dtype=float)


def descriptive_stats(configs, names):
    """Imprime media ± desviación típica muestral (ddof=1) de cada configuración, en las 3 métricas."""
    print("=" * 70)
    print("ESTADÍSTICA DESCRIPTIVA (media ± desv. típica, ddof=1)")
    print("=" * 70)
    header = f"{'Métrica':<22}" + "".join(f"{n:>16}" for n in names)
    print(header)
    print("-" * len(header))
    for idx, mname in enumerate(METRIC_NAMES):
        row = f"{mname:<22}"
        for d in configs:
            arr = to_array(d, idx)
            row += f"{arr.mean():>8.3f}±{arr.std(ddof=1):<7.3f}"
        print(row)
    print()


def holm_correction(pvals):
    p = np.asarray(pvals, dtype=float)
    m = p.size
    if m == 0:
        return p.copy()

    order = np.argsort(p)
    p_sorted = p[order]

    adj_sorted = np.empty(m, dtype=float)
    running_max = 0.0
    for i in range(m):
        candidate = min(p_sorted[i] * (m - i), 1.0)
        running_max = max(running_max, candidate)
        adj_sorted[i] = running_max

    adj = np.empty(m, dtype=float)
    adj[order] = adj_sorted
    return adj


def rank_biserial_r(x, y):
    d = x - y
    d_nonzero = d[d != 0]
    n_eff = len(d_nonzero)
    if n_eff == 0:
        return 0.0, 0

    ranks = rankdata(np.abs(d_nonzero))  
    W_plus = ranks[d_nonzero > 0].sum()
    W_minus = ranks[d_nonzero < 0].sum()
    total = W_plus + W_minus
    r = (W_plus - W_minus) / total
    return r, n_eff


def friedman_test(configs, names, label):
    print("#" * 70)
    print(f"# TEST DE FRIEDMAN (ómnibus) — {label}")
    print(f"#   Configuraciones: {', '.join(names)}")
    print("#" * 70)
    results = {}
    for idx, mname in enumerate(METRIC_NAMES):
        samples = [to_array(d, idx) for d in configs]
        stat, p = friedmanchisquare(*samples)
        sig = p < 0.05
        flag = "SIGNIFICATIVO" if sig else "no significativo"
        print(f"  {mname:<22}  chi2 = {stat:8.4f}   p = {p:.4e}   -> {flag}")
        results[mname] = (stat, p, sig)
    print()
    return results


def wilcoxon_posthoc(ref, others_dict, ref_name, label,
                     metrics_to_test=None, holm=True):
    if metrics_to_test is None:
        metrics_to_test = [0, 1, 2]

    print("#" * 70)
    print(f"# POST-HOC WILCOXON PAREADO — {label}  (referencia: {ref_name})")
    print("#" * 70)

    if len(metrics_to_test) == 0:
        print("  No hay métricas con Friedman significativo; se omite post-hoc.\n")
        return [], np.array([])

    rows = []
    raw_pvals = []
    for idx in metrics_to_test:
        mname = METRIC_NAMES[idx]
        x = to_array(ref, idx)
        for oname, odata in others_dict.items():
            y = to_array(odata, idx)
            W, p = wilcoxon(x, y)
            r, n_eff = rank_biserial_r(x, y)
            rows.append((mname, oname, W, p, r, n_eff))
            raw_pvals.append(p)

    raw_pvals = np.array(raw_pvals, dtype=float)
    adj_pvals = holm_correction(raw_pvals) if holm else raw_pvals.copy()

    head = (f"  {'Métrica':<22}{'Comparación':<16}{'W':>8}"
            f"{'p':>13}{'p_Holm':>13}{'r':>8}{'N_eff':>7}{'Sig.':>7}")
    print(head)
    print("  " + "-" * (len(head) - 2))
    for (mname, oname, W, p, r, n_eff), padj in zip(rows, adj_pvals):
        comp = f"{ref_name} vs {oname}"
        sig = "Sí" if padj < 0.05 else "No"
        print(f"  {mname:<22}{comp:<16}{W:>8.1f}{p:>13.4e}{padj:>13.4e}"
              f"{r:>8.3f}{n_eff:>7}{sig:>7}")
    print()

    return rows, adj_pvals


def run_ppo_analysis():
    """PPO: descriptiva -> Friedman -> post-hoc Wilcoxon (+r) contra k=2 (mejor config.),
    restringido a las métricas donde Friedman salió significativo."""
    print("\n" + "=" * 70)
    print(" ANÁLISIS PPO (history length k = 0, 2, 4, 8)")
    print("=" * 70)

    configs = [ppo_k0, ppo_k2, ppo_k4, ppo_k8]
    names = ["k=0", "k=2", "k=4", "k=8"]

    descriptive_stats(configs, names)
    fried = friedman_test(configs, names, label="PPO")

    sig_metrics_idx = [idx for idx, mname in enumerate(METRIC_NAMES)
                       if fried[mname][2]]
    sig_names = [METRIC_NAMES[i] for i in sig_metrics_idx]
    print(f"  Métricas con Friedman significativo (post-hoc aplicable): "
          f"{sig_names if sig_names else 'ninguna'}\n")

    wilcoxon_posthoc(
        ref=ppo_k2,
        others_dict={"k=0": ppo_k0, "k=4": ppo_k4, "k=8": ppo_k8},
        ref_name="k=2",
        label="PPO",
        metrics_to_test=sig_metrics_idx,
        holm=True,
    )


def run_sac_analysis():
    print("\n" + "=" * 70)
    print(" ANÁLISIS SAC (history length k = 0, 2, 4, 8)")
    print("=" * 70)

    configs = [sac_k0, sac_k2, sac_k4, sac_k8]
    names = ["k=0", "k=2", "k=4", "k=8"]

    descriptive_stats(configs, names)
    fried = friedman_test(configs, names, label="SAC")

    sig_metrics_idx = [idx for idx, mname in enumerate(METRIC_NAMES)
                       if fried[mname][2]]
    sig_names = [METRIC_NAMES[i] for i in sig_metrics_idx]
    print(f"  Métricas con Friedman significativo (post-hoc aplicable): "
          f"{sig_names if sig_names else 'ninguna'}\n")

    wilcoxon_posthoc(
        ref=sac_k4,
        others_dict={"k=0": sac_k0, "k=2": sac_k2, "k=8": sac_k8},
        ref_name="k=4",
        label="SAC",
        metrics_to_test=sig_metrics_idx,
        holm=True,
    )


def run_rppo_vs_bdot_analysis():
    print("\n" + "=" * 70)
    print(" ANÁLISIS RPPO vs BDOT CLÁSICO (sin comparación de historial)")
    print("=" * 70)

    descriptive_stats([rppo, bdot], ["RPPO", "Bdot"])


def run_ppo_vs_bdot_analysis():
    print("\n" + "=" * 70)
    print(" ANÁLISIS PPO k=2 vs BDOT CLÁSICO (sin comparación de historial)")
    print("=" * 70)

    descriptive_stats([ppo_k2, bdot], ["PPO k=2", "Bdot"])


def run_sac_opt_vs_bdot_analysis():
    print("\n" + "=" * 70)
    print(" ANÁLISIS SAC OPTIMIZADO vs BDOT CLÁSICO (sin comparación de historial)")
    print("=" * 70)

    descriptive_stats([sac_opt, bdot], ["SAC opt", "Bdot"])

if __name__ == "__main__":
    run_ppo_analysis()
    run_sac_analysis()
    run_rppo_vs_bdot_analysis()
    run_ppo_vs_bdot_analysis()
    run_sac_opt_vs_bdot_analysis()
    print("\n" + "=" * 70)
    print("Análisis completo.")
    print("=" * 70)