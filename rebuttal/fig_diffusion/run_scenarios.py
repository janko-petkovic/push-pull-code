import pickle, time
from dataclasses import replace
import numpy as np
from cam_kernel import Params, Model, fit_shapes, neighbour_weights

base = Params()
scenarios = {
    "low spread (D=13, tau=50 ms)":  dict(D_c=13.0, tau_c=0.05),
    "high spread (D=65, tau=200 ms)": dict(D_c=65.0, tau_c=0.2),
}
strengths = {"weak (peak ~0.5 uM)": 300.0, "strong (peak ~3 uM)": 1000.0}

results = {}
for sname, sp in scenarios.items():
    for iname, infl in strengths.items():
        p = replace(base, influx=infl, **sp)
        t = time.time(); r = Model(p).run(); dt = time.time() - t
        x = r["x"]
        r["fits"] = {sp_: fit_shapes(x, r[f"int_{sp_}"]) for sp_ in ("k", "n")}
        r["neigh"] = {sp_: neighbour_weights(x, r[f"int_{sp_}"]) for sp_ in ("k", "n")}
        results[(sname, iname)] = r
        i0 = x.size // 2
        print(f"\n{sname} | {iname} | peak c {r['peak_c'][i0]:.2f} uM | {dt:.0f} s")
        for sp_ in ("k", "n"):
            f = r["fits"][sp_]; nb = r["neigh"][sp_]
            print(f"  {sp_}: gauss sigma {f['gauss']['width']:.3f} um (1-R2 {f['gauss']['one_minus_r2']:.2e}) | "
                  f"exp lambda {f['exp']['width']:.3f} um (1-R2 {f['exp']['one_minus_r2']:.2e}) | "
                  f"neighbour @1um {nb[1.0]:.3f}, @2um {nb[2.0]:.4f}")

# grid convergence for one case
p = replace(base, influx=1000.0, D_c=65.0, tau_c=0.2)
for dx in (0.05, 0.025):
    r = Model(replace(p, dx=dx)).run()
    f = fit_shapes(r["x"], r["int_k"]); nb = neighbour_weights(r["x"], r["int_k"])
    print(f"convergence dx={dx}: k exp lambda {f['exp']['width']:.4f}, neighbour@1um {nb[1.0]:.4f}")

pickle.dump(results, open("results.pkl", "wb"))
