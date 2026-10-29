from dataclasses import replace
import numpy as np, pickle
from cam_kernel import Params, Model, fit_shapes, neighbour_weights
base = replace(Params(), D_c=13.0, tau_c=0.05)
rows = []
for D_m in (10.0, 20.0):
    for koff_m in (3.0, 10.0, 30.0, 100.0):
        for infl, lab in ((300.0, "weak"), (1000.0, "strong")):
            r = Model(replace(base, D_m=D_m, koff_m=koff_m, influx=infl)).run()
            fk = fit_shapes(r["x"], r["int_k"]); fn = fit_shapes(r["x"], r["int_n"])
            nk = neighbour_weights(r["x"], r["int_k"])
            rows.append((D_m, koff_m, lab, np.sqrt(D_m/koff_m), fk["exp"]["width"], fk["gauss"]["width"],
                         fk["gauss"]["one_minus_r2"], fk["exp"]["one_minus_r2"], fn["gauss"]["width"], nk[1.0]))
            print("D_m %4.0f koff_m %5.0f %-6s sqrt(D/koff) %.2f | K: lam %.2f sigma %.2f (1-R2 g %.1e, e %.1e) | N sigma %.2f | K neigh@1um %.2f" % rows[-1])
pickle.dump(rows, open("sweep.pkl", "wb"))
