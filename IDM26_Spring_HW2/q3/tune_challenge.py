import argparse
import itertools
import os
import re
import subprocess


def run_and_get_auc(train, test, k, wu, wi, wp, ws, cwd):
    # We run the module as __main__ with constants overridden.
    code = f"""
import runpy
import hw2_q3_challenge as m
m.SVD_N_COMPONENTS={k}
m.W_USER={wu}
m.W_ITEM={wi}
m.W_POP={wp}
m.W_SVD={ws}
import sys
sys.argv=['hw2_q3_challenge.py', r'{train}', r'{test}']
runpy.run_module('hw2_q3_challenge', run_name='__main__')
"""
    subprocess.run(["python", "-c", code], cwd=cwd, check=True, stdout=subprocess.DEVNULL)

    out = subprocess.check_output(["python", "ROC_AUC_score.py"], cwd=cwd, text=True)
    m = re.search(r"ROC-AUC Score:\s*([0-9.]+)", out)
    if not m:
        raise RuntimeError("Could not parse ROC-AUC from ROC_AUC_score.py output")
    return float(m.group(1))


def main():
    ap = argparse.ArgumentParser(description="Simple hyperparameter tuner for hw2_q3_challenge.py")
    ap.add_argument("--train", default="ratings.txt")
    ap.add_argument("--test", default="val_test.txt")

    ap.add_argument("--svd", default="32,64,96,128", help="Comma-separated SVD ranks")

    ap.add_argument(
        "--weights",
        default="0.30,0.30,0.10,0.30;0.25,0.25,0.05,0.45;0.20,0.20,0.05,0.55;0.25,0.25,0.10,0.40",
        help="Semicolon-separated weight tuples: W_USER,W_ITEM,W_POP,W_SVD",
    )

    args = ap.parse_args()
    cwd = os.getcwd()

    svd_ks = [int(x) for x in args.svd.split(",") if x.strip()]
    weight_tuples = []
    for t in args.weights.split(";"):
        parts = [p.strip() for p in t.split(",")]
        if len(parts) != 4:
            raise ValueError(f"Bad weight tuple: {t}")
        wu, wi, wp, ws = map(float, parts)
        weight_tuples.append((wu, wi, wp, ws))

    best_auc = -1.0
    best_params = None

    for k, (wu, wi, wp, ws) in itertools.product(svd_ks, weight_tuples):
        auc = run_and_get_auc(args.train, args.test, k, wu, wi, wp, ws, cwd)
        print(f"k={k:3d} weights=({wu:.2f},{wi:.2f},{wp:.2f},{ws:.2f}) -> AUC={auc:.6f}")
        if auc > best_auc:
            best_auc = auc
            best_params = (k, wu, wi, wp, ws)
            print(f"NEW BEST: AUC={best_auc:.6f} params={best_params}")

    print("\nBEST:")
    print(f"  AUC={best_auc:.6f}")
    print(f"  SVD_N_COMPONENTS={best_params[0]}")
    print(f"  W_USER={best_params[1]}")
    print(f"  W_ITEM={best_params[2]}")
    print(f"  W_POP={best_params[3]}")
    print(f"  W_SVD={best_params[4]}")


if __name__ == "__main__":
    main()
