"""
重算 results_master.csv 每一行的七点均值（mean7）。

用途：台账只存 clean / mid_band_mean / neg_0.3 / pos_0.3 四个字段，
mean7 需要由这四个数还原。本脚本把这个还原过程固定下来，避免手算出错
（例如 §12.11 与 §13.8 里就各出现过一次手算错误，见 论文.md §8 勘误 3、4）。

口径（与 experiments_ledger.md §0 一致）：
    mean7 = (acc(-0.3) + mid_band*4 + acc(0.0) + acc(+0.3)) / 7
    其中 mid_band = mean(acc(-0.2), acc(-0.1), acc(+0.1), acc(+0.2))

用法：
    python tools/recalc_mean7.py                     # 打印全部行
    python tools/recalc_mean7.py --filter budget     # 只看 run_id 含 "budget" 的行
    python tools/recalc_mean7.py --csv               # 输出带 mean7 列的新 CSV 到 stdout
"""
import argparse
import csv
import os

MASTER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "results_master.csv")


def mean7(clean, mid, neg, pos):
    return (neg + mid * 4.0 + clean + pos) / 7.0


def load_rows(path=MASTER):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def compute(row):
    """返回 mean7，字段缺失或非数值时返回 None。"""
    try:
        c = float(row["clean"])
        m = float(row["mid_band_mean"])
        n = float(row["neg_0.3"])
        p = float(row["pos_0.3"])
    except (KeyError, ValueError, TypeError):
        return None
    return mean7(c, m, n, p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--filter", default="", help="只保留 run_id 含该子串的行")
    ap.add_argument("--csv", action="store_true", help="输出整表（追加 mean7 列）到 stdout")
    args = ap.parse_args()

    rows = load_rows()
    if args.filter:
        rows = [r for r in rows if args.filter in r["run_id"]]

    if args.csv:
        fields = list(rows[0].keys()) + ["mean7"]
        w = csv.DictWriter(__import__("sys").stdout, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        for r in rows:
            v = compute(r)
            r = dict(r)
            r["mean7"] = "" if v is None else f"{v:.3f}"
            w.writerow(r)
        return

    print(f"{'run_id':<60} {'clean':>7} {'neg':>7} {'pos':>7} {'mid':>9} {'mean7':>8}")
    n_ok = 0
    for r in rows:
        v = compute(r)
        tag = "n/a" if v is None else f"{v:>8.3f}"
        if v is not None:
            n_ok += 1
        print(f"{r['run_id'][:60]:<60} {r['clean'] or '-':>7} {r['neg_0.3'] or '-':>7} "
              f"{r['pos_0.3'] or '-':>7} {r['mid_band_mean'] or '-':>9} {tag}")
    print(f"\n共 {len(rows)} 行，其中 {n_ok} 行可算 mean7。")


if __name__ == "__main__":
    main()
