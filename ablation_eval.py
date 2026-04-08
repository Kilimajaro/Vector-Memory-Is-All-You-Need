#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BIMS 小规模消融：在 LongMemEval 子集上对比完整模型与各组件移除变体。
指标与 eval_new 一致：检索 recall@k、QA correctness（Ollama 评判）。
"""
import argparse
import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_EVAL = _ROOT / "eval"
for _p in (_ROOT, _EVAL):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from eval_new import LongMemEvalEvaluator
from memory_manager import VectorMemoryManager as MemoryManager


# (run_tag, ablation_dict)；None 表示完整 BIMS
ABLATION_VARIANTS: List[Tuple[str, Optional[Dict[str, Any]]]] = [
    ("BIMS_full", None),
    ("BIMS_wo_time", {"no_temporal": True}),
    ("BIMS_wo_assoc", {"no_assoc": True}),
    ("BIMS_wo_dual_clust", {"single_stage_cluster": True}),
    ("BIMS_wo_semweight", {"balanced_sem_rec_weights": True}),
]


def _build_subset(
    evaluator: LongMemEvalEvaluator,
    dataset_type: str,
    sample_size: int,
    sample_seed: int,
) -> List[Dict[str, Any]]:
    all_inst = evaluator.load_benchmark_data(dataset_type)
    n = len(all_inst)
    if sample_size >= n:
        return list(all_inst)
    rng = np.random.default_rng(sample_seed)
    pick = rng.choice(n, size=sample_size, replace=False)
    return [all_inst[i] for i in pick]


def main():
    parser = argparse.ArgumentParser(description="BIMS 消融实验（LongMemEval）")
    parser.add_argument("--dataset", type=str, default="s", choices=["oracle", "s"])
    parser.add_argument("--sample_size", type=int, default=100)
    parser.add_argument("--sample_seed", type=int, default=42, help="子集无放回抽样种子")
    parser.add_argument("--config", type=str, default="eval_config.json")
    parser.add_argument("--model", type=str, default=None, help="覆盖 evaluation_llm")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/ablation",
        help="汇总 JSON 与各轮详细结果目录",
    )
    parser.add_argument("--save_detailed", action="store_true", help="保存每轮详细 JSON")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    base_eval = LongMemEvalEvaluator(MemoryManager(), args.config)
    if args.model:
        base_eval.config["evaluation_llm"] = args.model
    base_eval.config["output_dir"] = args.output_dir
    if args.save_detailed:
        base_eval.config["save_detailed_results"] = True
        base_eval.config["detailed_results_threshold"] = max(
            base_eval.config.get("detailed_results_threshold", 50),
            args.sample_size + 1,
        )

    retrieval_keys = ("min_summary_similarity", "knowledge_min_score")
    retrieval_params = {
        k: base_eval.config[k] for k in retrieval_keys if k in base_eval.config
    }

    subset = _build_subset(base_eval, args.dataset, args.sample_size, args.sample_seed)
    print(
        f"数据集 longmemeval_{args.dataset}.json | 子集 {len(subset)} 条 | seed={args.sample_seed}"
    )

    rows = []
    full_recall = full_qa = None

    for run_tag, ablation in ABLATION_VARIANTS:
        mm = MemoryManager(
            retrieval_params=retrieval_params if retrieval_params else None,
            ablation=ablation,
        )
        base_eval.memory_manager = mm
        metrics = base_eval.run_evaluation(
            dataset_type=args.dataset,
            instances=subset,
            run_tag=run_tag,
            skip_visualizations=True,
        )
        ov = metrics["overall"]
        recall = ov["avg_retrieval_recall"]
        qa = ov["avg_qa_correctness"]
        rows.append(
            {
                "variant": run_tag,
                "ablation": ablation or {},
                "avg_retrieval_recall": recall,
                "avg_qa_correctness": qa,
                "success_rate": ov.get("success_rate"),
            }
        )
        if run_tag == "BIMS_full":
            full_recall, full_qa = recall, qa

    summary = {
        "created_at": datetime.now().isoformat(),
        "dataset": args.dataset,
        "sample_size": len(subset),
        "sample_seed": args.sample_seed,
        "instance_ids": [x.get("instance_id") for x in subset],
        "evaluation_llm": base_eval.config["evaluation_llm"],
        "baseline": {"recall": full_recall, "qa_correctness": full_qa},
        "variants": rows,
        "qa_drop_pct_vs_full": [],
    }

    if full_recall is not None and full_qa is not None and full_qa > 1e-8:
        for r in rows:
            if r["variant"] == "BIMS_full":
                summary["qa_drop_pct_vs_full"].append(
                    {"variant": r["variant"], "qa_drop_pct": 0.0, "recall_delta": 0.0}
                )
            else:
                qa_drop = (full_qa - r["avg_qa_correctness"]) / full_qa * 100.0
                summary["qa_drop_pct_vs_full"].append(
                    {
                        "variant": r["variant"],
                        "qa_drop_pct": round(qa_drop, 3),
                        "recall_delta": round(r["avg_retrieval_recall"] - full_recall, 4),
                    }
                )

    out_path = os.path.join(
        args.output_dir,
        f"ablation_summary_{args.dataset}_n{len(subset)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 72)
    print("消融结果（均值）")
    print("=" * 72)
    print(f"{'变体':<22} {'Recall@k':>12} {'QA正确率':>12} {'QA降幅%(相对full)':>18}")
    print("-" * 72)
    for r in rows:
        drop = ""
        if r["variant"] != "BIMS_full" and full_qa and full_qa > 1e-8:
            drop = f"{(full_qa - r['avg_qa_correctness']) / full_qa * 100:.2f}%"
        else:
            drop = "—"
        print(
            f"{r['variant']:<22} {r['avg_retrieval_recall']:12.4f} {r['avg_qa_correctness']:12.4f} {drop:>18}"
        )
    print("-" * 72)
    print(f"汇总已写入: {out_path}")


if __name__ == "__main__":
    main()
