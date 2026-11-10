"""Render a completed XLinear water soft-sensor run into a Markdown report.

Only completed run directories containing summary.json are accepted.  This
script never starts training, changes weights, or reads in-progress station
outputs.  Example: python -B make_water_report.py --results results
"""

import argparse
import csv
import json
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent
TARGETS = [
    "高锰酸盐指数(mg/L)",
    "氨氮(mg/L)",
    "总磷(mg/L)",
    "总氮(mg/L)",
]
SHORT = {name: name.split("(")[0] for name in TARGETS}


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def number(value, digits=4):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "NA"
    return f"{value:.{digits}g}" if math.isfinite(value) else "NA"


def integer(value):
    return str(int(float(value)))


def md_table(headers, rows):
    def cell(value):
        return str(value).replace("|", "／").replace("\n", " ")

    return (
        "| " + " | ".join(map(cell, headers)) + " |\n"
        + "| " + " | ".join("---" for _ in headers) + " |\n"
        + "".join("| " + " | ".join(map(cell, row)) + " |\n" for row in rows)
    )


def aggregate_metadata(rows, key):
    values = [int(float(row[key])) for row in rows if row.get(key, "") != ""]
    if not values:
        return "NA"
    ordered = sorted(values)
    middle = len(ordered) // 2
    median = (
        ordered[middle]
        if len(ordered) % 2
        else (ordered[middle - 1] + ordered[middle]) / 2
    )
    return f"{min(values)}／{median:g}／{max(values)}"


def build_report(directory):
    required = [
        "summary.json",
        "config.json",
        "environment.json",
        "03_总体与岭回归对比.csv",
        "04_分流域误差与对比.csv",
        "05_逐站点训练与时间划分.csv",
    ]
    missing = [name for name in required if not (directory / name).is_file()]
    if missing:
        raise RuntimeError(
            "运行尚未完成或文件不足；报告不会读取训练中结果。缺少："
            + "、".join(missing)
        )

    summary = load_json(directory / "summary.json")
    configuration = load_json(directory / "config.json")
    conf = configuration["config"]
    environment = load_json(directory / "environment.json")
    overall = load_csv(directory / "03_总体与岭回归对比.csv")
    basin_rows = load_csv(directory / "04_分流域误差与对比.csv")
    station_rows = load_csv(directory / "05_逐站点训练与时间划分.csv")

    n_stations = int(summary["站点数"])
    n_tests = int(summary["测试记录数"])
    is_full = n_stations == 750
    if len(station_rows) != n_stations:
        raise RuntimeError("站点时间划分行数与已完成摘要不一致。")
    if len({r["station"] for r in station_rows}) != n_stations:
        raise RuntimeError("站点时间划分存在重复站点。")
    if len(overall) != 4 or {r["目标变量"] for r in overall} != set(TARGETS):
        raise RuntimeError("总体结果未完整包含四个化学目标。")
    if int(summary["目标指标评价数量"]) != n_stations * 4:
        raise RuntimeError("站点—目标评价数量与预期不一致。")
    if sum(int(r["n_test_windows"]) for r in station_rows) != n_tests:
        raise RuntimeError("逐站测试记录数之和与摘要不一致。")
    if summary["fingerprint"] != configuration["fingerprint"]:
        raise RuntimeError("完成摘要与配置指纹不一致。")
    if not all(summary["验证"].values()):
        raise RuntimeError("运行摘要包含未通过的验证项，不能生成完成报告。")

    lookup = {r["目标变量"]: r for r in overall}
    by_basin = {}
    for row in basin_rows:
        basin = row["流域"]
        target = row["目标变量"]
        if target in by_basin.setdefault(basin, {}):
            raise RuntimeError(f"流域目标重复：{basin}／{target}")
        by_basin[basin][target] = row
    for basin, rows in by_basin.items():
        if set(rows) != set(TARGETS):
            raise RuntimeError(f"流域目标缺失：{basin}")
        counts = {int(r["站点数"]) for r in rows.values()}
        if len(counts) != 1:
            raise RuntimeError(f"流域各目标的站点数不一致：{basin}")
    if sum(int(rows[TARGETS[0]]["站点数"]) for rows in by_basin.values()) != n_stations:
        raise RuntimeError("流域站点数量之和与总体不一致。")
    if is_full and (n_tests != 27326 or len(by_basin) != 13):
        raise RuntimeError("750站全量运行的测试量或流域数与既定数据集不一致。")

    title = (
        "XLinear骨干软测量适配：750站全量评估"
        if is_full
        else f"XLinear骨干软测量适配：{n_stations}站试运行（非全量）"
    )
    lines = [f"# {title}\n"]
    if not is_full:
        lines.append(
            f"> 本文件仅为{n_stations}站试运行结果，不代表全部750站。"
            "不得将试运行中的站点比例外推到全量，也不能把本文件称为全量报告。\n"
        )
    lines.extend([
        "## 任务和数据\n",
        "本轮保持用户的研究目标：直接测水温、pH、溶解氧、电导率、浊度5项，"
        "估计同一监测时刻的高锰酸盐指数、氨氮、总磷、总氮4项，总共9项。"
        "叶绿素、藻密度不参与输入或输出。没有使用任何化学目标历史作为输入。\n",
        f"本次已完成{n_stations}站、{n_stations * 4:,}个站点—目标评价，"
        f"测试观测{n_tests:,}条，覆盖{len(by_basin)}个流域。"
        "使用既有30天数据集；全部750站是总体研究数据集，533／406／85站的"
        "早期相关筛选或空间草案不用于本轮缩减站点。\n",
        "这属于XLinear骨干的**软测量任务适配**，不是官方原生未来序列预测任务的直接复现。"
        "原生MS模式会使用目标历史及其窗口归一化信息，本轮没有沿用该依赖。\n",
        "## 模型与输入窗口\n",
        "保留原版 `Forcast_multi` 的时间门控、跨变量门控及全局token骨干，"
        "原文件按字节复制，来源哈希记录在配置和 `source_provenance.json`。"
        "输入为 `[batch, 6, 5]`，骨干输出 `[batch, 5, 32]`，"
        "展平后经Dropout与 `Linear(160,4)` 回归头输出4个化学目标，"
        f"每站模型共{int(summary['参数量']):,}个参数。"
        "不调用原 `Model` 的RevIN、目标历史归一化或输入通道逆归一化。\n",
        "每站独立训练一个同时输出4项的模型。输入窗口为同站连续6次易测观测，"
        "包含待估时刻的当前易测值，前5次均不晚于该时刻；不跨站、无未来值。"
        "这是同期估计，不是提前若干小时预测未来水质。"
        "原始采样并非等间隔，6次观测不等于6小时；模型按记录序列使用这6次观测，"
        "没有显式输入时间间隔，也未插值补齐统一时间网格。"
        "每站最初5次观测不能组成完整历史窗口，只从有6次历史的端点构造训练样本。\n",
        "## 时间切分、选择轮数和重新拟合\n",
        "1. 初始拟合端点早于2025-04-25；验证端点为"
        "2025-04-25至2025-04-30之前。X、Y分别按站点仅在4月25日之前的原始观测拟合StandardScaler。\n"
        "2. 以4项标准化目标的未截断预测平均MSE选择最佳epoch并早停。"
        "测试集不参与轮数选择。该标准化多目标损失不具有mg/L单位，不能当作浓度误差。\n"
        "3. 选择epoch后丢弃选择阶段模型，重新初始化；"
        "X、Y的StandardScaler分别用4月30日之前的全部原始观测重新拟合，"
        "最终模型在此前20天窗口上训练选定轮数。\n"
        "4. 仅在以上过程锁定后测试2025-04-30至2025-05-10 00:00（含端点）的观测。"
        "后段测试窗口可使用已经观测到的测试期易测历史，但不使用测试期化学实测历史。\n",
        f"优化器Adam，固定学习率{conf['lr']}；最多{conf['max_epochs']}轮，"
        f"早停耐心{conf['patience']}轮；窗口长度{conf['seq_len']}，"
        f"d_model={conf['d_model']}，t_ff=c_ff=16，"
        f"门控／嵌入／回归头Dropout={conf['dropout']}。"
        "随机种子为2025加站点ID中的数字；重拟合使用同一站点种子。"
        "未以最终测试成绩挑模型、轮数或变量组合。\n",
        "## 原单位评价与岭回归比较\n",
        "输出先用该站最终训练期Y scaler逆变换回原单位，再将负浓度截为0；"
        "同时保留未截断预测及负预测次数。截零是确定性物理后处理，"
        "不是检测限填补。早停使用的仍是标准化未截断MSE，"
        "所以早停损失与最终原单位截零后的MAE不是同一指标。\n",
        "下表MAE、RMSE单位均为mg/L，R²无单位；先逐站计算，再报告站点中位数。"
        "R²不可计算的常量／近常量测试目标留空，不填0；保留负R²。"
        "R²为负表示平方误差大于使用该站测试实测均值作常数的误差，"
        "并不直接等于比可部署的训练均值或末次训练值基线更差。\n",
    ])

    overall_rows = []
    for target in TARGETS:
        row = lookup[target]
        overall_rows.append([
            SHORT[target],
            integer(row["R2可计算站点数"]),
            number(row["XLinear_MAE中位数"]),
            number(row["XLinear_RMSE中位数"]),
            number(row["XLinear_R2中位数"]),
            number(row["岭回归MAE中位数"]),
            number(row["岭回归R2中位数"]),
            f"{integer(row['XLinear_MAE低于岭回归站点数'])}／{integer(row['站点数'])}",
        ])
    lines.append(md_table(
        ["目标", "R²有效站点", "XLinear MAE", "XLinear RMSE", "XLinear R²",
         "岭回归MAE", "岭回归R²", "MAE低于岭回归站点"],
        overall_rows,
    ))
    lines.append(
        "\n与既有岭回归对齐相同站点、相同测试监测时刻和真实化学标签；"
        "全量运行均为27,326条测试观测。但两者并非只更换模型骨干的严格消融："
        "XLinear多使用前5次易测历史，岭回归仅用同期5项输入，"
        "且内部验证策略不同。改善不能全部归因于架构。"
        "“MAE低于岭回归”只是相对这个基线更好，不表示误差已满足检测用途，"
        "也不称为可推导成功率。\n"
    )

    lines.append("## 各流域结果\n")
    lines.append(
        "流域表按每项化学指标分别汇总，不把不同目标的原单位误差合成一个平均值。"
        "各表站点等权；流域站点数不相等，结果也不代表全流域随机抽样。\n"
    )
    basin_order = sorted(by_basin)
    for heading, key, display in [
        ("各站R²中位数", "XLinear_R2中位数", number),
        ("各站MAE中位数（mg/L）", "XLinear_MAE中位数", number),
        ("MAE低于岭回归的站点数", "XLinear_MAE低于岭回归站点数", integer),
    ]:
        lines.append(f"### {heading}\n")
        lines.append(md_table(
            ["流域", "站点数"] + [SHORT[t] for t in TARGETS],
            [
                [basin, integer(by_basin[basin][TARGETS[0]]["站点数"])]
                + [display(by_basin[basin][target][key]) for target in TARGETS]
                for basin in basin_order
            ],
        ))

    lines.extend([
        "\n## 样本量与证据限制\n",
        f"测试观测少于20条的站点有{int(summary['测试不足20条站点数'])}个；"
        f"常量或近常量测试目标导致R²无法计算的站点—目标组合共"
        f"{int(summary['常量或近常量测试目标R2不可计算数量'])}个。"
        "这些站点仍保留，不能因样本少或R²缺失而悄悄移出总体分母。"
        "测试目标变化很小时，较小绝对误差也可能对应极低R²；"
        "须结合目标变化范围、绝对误差和基线理解。\n",
        md_table(
            ["逐站样本量", "最小／中位／最大"],
            [
                ["选epoch初始训练窗口", aggregate_metadata(station_rows, "n_selection_train_windows")],
                ["验证窗口", aggregate_metadata(station_rows, "n_validation_windows")],
                ["前20天重拟合窗口", aggregate_metadata(station_rows, "n_refit_windows")],
                ["最终测试窗口", aggregate_metadata(station_rows, "n_test_windows")],
                ["验证选择的epoch", aggregate_metadata(station_rows, "best_epoch")],
            ],
        ),
        "\n本批30天数据此前已经做过相关性与站点筛选探索，"
        "这次属于回溯性时间留出评估，不是完全未经探索的新采集外部测试。"
        "同站时间序列有自相关，窗口还会共享易测历史，测试记录不是独立重复试验。"
        "没有据此计算独立样本显著性星号或长期可靠性置信结论。\n",
        "本轮逐站模型仍使用了该站前期化学实测作为训练标签和校准依据，"
        "不能理解为可以取消既有化学测量；后续仍需定期校准、独立时段检验和质量抽检。"
        "也未验证从未提供化学标签的新站迁移，不能外推2500站、其他季节或全年。"
        "MAE更低不等于已能替代化学检测，变量数量减少不直接等于等比例降本。\n",
        "用户尚未给出各化学指标的允许误差，本轮不设置合格站点阈值，"
        "不计算“测5项成功推导4项”的比例。若以后定义验收，应分别列"
        "至少1／2／3／4目标达标与证据不足站点；比例分母保留原750站，"
        "并另外说明可评估子集内的比例。\n",
        "## 运行环境、来源与复核\n",
        md_table(
            ["项目", "记录"],
            [
                ["设备", environment.get("device", "NA")],
                ["GPU", environment.get("gpu") or "未使用"],
                ["PyTorch", environment.get("torch", "NA")],
                ["NumPy", environment.get("numpy", "NA")],
                ["pandas", environment.get("pandas", "NA")],
                ["参数量", summary["参数量"]],
                ["本次运行汇总耗时（秒）", number(summary.get("总训练耗时秒"), 8)],
                ["配置与来源指纹", configuration["fingerprint"]],
                ["原始数据SHA256", configuration.get("source_dataset_sha256", "NA")],
                ["原版骨干副本SHA256", configuration.get("vendor_sha256", "NA")],
            ],
        ),
        "\n摘要中全部验证项均通过：\n\n"
        + "".join(f"- {key}：通过。\n" for key in summary["验证"]),
        "\n结果文件：\n\n"
        "- [逐站点四目标误差](01_逐站点四目标误差.csv)：原单位MAE、RMSE、R²、偏差、基线与样本量。\n"
        "- [测试实测与预测](02_测试实测与预测.csv)：输入、真实值、截零后与未截断预测、固定基线及窗口跨度。\n"
        "- [总体比较](03_总体与岭回归对比.csv)与[分流域比较](04_分流域误差与对比.csv)。\n"
        "- [逐站时间划分](05_逐站点训练与时间划分.csv)：初始拟合、验证、重拟合、测试样本量和模型轮数。\n"
        "- `stations/<站点ID>/`：逐站权重、训练历史、scaler、参数来源指纹及可复核预测。\n",
    ])
    audit_path = directory / "独立验证.json"
    if audit_path.exists():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if audit.get("passed") is not True:
            raise RuntimeError("独立验证未通过，不能生成完成报告")
        lines.append(
            "\n## 独立CPU复核\n\n"
            f"独立验证通过：{audit['station_count']}站、{audit['metric_rows']}组指标、"
            f"{audit['test_rows']}个测试窗口与源数据及岭回归的测试键和四项真值一致；"
            f"{audit['audited_scalers']}套标准化统计独立重算一致。"
            "所有窗口的五项输入和同站历史起点均与原数据一致，无未来或跨站窗口。\n\n"
            f"从保存权重在CPU恢复{len(audit['cpu_checkpoint_replay'])}个固定站点的预测通过，"
            f"最大标准化差{max(r['max_abs_standardized_difference'] for r in audit['cpu_checkpoint_replay']):.3g}。"
            "独立复核没有重新训练模型；重新初始化和不使用测试选模型还结合了代码审查。"
            "完整记录见[独立验证](独立验证.json)。\n"
        )
    if all(r['XLinear_MAE中位数'] > r['岭回归MAE中位数'] for r in summary['总体结果']):
        lines.append(
            "\n## 本轮结论\n\n"
            "四项目标的站点MAE中位数均高于岭回归，当前配置未表现出总体改善。"
            "这不证明所有XLinear配置、其他非线性模型或更多独立数据下均不能估计，"
            "也不能从当前比较得到可推导成功率。\n"
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        default="results",
        help="已完成的运行目录；相对路径以本脚本目录为基准。试运行可传pilot。",
    )
    args = parser.parse_args()
    directory = Path(args.results)
    if not directory.is_absolute():
        directory = HERE / directory
    if not directory.is_dir():
        raise SystemExit(f"运行目录尚不存在：{directory}")
    try:
        report = build_report(directory)
    except (RuntimeError, KeyError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    destination = directory / "水质软测量评估报告.md"
    destination.write_text(report, encoding="utf-8")
    print(str(destination))


if __name__ == "__main__":
    main()
