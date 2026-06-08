"""Compare baseline SpQR with module/depth-wise outlier sensitivity variants."""

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class TestConfig:
    """Конфигурация теста."""

    name: str
    use_module_preset: bool = False
    module_sensitivity: str = None
    use_layer_preset: bool = False
    layer_sensitivity: str = None
    nsamples: int = 128
    offload_activations: bool = False
    unquantized: bool = False

    def to_args(self) -> list[str]:
        """Преобразует конфигурацию в аргументы командной строки."""
        args = ["--nsamples", str(self.nsamples)]
        if self.use_module_preset:
            args.append("--outlier_module_preset")
        if self.module_sensitivity:
            args.extend(["--outlier_module_sensitivity", self.module_sensitivity])
        if self.use_layer_preset:
            args.append("--outlier_layer_preset")
        if self.layer_sensitivity:
            args.extend(["--outlier_layer_sensitivity", self.layer_sensitivity])
        if self.offload_activations:
            args.append("--offload_activations")
        return args


@dataclass
class TestResult:
    """Результат теста."""

    config: TestConfig
    ppl_wikitext2: float = None
    ppl_c4: float = None
    ppl_ptb: float = None
    global_ol_share: float = None
    compression_time: float = None
    exit_code: int = 0
    error: str = None
    nsamples: int = 128


def run_test(
    model_path: str,
    dataset: str,
    config: TestConfig,
    base_args: list[str],
    save_dir: str,
) -> TestResult:
    """Запускает один тест с заданной конфигурацией."""
    print(f"\n{'=' * 70}")
    print(f"Запуск теста: {config.name}")
    print(
        f"  module_preset={config.use_module_preset}, module_sensitivity={config.module_sensitivity}, "
        f"layer_preset={config.use_layer_preset}, layer_sensitivity={config.layer_sensitivity}, "
        f"nsamples={config.nsamples}, offload={config.offload_activations}"
    )
    print(f"{'=' * 70}\n")

    result = TestResult(config=config, nsamples=config.nsamples)

    if config.unquantized:
        effective_base_args = ["--wbits", "16"]
        if args_device := [a for i, a in enumerate(base_args) if i > 0 and base_args[i - 1] == "--device"]:
            effective_base_args.extend(["--device", args_device[0]])
        cmd = [
            sys.executable,
            "main.py",
            model_path,
            dataset,
            *effective_base_args,
            *config.to_args(),
        ]
    else:
        cmd = [
            sys.executable,
            "main.py",
            model_path,
            dataset,
            *base_args,
            *config.to_args(),
            "--save",
            f"{save_dir}/{config.name}",
        ]

    print(f"Команда: {' '.join(cmd)}\n")

    start_time = datetime.now()

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    output_lines = []
    for line in process.stdout:
        print(line, end="")
        output_lines.append(line)

    process.wait()
    result.exit_code = process.returncode
    result.compression_time = (datetime.now() - start_time).total_seconds()

    output = "".join(output_lines)
    result.ppl_wikitext2 = parse_perplexity(output, "wikitext2")
    result.ppl_c4 = parse_perplexity(output, "c4")
    result.ppl_ptb = parse_perplexity(output, "ptb")
    result.global_ol_share = parse_global_ol_share(output)

    if result.exit_code != 0:
        result.error = "Тест завершился с ошибкой"

    return result


def parse_perplexity(output: str, dataset: str) -> float | None:
    """Извлекает perplexity из вывода."""
    patterns = [
        rf"{re.escape(dataset)}\s+perplexity\s*=\s*(\d+\.?\d*)",
        rf"{re.escape(dataset)}\s+(\d+\.?\d*)",
    ]
    for pattern in patterns:
        match = re.search(pattern, output, re.IGNORECASE)
        if match:
            return float(match.group(1))
    return None


def parse_global_ol_share(output: str) -> float | None:
    """Извлекает global_ol_share из вывода main.py как процент, например 0.350 для 0.350%."""
    match = re.search(r"global_ol_share:\s*([+-]?\d+(?:\.\d+)?)\s*%", output, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return None


def format_ol_share(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.3f}%"


def print_results_table(results: list[TestResult]):
    """Выводит таблицу результатов."""
    print("\n" + "=" * 100)
    print("РЕЗУЛЬТАТЫ СРАВНЕНИЯ")
    print("=" * 100)

    header = (
        f"{'Тест':<30} {'N':<6} {'PPL Wiki2':<12} {'PPL C4':<12} {'PPL PTB':<12} "
        f"{'OL share':<10} {'Время (мин)':<12}"
    )
    print(header)
    print("-" * len(header))

    unquantized_result = None
    baselines = {}
    for result in results:
        if result.exit_code != 0:
            row = f"{result.config.name:<30} {result.nsamples:<6} {'ERROR':<12} {'':12} {'':12} {'':10} {'':12}"
            print(row)
            continue

        row = (
            f"{result.config.name:<30} "
            f"{result.nsamples:<6} "
            f"{result.ppl_wikitext2 or 'N/A':<12} "
            f"{result.ppl_c4 or 'N/A':<12} "
            f"{result.ppl_ptb or 'N/A':<12} "
            f"{format_ol_share(result.global_ol_share):<10} "
            f"{result.compression_time / 60:.1f}{' мин':<6}"
        )
        print(row)

        if result.config.unquantized:
            unquantized_result = result
        elif (
            not result.config.use_module_preset
            and result.config.module_sensitivity is None
            and not result.config.use_layer_preset
            and result.config.layer_sensitivity is None
        ):
            baselines[result.nsamples] = result

    if unquantized_result and unquantized_result.ppl_wikitext2:
        print("\nДеградация perplexity относительно FP16 (без квантизации):")
        print("-" * len(header))
        for result in results:
            if result.config.unquantized or result.exit_code != 0 or result.ppl_wikitext2 is None:
                continue
            change = (result.ppl_wikitext2 - unquantized_result.ppl_wikitext2) / unquantized_result.ppl_wikitext2 * 100
            print(f"    {result.config.name:<28} {change:+.2f}%")

    if baselines:
        print("\nОтносительное изменение perplexity к quantized baseline:")
        print("-" * len(header))

        by_nsamples = {}
        for result in results:
            if result.config.unquantized:
                continue
            by_nsamples.setdefault(result.nsamples, []).append(result)

        for nsamples in sorted(by_nsamples.keys()):
            baseline = baselines.get(nsamples)
            if not baseline or not baseline.ppl_wikitext2:
                continue

            print(f"\n  [nsamples={nsamples}]")
            for result in by_nsamples[nsamples]:
                if result == baseline or result.ppl_wikitext2 is None:
                    continue

                change = (result.ppl_wikitext2 - baseline.ppl_wikitext2) / baseline.ppl_wikitext2 * 100
                status = "✓ УЛУЧШЕНИЕ" if change < 0 else "✗ УХУДШЕНИЕ" if change > 0 else "= БЕЗ ИЗМЕНЕНИЙ"
                ol_share_delta = ""
                if result.global_ol_share is not None and baseline.global_ol_share is not None:
                    ol_share_delta = f", OL Δ {result.global_ol_share - baseline.global_ol_share:+.3f} pp"

                print(f"    {result.config.name:<28} {change:+.2f}% ({status}{ol_share_delta})")

    print("=" * 100 + "\n")


def save_results(results: list[TestResult], save_dir: str):
    """Сохраняет результаты в JSON."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = Path(save_dir) / f"comparison_results_{timestamp}.json"

    data = {
        "timestamp": timestamp,
        "results": [
            {
                "name": result.config.name,
                "unquantized": result.config.unquantized,
                "use_module_preset": result.config.use_module_preset,
                "module_sensitivity": result.config.module_sensitivity,
                "use_layer_preset": result.config.use_layer_preset,
                "layer_sensitivity": result.config.layer_sensitivity,
                "nsamples": result.nsamples,
                "offload_activations": result.config.offload_activations,
                "perplexity_wikitext2": result.ppl_wikitext2,
                "perplexity_c4": result.ppl_c4,
                "perplexity_ptb": result.ppl_ptb,
                "global_ol_share_percent": result.global_ol_share,
                "compression_time_min": result.compression_time / 60 if result.compression_time else None,
                "exit_code": result.exit_code,
                "error": result.error,
            }
            for result in results
        ],
    }

    with open(results_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Результаты сохранены в: {results_path}")


def slugify_sensitivity(spec: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", spec).strip("_").lower()
    return slug[:80] or "custom"


def get_module_variants(args):
    variants = []
    if not args.skip_preset:
        variants.append(("module_preset", True, None))
    for sensitivity in args.module_sensitivities or []:
        variants.append((f"module_{slugify_sensitivity(sensitivity)}", False, sensitivity))
    return variants


def get_layer_variants(args):
    variants = []
    if args.layer_preset:
        variants.append(("layer_preset", True, None))
    for sensitivity in args.layer_sensitivities or []:
        variants.append((f"layer_{slugify_sensitivity(sensitivity)}", False, sensitivity))
    return variants


def main():
    parser = argparse.ArgumentParser(
        description="Сравнительное тестирование module/depth-wise outlier sensitivity для SpQR"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Путь к модели",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="c4",
        help="Датасет для калибровки",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./comparison_results",
        help="Директория для сохранения результатов",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Быстрый тест: только nsamples=128",
    )
    parser.add_argument(
        "--skip_baseline",
        action="store_true",
        help="Пропустить baseline тест",
    )
    parser.add_argument(
        "--skip_preset",
        action="store_true",
        help="Пропустить встроенный module-wise preset",
    )
    parser.add_argument(
        "--module_sensitivities",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Дополнительные custom sensitivity строки, например "
            "'down_proj=1.6,o_proj=1.3,up_proj=0.8,gate_proj=0.8'"
        ),
    )
    parser.add_argument(
        "--layer_preset",
        action="store_true",
        help="Добавить depth-wise preset: first 25%%=0.95, middle 50%%=1.0, last 25%%=1.10",
    )
    parser.add_argument(
        "--layer_sensitivities",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Дополнительные depth sensitivity строки, например "
            "'0-7=0.95,8-23=1.0,24-31=1.15' или 'early=0.95,mid=1.0,late=1.15'"
        ),
    )
    parser.add_argument(
        "--combine_module_depth",
        action="store_true",
        help="Дополнительно запустить все комбинации module variants и depth variants",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (e.g. cuda:0, cpu)",
    )
    parser.add_argument(
        "--offload_activations",
        action="store_true",
        help="Offload activations to CPU to save VRAM",
    )
    parser.add_argument(
        "--adaptive_outlier_threshold",
        action="store_true",
        help="Включить адаптивный расчет порога на основе статистики слоя",
    )
    parser.add_argument(
        "--outlier_adaptation_factor",
        type=float,
        default=1.0,
        help="Множитель для адаптивного порога (по умолчанию 1.0)",
    )

    args = parser.parse_args()

    nsamples_values = [128] if args.quick else [128, 256]

    base_args = [
        "--wbits",
        "4",
        "--groupsize",
        "16",
        "--perchannel",
        "--qq_scale_bits",
        "3",
        "--qq_zero_bits",
        "3",
        "--qq_groupsize",
        "16",
        "--outlier_threshold",
        "0.2",
        "--permutation_order",
        "act_order",
        "--percdamp",
        "1.0",
    ]

    if args.device:
        base_args.extend(["--device", args.device])

    if args.adaptive_outlier_threshold:
        base_args.append("--adaptive_outlier_threshold")
        if args.outlier_adaptation_factor != 1.0:
            base_args.extend(["--outlier_adaptation_factor", str(args.outlier_adaptation_factor)])

    test_configs = []

    test_configs.append(
        TestConfig(
            name="unquantized_fp16",
            unquantized=True,
            nsamples=nsamples_values[0],
            offload_activations=args.offload_activations,
        )
    )

    for nsamples in nsamples_values:
        if not args.skip_baseline:
            test_configs.append(
                TestConfig(
                    name=f"baseline_n{nsamples}",
                    nsamples=nsamples,
                    offload_activations=args.offload_activations,
                )
            )

        if not args.skip_preset:
            test_configs.append(
                TestConfig(
                    name=f"module_preset_n{nsamples}",
                    use_module_preset=True,
                    nsamples=nsamples,
                    offload_activations=args.offload_activations,
                )
            )

        if args.layer_preset:
            test_configs.append(
                TestConfig(
                    name=f"layer_preset_n{nsamples}",
                    use_layer_preset=True,
                    nsamples=nsamples,
                    offload_activations=args.offload_activations,
                )
            )

        for sensitivity in args.module_sensitivities or []:
            test_configs.append(
                TestConfig(
                    name=f"module_{slugify_sensitivity(sensitivity)}_n{nsamples}",
                    module_sensitivity=sensitivity,
                    nsamples=nsamples,
                    offload_activations=args.offload_activations,
                )
            )

        for sensitivity in args.layer_sensitivities or []:
            test_configs.append(
                TestConfig(
                    name=f"layer_{slugify_sensitivity(sensitivity)}_n{nsamples}",
                    layer_sensitivity=sensitivity,
                    nsamples=nsamples,
                    offload_activations=args.offload_activations,
                )
            )

        if args.combine_module_depth:
            for module_name, use_module_preset, module_sensitivity in get_module_variants(args):
                for layer_name, use_layer_preset, layer_sensitivity in get_layer_variants(args):
                    test_configs.append(
                        TestConfig(
                            name=f"{module_name}__{layer_name}_n{nsamples}",
                            use_module_preset=use_module_preset,
                            module_sensitivity=module_sensitivity,
                            use_layer_preset=use_layer_preset,
                            layer_sensitivity=layer_sensitivity,
                            nsamples=nsamples,
                            offload_activations=args.offload_activations,
                        )
                    )

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    results = []
    for config in test_configs:
        result = run_test(
            model_path=args.model_path,
            dataset=args.dataset,
            config=config,
            base_args=base_args,
            save_dir=args.save_dir,
        )
        results.append(result)

    print_results_table(results)
    save_results(results, args.save_dir)

    failed = any(result.exit_code != 0 for result in results)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
