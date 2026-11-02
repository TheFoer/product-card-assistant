"""Командная строка: serve, check, generate."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from cardgen.analysis import parse_json, parse_product, preflight
from cardgen.config import Settings
from cardgen.errors import CardGenError
from cardgen.llm import AnthropicClient
from cardgen.service import CardGenerator

EXIT_OK, EXIT_FAILED, EXIT_BAD_INPUT = 0, 1, 2


def _load_items(path: Path) -> list[Any]:
    try:
        data = parse_json(path.read_bytes())
    except OSError as exc:
        raise CardGenError(f"Не удалось прочитать {path}: {exc.strerror}") from None
    return data if isinstance(data, list) else [data]


def _print_error(label: str, exc: CardGenError) -> None:
    print(f"{label}: {exc.message}", file=sys.stderr)
    for detail in exc.details:
        print(f"  - {detail['field']}: {detail['message']}", file=sys.stderr)


def _cmd_check(args: argparse.Namespace) -> int:
    code = EXIT_OK
    for path in args.files:
        try:
            for item in _load_items(path):
                report = preflight(parse_product(item))
                print(json.dumps(report.model_dump(), ensure_ascii=False, indent=2))
        except CardGenError as exc:
            _print_error(str(path), exc)
            code = EXIT_BAD_INPUT
    return code


def _cmd_generate(args: argparse.Namespace) -> int:
    settings = Settings.from_env()
    generator = CardGenerator(AnthropicClient(settings), settings.max_repairs)
    code = EXIT_OK
    for path in args.files:
        try:
            items = _load_items(path)
        except CardGenError as exc:
            _print_error(str(path), exc)
            code = EXIT_BAD_INPUT
            continue
        for index, item in enumerate(items, start=1):
            label = path.stem if len(items) == 1 else f"{path.stem}-{index}"
            try:
                result = generator.generate(parse_product(item))
            except CardGenError as exc:
                # Сбой одного товара не должен останавливать остальную пачку.
                _print_error(label, exc)
                code = max(code, EXIT_FAILED)
                continue
            if args.out_dir:
                args.out_dir.mkdir(parents=True, exist_ok=True)
                (args.out_dir / f"{label}.md").write_text(result.markdown, encoding="utf-8")
                (args.out_dir / f"{label}.json").write_text(
                    json.dumps(result.model_dump(exclude={"markdown"}), ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                print(f"{label}: {result.status}, файлы в {args.out_dir}")
            elif args.format == "json":
                print(json.dumps(result.model_dump(exclude={"markdown"}), ensure_ascii=False, indent=2))
            else:
                print(result.markdown)
    return code


def _cmd_serve(args: argparse.Namespace) -> int:
    from cardgen.web import create_app

    create_app().run(host=args.host, port=args.port, threaded=True)
    return EXIT_OK


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cardgen", description="Черновики карточек товаров")
    commands = parser.add_subparsers(dest="command", required=True)

    serve = commands.add_parser("serve", help="запустить веб-форму и API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.set_defaults(func=_cmd_serve)

    check = commands.add_parser("check", help="проверить входные данные без обращения к LLM")
    check.add_argument("files", nargs="+", type=Path)
    check.set_defaults(func=_cmd_check)

    generate = commands.add_parser("generate", help="создать карточки из JSON-файлов")
    generate.add_argument("files", nargs="+", type=Path)
    generate.add_argument("--out-dir", type=Path, help="сохранить .md и .json в каталог")
    generate.add_argument("--format", choices=("markdown", "json"), default="markdown")
    generate.set_defaults(func=_cmd_generate)
    return parser


def main(argv: list[str] | None = None) -> int:
    load_dotenv(Path.cwd() / ".env")
    args = _parser().parse_args(argv)
    try:
        logging.basicConfig(
            level=Settings.from_env().log_level,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
            stream=sys.stderr,
        )
        return args.func(args)
    except CardGenError as exc:
        print(f"Ошибка: {exc.message}", file=sys.stderr)
        return EXIT_FAILED
    except ValueError as exc:
        print(f"Ошибка настройки: {exc}", file=sys.stderr)
        return EXIT_BAD_INPUT
    except KeyboardInterrupt:
        return EXIT_FAILED
