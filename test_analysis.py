import pytest

from cardgen.analysis import parse_json, parse_product, preflight
from cardgen.errors import InvalidJsonError, InvalidProductError


@pytest.mark.parametrize("text", ["", "   ", "{", '{"name": "x",}', '{"a": NaN}', b"\xff\xfe"])
def test_invalid_json_is_rejected(text):
    with pytest.raises(InvalidJsonError):
        parse_json(text)


def test_json_error_points_to_position():
    with pytest.raises(InvalidJsonError, match="строка 2"):
        parse_json('{\n  "name" "x"\n}')


def test_bom_is_accepted():
    assert parse_json('﻿{"name": "x"}'.encode()) == {"name": "x"}


@pytest.mark.parametrize("raw", [[], "строка", 5, None])
def test_non_object_is_rejected(raw):
    with pytest.raises(InvalidProductError):
        parse_product(raw)


@pytest.mark.parametrize(
    "raw, field",
    [
        ({"specs": {"a": "1"}}, "name"),
        ({"name": "   ", "description": "текст"}, "name"),
        ({"name": "н/д", "description": "текст"}, "name"),
        ({"name": "Прибор"}, "specs, description"),
        ({"name": "Прибор", "specs": {"a": ""}, "description": " "}, "specs, description"),
        ({"name": 5, "description": "текст"}, "name"),
        ({"name": "П", "specs": ["a"]}, "specs"),
        ({"name": "П", "specs": {"a": {"b": 1}}}, "specs.a"),
        ({"name": "П", "specs": {"a": True}}, "specs.a"),
        ({"name": "П", "description": "x" * 20_001}, "description"),
    ],
)
def test_invalid_product_reports_field(raw, field):
    with pytest.raises(InvalidProductError) as exc:
        parse_product(raw)
    assert field in [detail["field"] for detail in exc.value.details]


def test_numbers_are_kept_and_placeholders_become_empty():
    product = parse_product({"name": "П", "specs": {"масса": 2.5, "питание": "н/д", "x": None}})
    assert product.specs == {"масса": "2.5", "питание": "", "x": ""}
    assert product.filled_specs == {"масса": "2.5"}


def test_preflight_reports_gaps(dm02_input):
    report = preflight(parse_product(dm02_input))
    assert report.missing_fields == ["Питание", "Тип корпуса", "Интерфейсы"]
    assert "truncated_description" in {issue.code for issue in report.issues}


def test_preflight_handles_aliases_and_empty_values():
    report = preflight(parse_product({
        "name": "П", "specs": {"напряжение питания": "24 В", "точность": "—"}, "extra": 1,
    }))
    assert "Питание" not in report.missing_fields
    assert report.missing_fields.count("точность") == 1
    assert "Класс точности" not in report.missing_fields
    assert "Производитель" in report.missing_fields
    assert "ignored_fields" in {issue.code for issue in report.issues}
