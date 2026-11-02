import pytest

from cardgen.analysis import parse_product
from cardgen.factcheck import build_specs, check_card, numbers_in, unsupported_facts
from cardgen.schemas import MISSING_MARK, ExtractedSpec, LlmCard

SOURCE = "Манометр DM 02, 0–1,6 МПа, класс 1,5, присоединение G1/2, корпус 100 мм"


def test_number_forms_are_equivalent():
    assert numbers_in("1,6 и 1.60 и G1/2") == {"1.6", "1/2"}


@pytest.mark.parametrize("text", [
    "Диапазон 0–1,6 МПа, класс 1.5",
    "манометр dm 02 g1/2",
    "Подходит инженеру КИП",
])
def test_supported_text_passes(text):
    assert unsupported_facts(text, SOURCE) == []


@pytest.mark.parametrize("text, expected", [
    ("диапазон 0–2,5 МПа", "2.5"),
    ("16 бар", "16"),
    ("степень защиты IP65", "IP65"),
    ("интерфейс RS-485", "RS-485"),
    ("соответствует ГОСТ", "ГОСТ"),
    ("корпус из нержавеющей стали", "нержавеющая сталь"),
    ("заполнен глицерином", "заполнение корпуса"),
    ("прибор внесён в госреестр", "внесение в реестр"),
    ("с первичной поверкой", "поверка"),
    ("надёжный прибор", "надёжность"),
    ("для нефтегазовой отрасли", "отрасль"),
])
def test_invented_facts_are_caught(text, expected):
    assert expected in unsupported_facts(text, SOURCE)


def test_claim_present_in_source_is_allowed():
    assert unsupported_facts("корпус из нержавеющей стали", "Корпус из нержавеющей стали") == []


def _card(response, **changes):
    return LlmCard.model_validate({**response, **changes})


def test_valid_fixtures_have_no_errors(dm02_input, dm02_response, demo_input, demo_response):
    assert check_card(_card(dm02_response), parse_product(dm02_input)) == []
    assert check_card(_card(demo_response), parse_product(demo_input)) == []


@pytest.mark.parametrize("changes, code", [
    ({"seo_title": "М" * 71}, "title_too_long"),
    ({"seo_title": ""}, "title_empty"),
    ({"seo_description": "М" * 799}, "description_length"),
    ({"seo_description": "М" * 1201}, "description_length"),
    ({"search_queries": ["а", "б"]}, "queries_count"),
    ({"search_queries": list("абвгде")}, "queries_count"),
    ({"search_queries": ["а", "А", "б"]}, "queries_duplicate"),
    ({"target_audience": []}, "audience_empty"),
    ({"tasks": []}, "tasks_empty"),
    ({"seo_title": "Манометр IP65"}, "unsupported_fact"),
])
def test_rule_violations(dm02_input, dm02_response, changes, code):
    issues = check_card(_card(dm02_response, **changes), parse_product(dm02_input))
    assert code in {issue.code for issue in issues}


@pytest.mark.parametrize("length, ok", [(70, True), (71, False)])
def test_title_boundary(dm02_input, dm02_response, length, ok):
    title = ("Манометр " * 10)[:length - 1] + "М"
    codes = {i.code for i in check_card(_card(dm02_response, seo_title=title), parse_product(dm02_input))}
    assert ("title_too_long" not in codes) is ok


def test_specs_come_from_input_and_gaps_are_marked(dm02_input):
    rows, missing, issues = build_specs(parse_product(dm02_input), [])
    by_name = {row.name: row for row in rows}
    assert by_name["Диапазон измерений"].value == "0–1,6 МПа"
    assert by_name["Питание"].value == MISSING_MARK
    assert by_name["Производитель"].value == "Манотомь"
    assert set(missing) == {"Питание", "Тип корпуса", "Интерфейсы"}
    assert issues == []


def _product():
    return parse_product({
        "name": "Датчик",
        "specs": {"диапазон измерений": "0–10 бар", "питание": ""},
        "description": "Питание 24 В. Диапазон измерений 0–16 бар. Корпус из нержавеющей стали.",
    })


def test_extracted_value_fills_gap():
    rows, missing, _ = build_specs(_product(), [
        ExtractedSpec(name="Напряжение питания", value="24 В", quote="Питание 24 В"),
    ])
    row = next(r for r in rows if r.name == "Питание")
    assert (row.value, row.source) == ("24 В", "description")
    assert "Питание" not in missing


def test_inflected_value_is_accepted():
    rows, _, issues = build_specs(_product(), [
        ExtractedSpec(name="Материал", value="нержавеющая сталь", quote="Корпус из нержавеющей стали"),
    ])
    assert any(r.name == "Материал" for r in rows) and issues == []


@pytest.mark.parametrize("spec", [
    ExtractedSpec(name="Масса", value="2 кг", quote="Масса 2 кг"),
    ExtractedSpec(name="Питание", value="36 В", quote="Питание 24 В"),
    ExtractedSpec(name="Материал", value="латунь", quote="Корпус из нержавеющей стали"),
])
def test_unconfirmed_extraction_is_dropped(spec):
    rows, _, issues = build_specs(_product(), [spec])
    assert spec.value not in [row.value for row in rows]
    assert issues[0].code == "extraction_rejected"


def test_conflict_keeps_input_value_and_warns():
    rows, _, issues = build_specs(_product(), [
        ExtractedSpec(name="Диапазон", value="0–16 бар", quote="Диапазон измерений 0–16 бар"),
    ])
    assert next(r for r in rows if r.name == "Диапазон измерений").value == "0–10 бар"
    assert issues[0].code == "data_conflict"
