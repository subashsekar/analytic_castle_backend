from app.core.slugs import slugify


def test_slugify_normalizes_case_and_spaces() -> None:
    assert slugify("Sales Analytics") == "sales-analytics"


def test_slugify_strips_special_characters() -> None:
    assert slugify("  Acme & Co.  ") == "acme-co"


def test_slugify_handles_unicode() -> None:
    assert slugify("Café Labs") == "cafe-labs"


def test_slugify_empty_or_symbols_returns_empty() -> None:
    assert slugify("@@@") == ""
    assert slugify("   ") == ""
