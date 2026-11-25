from wands_search.llm import parse


def test_parse_clean_json():
    raw = '{"corrected": "bar stool", "product_type": "bar stool", "color": null, "material": "wood",' \
          ' "size": null, "style": null, "rewrite": "wooden bar stool"}'
    r = parse(raw, "bar stol")
    assert r["corrected"] == "bar stool" and r["material"] == "wood" and r["color"] is None
    assert r["parse_ok"]


def test_parse_falls_back_to_query():
    r = parse("sorry, I cannot help", "oak desk")
    assert r["corrected"] == "oak desk" and r["rewrite"] == "oak desk" and not r["parse_ok"]


def test_parse_embedded_json_and_lists():
    r = parse('Here you go: {"color": ["white", "gold"], "rewrite": ""}', "white gold lamp")
    assert r["color"] == "white gold" and r["rewrite"] == "white gold lamp"
