import regex as re

def dehyphenate(text: str) -> str:
    # join line-break hyphenations: exam-\nple -> example
    return re.sub(r"(\w)-\n(\w)", r"\1\2", text)

def normalize_ws(text: str) -> str:
    # collapse weird spaces but keep paragraphs
    text = text.replace("\r", "")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def strip_headers_footers(text: str, header_regex=None, footer_regex=None) -> str:
    lines = text.splitlines()
    cleaned = []
    for i, line in enumerate(lines):
        if header_regex and re.match(header_regex, line):
            continue
        if footer_regex and re.match(footer_regex, line):
            continue
        cleaned.append(line)
    return "\n".join(cleaned)

def clean_page(text: str, header_regex=None, footer_regex=None) -> str:
    text = dehyphenate(text)
    text = strip_headers_footers(text, header_regex, footer_regex)
    text = normalize_ws(text)
    return text
