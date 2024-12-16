from pathlib import Path
from astroquery.eso import Eso

import PyPDF2
from tqdm import tqdm


def search_word_in_pdf(pdf_path, word):
    results = []
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text()
            if text and word.lower() in text.lower():
                results.append((page_num, text.count(word)))
    return results


if __name__ == "__main__":
    # word_to_search = "97048"
    # directory = Path().home() / "documents" / "VLTI_reports"
    # matches = {}
    # for pdf in tqdm(list(directory.glob("*.pdf")), desc="Searching pdfs..."):
    #     match = search_word_in_pdf(pdf, word_to_search)
    #     matches[pdf.name] = match
    #
    # for file_name, match in matches.items():
    #     if match:
    #         print(f"'{word_to_search}' found in file {file_name}:")
    #         print("For pages:")
    #         for page, count in match:
    #             print(f"- Page {page}: {count} occurrence(s)")

    eso = Eso()
    eso.login(username="MbS")

    table = eso.query_instrument(
        "matisse",
        column_filters={
            "target": "HD 97048",
            "stime": "2023-01-01",
            "etime": "2024-12-31",
        },
        columns=["night"],
        cache=False,
    )
    breakpoint()
