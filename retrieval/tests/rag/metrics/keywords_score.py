import keyword
from typing import Any, Dict, Union
from httpx import get
from rag_test_output import get_required_vals
from langchain_core.documents import Document
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from helper import calculate_documents_score, ScoreCalculationType

METRIC = 'keywords_score'
    

def keyword_score(retrieved: str, expexted: str, embedding: SentenceTransformerEmbeddings | None = None, keywords: list[str] | None = None) -> float:
    if keywords is None:
        raise ValueError("Keywords are not provided") 
    
    retrieved = retrieved.replace(" ", "").lower()
    retrieved = retrieved.replace("\n", "")
    keywords = [keyword.lower().replace(" ", "") for keyword in keywords]
    keyword_count = 0
    for keyword in keywords:
        if keyword in retrieved:
            keyword_count += 1
    return keyword_count / len(keywords)


def get_assert(output, context) -> Union[bool, float, Dict[str, Any]]:
    test_config = get_required_vals(output, context, 0.75, METRIC)

    return calculate_documents_score(test_config, keyword_score, METRIC, ScoreCalculationType.MAX)

# 
# output = "{\n    \"documents\": [\n        {\n            \"metadata\": {\n                \"file_name\": \"119_ZuSMa_Senat_18012022.pdf\",\n                \"file_path\": \"/home/tpllmws23/Chatbot-LLama-Pruefungsamt/main_data/119_ZuSMa_Senat_18012022.pdf\",\n                \"page_number\": 3\n            },\n            \"page_content\": \"deutschsprachigen  Einrichtung erworben haben, ein Nachweis \\u00fcber die erforderlichen \\nSprachkenntnisse  entsprechend \\u00a7 4.  \\n2Die in Satz 1 genannten Nachweise sind in einfacher Kopie vorzulegen. 3Sind die Nachweise \\ngem\\u00e4\\u00df  Satz 1 nicht in deutscher oder englischer Sprache abgefasst, bedarf es einer \\n\\u00dcbersetzung in deutscher oder englischer Sprache durch eine Person oder Institution, die zu \\neiner vereidigten oder  gerichtlich zugelassenen \\u00dcbersetzung berechtigt ist. \\n \\n\\u00a7 4 Sprachkenntnisse  \\n(1) 1Neben den allgemeinen Zugangsvoraussetzungen (\\u00a7 59 LHG) sind f\\u00fcr die in \\u00a7 1 Abs. 1 \\nS. 1 genannten Studieng\\u00e4nge deutsche Sprach kenntnisse nachzuweisen. 2Diese k\\u00f6nnen \\ndurch eine deutsche Hochschulzugangsberechtigung (u. a. erfolgreich abgeschlossenes \\ngrundst\\u00e4ndiges Hochschulstudium) nachgewiesen werden. 3Ferner kann der \\nSprachnachweis durch die Vorlage eines der folgenden Dokumente erbracht werden:  \\n1. Feststellungspr\\u00fcfung f\\u00fcr ein Bachelorstudium durch Vorlage der Zugangsberechtigung\",\n            \"type\": \"Document\"\n        },\n        {\n            \"metadata\": {\n                \"file_name\": \"124_SPOMa_AT_Senat_08112022.pdf\",\n                \"file_path\": \"/home/tpllmws23/Chatbot-LLama-Pruefungsamt/main_data/124_SPOMa_AT_Senat_08112022.pdf\",\n                \"page_number\": 13\n            },\n            \"page_content\": \"r\\u00fchrt. Im Antrag muss die Studienzeit und jede Studien-  und Pr\\u00fcfungsleistung, die anerkannt werden soll, \\neinzeln aufgef\\u00fchrt werden. Es obliegt dem/der Antragssteller/in, die erforderlichen Inform ationen \\u00fcber die \\nanzuerkennende Leistung bereitzustellen. Die Beweislast daf\\u00fcr, dass ein Antrag die Voraussetzungen f\\u00fcr die \\nAnerkennung nicht erf\\u00fcllt, liegt bei der Hochschule Konstanz. Ganz oder teilweise ablehnende Entscheidun-\\ngen werden vom  Zentralen Pr\\u00fcfungsamt schriftlich begr\\u00fcndet und mit einer Rechtsbehelfsbelehrung verse-\\nhen. \\n(5) Soweit Vereinbarungen und Abkommen der Bundesrepublik Deutschland mit anderen Staaten \\u00fcber \\nGleichwerti gkeiten im Hochschulbereich (\\u00c4quivalenzabkommen) Studieren de ausl\\u00e4ndischer Staaten abwei-\\nchend von den Abs\\u00e4tzen 1 bis 4 beg\\u00fcnstigen, gehen die Regelungen der \\u00c4quivalenzabkommen vor. Von der Kultusministerkonferenz und von der Hochschulrektorenkonferenz gebilligte \\u00c4quivalenzvereinbarungen so-wie Absprachen im Rahmen von Hochschul partnerschaften sind zu beachten.\",\n            \"type\": \"Document\"\n        },\n        {\n            \"metadata\": {\n                \"file_name\": \"124_SPOMa_AT_Senat_08112022.pdf\",\n                \"file_path\": \"/home/tpllmws23/Chatbot-LLama-Pruefungsamt/main_data/124_SPOMa_AT_Senat_08112022.pdf\",\n                \"page_number\": 2\n            },\n            \"page_content\": \"(5)  Durch Beschluss der Fakult\\u00e4t kann die im Besonderen Teil festgelegte Reihenfolge und Art der Lehrver-\\nanstaltun gen und der zugeh\\u00f6rigen Pr\\u00fcfungen aus zwingenden Gr\\u00fcnden im Ein zelfall f\\u00fcr ein St udiensemes-\\nter abge\\u00e4ndert werden.  \\n(6)  Jedem Masterstudiengang ist eines der beiden Studiengangsprofile \\u201est\\u00e4rker anwendungsorientiert\\u201c oder \\n\\u201est\\u00e4rker forschungsorientiert\\u201c zuzuordnen. Der Profiltyp wird jeweils im Besonderen Teil  beschri eben.\",\n            \"type\": \"Document\"\n        },\n        {\n            \"metadata\": {\n                \"file_name\": \"124_SPOMa_AT_Senat_08112022.pdf\",\n                \"file_path\": \"/home/tpllmws23/Chatbot-LLama-Pruefungsamt/main_data/124_SPOMa_AT_Senat_08112022.pdf\",\n                \"page_number\": 13\n            },\n            \"page_content\": \"fungsleist ungen tragen, diese in wesentlicher Tiefe umfassen und inhaltlich ausreichend auf die Erg\\u00e4nzung \\ndurch weitere zentrale Studieninhal te vorbereiten sowie deren Aufbau erm\\u00f6glichen.  \\nAnzurechnende Kenntnisse und F\\u00e4higkeiten m\\u00fcssen in einer klar abgrenzbaren Lei stung erkennbar sein. \\nDer zeitliche Aufwand f\\u00fcr ihren Erwerb oder ihre Anwendung sowie die dazu erforderlichen Vorkenntnisse\",\n            \"type\": \"Document\"\n        },\n        {\n            \"metadata\": {\n                \"file_name\": \"124_SPOMa_AT_Senat_08112022.pdf\",\n                \"file_path\": \"/home/tpllmws23/Chatbot-LLama-Pruefungsamt/main_data/124_SPOMa_AT_Senat_08112022.pdf\",\n                \"page_number\": 13\n            },\n            \"page_content\": \"sich dieser Modul - bzw. Modulteilpr\\u00fcfung an der Hochschule Konstanz erstmals unterzogen hat.  \\n(8) Die Abs\\u00e4tze 1 und 4 bis 7 gelten bei einem Wechsel des Stud iengangs innerhalb der Hochschule Kon-\\nstanz entspr echend.  \\n(9) Kenntnisse und F\\u00e4higkeiten, die au\\u00dferhalb des Hochschulsystems erworben wurden, sind auf ein Hoch-\\nschulstud ium anzurechnen, wenn  \\n1. zum Zeitpunkt der Anrechnung die f\\u00fcr den Hochschulzugang geltenden Voraussetzungen erf\\u00fcllt sind,  \\n2. die auf das Hochschulstudium anzurechnenden Kenntnisse und F\\u00e4higkeiten den Studien-  und Pr\\u00fcfungs-\\nleistungen, die sie ersetzen sollen, nach Inhalt und Niveau gleichwertig sind und  \\n3. die Kriterien f\\u00fcr die Anrechnung im Rahmen einer Akkreditierung \\u00fcberpr\\u00fcft worden sind.  \\nGleichwertigkeit im Sinne von Satz 1 Nr. 2 besteht dann, wenn die fachlichen Auspr\\u00e4gungen der anzurech-nenden Kenntnisse und F\\u00e4higkeiten \\u00fcberwiegend die Wesensz\\u00fcge der zu ersetzenden Studien-  und Pr\\u00fc-\",\n            \"type\": \"Document\"\n        }\n    ],\n    \"embedding_model\": \"sentence-transformers/all-MiniLM-L6-v2\",\n    \"tags\": []\n}"
# context = {
#     "vars": {
#           "question": "Welche Dokumente können als Nachweis für deutsche Sprachkenntnisse akzeptiert werden?",
#           "expected_response_data": {
#             "keywords": [
#               "Zugangsberechtigung des Studienkollegs",
#               "TestDaF",
#               "TDN 4",
#               "DSH-2",
#               "Telc Deutsch C1"
#             ],
#             "page": [
#               3,
#               4
#             ],
#             "answer": "Als Nachweis für deutsche Sprachkenntnisse können folgende Dokumente akzeptiert werden: (1) Feststellungsprüfung für ein Bachelorstudium durch Vorlage der Zugangsberechtigung des Studienkollegs an der Hochschule Konstanz, (2) Test Deutsch als Fremdsprache (TestDaF) mit mindestens der Stufe TDN 4, (3) Deutsche Sprachprüfung für den Hochschulzugang (DSH) mit mindestens der Stufe DSH-2, (4) „Telc Deutsch C1 Hochschule“ oder eine äquivalente Sprachprüfung gemäß der Rahmenordnung über Deutsche Sprachprüfungen für das Studium an deutschen Hochschulen (RO-DT)."
#           }
#         },
#     "test": {
#         "assert": [
#             {
#                 "type": "python",
#                 "value": "file://metrics/keywords_score.py",
#                 "threshold": 0.2
#             }
#         ]
#     },
# }
# get_assert(output, context) == 0.0