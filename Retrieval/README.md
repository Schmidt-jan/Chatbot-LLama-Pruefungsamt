# Retrieval

In diesem Ordner liegen die benötigten Skripte um Vektordatenbanken zu erstellen und diese zu Testen.

## Datenbank erstellen
Um eine neue Datenbank zu erstellen, kann das [database_creator.py](./database_creator.py)-Skript genutzt werden. 
Es können verschiedene Einstellungen gesetzt werden. Daten können vorbereitet werden (Entfernen der Header, ...) oder in ihrere rohen Form geladen werden. Die so extrahierten Daten werden dann in Chunks unterteilt und diese in einer Datenbank gespeichert. Sollten die Daten vorbereitet werden, müssen die Parameter für Header und Footer auf das Dokument angepasst werden.

Aktuell wird für jedes Modell mehrere Datenbanken mit verschiedenen Einstellungen erstellt:
- embedding_list = [ ... ]
- Chunk size: 1024, 2048, 4096
- Chunk overlap: 128, 256
- seperators: ["\n\n", "\n§", "  §"]
- distance function: ["cosine", "l2"]
Je nach Anwendungszweck kann es sinnvoll sein diese Parameter zu ändern. Bei Änderung sollte aber darauf geachtet werden, dass die [custom-rag-loader-library](../libs/custom_rag_loader/) aktuell gehalten wird. Diese haben wir zum laden der Datenbanken, und LLMs verwendet.

Alle Daten werden in einer ChromaDB unter dem [databases](./databases/) Verzeichnis gespeichert. Die ChromaDB kann direkt über deren API, als auch über LanChain oder LlamaIndex angesprochen werden. Somit hat man viel flexibilität.


## Datenbank testen
Zum Testen der Datenbanken wird der [tests](./tests) Ordner verwendet. Die Tests werden mir Promptfoo ausgeführt. Das Verzeichnis enthalt drei Ordner:
- [_input](./tests/_input/) - Input Fragen und skript zum erstellen einer formatierten json Datei mit den verschiedenen Test-Cases
- [_output](./tests/_output/) - Ergebnisse eines Testruns. Die [output.json](./tests/_output/output.json) enthält die Ergebnisse des letzten Testruns (Da diese Datei sehr groß ist, sollte diese mit keinem Editor geöffnet werden. Zudem kann sie nicht auf github hochgeladen werden). Um die Daten in ein besser verwendbares Format zu bringen, kann das Skript [promptfoo_output_converter.py](./tests/_output/promptfoo_output_converter.py) verwendet werden. Dieses erstellt ein kompaktere json Datei mit allen benötigten Informationen. Um nun die Daten auszuwerten und sich die Ergebnisse zu plotten, kann das Jupyter-Notebook [plot_generation](./tests/_output/plot_generation.ipynb) verwendet werden.
- [rag](./tests/rag/) - In diesem Verzeichnis liegen die ganzen promptfoo Konfigurationen für die Tests. Die einzelnen Testcases sind in der Datei [tests.yaml](./tests/rag/tests.yaml). In der Datei [promptfooconfig.yaml](./tests/rag/promptfooconfig.yaml) liegen die ganzen Einstellungen für die verschiedenen Testeinstellungen. Hierbei muss darauf geachtet werden, dass die ganzen Konfigurationen einmal im yaml-Format für den Test eingegeben werden, und diese aber auch im json-Format dem `label` übergeben werden müssen. Dies wird für die spätere Auswertung benötigt. Um einen Testrun zu starten muss folgender Befehl ausgeführt werden: `promptfoo eval -j 15 --no-cache`.