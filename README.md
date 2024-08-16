# Chatbot-LLama-Pruefungsamt
![_c5aac732-7c20-46f2-93ac-5c152cdfc89c](https://github.com/Schmidt-jan/Chatbot-LLama-Pruefungsamt/assets/73313922/01ddc0ca-5e5a-4782-8124-d9c1da357372)

In diesem Repository sind die Skripte und Ergebnisse des Teamprojekts 'Ein Chatbot für das Prüfungsamt'.

Die [Abschlusspräsentation](./Abschlussbericht/praesentation.pptx) und die eine kurze [Zusammenfassung]() sind in dem Ordner [Abschlussbericht/](./Abschlussbericht/) zu finden.

## Datenbasis
Als Datenbasis wurden die Dokumente [Zulassungssatzung für die Masterstudiengänge (ZuSMa)](./main_data_filtered/119_ZuSMa_Senat_18012022.pdf) und [SPO Nr. 5 - Studiengang Informatik (MSI)](./main_data_filtered/SPO_MSI_SPONr5_Senat_10122019.pdf) verwendet.

## Libraries
In dem Ordner [libs](./libs/) sind einige Libraries enthalten welche wir erstellt haben.   
Die Bibliothek [custom_rag_loader](./libs/custom_rag_loader/) haben wir dazu verwendet um die Datenbank sowie die LLMs zu laden.

## Evaluierung - Retrieval Augmented Generation (RAG)
Für die Evaluierung des Retrieval Augmented Generation Ansatzes sind die folgenden Ordner wichtig:
- [Retrieval](./Retrieval/) - Alles was mit dem Erstellen der Datenbasis und deren Tests zu tun hat
- [Evaluierung des RAGs]()
- [prototype](./prototype/) - Ein Prototyp eines einfachen Chatbots

## Finetuning
Alles zu dem Finetuning ist in dem Ordner [finetune](./finetune/) zu finden.