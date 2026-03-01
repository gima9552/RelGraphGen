# RelGraphGen

In this repo, you can find an example of the pipeline described in "ReX-GG: a LLM Ensemble Pipeline for Relation-extraction and Graph Generation". Get summaries of scientific documents and the relations presented in them, all in the form of colour-coded, interactive graphs, powered by a customisable ensemble of Large Language Models.

### Structure of the repo
...
  main_folder
  ├── data
  │   ├── logs           # Destination folder for the error logs while running the pipeline.
  │   └── queries        # Destination folder for the stored queries for all LLM communication.
  ├── html_files         # Destination folder for the output graphs in HTML format.
  ├── .env               # Environment file with the OpenRouter API key (modify it before using the pipeline).
  ├── config.py          # Configuration information for the main file. Feel free to modify at your convenience.
  └── main.py            # Executable for the pipeline, to be run via command line.
