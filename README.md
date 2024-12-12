# Multidimensional-idiographic-SLD-software
Directory Structure for the complete system:

linguistic-analyzer/
├── setup.py
├── requirements.txt
├── README.md
├── linguistic_analyzer/
│   ├── __init__.py
│   ├── shared_types.py
│   ├── config.py
│   ├── logger_config.py
│   ├── chat_client.py
│   ├── chat_server.py
│   ├── integrated_analyzer.py
│   ├── analyzers/
│   │   ├── __init__.py
│   │   ├── lexical_analyzer.py
│   │   ├── burst_analyzer.py
│   │   ├── dependency_analyzer.py
│   │   ├── error_classifier.py
│   │   └── academic_vocabulary.py
│   ├── data/
│   │   └── avl_data.py
│   └── visualizer.py
├── tests/
│   ├── __init__.py
│   └── integration_test.py
├── logs/
├── data/
└── output/
