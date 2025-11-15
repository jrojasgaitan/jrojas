# Laboratorio AutoML - Dataset Farmacia 
 
## Descripci¢n 
Pipeline de AutoML para predecir ventas en farmacia usando DVC y Git. 
 
## Estructura del Proyecto 
 
\`\`\` 
laboratorio_farmacia/ 
ÃÄÄ data/               # Datasets versionados 
ÃÄÄ src/                # C¢digo del pipeline 
ÃÄÄ models/             # Modelos entrenados 
ÃÄÄ params.yaml         # Configuraci¢n 
ÃÄÄ dvc.yaml           # Pipeline DVC 
ÀÄÄ requirements.txt    # Dependencias 
\`\`\` 
 
## Resultados 
- **Mejor modelo**: GradientBoosting 
- **R2 score**: -4.8249 
- **Modelos comparados**: LinearRegression, RandomForest, GradientBoosting 
 
## Ejecuci¢n 
\`\`\`bash 
dvc repro              # Ejecutar pipeline completo 
dvc metrics show       # Ver m‚tricas 
dvc metrics diff       # Comparar entre versiones 
\`\`\` 
