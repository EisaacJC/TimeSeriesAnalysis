# Framework para el análisis de la volatilidad en mercados financieros.

Este framework tiene como objetivo hallar comparaciones dentro de métricas del estado 
del arte para poder observar como se desempeñan los algoritmos de aprendizaje
profundo, así como una diversa serie de modelos
que típicamente se usan en el análisis clásico de los mercados financieros.

## Características del algoritmo.

* Métodos para la estimación de la volatilidad
  * Ventanas móviles: Es un proceso en el cual se estima la volatilidad del mercado financiero por ventanas de tiempo, finalmente se observa un proceso que se comporta segíun lo siguiente..
  $$\sigma=\sqrt{\frac{\sum (x_i -\mu)^2}{n-1}}*252$$
  * Parkinson: Es un método que asume que la distribución de los precios de las acciones siguen un comportamiento geométrico browniano.
  * Yang-Zhang:Combina lo que es la volatilidad en la noche con la volatilidad intra-día
  * GARCH: Es un método estadístico que captura la volatilidad clusteriszando y promediando el redimiento en series de tiempo financieras.
  * Deep Learning: Usa una LSTM para poder encontrar el comportamiento del mercado financiero.
## Cómo instalar el repositorio?
1. ### **Clonar el repositorio.**
> git clone https://github.com/yourusername/TimesSeriesAnalysis.git

> cd TimeSeriesAnalysis
2. ### **Crear un entorno en python** 
> python -m venv venv 
> source venv/bin/activate 
3. ### **Instalar las dependencias.**
> pip install -r requirements.txt 
4. ### **Uso básico del código.**
> from sector_analyzer import SectorAnalyzer
> analyzer=SectorAnalyzer()
> all_sector_results=analyser.generate_reports()
## Estructura del repositorio
volatility-analysis/

├── src/
│   ├── models/
│   │   ├── traditional.py
│   │   ├── parkinson.py
│   │   ├── yang_zhang.py
│   │   ├── garch.py
│   │   ├── deep_learning.py
│   │   └── gaussian_process.py
│   ├── visualization/
│   │   └── plotter.py
│   └── utils/
│       ├── config.py
│       └── metrics.py
├── config/
│   └── settings.json
├── requirements.txt
└── README.md

## Requierements
* numpy>=1.21.0
* pandas>=1.3.0
* yfinance>=0.1.63
* torch>=1.9.0
* gpytorch>=1.4.0
* matplotlib>=3.4.0
* seaborn>=0.11.0
* arch>=4.19.0
* scipy>=1.7.0
* scikit-learn>=0.24.0
* tensorflow>=2.6.0