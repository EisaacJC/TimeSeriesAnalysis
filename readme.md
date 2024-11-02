# Framework para el análisis de la volatilidad en mercados financieros

Este framework tiene como objetivo hallar comparaciones dentro de métricas del estado del arte para poder observar cómo se desempeñan los algoritmos de aprendizaje profundo, así como una diversa serie de modelos que típicamente se usan en el análisis clásico de los mercados financieros.

## Características del algoritmo

* **Métodos para la estimación de la volatilidad:**
  - **Ventanas móviles:** Es un proceso en el cual se estima la volatilidad del mercado financiero por ventanas de tiempo. Finalmente, se observa un proceso que se comporta según lo siguiente:
    $\sigma=\sqrt{\frac{\sum (x_i -\mu)^2}{n-1}} \times \sqrt{252}$
  
  - **Parkinson:** Es un método que asume que la distribución de los precios de las acciones sigue un comportamiento geométrico browniano. La fórmula es:
    $\sigma = \frac{1}{4 \ln(2)} \cdot \frac{1}{n} \sum_{i=1}^{n} \left( \ln\left(\frac{H_i}{L_i}\right) \right)^2 \times \sqrt{252}$
  
    donde $H_i$ y $L_i$ son el precio máximo y mínimo en el periodo $i$, respectivamente.

  - **Yang-Zhang:** Combina la volatilidad nocturna con la volatilidad intra-día. Su fórmula es:
    $ \sigma^2 = Vol_{overnight}^2 + Vol_{open-close}^2 + Vol_{close-close}^2$

    donde:
    - $\text{Vol}_{\text{overnight}}$ es la volatilidad entre el precio de cierre y apertura del siguiente día.
    - $\text{Vol}_{\text{open-close}}$ es la volatilidad durante el día.
    - $\text{Vol}_{\text{close-close}}$ es la volatilidad de cierre a cierre.

  - **GARCH:** Es un método estadístico que captura la volatilidad al clústerizar y promediar el rendimiento en series de tiempo financieras. La fórmula básica para GARCH(1,1) es:
    $\sigma_t^2 = \alpha_0 + \alpha_1 \epsilon_{t-1}^2 + \beta_1 \sigma_{t-1}^2$

    donde:
    - $\sigma_t^2$ es la varianza condicional en el tiempo $t$.
    - $\alpha_0$, $\alpha_1$, y $\beta_1$ son parámetros del modelo.
    - $\epsilon_{t-1}$ es el residuo en el tiempo $t-1$.

  - **Deep Learning:** Usa una LSTM para poder encontrar el comportamiento del mercado financiero.

## ¿Cómo instalar el repositorio?

1. **Clonar el repositorio:**
   ```bash
   git clone https://github.com/yourusername/TimesSeriesAnalysis.git
   cd TimeSeriesAnalysis
   ```

2. **Crear un entorno en Python:**
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. **Instalar las dependencias:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Uso básico del código:**
   ```python
   from sector_analyzer import SectorAnalyzer
   analyzer = SectorAnalyzer()
   all_sector_results = analyzer.generate_reports()
   ```

## Estructura del repositorio

```
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
```

## Requisitos

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
