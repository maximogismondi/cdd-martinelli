# TP1 – Análisis Exploratorio con Pandas

* Padrón: 110119
* Nombre completo: Gismondi Máximo
* Corrector: Martín Stefanelli
* Materia: Ciencia de Datos
* Catedra: Martinelli
* Cuatrimestre: 2do cuatrimestre 2025


Guía de lo hecho con los links a lo hecho... los links funcionan desde [Github](https://github.com/maximogismondi/cdd-martinelli/blob/main/tps/tp1/informe.md)

## Limpieza de datos

Se realizó una limpieza inicial con los datos crudos (disponibles en data/raw) para dejar los datasets listos para análisis y consultas. Los archivos limpios se encuentran en data/clean.

* [Categorías](cleaning_data/categories.ipynb)
* [Clientes](cleaning_data/customers.ipynb)
* [Inventario](cleaning_data/inventory_logs.ipynb)
* [Ítems de órdenes](cleaning_data/order_items.ipynb)
* [Órdenes](cleaning_data/orders.ipynb)
* [Productos](cleaning_data/products.ipynb)
* [Reseñas](cleaning_data/reviews.ipynb)

Notas:
- Los .pkl se usan para mantener tipos de datos y estructuras complejas.

## Consultas

1) [Descuentos por estado](querys/01_descuentos_estado.ipynb)
    Cálculo de descuentos por estado de envío y de pago; agregaciones por estado; comparación Top/Bottom; validación y conclusiones.

2) [Devueltos por código postal](querys/02_devueltos_codigo_postal.ipynb)
    Tasa de devolución por código postal; agrupaciones y tablas resumen; identificación de códigos con mayores devoluciones.

3) [Segmentos x tipo de pago](querys/03_segmentos_tipo_de_pago.ipynb)
    Pivots de clientes por segmento y método de pago; métricas de % activos y % consentimiento; análisis de distribución.

4) [Peso de inventario “stuff” por marca](querys/04_peso_inventario_stuff_por_marca.ipynb)
    Integración productos+inventario; filtro por “stuff”; peso acumulado por marca; Top marcas; variación trimestral; comparativas y correlaciones.

5) [Ventas por categoría y subcategorías](querys/05_ventas_categoria.ipynb)
    Integración productos+ítems de órdenes; revenue por producto; revenue por categoría y subcategoría; análisis de distribución.

6) [Reviews por segmento de cliente](querys/06_reviews_by_customer_segment.ipynb)
    Integración clientes+reseñas; análisis de reviews por segmento y género; distribución de ratings

## Visualizaciones según consigna (casos de uso + links)

1. Una continua con una línea de tiempo
    - Caso de uso: variación trimestral del peso de inventario por marca.
    - [04_peso_inventario_stuff_por_marca](querys/04_peso_inventario_stuff_por_marca.ipynb)

2. Una discreta con una continua
    - Caso de uso: barras por marca (discreta) vs peso acumulado (continua).
    - [04_peso_inventario_stuff_por_marca](querys/04_peso_inventario_stuff_por_marca.ipynb)

3. Una discreta con una discreta
    - Caso de uso: segmento de cliente (discreta) vs rating de reviews (discreta).
    - [06_reviews_by_customer_segment](querys/06_reviews_by_customer_segment.ipynb)

4. Una continua con otra continua
    - Caso de uso: dispersión de métricas continuas de inventario/peso.
    - [04_peso_inventario_stuff_por_marca](querys/04_peso_inventario_stuff_por_marca.ipynb)

5. Un heatmap
    - Caso de uso: pivots de totales, % activos y % consentimiento.
    - [03_segmentos_tipo_de_pago](querys/03_segmentos_tipo_de_pago.ipynb)

6. Dos visualizaciones a elección
    - 100% Stacked Barplot
        - Barras apiladas por período (participación porcentual por marca) del "peso" del inventario.
        - [04_peso_inventario_stuff_por_marca](querys/04_peso_inventario_stuff_por_marca.ipynb)
    
    - Violin Barplot
        - Gráfica de la distribución de los pesos de inventario por marca
        - [04_peso_inventario_stuff_por_marca](querys/04_peso_inventario_stuff_por_marca.ipynb)

7. Una visualización que use una de las siguientes: Treemap / Sankey / Joint / Custom
    - Caso de uso: Treemap de revenue por categoría y subcategorías.
    - [05_ventas_categoria](querys/05_ventas_categoria.ipynb)