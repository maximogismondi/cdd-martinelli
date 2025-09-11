# TP1 – Mapa de desarrollo y checklist

Guía de lo hecho y lo pendiente, con links a los notebooks y ejemplos de gráficos.

## Limpieza de datos

* [Categorías](cleaning_data/categories.ipynb)
* [Clientes](cleaning_data/customers.ipynb)
* [Inventario](cleaning_data/inventory_logs.ipynb)
* [Ítems de órdenes](cleaning_data/order_items.ipynb)
* [Órdenes](cleaning_data/orders.ipynb)
* [Productos](cleaning_data/products.ipynb)
* [Reseñas](cleaning_data/reviews.ipynb)

Notas:
- Los .pkl en data/clean confirman la limpieza completada para cada conjunto.

## Consultas

1) [Descuentos por estado](querys/01_descuentos_estado.ipynb)
    Cálculo de descuentos por estado de envío y de pago; agregaciones por estado; comparación Top/Bottom; validación y conclusiones.

2) [Devueltos por código postal](querys/02_devueltos_codigo_postal.ipynb)
    Tasa de devolución por código postal; agrupaciones y tablas resumen; identificación de códigos con mayores devoluciones.

3) [Segmentos x tipo de pago](querys/03_segmentos_tipo_de_pago.ipynb)
    Pivots de clientes por segmento y método de pago; métricas de % activos y % consentimiento; análisis de distribución.

4) [Peso de inventario “stuff” por marca](querys/04_peso_inventario_stuff_por_marca.ipynb)
    Integración productos+inventario; filtro por “stuff”; peso acumulado por marca; Top marcas; variación trimestral; comparativas y correlaciones.

## Visualizaciones según consigna (casos de uso + links)

1. Una continua con una línea de tiempo
    - Caso de uso: variación trimestral del peso de inventario por marca.
    - [04_peso_inventario_stuff_por_marca](querys/04_peso_inventario_stuff_por_marca.ipynb)

2. Una discreta con una continua
    - Caso de uso: barras por marca (discreta) vs peso acumulado (continua).
    - [04_peso_inventario_stuff_por_marca](querys/04_peso_inventario_stuff_por_marca.ipynb)

3. Una discreta con una discreta
    - Caso de uso: segmento de cliente vs método de pago (matriz de categorías).
    - Notebook/celda: [03_segmentos_tipo_de_pago](querys/03_segmentos_tipo_de_pago.ipynb)

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
    - TODO

## Próximos pasos mínimos

- [ ] Añadir 1 visualización “especial” (treemap/sankey/joint/custom) sin repetir tipos. Sugerencia: Sankey en el notebook 04.
- [ ] Dejar capturas o exportar a html/png las figuras finales clave para el informe.