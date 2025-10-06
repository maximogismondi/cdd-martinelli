# TP2 - Queries con Spark

* Padrón: 110119
* Nombre completo: Gismondi Máximo
* Corrector: Martín Stefanelli
* Materia: Ciencia de Datos
* Catedra: Martinelli
* Cuatrimestre: 2do cuatrimestre 2025

Para que funcionen bien los vínculos recomiendo entrar vía Github donde los mismos funcionan correctamente: [Github](https://github.com/maximogismondi/cdd-martinelli/blob/main/tps/tp2/informe.md)

## Dataset

Contamos con un set de datos de un e-commerce que cuenta con una serie de datasets. Los mismos deberían estar en la carpeta [../data/raw/](../data/raw/) para poder ser utilizados en las distintas querys.


### Categorías

El dataset deberá estar bajo el nombre `categories.csv` y contiene la información sobre las categorías de los distintos productos.

| Nombre | Tipo | Descripción |
| --- | --- | --- |
| category_id | integer | Identificador único de la categoría. |
| category_name | string | Nombre de la categoría. |
| parent_category_name | string (nullable) | Nombre de la categoría padre, si aplica. |
| created_at | datetime (nullable) | Fecha y hora de creación del registro. |


### Productos

El dataset deberá estar bajo el nombre `products.csv` y contiene la información sobre los productos disponibles en la tienda.

| Nombre | Tipo | Descripción |
| --- | --- | --- |
| product_id | integer | Identificador único del producto. |
| product_name | string | Nombre del producto. |
| category_id | integer (nullable) | Referencia a la categoría del producto. |
| brand | string | Marca del producto (puede tratarse como categórica). |
| price | float | Precio de venta. |
| cost | float | Costo asociado al producto. |
| stock_quantity | integer/float | Cantidad disponible en stock. |
| weight_kg | float (nullable) | Peso en kilogramos. |
| dimensions | string (nullable) | Dimensiones en formato libre. |
| description | string (nullable) | Descripción larga del producto. |
| is_active | boolean (nullable) | Indica si el producto está activo. |
| created_at | datetime (nullable) | Fecha de creación del registro. |


### Clientes

El dataset deberá estar bajo el nombre `customers.csv` y contiene la información sobre los clientes de la tienda.

| Nombre | Tipo | Descripción |
| --- | --- | --- |
| customer_id | integer | Identificador único del cliente. |
| first_name | string (nullable) | Nombre del cliente. |
| last_name | string (nullable) | Apellido del cliente. |
| email | string (nullable) | Correo electrónico. |
| phone | string (nullable) | Teléfono de contacto. |
| address | string (nullable) | Dirección postal. |
| postal_code | string (nullable) | Código postal. |
| city | string (nullable) | Ciudad. |
| country | string (nullable) | País. |
| date_of_birth | datetime (nullable) | Fecha de nacimiento. |
| registration_date | datetime (nullable) | Fecha de registro en la plataforma. |
| last_login | datetime (nullable) | Último inicio de sesión. |
| is_active | boolean (nullable) | Indica si la cuenta está activa. |
| gender | string/category | Género (valores comunes: `M`, `F`, otros/NULL según origen). |
| customer_segment | string/category | Segmento del cliente (ej.: `Regular`, `Premium`, `Budget`). |
| marketing_consent | boolean (nullable) | Consentimiento para comunicaciones de marketing. |



### Órdenes

El dataset deberá estar bajo el nombre `orders.csv` y contiene la información sobre las órdenes realizadas por los clientes, con su estado y fechas relevantes.

| Nombre | Tipo | Descripción |
| --- | --- | --- |
| order_id | integer | Identificador único de la orden. |
| customer_id | integer (nullable) | Referencia al cliente que hizo la orden. |
| order_date | datetime | Fecha en que se realizó la orden. |
| status | string/category | Estado de la orden. Valores comunes: `CANCELLED`, `COMPLETED`, `PROCESSING`, `REFUNDED`, `RETURNED`, `SHIPPED`. |
| payment_method | string/category | Método de pago. Valores comunes: `BANK TRANSFER`, `CASH ON DELIVERY`, `CREDIT CARD`, `DEBIT CARD`, `DIGITAL WALLET`, `PAYPAL`. |
| shipping_address | string (nullable) | Dirección de envío. |
| billing_address | string (nullable) | Dirección de facturación. |
| discount_amount | float (nullable) | Monto de descuento aplicado. |
| shipping_cost | float (nullable) | Costo de envío. |
| total_amount | float (nullable) | Total cobrado en la orden. |
| currency | string/category | Moneda de la transacción. Valores comunes: `USD`, `EUR`, `GBP`, `CAD`. |
| created_at | datetime (nullable) | Fecha de creación del registro. |
| updated_at | datetime (nullable) | Fecha de la última actualización. |


### Detalles de órdenes

El dataset deberá estar bajo el nombre `order_items.csv` y contiene la información sobre los ítems de cada orden.

| Nombre | Tipo | Descripción |
| --- | --- | --- |
| order_item_id | integer | Identificador único del ítem de la orden. |
| order_id | integer | Referencia a la orden. |
| product_id | integer | Referencia al producto. |
| quantity | integer | Cantidad del producto en la línea. |
| unit_price | float | Precio unitario. |
| line_total | float | Total de la línea (cantidad * unit_price - descuento si aplica). |
| discount_amount | float (nullable) | Descuento aplicado a la línea, si corresponde. |


### Inventario

El dataset deberá estar bajo el nombre `inventory_logs.csv` y contiene la información sobre los movimientos de inventario de los productos dentro de la tienda.

| Nombre | Tipo | Descripción |
| --- | --- | --- |
| log_id | integer | Identificador del movimiento de inventario. |
| product_id | integer | Referencia al producto afectado. |
| movement_type | string/category | Tipo de movimiento (ej.: `IN`, `OUT`, `ADJUSTMENT`, etc.). |
| quantity_change | integer | Cambio en cantidad (positivo para entrada, negativo para salida). |
| reason | string (nullable) | Motivo del movimiento. |
| timestamp | datetime | Fecha y hora del registro del movimiento. |
| reference_id | integer (nullable) | Identificador de referencia (p.ej. orden relacionada). |
| notes | string (nullable) | Observaciones adicionales. |


### Reseñas

El dataset deberá estar bajo el nombre `reviews.csv` y contiene la información sobre las reseñas dejadas por los clientes sobre los distintos productos.

| Nombre | Tipo | Descripción |
| --- | --- | --- |
| review_id | integer | Identificador único de la reseña. |
| customer_id | integer (nullable) | Referencia al autor de la reseña. |
| product_id | integer (nullable) | Referencia al producto reseñado. |
| rating | integer (nullable) | Puntuación numérica (p.ej. 1-5, según la fuente). |
| title | string (nullable) | Título breve de la reseña. |
| comment | string (nullable) | Texto completo de la reseña. |
| is_verified_purchase | boolean (nullable) | Indica si la compra fue verificada. |
| helpful_votes | integer (nullable) | Votos de utilidad recibidos. |
| created_at | datetime (nullable) | Fecha de creación de la reseña. |

## Consultas

Para el Trabajo Práctico se piden pide sobre el dataset hacer una serie de consutlas utilizando PySpark. La idea de las mismas es utilizar las utilidades básicas de Spark (map, filter, reduceByKey, etc) para obtener los resultados pedidos.

Para cada consulta se hizo el desarrollo en un notebook aparte, los cuales están en la carpeta [queries](../queries). En cada notebook:

1. Se detalla la consigna + detalles consultados con el profesor
2. Se detalla el plan de ejecución de la consulta
3. Se realiza la consulta con PySpark
4. Se muestran los resultados obtenidos ya sea con Pandas en el caso de las tablas o con python base cuando se piden simples datos sueltos.

### Query 1 - Descuentos por estado

Al igual que en el TP1, pero utilizando PySpark los resultados, obtener el total de descuentos aplicados en órdenes agrupados por estado de la orden.

De la misma manera que antes al estar los datos randomizados los resultados prácticamente no varían por estado de US.

Resultados:

| Métrica | Dirección | Estado | Valor |
| :-: | :-: | :-: | :-: |
| Total descuento | Shipping | UT | 134728.09 |
| Promedio descuento | Shipping | UT | 2.75 |
| Total descuento | Billing | MO | 134500.40 |
| Promedio descuento | Billing | MO | 2.73 |

### Query 2 - Ordenes devueltas

Al igual que en las query con Pandas pero ahora manejando todo con PySpark, el resultado no es muy abultado ya que las ordenes devueltas se encuentran muy distribuidas entre códigos postales. Tomamos la dirección de shipping ya que es la más relevante para devoluciones.

Resultados:

| Código postal | Cantidad órdenes devueltas |
| :-: | :-: |
| 70696 | 6 |
| 47612 | 5 |
| 11954 | 5 |
| 83755 | 5 |
| 59883 | 5 |

Por otro lado el nombre más frecuente en este caso tomado de la columna `first_name` no es ningúno en particular ya que no hay duplicados.

En mi caso me dio `Amy` con una orden devuelta.

### Query 3 - Métodos de pago por segmento de cliete, actividad y consentimiento

Nuevamente al igual que en el otro TP los resultados son muy uniformes a lo largo de los métodos de pago variando solo en la dimensión de los segmentos de cliente ya que la distribución de clientes por segmento NO es uniforme.

Resultados:

metodo_pago	segmento	total_clientes	activos	activos (%)	consentidos	consentidos (%)

| metodo_pago | segmento | total_clientes | activos | activos (%) | consentidos | consentidos (%) |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| BANK TRANSFER | BUDGET | 17867 | 16041 | 89.78% | 12451 | 69.69% |
| BANK TRANSFER | PREMIUM | 18292 | 16441 | 89.88% | 12808 | 70.02% |
| BANK TRANSFER | REGULAR | 54606 | 49126 | 89.96% | 38297 | 70.13% |
| CASH ON DELIVERY | BUDGET | 17863 | 16040 | 89.79% | 12449 | 69.69% |
| CASH ON DELIVERY | PREMIUM | 18288 | 16437 | 89.88% | 12807 | 70.03% |
| CASH ON DELIVERY | REGULAR | 54620 | 49140 | 89.97% | 38307 | 70.13% |
| CREDIT CARD | BUDGET | 17865 | 16039 | 89.78% | 12448 | 69.68% |
| CREDIT CARD | PREMIUM | 18286 | 16434 | 89.87% | 12807 | 70.04% |
| CREDIT CARD | REGULAR | 54614 | 49132 | 89.96% | 38301 | 70.13% |
| DEBIT CARD | BUDGET | 17860 | 16034 | 89.78% | 12441 | 69.66% |
| DEBIT CARD | PREMIUM | 18292 | 16442 | 89.89% | 12810 | 70.03% |
| DEBIT CARD | REGULAR | 54606 | 49130 | 89.97% | 38295 | 70.13% |
| DIGITAL WALLET | BUDGET | 17863 | 16039 | 89.79% | 12445 | 69.67% |
| DIGITAL WALLET | PREMIUM | 18290 | 16440 | 89.89% | 12808 | 70.03% |
| DIGITAL WALLET | REGULAR | 54606 | 49127 | 89.97% | 38299 | 70.14% |
| PAYPAL | BUDGET | 17863 | 16039 | 89.79% | 12446 | 69.67% |
| PAYPAL | PREMIUM | 18286 | 16435 | 89.88% | 12808 | 70.04% |
| PAYPAL | REGULAR | 54620 | 49138 | 89.96% | 38306 | 70.13% |

### Query 4 - Peso por marca de productos "stuff"

Al igual que en el TP1, se pedía agrupar por marca los productos que tengan en la descripción la palabra "stuff" y calcular el peso a apartir de los logs de inventario.

Hicimos una sumatoria del flujo de entrada y salida y descartamos aquellos que hayan quedado con flujo negativo ya que representarían un peso negativo. y no tiene demasiado sentido.

Resultados:

| Marca | Peso total (kg) |
| :-: | :-: |
| Undefined | 385686.67 |
| Gardena | 269784.97 |
| Stubhub | 255881.28 |
| Leuchtturm1917 | 237153.79 |
| Adidas | 229815.13 |

En este caso llama la atención que la marca `Undefined` tenga tanto peso, lo cual puede deberse a que muchos productos no tienen marca y por ende quedan en esa categoría o algún error en el dataset.

### Query 5 - Productos con más stock que la media

Se pide saber que porcentaje de los productos tienen más stock que 120% * el stock promedio de la marca.

Para ello primero se calcula el stock promedio por marca, luego se cuentan cuantos productos tienen más stock que 120% del promedio y finalmente se calcula el porcentaje.

Resultados:

* Productos totales: 84869
* Productos con stock >= 120% del promedio de su marca: 28256
* Porcentaje: 33.29%

### Query 6 - Ordenes sin productos del top 10 más famosos

Se tomó la métrica que es el top 10 como los 10 productos con más UNIDADES vendidas. Entonces primero se agruparon los detalles de las ordenes como (product_id, sum(cantidad)) y se tomó el top 10. Luego los valores se broadcastearon para filtrar las órdenes que no tuvieran ninguno de esos productos.

El filtro de si tiene no se hace mapeando a (order_id, product_id) a (order_id, tiene_top_10), luego sumando por order_id y filtrando los que tienen 0.

Hay 2 maneras de hacer esto... uno es partiendo desde el dataset de órdenes haciendo un left outer join con los detalles de órdenes ya procesados. Y otro es partir desde los detalles directamente y no agregar ningún join.

Si hacemos la segunda opción nos da:

* Total órdenes: 99507

Ahora si partimos desde las ordenes directamente y hacemos el left outer join nos da:

* Total órdenes: 4700000

Que casualmente es igual que la cantidad de ordenes. Eso es en realidad porque los ids de las ordenes en el dataset de ordenes no coinciden con los de detalles de ordenes.

El primero se mueve desde 1 a 4700000 y el segundo desde 19900399 a 20000000 por lo que no hay superposición y por ende todas las ordenes quedan como que no tienen productos del top 10.

### Query 7 - Clientes inactivos