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
