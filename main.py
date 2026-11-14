from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


DATA_DIR = Path(__file__).parent / "data"

spark = (
    SparkSession.builder
    .master("local[*]")
    .appName("goit-de-hw-03")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR")


def show_df(title, df):
    print(f"\n--- {title} ---")
    df.show(truncate=False)


try:
    # Етап 1. Читаємо три CSV-файли
    users_raw = (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .csv(str(DATA_DIR / "users.csv"))
    )

    purchases_raw = (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .csv(str(DATA_DIR / "purchases.csv"))
    )

    products_raw = (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .csv(str(DATA_DIR / "products.csv"))
    )

    show_df("Етап 1 — users.csv", users_raw)
    show_df("Етап 1 — purchases.csv", purchases_raw)
    show_df("Етап 1 — products.csv", products_raw)

    # Етап 2. Видаляємо рядки з пропущеними значеннями
    users = users_raw.na.drop()
    purchases = purchases_raw.na.drop()
    products = products_raw.na.drop()

    show_df("Етап 2 — очищені користувачі", users)
    show_df("Етап 2 — очищені покупки", purchases)
    show_df("Етап 2 — очищені продукти", products)

    # Додаємо до кожної покупки категорію, ціну й суму покупки
    purchase_details = (
        purchases
        .join(products, on="product_id", how="inner")
        .withColumn(
            "purchase_amount",
            F.col("quantity") * F.col("price")
        )
    )

    # Етап 3. Загальна сума покупок за категоріями
    category_totals = (
        purchase_details
        .groupBy("category")
        .agg(F.round(F.sum("purchase_amount"), 2).alias("total_spent"))
        .orderBy(F.desc("total_spent"))
    )

    show_df("Етап 3 — витрати за категоріями", category_totals)

    # Відбираємо покупки користувачів віком від 18 до 25 років включно
    purchases_18_25 = (
        purchase_details
        .join(users.select("user_id", "age"), on="user_id", how="inner")
        .filter(F.col("age").between(18, 25))
    )

    # Етап 4. Витрати цієї вікової групи за категоріями
    category_totals_18_25 = (
        purchases_18_25
        .groupBy("category")
        .agg(F.round(F.sum("purchase_amount"), 2).alias("total_spent"))
    )

    show_df("Етап 4 — витрати віком 18–25 за категоріями", category_totals_18_25)

    # Етап 5. Частка кожної категорії від усіх витрат віком 18–25
    total_spent_18_25 = (
        category_totals_18_25
        .agg(F.sum("total_spent").alias("age_group_total"))
    )

    category_shares_18_25 = (
        category_totals_18_25
        .crossJoin(total_spent_18_25)
        .withColumn(
            "share_percent",
            F.round(F.col("total_spent") / F.col("age_group_total") * 100, 2)
        )
        .drop("age_group_total")
        .orderBy(F.desc("share_percent"))
    )

    category_shares_display = category_shares_18_25.withColumn(
        "share_percent",
        F.format_number(F.col("share_percent"), 2)
    )
    show_df("Етап 5 — частка витрат віком 18–25, %", category_shares_display)

    # Етап 6. Три категорії з найбільшою часткою витрат
    top_3_categories = category_shares_18_25.limit(3)
    top_3_display = top_3_categories.withColumn(
        "share_percent",
        F.format_number(F.col("share_percent"), 2)
    )

    show_df("Етап 6 — топ-3 категорії", top_3_display)

finally:
    spark.stop()