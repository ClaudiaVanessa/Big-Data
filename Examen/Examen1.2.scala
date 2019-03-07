//1. Sesion de Spark
import org.apache.spark.sql.SparkSession 
val spark = SparkSession.builder().getOrCreate()

//2. Cargar archivo
val df = spark.read.option("header", "true").option("inferSchema","true")csv("Netflix_2011_2016") 

//3. Nombre de las columnas
df.printSchema() 

//4. Esquema
df.printSchema() 

//5. Imprime las primeras 5 columnas/renglones
df.head(5)
for(row <- df.head(5)){  
    println(row)
}

//6. Usa describe()
df.describe().show() 

//7. Columna nueva, HV Ratio
val df2 = df.withColumn("HV Ratio", df("High")/df("Volume"))
df2.show()

//8. Pico más alto de la columna en la columna "Hight"
df.orderBy($"High".desc).show()

//9. Significado de la columna cerrar "Close"
//Es el valor de las acciones al termino del dia que cierran en dicho periodo de tiempo

//10. Máximo y Minimo de la columna Volume
df.select(min("Volume")).show()
df.select(max("Volume")).show()

//11. Con sintaxis Scala/Spark $ contestar lo siguiente:
//11.a Cuántos días fue la columna "Close" ingferior a $600
val df3 = df.filter($"Close" < 600)
df3.count() 

//11.b Porcentaje del tiempo fue la columna "Hight" mayor a $500 
val df4 = df.filter($"Close" > 500)
df4.count()

//11.c Correlación de Pearson de columna "High" y columna "Volumen"
df.select(corr("High", "Volume")).show() 

//11.d Máximo de la columna "High" por año
val df5 = df.withColumn("Year", year(df("Date")))
val df5max = df5.groupBy("Year").max()
df5max.select($"Year", $"max(High)").show() 

//11.e Promedio de la columna "Close" para cada mes del calendario
val df6 = df.withColumn("Month", month(df("Date")))
val df6mean = df6.groupBy("Month").mean()
df6mean.select($"Month", $"avg(Close)").show()    