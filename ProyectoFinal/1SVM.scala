//Libreria VectorAssembler
import org.apache.spark.ml.feature.VectorAssembler

//Empieza la limpieza de los datos del csv "bank-full"
//Se carga Bank, el dataframe
val datosBank = spark.read.option("header","true").option("inferSchema","true").format("csv").load("bank-full.csv")
datosBank.show() //Muestra tabla 

val datosBank = spark.read.option("header","true").option("inferSchema","true").option("delimiter",";").format("csv").load("bank-full.csv")
datosBank.show() //Muestra tabla 

datosBank.printSchema()

val cambio1 = datosBank.withColumn("y",when(col("y").equalTo("yes"),1).otherwise(col("y")))
val cambio2 = cambio1.withColumn("y",when(col("y").equalTo("no"),2).otherwise(col("y")))
//Ahora la columna sigue funcionando como un string 
val nuevacolum = cambio2.withColumn("y",'y.cast("Int"))
nuevacolum.show(1) //Muestra 1

//Arreglo de las caracteristicas con VectorAssembler
val assembler = (new VectorAssembler().setInputCols(Array("balance","day","duration","pdays","previous")).setOutputCol("features"))
val Ldata = assembler.transform(nuevacolum)
Ldata.show(1) //Muestra 1

//Se cambia el nombre de columna por "y"
val cambio = Ldata.withColumnRenamed("y", "label") // Se renombra la columna
val feat = cambio.select("label","features") 

feat.show() //Muestra el dataframe limpio

///// - SVM 
//Libreria LinearSVM
import org.apache.spark.ml.classification.LinearSVC

val c1 = feat.withColumn("label",when(col("label").equalTo("1"),0).otherwise(col("label")))
val c2 = c1.withColumn("label",when(col("label").equalTo("2"),1).otherwise(col("label")))
val c3 = c2.withColumn("label",'label.cast("Int"))

val lsvc = new LinearSVC().setMaxIter(10).setRegParam(0.1)

// Entrenamos el modelo
val lsvcModel = lsvc.fit(c3)  

//Imprima los coeficientes e intercepte para svc lineal
println(s"Coefficients: ${lsvcModel.coefficients} Intercept: ${lsvcModel.intercept}")