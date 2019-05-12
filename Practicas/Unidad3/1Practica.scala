import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.feature.{VectorAssembler,StringIndexer,VectorIndexer,OneHotEncoder}
import org.apache.spark.ml.linalg.Vectors

//para que no marque errores 
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)

// cargamos el csv 
val spark = SparkSession.builder().getOrCreate()
val df = spark.read.option("header","true").option("inferSchema","true").csv("Wholesale customers data.csv")

//selecionamos las columnas que vamos a entrenar
val feature_data = df.select($"Fresh", $"Milk", $"Grocery", $"Frozen", $"Detergents_Paper", $"Delicassen")
val assembler = new VectorAssembler().setInputCols(Array("Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen")).setOutputCol("features")

//entrenamos la data que colocamos en el vectorassembler
val traning = assembler.transform(feature_data)

//la data entrenada la pasamos al modelo 
val kmeans = new KMeans().setK(2).setSeed(1L)
val model = kmeans.fit(traning)

// Evaluate clustering by calculate Within Set Sum of Squared Errors.
val WSSSE = model.computeCost(traning)
println(s"Within Set Sum of Squared Errors = $WSSSE")

//resultado
println("Cluster Centers: ")
model.clusterCenters.foreach(println)