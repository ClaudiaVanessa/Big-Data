//Conjunto de librerias a utilizar
import org.apache.spark.mllib.classification.{SVMModel, SVMWithSGD}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.util.MLUtils
import org.apache.log4j._
import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer,VectorIndexer, OneHotEncoder}
import org.apache.spark.ml.classification.LinearSVC
import org.apache.spark.ml.Pipeline
import org.apache.spark.mllib.evaluation.MulticlassMetrics

//Se inicia la sesion en Spark
val spark = SparkSession.builder().getOrCreate()

//Se carga el dataset 
val databank = spark.read.option("header","true").option("inferSchema", "true").option("delimiter",";").format("csv").load("bank-full.csv")

val labelIndexer = new StringIndexer().setInputCol("y").setOutputCol("label").fit(databank)

//Se Crea un nuevo objecto VectorAssembler llamado assembler para los feature
val assembler = (new VectorAssembler().setInputCols(Array("age","balance","day","duration","campaign","pdays","previous")).setOutputCol("features"))

//Utilice randomSplit para crear datos de train y test divididos en 70/30
val Array(training, test) = databank.randomSplit(Array(0.7, 0.3), seed = 11L)

val lsvc = new LinearSVC().setLabelCol("label").setFeaturesCol("features").setPredictionCol("prediction").setMaxIter(10).setRegParam(0.1)

//Se crea un nuevo  pipeline con los elementos: assembler, lr
val pipeline = new Pipeline().setStages(Array(labelIndexer,assembler,lsvc))

//Ajuste (fit) el pipeline para el conjunto de training
val model = pipeline.fit(training)

//Tome los Resultados en el conjuto Test con transform
val result = model.transform(test)

//Tome los Resultados en el conjuto Test con transform
val predictionAndLabels = result.select("prediction", "label")

//Convierta los resutalos de prueba (test) en RDD utilizando .as y .rdd
val predictionAndLabels = results.select($"prediction",$"label").as[(Double, Double)].rdd

//Inicialice un objeto MulticlassMetrics 
val metrics = new MulticlassMetrics(predictionAndLabels)

//Imprima la  Confusion matrix
println("Confusion matrix:")
println(metrics.confusionMatrix)
metrics.accuracy

/*
Confusion matrix:
11989.0  175.0                                                                  
1331.0   246.0  
res37: Double = 0.890400989738738
*/