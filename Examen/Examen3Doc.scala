import org.apache.spark.ml.classification.MultilayerPerceptronClassifier 
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator 
import org.apache.spark.sql.types._ 
import org.apache.spark.ml.Pipeline 
import org.apache.spark.ml.PipelineStage
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.feature.StringIndexer

//Estructura que se le da al excel
val estructuraIris = StructType(StructField("Datos0", DoubleType, true) ::StructField("Datos1", DoubleType, true) ::StructField("Datos2", DoubleType, true) ::StructField("Datos3",DoubleType, true) ::StructField("Datos4", StringType, true) :: Nil)

//Se carga Iris, dataframe
val datosIris = spark.read.option("header", "true").schema(estructuraIris)csv("Iris.csv")
//val datosIris = spark.read.option("header", "false").format("csv").load("Iris.csv")
//datosIris.show()  

//label son los diferentes tipos de string
//se tiene que tener un label para poder procesarlos por el clasificador
val etiqueta = new StringIndexer().setInputCol("Datos4").setOutputCol("label")

//el arreglo de las 4 columnas de datos se estan mandando a features
//features es una nueva tabla para poder realizar el procesamiento
val ensamblador = new VectorAssembler().setInputCols(Array("Datos0", "Datos1", "Datos2", "Datos3")).setOutputCol("features")

//Datos de entrenamiento:85 y prueba:15
val splits = datosIris.randomSplit(Array(0.85, 0.15), seed = 1234L) //partiendo los datos
val entrenar = splits(0) 
val prueba = splits(1)

//Capa entrada:4 neuronas. Capa intermedia:8,7 neuronas. Capa salida:3 neuronas
//Son 4 columnas y 3 salidas: setosa, versicolor, virginica
val capasN = Array[Int](4, 8, 7, 3) //red neuronal

//Se hace la creacion del entrenador y se le dan los parametros
//se esta usando el modelo del clasificador
val entrenador = new MultilayerPerceptronClassifier().setLayers(capasN).setBlockSize(128).setSeed(1234L).setMaxIter(100)

//se estan agrupando los datos
val pipeline = new Pipeline().setStages(Array(etiqueta,ensamblador,entrenador))

//Se junta el modelo y lo entrena 
val modelo = pipeline.fit(entrenar)
//val modelo = entrenado.fit(train)

//se crea la variable de los resultados de la prueba
val resultados = modelo.transform(prueba)
resultados.show() //muestra los resultados
//prediciendo la exactitud con un evaluador
val predictionAndLabels = resultados.select("prediction", "label")
val evaluator = new MulticlassClassificationEvaluator().setMetricName("accuracy")

//se imprime la exactitud
println(s"Test set accuracy = ${evaluator.evaluate(predictionAndLabels)}")