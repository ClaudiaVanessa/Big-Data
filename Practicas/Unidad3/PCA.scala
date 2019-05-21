//Importamos librería de spark e iniciamos sesión de spark
import org.apache.spark.sql.SparkSession
val spark = SparkSession.builder().appName("PCA_Example").getOrCreate()

//Cargamos el csv "Cancer_Data"
val data = spark.read.option("header","true").option("inferSchema","true").format("csv").load("Cancer_Data")

//Se imprime el schema para ver los tipos de datos
data.printSchema()

//Importamos librería para uso de PCA
import org.apache.spark.ml.feature.{PCA,StandardScaler,VectorAssembler}
import org.apache.spark.ml.linalg.Vectors

//Columnas que serán las caracteristicas para saber si se tiene cancer o no, es un arreglo 
val colnames = (Array("mean radius", "mean texture", "mean perimeter", "mean area", "mean smoothness",
"mean compactness", "mean concavity", "mean concave points", "mean symmetry", "mean fractal dimension",
"radius error", "texture error", "perimeter error", "area error", "smoothness error", "compactness error",
"concavity error", "concave points error", "symmetry error", "fractal dimension error", "worst radius",
"worst texture", "worst perimeter", "worst area", "worst smoothness", "worst compactness", "worst concavity",
"worst concave points", "worst symmetry", "worst fractal dimension"))

//Juntamos las caracteristicas para pasarla a una columna de salida que serán los features
val assembler = new VectorAssembler().setInputCols(colnames).setOutputCol("features")

//Transformamos la data
val output = assembler.transform(data).select($"features")

//Columna de entrada y de salida
val scaler = (new StandardScaler()
  .setInputCol("features")
  .setOutputCol("scaledFeatures")
  .setWithStd(true)
  .setWithMean(false))

//Entrenamos el modelo
val scalerModel = scaler.fit(output)

//Salida
val scaledData = scalerModel.transform(output)

//Ponemos como los parametros los scaledfeature  y los fiteamos
val pca = (new PCA()
  .setInputCol("scaledFeatures")
  .setOutputCol("pcaFeatures")
  .setK(4)
  .fit(scaledData))

val pcaDF = pca.transform(scaledData)

//Imprimimos el resultado
val result = pcaDF.select("pcaFeatures")
result.show()

result.head(1)
