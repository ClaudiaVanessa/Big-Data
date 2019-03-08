//Conversiones Practica #1
object Practica1 
{
   def main(args: Array[String]) 
   {
      val metros = 0.3048
      val pies = 3

      println(metros * pies  + "metros") //Imprimimos el valor en metros

      pies = 10

      println(metros * pies  + "metros")

      pies = 15

      println(metros * pies  + "metros")

//Conversión de dolares a pesos
      val pesos = 19.30
      val dolar = 3

      println(dolar * pesos  + "pesos") //Imprimimos el valor en pesos

      dolar = 10

      println(dolar * pesos  + "pesos")

      dolar = 15

      println(dolar * pesos  + "pesos")

//Conversión de millas a km
     val km = 1.60934 
     val milla = 3

     println(milla * km  + "km") //Imprimimos el valor en km

     milla = 10

     println(milla * km  + "km")

     milla = 15

     println(milla * km  + "km")

   }
}
