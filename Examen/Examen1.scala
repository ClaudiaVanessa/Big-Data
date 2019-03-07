//1. Verifique solo número par

def isEven(num:Int): Boolean = {
     return num%2 == 0
 }
 def isEven(num:Int): num%2 == 0
 println(isEven(6))
 println(isEven(3))

//2. Buscar números pares en lista

def listEvens(list:List[Int]): String = {
    for(n <- list){
        if(n%2==0){
            println(s"$n is even")
        }else{
            println(s"$n is odd")
        }
    }
    return "Done"
}

val l = List(1,2,3,4,5,6,7,8)
val l2 = List(4,3,22,55,7,8)
listEvens(l)
listEvens(l2)

//3. Afortunado número 7

def afortunado(list:List[Int]): Int = {
    var res=0
    for(n <- list){
        if(n==7){
            res = res + 14
        }else{
            res = res + n
        }
    }
    return res
}

val af= List(1,7,7)
println(afortunado(af))

//4. ¿Puedes equilibrar?

def balance(list:List[Int]): Boolean = {
    var primera = 0
    var segunda = 0

    segunda = list.sum

    for(i <- Range(0,list.length)){
        primera = primera + list(i)
        segunda = segunda - list(i)

        if(primera == segunda){
            return true
        }
    }
    return false 
}

//5. Verificar palíndromo

def palindromo(palabra:String):Boolean = {
    return (palabra == palabra.reverse)
}

val palabra = "OSO"
val palabra2 = "ANNA"
val palabra3 = "JUAN"

println(palindromo(palabra))
println(palindromo(palabra2))
println(palindromo(palabra3))