import genetic.evolve
import neuralnetwork.InputNode
import neuralnetwork.NeuralNetwork
import neuralnetwork.Neuron
import kotlin.math.abs

fun main() {
    val i0 = InputNode(0f)
    val i1 = InputNode(0f)
    val n0 = Neuron(0f)
    val n1 = Neuron(0f)
    val n2 = Neuron(0f)
    val n3 = Neuron(0f)
    val n4 = Neuron(0f)
    val n5 = Neuron(0f)
    val individual = GeneticNeuralNetwork(
//    NeuralNetwork.buildRandom(5, 5, 2, 2, 2, 1)
        NeuralNetwork(
            listOf(i0, i1),
            mutableMapOf(
                n0 to mutableListOf(i0, i1),
                n1 to mutableListOf(i0, i1),
                n2 to mutableListOf(n0, n1),
            ),
            listOf(n2)
        )
    )
    individual.printAsPNG("original", displayWeights = true, removeDotFile = true, displayId = true)
    val population = (0 until 200).map { individual.clone() }
    val winner = evolve(population, 300, log = true, print = false).minByOrNull { it.fitness() }
    if (winner != null) {
        val a = (0 until 100).random().toFloat()
        val b = (0 until 100).random().toFloat()
        val result = winner.innerInstance.compute(listOf(a, b), true)[0]
        val answer = a + b
        println("$a + $b == $result")
        println("Answer : $answer")
        println("Error : ${abs(answer - result)}")
        winner.innerInstance.printGraphPNG("winner", displayWeights = true, removeDotFile = true, displayId = true)

        println("0 + 0 == ${winner.innerInstance.compute(listOf(0f, 0f), false)[0]}")
        println("100 + 100 == ${winner.innerInstance.compute(listOf(100f, 100f), false)[0]}")
        println("-100 + 100 == ${winner.innerInstance.compute(listOf(-100f, 100f), false)[0]}")
        println("100 + -100 == ${winner.innerInstance.compute(listOf(100f, -100f), false)[0]}")
    }
}
