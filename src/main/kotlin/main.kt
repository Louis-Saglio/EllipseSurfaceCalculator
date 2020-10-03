import genetic.evolve
import neuralnetwork.InputNode
import neuralnetwork.NeuralNetwork
import neuralnetwork.Neuron

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
        NeuralNetwork(
            listOf(i0, i1),
            mutableMapOf(
                n0 to mutableListOf(i0, i1),
                n1 to mutableListOf(i0, i1),
                n2 to mutableListOf(n0, n1),
                n3 to mutableListOf(n0, n1),
                n4 to mutableListOf(n0, n1),
                n5 to mutableListOf(n2, n3, n4),
            ),
            listOf(n5)
        )
    )
    individual.printAsPNG("original", true)
    val population = (0 until 200).map { individual.clone() }
    val winner = evolve(population, 100, true).minByOrNull { it.fitness() }
    if (winner != null) {
        val a = (0 until 100).random().toFloat()
        val b = (0 until 100).random().toFloat()
        println("$a + $b == ${winner.innerInstance.compute(listOf(a, b), true)}")
        winner.innerInstance.printGraphPNG("winner", true)
    }
}
