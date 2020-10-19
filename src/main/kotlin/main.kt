import genetic.evolve
import neuralnetwork.InputNode
import neuralnetwork.NeuralNetwork
import neuralnetwork.Neuron
import neuralnetwork.random
import kotlin.math.abs

abstract class BasicOperation(private val symbol: String) : NeuralNetworkProblem() {
    private var a = 0f
    private var b = 0f

    abstract fun compute(a: Float, b: Float): Float

    override fun getInput(): List<Float> {
        a = (-100..100).random(random).toFloat()
        b = (-100..100).random(random).toFloat()
        return listOf(a, b)
    }

    override fun getOutput(): List<Float> {
        return listOf(compute(a, b))
    }

    override fun toString(input: List<Float>, result: List<Float>): String {
        val answer = input.sum()
        return "${input[0]} $symbol ${input[1]} == ${result[0]}\nAnswer : $answer\nError : ${abs(answer - result[0])}"
    }
}

class Addition : BasicOperation("+") {
    override fun compute(a: Float, b: Float): Float {
        return a + b
    }
}

class Multiplication : BasicOperation("*") {
    override fun compute(a: Float, b: Float): Float {
        return a * b
    }
}

fun main() {
    val i0 = InputNode(0f)
    val i1 = InputNode(0f)
    val n0 = Neuron(0f)
    val n1 = Neuron(0f)
    val n2 = Neuron(0f)
    val problem = Addition()
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
        ),
            problem
    )
    individual.printAsPNG("original", displayWeights = true, removeDotFile = true, displayId = true)
    val population = (0 until 1000).map { individual.clone() }
    evolve(population, 500, log = true).minByOrNull { it.fitness() }?.showOff()
}
