import genetic.evolve
import neuralnetwork.*
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.pow

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

abstract class OperationChain(private val size: Int, private val symbol: String) : NeuralNetworkProblem() {
    private var numbers: List<Float> = listOf()

    abstract fun compute(numbers: List<Float>): List<Float>

    override fun getInput(): List<Float> {
        numbers = (0 until size).map { (-100..100).random(random).toFloat() }
        return numbers
    }

    override fun getOutput(): List<Float> {
        return compute(numbers)
    }

    override fun toString(input: List<Float>, result: List<Float>): String {
        val goodAnswer = getOutput().first()
        return "${numbers.joinToString(" $symbol ")} == ${result[0]}\nAnswer : $goodAnswer\nError : ${abs(result.first() - goodAnswer)}"
    }
}

class AdditionChain(size: Int): OperationChain(size, "+") {
    override fun compute(numbers: List<Float>): List<Float> {
        return listOf(numbers.sum())
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

class BinaryReader(private val size: Int) : NeuralNetworkProblem() {
    private var number = 0
    override fun getInput(): List<Float> {
        number = (0 until 2f.pow(size).toInt()).random()
        return (0 until 2f.pow(size).toInt()).random().toString(2).padStart(size, '0').map { if (it == '0') 0f else 1f }
    }

    override fun getOutput(): List<Float> {
        return listOf(number.toFloat())
    }

    override fun toString(input: List<Float>, result: List<Float>): String {
        return "${number.toString(2)}\nAnswer : $number\nPrediction :${result[0]}"
    }

//    override fun computeError(predictions: List<Float>, expectedOutputs: List<Float>): Float {
//        val a = predictions.first().toInt().toString(2)
//        val b = expectedOutputs.first().toInt().toString(2)
//        return (a zip b).sumBy { if (it.first == it.second) 1 else 0 }.toFloat()
//    }
}

class EllipseSurface : NeuralNetworkProblem() {
    private var ab = listOf(0f, 0f)

    override fun getInput(): List<Float> {
        ab = listOf(random.nextFloat() * 200 - 100, random.nextFloat() * 200 - 100).sorted()
        return ab
    }

    override fun getOutput(): List<Float> {
        val (a, b) = ab
        val h = (a - b).pow(2) / (a + b).pow(2)
        return listOf(
            (
                PI * (a + b) * (
                    h.pow(0) / 1 +
                        h.pow(1) / 4 +
                        h.pow(2) / 64 +
                        h.pow(3) / 256 +
                        h.pow(4) * 25 / 16384 +
                        h.pow(5) * 49 / 65536 +
                        h.pow(6) * 441 / 1048576
                    )
                ).toFloat()
        )
    }

    override fun toString(input: List<Float>, result: List<Float>): String {
        return "Surface of Ellipse(a=${input[0]}, b=${input[1]}) == ${result[0]}"
    }
}

fun build(inputSize: Int, outputSize: Int, hiddenSize: Int): NeuralNetwork {
    val inputNodes = (0 until inputSize).map { InputNode(0f) }
    val inputLayer = (0 until hiddenSize).map { Neuron(0f) }
    val hiddenLayer0 = (0 until hiddenSize).map { Neuron(0f) }
    val hiddenLayer1 = (0 until hiddenSize).map { Neuron(0f) }
    val outputLayer = (0 until outputSize).map { Neuron(0f) }
    val map = mutableMapOf<Neuron, MutableList<Inputable<Float>>>()
    connectLayers(map, inputNodes, inputLayer)
    connectLayers(map, inputLayer, hiddenLayer0)
    connectLayers(map, hiddenLayer0, hiddenLayer1)
    connectLayers(map, hiddenLayer1, outputLayer)
    return NeuralNetwork(
        inputNodes = inputNodes,
        inputsByNeuron = map,
        outputNeurons = outputLayer,
    )
}

fun connectLayers(map: MutableMap<Neuron, MutableList<Inputable<Float>>>, inputLayer: List<Inputable<Float>>, outputLayer: List<Neuron>) {
    outputLayer.forEach { neuron ->
        map[neuron] = mutableListOf()
        inputLayer.forEach {
            map[neuron]!!.add(it)
        }
    }
}

fun main() {
    val i0 = InputNode(0f)
    val i1 = InputNode(0f)
    val i2 = InputNode(0f)
    val n0 = Neuron(0f)
    val n1 = Neuron(0f)
    val n2 = Neuron(0f)
//    val problem = BinaryReader(10)
    val problem = AdditionChain(3)
//    val problem = Addition()
//    val problem = EllipseSurface()
    val individual = GeneticNeuralNetwork(
//    NeuralNetwork.buildRandom(10, 15, 2, 4, 10, 1),
//        build(2, 1, 5),
        NeuralNetwork(
            listOf(i0, i1, i2),
            mutableMapOf(
                n0 to mutableListOf(i0, i1, i2),
                n1 to mutableListOf(i0, i1, i2),
                n2 to mutableListOf(n0, n1),
            ),
            listOf(n2)
        ),
        problem
    )
    individual.printAsPNG("original", displayWeights = true, removeDotFile = true, displayId = true)
    val population = (0 until 2000).map { individual.clone() }
    val winner = evolve(population, 500, 2, log = true).minByOrNull { it.fitness() }
    winner?.showOff()
    var stop = false
    do {
        try {
            val input = readLine()!!.split(' ').map(String::toFloat)
            winner!!.log = true
            println(winner.compute(input))
            winner.printAsPNG("demo", displayWeights = true, removeDotFile = true, displayId = true)
            if (input.first() == 666666f) stop = true
        } catch (e: Exception) {
            println(e)
        }
    } while (!stop)
}
