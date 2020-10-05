import genetic.Individual
import neuralnetwork.Identifier
import neuralnetwork.NeuralNetwork
import java.util.concurrent.locks.ReentrantLock
import kotlin.concurrent.withLock
import kotlin.math.pow

class GeneticNeuralNetwork(
    override val innerInstance: NeuralNetwork
) : Individual<NeuralNetwork, GeneticNeuralNetwork> {
    private var fitness: Float? = null
    private val fitnessLock = ReentrantLock()

    override fun fitness(): Float {
        fitnessLock.withLock {
            if (fitness == null) {
                val results = mutableSetOf<Float>()
                repeat(10) {
                    val a = (0 until 100).random().toFloat()
                    val b = (0 until 100).random().toFloat()
                    val expectedResult = a + b
                    val prediction = innerInstance.compute(listOf(a, b))[0]
                    val squaredError = (expectedResult - prediction).pow(2)
                    results.add(squaredError)
                }
                fitness = results.average().toFloat()
            }
            return fitness!!
        }
    }

    override fun clone(): GeneticNeuralNetwork {
        return GeneticNeuralNetwork(innerInstance.clone())
    }

    fun printAsPNG(fileName: String, displayWeights: Boolean, removeDotFile: Boolean, displayId: Boolean) {
        innerInstance.printGraphPNG(fileName, displayWeights, removeDotFile, displayId)
    }

    override fun mutate() {
        innerInstance.mutate()
    }

    override fun print() {
        innerInstance.printGraphPNG(
            "data/individual${Identifier.idOf(this)}.png",
            displayWeights = true,
            removeDotFile = true,
                displayId = false
        )
    }
}
