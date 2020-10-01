import genetic.Individual
import neuralnetwork.NeuralNetwork
import java.util.concurrent.locks.ReentrantLock
import kotlin.concurrent.withLock
import kotlin.math.abs

class GeneticNeuralNetwork(
    override val innerInstance: NeuralNetwork
) : Individual<NeuralNetwork, GeneticNeuralNetwork> {
    private var fitness: Float? = null
    private val fitnessLock = ReentrantLock()

    override fun fitness(): Float {
        fitnessLock.withLock {
            if (fitness == null) {
                val a = (0 until 100).random().toFloat()
                val b = (0 until 100).random().toFloat()
                fitness = 1 / abs(innerInstance.compute(listOf(a, b))[0] - (a + b))
            }
            return fitness!!
        }
    }

    override fun clone(): GeneticNeuralNetwork {
        return GeneticNeuralNetwork(innerInstance.clone())
    }

    fun printAsPNG(fileName: String, displayWeights: Boolean) {
        innerInstance.printGraphPNG(fileName, displayWeights)
    }

    override fun mutate() {
        innerInstance.mutate()
    }
}
