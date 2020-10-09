import genetic.Individual
import genetic.Problem
import neuralnetwork.NeuralNetwork
import kotlin.math.pow

abstract class NeuralNetworkProblem : Problem<List<Float>, List<Float>>() {

    override fun computeError(predictions: List<Float>, expectedOutput: List<Float>): Float {
        return (expectedOutput zip predictions).map { (output, prediction) -> (output - prediction).pow(2) }.average().toFloat()
    }
}

class GeneticNeuralNetwork(
    val innerInstance: NeuralNetwork,
    private val problem: NeuralNetworkProblem
) : Individual<GeneticNeuralNetwork, List<Float>, List<Float>>(problem) {

    override fun clone(): GeneticNeuralNetwork {
        return GeneticNeuralNetwork(innerInstance.clone(), problem)
    }

    fun printAsPNG(fileName: String, displayWeights: Boolean, removeDotFile: Boolean, displayId: Boolean) {
        innerInstance.printGraphPNG(fileName, displayWeights, removeDotFile, displayId)
    }

    override fun mutate() {
        innerInstance.mutate()
    }

    override fun compute(input: List<Float>): List<Float> {
        return innerInstance.compute(input)
    }
}
