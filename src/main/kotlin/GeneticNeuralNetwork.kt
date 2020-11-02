import genetic.Individual
import genetic.Problem
import neuralnetwork.NeuralNetwork
import kotlin.math.pow

abstract class NeuralNetworkProblem : Problem<List<Float>, List<Float>>() {

    override fun computeError(predictions: List<Float>, expectedOutputs: List<Float>): Float {
        return (expectedOutputs zip predictions).map { (output, prediction) -> (output - prediction).pow(2) }.average().toFloat()
    }
}

// todo : inherit from NeuralNetwork
class GeneticNeuralNetwork(
    private val innerInstance: NeuralNetwork,
    private val problem: NeuralNetworkProblem,
    var log: Boolean = false
) : Individual<GeneticNeuralNetwork, List<Float>, List<Float>>(problem) {

    fun getSize(): Int = innerInstance.size

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
        return innerInstance.predict(input, log)
    }

    override fun fitness(): Float {
        return super.fitness().pow(1) * innerInstance.size.toFloat().pow(2)
    }

    override fun showOff() {
        val input = problem.getInput()
        println(problem.toString(input, innerInstance.predict(input, true)))
        printAsPNG("optimal", displayWeights = true, removeDotFile = true, displayId = true)
    }
}
