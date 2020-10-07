import genetic.Individual
import genetic.Problem
import neuralnetwork.NeuralNetwork
import neuralnetwork.random
import kotlin.math.pow

abstract class NeuralNetworkProblem : Problem<List<Float>, List<Float>>() {

    override fun computeError(predictions: List<Float>, expectedOutput: List<Float>): Float {
        return (expectedOutput zip predictions).map { (output, prediction) -> (output - prediction).pow(2) }.average().toFloat()
    }
}

class Addition : NeuralNetworkProblem() {
    private var a = 0f
    private var b = 0f
    override fun getInput(): List<Float> {
        a = (-100..100).random(random).toFloat()
        b = (-100..100).random(random).toFloat()
        return listOf(a, b)
    }

    override fun getOutput(): List<Float> {
        return listOf(a + b)
    }
}

class GeneticNeuralNetwork(val innerInstance: NeuralNetwork) : Individual<GeneticNeuralNetwork, List<Float>, List<Float>>(Addition()) {

    override fun clone(): GeneticNeuralNetwork {
        return GeneticNeuralNetwork(innerInstance.clone())
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
