import genetic.Individual
import neuralnetwork.NeuralNetwork

class GeneticNeuralNetwork(
    override val innerInstance: NeuralNetwork
) : Individual<NeuralNetwork, GeneticNeuralNetwork> {

    override fun fitness(): Float {
        TODO("Not yet implemented")
    }

    override fun clone(mutationProbability: Float): GeneticNeuralNetwork {
        return GeneticNeuralNetwork(innerInstance.clone(mutationProbability))
    }

    fun printAsPNG(fileName: String, displayWeights: Boolean) {
        innerInstance.printGraphPNG(fileName, displayWeights)
    }
}
