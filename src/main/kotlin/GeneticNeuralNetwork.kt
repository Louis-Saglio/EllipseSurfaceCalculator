import genetic.Individual
import neuralnetwork.NeuralNetwork

class GeneticNeuralNetwork(private val neuralNetwork: NeuralNetwork) : Individual {

    override fun fitness(): Float {
        TODO("Not yet implemented")
    }

    override fun clone(mutationProbability: Float): Individual {
        return GeneticNeuralNetwork(neuralNetwork.clone(mutationProbability))
    }

    fun printAsPNG(fileName: String, displayWeights: Boolean) {
        neuralNetwork.printGraphPNG(fileName, displayWeights)
    }
}
