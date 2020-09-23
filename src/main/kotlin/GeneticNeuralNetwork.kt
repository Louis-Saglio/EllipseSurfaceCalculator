import genetic.Individual
import neuralnetwork.NeuralNetwork


class GeneticNeuralNetwork(private val neuralNetwork: NeuralNetwork) : Individual {

    override fun mutate() {
        TODO("Not yet implemented")
    }

    override fun fitness(): Float {
        TODO("Not yet implemented")
    }

    override fun clone(): Individual {
        return GeneticNeuralNetwork(neuralNetwork.clone())
    }

    fun printAsPNG(fileName: String, displayWeights: Boolean) {
        neuralNetwork.printGraphPNG(fileName, displayWeights)
    }
}
