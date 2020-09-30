import neuralnetwork.NeuralNetwork

fun main() {
    val individual = GeneticNeuralNetwork(
        NeuralNetwork.buildRandom(
            minNeuronNbr = 5,
            maxNeuronNbr = 15,
            minConnexionNbr = 1,
            maxConnexionNbr = 2,
            inputNbr = 2,
            outputNbr = 2
        )
    )
    individual.printAsPNG("original", true)
    val clone = individual.clone(0f)
    clone.printAsPNG("clone", true)
    val inputs = listOf(2f, 3f)
    individual.innerInstance.compute(inputs, true)
    clone.innerInstance.compute(inputs, true)
}
