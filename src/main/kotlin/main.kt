import neuralnetwork.NeuralNetwork

fun main() {
    val neuralNetwork = NeuralNetwork.buildRandom(
        minNeuronNbr = 5,
        maxNeuronNbr = 15,
        minConnexionNbr = 1,
        maxConnexionNbr = 2,
        inputNbr = 2,
        outputNbr = 2
    )
    println(neuralNetwork.compute(listOf(2f, 3f), true))
    neuralNetwork.printGraphPNG("neuralNetwork", true)
    val individual = GeneticNeuralNetwork(neuralNetwork)
    individual.clone()
    individual.printAsPNG("clone", true)
}
