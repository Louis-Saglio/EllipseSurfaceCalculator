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
    val individual = GeneticNeuralNetwork(neuralNetwork)
    val clone = individual.clone(1f)
    individual.printAsPNG("original", true)
    clone.printAsPNG("clone", true)
}
