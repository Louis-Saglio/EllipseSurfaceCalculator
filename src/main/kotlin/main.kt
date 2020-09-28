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
    val clone = individual.clone(0f)
    individual.printAsPNG("original", true)
    clone.printAsPNG("clone", true)
    val inputs = listOf(2f, 3f)
    println(individual.innerInstance.compute(inputs, true))
    println(clone.innerInstance.compute(inputs, true))
}
