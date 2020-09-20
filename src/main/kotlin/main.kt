import neuralnetwork.InputNode
import neuralnetwork.NeuralNetwork
import neuralnetwork.Neuron

fun main() {
    val n0 = Neuron(0f)
    val n1 = Neuron(0f)
    val n2 = Neuron(0f)
    val n3 = Neuron(0f)
    val n4 = Neuron(0f)
    val n5 = Neuron(0f)
    val iN0 = InputNode(0f)
    val iN1 = InputNode(0f)
    val links = mapOf(
        n0 to listOf(iN0, iN1),
    )
    val neuralNetwork = NeuralNetwork(
        inputNodes = listOf(iN0, iN1),
        links = links,
        outputNeurons = listOf(n2),
    )
    neuralNetwork.printGraphPNG(true)
    println(neuralNetwork.compute(listOf(13f, 2f), true))
    neuralNetwork.printGraphPNG(true)
}
