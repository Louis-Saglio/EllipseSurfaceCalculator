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
    val n6 = Neuron(0f)
    val n7 = Neuron(0f)
    val n8 = Neuron(0f)
    val n9 = Neuron(0f)
    val n10 = Neuron(0f)
    val iN0 = InputNode(0f)
    val iN1 = InputNode(0f)
    val links = mapOf(
        n0 to listOf(iN0, iN1),
        n1 to listOf(iN0, iN1),
        n2 to listOf(iN0, iN1, n9),
        n3 to listOf(iN0, iN1, n6),
        n4 to listOf(n0, n1, n2, n3, iN0, n5),
        n5 to listOf(n0, n1, n2, n3, n4),
        n6 to listOf(n0, n1, n2, n3),
        n7 to listOf(n0, n1, n2, n3),
        n8 to listOf(n0, n1, n2, n3),
        n9 to listOf(n4, n5, n6, n7, n8),
        n10 to listOf(n4, n5, n6, n7, n8, n1),
    )
    val neuralNetwork = NeuralNetwork(
        inputNodes = listOf(iN0, iN1),
        links = links,
        outputNeurons = listOf(n9, n10),
    )
    neuralNetwork.printGraphPNG(true)
    println(neuralNetwork.compute(listOf(2f, 3f), true))
    neuralNetwork.printGraphPNG(false)
}
