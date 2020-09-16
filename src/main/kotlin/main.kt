import neuralnetwork.InputNode
import neuralnetwork.NeuralNetwork
import neuralnetwork.Neuron

fun main(args: Array<String>) {
    val n0 = Neuron(1f)
    val n1 = Neuron(0f)
    val n2 = Neuron(2f)
    val iN0 = InputNode(0f)
    val iN1 = InputNode(0f)
    val links = mapOf(
        n2 to listOf(n0, n1),
        n0 to listOf(iN0, iN1),
        n1 to listOf(iN0, iN1),
    )
    println(NeuralNetwork(listOf(iN0, iN1), links, listOf(n0, n1, n2), listOf(n2)).run(listOf(1f, 2f)))
}
