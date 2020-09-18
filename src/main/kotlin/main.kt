import neuralnetwork.InputNode
import neuralnetwork.NeuralNetwork
import neuralnetwork.Neuron
import java.io.File

fun main() {
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
    val neuralNetwork = NeuralNetwork(
        listOf(iN0, iN1),
        links,
        listOf(n0, n1, n2),
        listOf(n2),
    )
    println(neuralNetwork.run(listOf(1f, 2f)))
    File("nn.dot").writeText(neuralNetwork.asGraphviz())
    Runtime.getRuntime().exec("dot -Tpng nn.dot -o neural_network.png")
}
