import neuralnetwork.InputNode
import neuralnetwork.NeuralNetwork
import neuralnetwork.Neuron
import java.io.File

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
        n0 to listOf(iN0, iN1, n2),
        n1 to listOf(iN0, iN1, n4),
        n3 to listOf(n0, n1),
        n4 to listOf(n0, n1),
        n5 to listOf(n0, n1, iN0),
        n2 to listOf(n3, n4, n5)
    )
    n2.setInputSize(3)
    n0.setInputSize(3)
    n1.setInputSize(3)
    n3.setInputSize(2)
    n4.setInputSize(2)
    n5.setInputSize(3)
    val neuralNetwork = NeuralNetwork(
        inputNodes = listOf(iN0, iN1),
        links = links,
        neurons = listOf(n0, n1, n2),
        outputNeurons = listOf(n2),
    )
    println(neuralNetwork.run(listOf(13f, 2f)))
    val text = neuralNetwork.asGraphviz(displayWeights = true)
    println(text)
    File("nn.dot").writeText(text)
    Runtime.getRuntime().exec("dot -Tpng nn.dot -o neural_network.png")
}
