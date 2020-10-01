package neuralnetwork

import org.junit.jupiter.api.Test
import kotlin.test.assertEquals

internal class NeuralNetworkTest {

    @Test
    fun testClone() {
        val i0 = InputNode(0f)
        val i1 = InputNode(0f)
        val n0 = Neuron(0f)
        val n1 = Neuron(0f)
        val n2 = Neuron(0f)
        val nn = NeuralNetwork(
            listOf(i0, i1),
            mutableMapOf(
                n0 to mutableListOf(i0, i1),
                n1 to mutableListOf(i0, i1),
                n2 to mutableListOf(n0, n1)
            ),
            listOf(n2),
        )
        val clone = nn.clone()

        assertEquals(nn.compute(listOf(0f, 1f)), clone.compute(listOf(0f, 1f)))
    }
}
