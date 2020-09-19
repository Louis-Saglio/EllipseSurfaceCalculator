package neuralnetwork

import java.util.concurrent.locks.ReentrantLock
import kotlin.concurrent.withLock

object Identifier {
    private var lastId = -1
    private val idLock = ReentrantLock()
    private val idByObjectLock = ReentrantLock()
    private val idByObject: MutableMap<Any, Int> = mutableMapOf()

    fun idOf(any: Any): Int {
        idByObjectLock.withLock {
            return idByObject.getOrPut(any, Identifier::generateId)
        }
    }

    private fun generateId(): Int {
        idLock.withLock {
            lastId += 1
            return lastId
        }
    }
}

class InputNode(var output: Float) : Inputable<Float> {
    override fun getOutput(): Float {
        return output
    }

    fun asGraphvizNode(): String {
        return "\"${Identifier.idOf(this)}\" [label=\"${String.format("%.2f", output)}\" color=green]"
    }
}

class NeuralNetwork(
    private val inputNodes: List<InputNode> = listOf(),
    private val links: Map<Neuron, List<Inputable<Float>>>,
    private val neurons: List<Neuron>,
    private val outputNeurons: List<Neuron>
) {
    fun run(inputs: List<Float>): List<Float> {
        (inputNodes zip inputs).forEach { it.first.output = it.second }
        neurons.forEach {
            it.compute(links[it]?.map(Inputable<Float>::getOutput) ?: (0 until it.getExpectedInputSize()).map { 0f })
        }
        neurons.forEach(Neuron::update)
        return outputNeurons.map(Neuron::getOutput)
    }

    fun asGraphviz(displayWeights: Boolean = true): String {
        val rows = mutableListOf("digraph {rankdir=LR")
        inputNodes.forEach { rows.add(it.asGraphvizNode()) }
        links.forEach { (output, inputs) ->
            rows.add(output.asGraphvizNode(if (output in outputNeurons) "red" else null))
            rows.addAll(output.asGraphvizLinks(inputs, displayWeights))
        }
        rows.add("}")
        return rows.joinToString("\n")
    }
}
