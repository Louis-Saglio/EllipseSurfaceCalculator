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

    fun asGraphviz(): String {
        val rows = mutableListOf("digraph {")
        inputNodes.forEach { rows.add("${Identifier.idOf(it)} [color=green]") }
        links.forEach { (output, inputs) ->
            rows.add(output.asGraphvizNode())
            rows.addAll(output.asGraphvizLinks(inputs))
        }
        rows.add("}")
        return rows.joinToString("\n")
    }
}
