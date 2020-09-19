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
    val outputNeuronsByInputable: Map<Inputable<Float>, List<Neuron>> = buildOutputNeuronsByInputable()

    init {
        links.forEach { (neuron, inputs) ->
            neuron.setInputSize(inputs.size)
        }
    }

    private fun buildOutputNeuronsByInputable(): Map<Inputable<Float>, List<Neuron>> {
        val data: MutableMap<Inputable<Float>, MutableList<Neuron>> = mutableMapOf()
        for ((neuron, inputs) in links) {
            for (input in inputs) {
                (data.getOrPut(input) { mutableListOf() }).add(neuron)
            }
        }
        return data
    }

    fun run(inputs: List<Float>): List<Float> {
        (inputNodes zip inputs).forEach { it.first.output = it.second }
        neurons.forEach {
            it.compute(links[it]?.map(Inputable<Float>::getOutput) ?: (0 until it.getExpectedInputSize()).map { 0f })
        }
        neurons.forEach(Neuron::update)
        return outputNeurons.map(Neuron::getOutput)
    }

    fun compute(inputs: List<Float>, log: Boolean = false) {
        val alreadyComputedNeurons = mutableSetOf<Neuron>()
        (inputNodes zip inputs).forEach { it.first.output = it.second }
        var layer = inputNodes.flatMap { outputNeuronsByInputable[it] ?: error("Output of $it not found") }
        while (layer.isNotEmpty()) {
            for (it in layer) {
                if (it !in alreadyComputedNeurons) {
                    it.compute((links[it] ?: error("Input of $it not found")).map { input -> input.getOutput() }, log)
                    alreadyComputedNeurons.add(it)
                }
            }
            layer.forEach { it.update() }
            layer = layer
                .flatMap { outputNeuronsByInputable[it] ?: error("Output of $it not found") }
                .filter { it !in alreadyComputedNeurons }
        }
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
