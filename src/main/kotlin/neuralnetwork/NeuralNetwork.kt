package neuralnetwork

import java.io.File
import java.util.concurrent.locks.ReentrantLock
import kotlin.concurrent.withLock
import kotlin.random.Random

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
        private val connexions: Map<Neuron, List<Inputable<Float>>>,
        private val outputNeurons: List<Neuron>
) {
    private val outputNeuronsByInputable: Map<Inputable<Float>, List<Neuron>> = buildOutputNeuronsByInputable()

    init {
        connexions.forEach { (neuron, inputs) ->
            neuron.setInputSize(inputs.size)
        }
    }

    private fun buildOutputNeuronsByInputable(): Map<Inputable<Float>, List<Neuron>> {
        val data: MutableMap<Inputable<Float>, MutableList<Neuron>> = mutableMapOf()
        for ((neuron, inputs) in connexions) {
            for (input in inputs) {
                (data.getOrPut(input) { mutableListOf() }).add(neuron)
            }
        }
        return data
    }

    fun compute(inputs: List<Float>, log: Boolean = false): List<Float> {
        val alreadyComputedNeurons = mutableSetOf<Neuron>()
        (inputNodes zip inputs).forEach { it.first.output = it.second }
        var layer = inputNodes.flatMapTo(mutableSetOf()) { outputNeuronsByInputable[it] ?: error("Output of $it not found") }
        while (layer.isNotEmpty()) {
            if (log) println("---------------------------------------------------")
            for (it in layer) {
                if (it !in alreadyComputedNeurons) {
                    it.compute((connexions[it] ?: error("Input of $it not found")).map { input -> input.getOutput() }, log)
                    alreadyComputedNeurons.add(it)
                }
            }
            layer.forEach { it.update() }
            layer = layer
                .flatMap { outputNeuronsByInputable.getOrDefault(it, listOf()) }
                .filterTo(mutableSetOf()) { it !in alreadyComputedNeurons }
        }
        return outputNeurons.map(Neuron::getOutput)
    }

    private fun asGraphviz(displayWeights: Boolean = true): String {
        val rows = mutableListOf("digraph {rankdir=LR")
        inputNodes.forEach { rows.add(it.asGraphvizNode()) }
        connexions.forEach { (output, inputs) ->
            rows.add(output.asGraphvizNode(if (output in outputNeurons) "red" else null))
            rows.addAll(output.asGraphvizLinks(inputs, displayWeights))
        }
        rows.add("}")
        return rows.joinToString("\n")
    }

    fun printGraphPNG(displayWeights: Boolean) {
        File("nn.dot").writeText(asGraphviz(displayWeights = displayWeights))
        Runtime.getRuntime().exec("dot -Tpng nn.dot -o neural_network.png")
    }

    companion object {
        fun buildRandom(
            minNeuronNbr: Int,
            maxNeuronNbr: Int,
            minConnexionNbr: Int,
            maxConnexionNbr: Int,
            inputNbr: Int,
            outputNbr: Int,
        ): NeuralNetwork {
            val neurons = (0 until (minNeuronNbr..maxNeuronNbr + 1).random()).map { Neuron(0f) }
            val inputNodes = (0 until inputNbr).map { InputNode(0f) }
            val connexions = neurons.associateWith {
                (0 until (minConnexionNbr..maxConnexionNbr + 1).random()).map { (neurons + inputNodes).random() }
            }
            // todo : may choose the same neuron multiple times
            val outputNeurons = (0 until outputNbr).map { neurons.random() }
            return NeuralNetwork(inputNodes, connexions, outputNeurons)
        }
    }
}
