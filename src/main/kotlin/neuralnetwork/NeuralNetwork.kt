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
    private val connexions: MutableMap<Neuron, MutableList<Inputable<Float>>>,
    private val outputNeurons: List<Neuron>
) {
    private val outputNeuronsByInputable: Map<Inputable<Float>, Set<Neuron>> = buildOutputNeuronsByInputable()

    init {
        connexions.forEach { (neuron, inputs) ->
            neuron.setInputSize(inputs.size)
        }
    }

    private fun buildOutputNeuronsByInputable(): Map<Inputable<Float>, Set<Neuron>> {
        val data: MutableMap<Inputable<Float>, MutableSet<Neuron>> = mutableMapOf()
        for ((neuron, inputs) in connexions) {
            for (input in inputs) {
                (data.getOrPut(input) { mutableSetOf() }).add(neuron)
            }
        }
        return data
    }

    private fun getOutputNeuronsOf(inputable: Inputable<Float>): Set<Neuron> {
        return outputNeuronsByInputable.getOrDefault(inputable, setOf())
    }

    fun compute(inputs: List<Float>, log: Boolean = false): List<Float> {
        val alreadyComputedNeurons = mutableSetOf<Neuron>()
        (inputNodes zip inputs).forEach { it.first.output = it.second }
        var layer = inputNodes.flatMapTo(mutableSetOf()) { getOutputNeuronsOf(it) }
        while (layer.isNotEmpty()) {
            if (log) println("---------------------------------------------------")
            layer.forEach {
                if (it !in alreadyComputedNeurons) {
                    it.compute(
                        inputs = (connexions[it] ?: error("Input of $it not found")).map { input -> input.getOutput() },
                        log = log
                    )
                    alreadyComputedNeurons.add(it)
                }
            }
            layer.forEach { it.update() }
            layer = layer
                .flatMap { getOutputNeuronsOf(it) }
                .filterTo(mutableSetOf()) { it !in alreadyComputedNeurons }
        }
        if (log) println("===================================================")
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

    fun printGraphPNG(fileName: String, displayWeights: Boolean) {
        File("$fileName.dot").writeText(asGraphviz(displayWeights = displayWeights))
        Runtime.getRuntime().exec("dot -Tpng $fileName.dot -o $fileName.png")
    }

    fun clone(): NeuralNetwork {
        val inputNodes = inputNodes.map { InputNode(0f) }
        val cloneByNeuron = mutableMapOf<Neuron, Neuron>()
        val outputNeurons = outputNeurons.map { cloneByNeuron.getOrPut(it) { it.clone() } }
        val connexions = connexions.map { (neuron, inputs) ->
            val newNeuron = cloneByNeuron.getOrPut(neuron) { neuron.clone() }
            val newInputs = inputs.mapTo(mutableListOf()) {
                when (it) {
                    is Neuron -> cloneByNeuron.getOrPut(it) { it.clone() }
                    is InputNode -> inputNodes[this.inputNodes.indexOf(it)]
                    else -> error("should not happen")
                }
            }
            newNeuron to newInputs
        }.toMap(mutableMapOf())
        return NeuralNetwork(inputNodes, connexions, outputNeurons)
    }

    private fun mutateWeightOrBias() {
        connexions.keys.random().mutate()
    }

    private fun removeNeuron() {
        connexions.remove(connexions.keys.filter { it !in outputNeurons }.random())
    }

    private fun addNeuron() {
        val neuron = Neuron(0f)
        val (outputNeuron, inputNodes) = connexions.toList().random()
        inputNodes.add(neuron)
        // todo output weight should be low
        outputNeuron.setInputSize(connexions[outputNeuron]!!.size + 1) // Can't fail because outputNeuron is taken from connexions
        connexions[neuron] = mutableListOf(connexions.values.flatMapTo(mutableSetOf()) { it }.random())
    }

    private fun removeConnexion() {
        val (neuron, inputables) = connexions.toList().random()
        // todo : do not remove entry point
        inputables.removeAt((0 until inputables.size).random())
        neuron.setInputSize(inputables.size)
    }

    private fun addConnexion() {
        val asList = connexions.keys.toList()
        connexions[asList.random()]!!.add(asList.random())
    }

    fun mutate() {
        when (Random.nextInt(100)) {
            in 0 until 80 -> mutateWeightOrBias()
//            in 80 until 85 -> removeNeuron()
//            in 85 until 90 -> addNeuron()
//            in 90 until 95 -> removeConnexion()
//            in 95 until 100 -> addConnexion()
        }
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
            val neurons = (0 until (minNeuronNbr..maxNeuronNbr).random()).map { Neuron(0f) }
            val inputNodes = (0 until inputNbr).map { InputNode(0f) }
            val inputables: MutableList<Inputable<Float>> = inputNodes.toMutableList()
            return NeuralNetwork(
                inputNodes,
                neurons.associateWithTo(mutableMapOf()) { neuron ->
                    (0 until (minConnexionNbr..maxConnexionNbr).random()).mapTo(mutableListOf()) {
                        val inputable = inputables.random()
                        inputables.add(neuron)
                        inputable
                    }
                },
                neurons.choice(outputNbr).toList()
            )
        }
    }
}

// todo : make generic for all Collection children
private fun <E> Collection<E>.choice(size: Int): Collection<E> {
    if (size > this.size) error("Cannot choice $size elements from collection of size ${this.size}")
    val collection = toMutableList()
    val rep = mutableListOf<E>()
    repeat(size) {
        val next = collection.random()
        collection.remove(next)
        rep.add(next)
    }
    return rep
}
