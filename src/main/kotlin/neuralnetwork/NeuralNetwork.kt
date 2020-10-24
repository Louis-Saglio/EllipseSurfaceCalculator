package neuralnetwork

import java.io.File
import java.util.concurrent.locks.ReentrantLock
import kotlin.concurrent.withLock
import kotlin.random.Random

val random = Random(0)

object Identifier {
    // May cause memory leak
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
    private val inputsByNeuron: MutableMap<Neuron, MutableList<Inputable<Float>>>,
    private val outputNeurons: List<Neuron>
) {
    private val outputNeuronsByInput: MutableMap<Inputable<Float>, MutableSet<Neuron>> = buildOutputNeuronsByInput()
    private val neurons
        // todo : use cache
        get() = outputNeuronsByInput.values.flatten().toMutableSet()
    val size: Int
        get() = inputsByNeuron.flatMap { it.value }.size

    init {
        inputsByNeuron.forEach { (neuron, inputs) ->
            neuron.setInputSize(inputs.size)
        }
    }

    private fun buildOutputNeuronsByInput(): MutableMap<Inputable<Float>, MutableSet<Neuron>> {
        val data: MutableMap<Inputable<Float>, MutableSet<Neuron>> = mutableMapOf()
        for ((neuron, inputs) in inputsByNeuron) {
            data.getOrPut(neuron) { mutableSetOf() }
            for (input in inputs) {
                data.getOrPut(input) { mutableSetOf() }.add(neuron)
            }
        }
        return data
    }

    private fun getOutputNeuronsOf(inputable: Inputable<Float>): MutableSet<Neuron> {
        return outputNeuronsByInput.getOrDefault(inputable, mutableSetOf())
    }

    fun predict(inputs: List<Float>, log: Boolean = false): List<Float> {
        val alreadyComputedNeurons = mutableSetOf<Neuron>()
        (inputNodes zip inputs).forEach { it.first.output = it.second }
        var layer = inputNodes.flatMapTo(mutableSetOf()) { getOutputNeuronsOf(it) }
        while (layer.isNotEmpty()) {
            if (log) println("---------------------------------------------------")
            layer.forEach {
                if (it !in alreadyComputedNeurons) {
                    it.compute(
                        inputs = (inputsByNeuron[it] ?: error("Input of $it not found")).map { input -> input.getOutput() },
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

    private fun asGraphviz(displayWeights: Boolean = true, displayId: Boolean): String {
        val rows = mutableListOf("digraph {rankdir=LR")
        inputNodes.forEach { rows.add(it.asGraphvizNode()) }
        inputsByNeuron.forEach { (output, inputs) ->
            rows.add(output.asGraphvizNode(if (output in outputNeurons) "red" else null, displayId))
            rows.addAll(output.asGraphvizLinks(inputs, displayWeights))
        }
        rows.add("}")
        return rows.joinToString("\n")
    }

    fun printGraphPNG(fileName: String, displayWeights: Boolean, removeDotFile: Boolean, displayId: Boolean) {
        val dotFile = File("$fileName.dot")
        dotFile.writeText(asGraphviz(displayWeights = displayWeights, displayId))

        // for some unknown reasons, the png file is not generated , except if we try a lot of times (8 most of the times)
        val pngFile = File("$fileName.png")
        pngFile.delete()
        do {
            Runtime.getRuntime().exec("dot -Tpng $fileName.dot -o $fileName.png")
        } while (!pngFile.exists())

        if (removeDotFile) {
            dotFile.deleteOnExit()
        }
    }

    fun clone(): NeuralNetwork {
        val inputNodes = inputNodes.map { InputNode(0f) }
        val cloneByNeuron = mutableMapOf<Neuron, Neuron>()
        val outputNeurons = outputNeurons.map { cloneByNeuron.getOrPut(it) { it.clone() } }
        val connexions = inputsByNeuron.map { (neuron, inputs) ->
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

    private fun addNeuron(inputs: MutableList<Inputable<Float>>, outputs: MutableSet<Neuron>) {
        // todo: optimize by crossing data from connexions and outputNeuronsByInputable
        val neuron = Neuron(0f)
        inputsByNeuron[neuron] = mutableListOf()
        outputNeuronsByInput[neuron] = mutableSetOf()
        inputs.forEach { addConnexion(it, neuron) }
        outputs.forEach { addConnexion(neuron, it) }
    }

    private fun removeConnexion(from: Inputable<Float>, to: Neuron) {
        assert(!(from is InputNode && outputNeuronsByInput[from]!!.size == 1))
        val inputs = inputsByNeuron[to] ?: error("Neuron not found")
        inputs.remove(from)
        to.setInputSize(inputs.size)
        (outputNeuronsByInput[from] ?: error("Input not found")).remove(to)
    }

    private fun removeNeuron(neuron: Neuron) {
        // todo: optimize by crossing data from connexions and outputNeuronsByInputable
        inputsByNeuron.remove(neuron)
        inputsByNeuron.forEach { (output, inputs) ->
            inputs.remove(neuron)
            output.setInputSize(inputs.size)
        }
        outputNeuronsByInput.remove(neuron)
        outputNeuronsByInput.forEach { (_, neurons) -> neurons.remove(neuron) }
    }

    private fun addConnexion(from: Inputable<Float>, to: Neuron) {
        val inputs = inputsByNeuron[to] ?: error("Neuron not found")
        inputs.add(from)
        getOutputNeuronsOf(from).add(to)
        to.setInputSize(inputs.size)
    }

    fun mutate() {
        when (random.nextInt(115)) {
            in 0 until 80 -> {
                if (neurons.isNotEmpty()) {
                    val neuron = neurons.random(random)
                    when (random.nextInt(100)) {
                        in 0 until 80 -> neuron.mutateWeight(
                            (0 until inputsByNeuron[neuron]!!.size).random(random),
                            (random.nextFloat() * 2) - 1,
                        )
                        in 80 until 95 -> neuron.mutateBias((random.nextFloat() * 2) - 1)
                        in 95 until 100 -> neuron.mutateActivationFunction(listOf(sigmoid, tanh, identity).choice(1).toList()[0])
                    }
                }
            }
            in 80 until 85 -> {
                val output = if (neurons.isEmpty()) mutableSetOf() else mutableSetOf(neurons.random(random))
                addNeuron(
                    mutableListOf((neurons + inputNodes).random(random)),
                    output,
                )
            }
            in 85 until 95 -> {
                if (neurons.isNotEmpty()) {
                    removeConnexion(
                        outputNeuronsByInput.keys.filter { !(it is InputNode && outputNeuronsByInput[it]!!.size == 1) }.random(random),
                        inputsByNeuron.keys.random(random),
                    )
                }
            }
            in 95 until 100 -> {
                if (neurons.isNotEmpty()) {
                    addConnexion(
                        outputNeuronsByInput.keys.random(random),
                        inputsByNeuron.keys.random(random),
                    )
                }
            }
            in 100 until 105 -> {
                val removableNeurons = inputsByNeuron.keys.filter { it !in outputNeurons }
                if (removableNeurons.isNotEmpty()) {
                    removeNeuron(removableNeurons.random(random))
                }
            }
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
            val neurons = (0 until (minNeuronNbr..maxNeuronNbr).random(random)).map { Neuron(0f) }
            val inputNodes = (0 until inputNbr).map { InputNode(0f) }
            val inputables: MutableList<Inputable<Float>> = inputNodes.toMutableList()
            return NeuralNetwork(
                inputNodes,
                neurons.associateWithTo(mutableMapOf()) { neuron ->
                    (0 until (minConnexionNbr..maxConnexionNbr).random(random)).mapTo(mutableListOf()) {
                        val inputable = inputables.random(random)
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
        val next = collection.random(random)
        collection.remove(next)
        rep.add(next)
    }
    return rep
}
