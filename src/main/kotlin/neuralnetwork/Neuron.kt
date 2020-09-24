package neuralnetwork

import java.lang.RuntimeException
import java.util.concurrent.locks.ReentrantLock
import kotlin.concurrent.withLock
import kotlin.random.Random

interface Inputable<T> {
    fun getOutput(): T
}

class Neuron(private var bias: Float) : Inputable<Float> {
    private val weights = mutableListOf<Float>()
    private var activationFunction: (Float) -> Float = { it }
    private var nextOutput: Float? = null
    private val nextOutputLock = ReentrantLock()
    private var output = 0f

    fun compute(inputs: List<Float>, log: Boolean = false) {
        if (inputs.size != weights.size) error("${inputs.size} input(s) but input size is ${weights.size}")
        nextOutputLock.withLock {
            nextOutput = activationFunction((weights zip inputs).map { it.first * it.second }.sum() + bias)
        }
        if (log) {
            println(
                (weights zip inputs).joinToString(
                    " + ",
                    "$this :\t",
                    " + $bias == ${String.format("%.2f", nextOutput)}"
                ) {
                    "(${String.format("%.2f", it.first)} x ${String.format("%.2f", it.second)})"
                }
            )
        }
    }

    fun setInputSize(size: Int) {
        if (size > weights.size) {
            repeat(size - weights.size) {
                weights.add(Random.nextFloat())
            }
        } else if (size < weights.size) {
            repeat(weights.size - size) {
                weights.removeAt(weights.size - 1 - it)
            }
        }
    }

    override fun getOutput(): Float {
        return output
    }

    fun update() {
        nextOutputLock.withLock {
            output = if (nextOutput != null) nextOutput!! else throw RuntimeException("Not yet computed")
        }
    }

    fun asGraphvizNode(color: String? = null): String {
        return "\"${Identifier.idOf(this)}\" [label=\"${Identifier.idOf(this)}\\n${String.format("%.2f", output)}\\n${String.format("%.2f", bias)}\" color=${color ?: "blue"}]"
    }

    fun asGraphvizLinks(inputs: Collection<Any>, displayWeights: Boolean = true): List<String> {
        return (weights zip inputs).map { (weight, input) ->
            "\"${Identifier.idOf(input)}\" -> \"${Identifier.idOf(this)}\" [label=\"${if (displayWeights) String.format("%.2f", weight) else ""}\"]"
        }
    }

    override fun toString(): String {
        return "${this.javaClass.simpleName}(id=${Identifier.idOf(this)})"
    }

    fun clone(): Neuron {
        val neuron = Neuron(bias)
        neuron.weights.addAll(weights)
        return neuron
    }

    fun mutate() {
        val index = Random.nextInt(weights.size + 1)
        val increment = (Random.nextFloat() - 0.5f) * 2
        if (index < weights.size) {
            weights[index] += increment
        } else {
            bias += increment
        }
    }
}
