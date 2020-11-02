package neuralnetwork

import java.util.concurrent.locks.ReentrantLock
import kotlin.concurrent.withLock
import kotlin.math.exp
import kotlin.math.sin
import kotlin.math.tanh as tanh_

interface Inputable<T> {
    fun getOutput(): T
}

class ActivationFunction(val name: String, val compute: (Float) -> Float)

val sigmoid = ActivationFunction("sigmoid") { 1.0f / (1.0f + exp(-it)) }
val tanh = ActivationFunction("tanh", ::tanh_)
val identity = ActivationFunction("Identity") { it }
val sin = ActivationFunction("sin") { sin(it) }

class Neuron(private var bias: Float) : Inputable<Float> {

    companion object {
        val possibleActivationFunctions = listOf(identity, sigmoid, tanh, sin)
    }

    private val weights = mutableListOf<Float>()
//    private var activationFunction: ActivationFunction = identity
    private var activationFunction: ActivationFunction = possibleActivationFunctions.random(random)
    private var nextOutput: Float? = null
    private val nextOutputLock = ReentrantLock()
    private var output = 0f

    fun compute(inputs: List<Float>, log: Boolean = false) {
        if (inputs.size != weights.size) error("${inputs.size} input(s) but input size is ${weights.size}")
        nextOutputLock.withLock {
            nextOutput = activationFunction.compute((weights zip inputs).map { it.first * it.second }.sum() + bias)
        }
        if (log) {
            println(
                (weights zip inputs).joinToString(
                    " + ",
                    "$this :\t",
                    " + $bias == ${String.format("%.5f", nextOutput)}"
                ) {
                    "(${String.format("%.2f", it.first)} x ${String.format("%.2f", it.second)})"
                }
            )
        }
    }

    fun setInputSize(size: Int) {
        if (size > weights.size) {
            repeat(size - weights.size) {
                weights.add(random.nextFloat())
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

    fun asGraphvizNode(color: String? = null, displayId: Boolean): String {
        val id = if (displayId) Identifier.idOf(this).toString() else ""
        val bias = String.format("%.2f", bias)
        return "\"${Identifier.idOf(this)}\" [label=\"${listOf(id, bias, activationFunction.name, output).joinToString("\\n")}\", color=${color ?: "blue"}]"
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
        neuron.activationFunction = activationFunction
        return neuron
    }

    // todo : rethink who should determine which internal part must mutate

    fun mutateWeight(weightIndex: Int, delta: Float) {
        weights[weightIndex] += delta
    }

    fun mutateBias(delta: Float) {
        bias += delta
    }

    fun mutateActivationFunction(function: ActivationFunction) {
        activationFunction = function
    }
}
