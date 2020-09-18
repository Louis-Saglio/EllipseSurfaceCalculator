package neuralnetwork

import java.lang.RuntimeException
import java.util.concurrent.locks.ReentrantLock
import kotlin.concurrent.withLock

interface Inputable<T> {
    fun getOutput(): T
}

class Neuron(private var bias: Float) : Inputable<Float> {
    private val weights = mutableListOf<Float>()
    private var activationFunction: (Float) -> Float = { it }
    private var nextOutput: Float? = null
    private val nextOutputLock = ReentrantLock()
    private var output = 0f

    fun compute(inputs: List<Float>) {
        nextOutputLock.withLock {
            nextOutput = activationFunction((weights zip inputs).map { it.first * it.second }.sum() + bias)
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

    fun getExpectedInputSize() = weights.size

    fun asGraphvizNode(): String {
        return "\"${Identifier.idOf(this)}\" [color=blue]"
    }

    fun asGraphvizLinks(inputs: Collection<Any>): List<String> {
        return (weights zip inputs).map { (weight, input) ->
            "\"${Identifier.idOf(input)}\" -> \"${Identifier.idOf(this)}\" [label=\"$weight\"]"
        }
    }
}
