package neuralnetwork

import java.lang.RuntimeException
import java.util.concurrent.locks.ReentrantLock
import kotlin.concurrent.withLock

interface PreSettable {
    fun preSet(field: String, value: Any)
    fun update()
}

open class PreSetter : PreSettable {
    private val data: HashMap<String, Any> = hashMapOf()
    private lateinit var kObject: Any

    fun setNeuron(kObject: Any) {
        this.kObject = kObject
    }

    override fun preSet(field: String, value: Any) {
        data[field] = value
    }

    override fun update() {
        for ((fieldName, fieldValue) in data) {
            kObject.javaClass.getDeclaredField(fieldName).set(kObject, fieldValue)
            data.remove(fieldName)
        }
    }
}

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
}
