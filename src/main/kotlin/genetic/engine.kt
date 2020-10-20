package genetic

import java.util.concurrent.locks.ReentrantLock
import kotlin.concurrent.withLock

abstract class Problem<InputType, OutputType> {
    // todo : should merge getInput/Output
    abstract fun getInput(): InputType
    abstract fun getOutput(): OutputType
    abstract fun computeError(predictions: OutputType, expectedOutputs: OutputType): Float
    fun computeAverageError(precision: Int, predict: (InputType) -> OutputType): Float {
        val results = mutableSetOf<Float>()
        repeat(precision) {
            val predictions = predict(getInput())
            results.add(computeError(predictions, getOutput()))
        }
        return results.average().toFloat()
    }
    abstract fun toString(input: InputType, result: OutputType): String
}

abstract class Individual<T : Individual<T, InputType, OutputType>, InputType, OutputType>(
    private val problem: Problem<InputType, OutputType>
) {
    private var fitness: Float? = null
    private val fitnessLock = ReentrantLock()

    abstract fun clone(): T
    abstract fun mutate()
    abstract fun compute(input: InputType): OutputType
    abstract fun showOff()

    open fun fitness(): Float {
        fitnessLock.withLock {
            if (fitness == null) {
                fitness = problem.computeAverageError(4, this::compute)
            }
            return fitness!!
        }
    }
}

fun <T : Individual<T, U, V>, U, V> evolve(
    individuals: List<Individual<T, U, V>>,
    generationNumber: Int,
    log: Boolean = false,
): List<T> {
    var population = mutableListOf<T>()
    @Suppress("UNCHECKED_CAST")
    population.addAll(individuals as Collection<T>)
    repeat(generationNumber) {
        population = population
                .onEach { it.mutate() }
                .sortedBy(Individual<T, U, V>::fitness)
                .subList(0, population.size / 2)
                .flatMapTo(mutableListOf()) { listOf(it.clone(), it.clone()) }
        if (log) println(population.map(Individual<T, U, V>::fitness).minOrNull())
    }
    return population
}
