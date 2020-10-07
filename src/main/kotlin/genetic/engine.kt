package genetic

import java.util.concurrent.locks.ReentrantLock
import kotlin.concurrent.withLock

abstract class Problem<InputType, OutputType> {
    abstract fun getInput(): InputType
    abstract fun getOutput(): OutputType
    abstract fun computeError(predictions: OutputType, expectedOutput: OutputType): Float
    fun evaluate(precision: Int, function: (InputType) -> OutputType): Float {
        val results = mutableSetOf<Float>()
        repeat(precision) {
            val predictions = function(getInput())
            results.add(computeError(predictions, getOutput()))
        }
        return results.average().toFloat()
    }
}

abstract class Individual<T : Individual<T, InputType, OutputType>, InputType, OutputType>(
    private val problem: Problem<InputType, OutputType>
) {
    private var fitness: Float? = null
    private val fitnessLock = ReentrantLock()

    abstract fun clone(): T
    abstract fun mutate()
    abstract fun compute(input: InputType): OutputType

    fun fitness(): Float {
        fitnessLock.withLock {
            if (fitness == null) {
                fitness = problem.evaluate(10) { compute(it) }
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
    for (generationIndex in 0 until generationNumber) {
        population = population
            .onEach { it.mutate() }
            .sortedBy(Individual<T, U, V>::fitness)
            .subList(0, population.size / 2)
            .flatMapTo(mutableListOf()) { listOf(it.clone(), it.clone()) }
        if (log) println(population.map(Individual<T, U, V>::fitness).average())
    }
    return population
}
