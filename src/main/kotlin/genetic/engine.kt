package genetic

import GeneticNeuralNetwork
import java.util.concurrent.locks.ReentrantLock
import kotlin.concurrent.withLock
import kotlin.math.roundToInt

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
    individuals: List<T>,
    generationNumber: Int,
    keepAlivePct: Float = 0.5f,
    keepBestsAlive: Int = 1,
    log: Boolean = false,
): List<T> {
    var population = mutableListOf<Individual<T, U, V>>()
    population.addAll(individuals)
//    val bests = mutableSetOf<Individual<T, U, V>>()
    repeat(generationNumber) { index ->
        // Sort population by fitness (best first)
        population = population.sortedBy(Individual<T, U, V>::fitness).toMutableList()
        // Retrieve the individuals with the best fitness
        val bests = population.subList(0, keepBestsAlive)
        population = population.subList(0, (population.size * keepAlivePct).roundToInt() - keepBestsAlive)
        population = population.flatMapTo(mutableListOf()) { individual -> (0 until (1 / keepAlivePct).roundToInt()).map { individual.clone() } }
        if (index < generationNumber - 1) {
            // Do not mutate the final result
            population = population.onEach { it.mutate() }
        }
        population.addAll(bests)
        if (log) {
            val best = population.minByOrNull { it.fitness() }
            if (best is GeneticNeuralNetwork) {
                println("$index, ${best.fitness()}, ${best.getSize()}")
            } else {
                if (log) println("$index, ${population.map(Individual<T, U, V>::fitness).minOrNull()}, ${population.map(Individual<T, U, V>::fitness).average()}")
            }
        }
    }
    @Suppress("UNCHECKED_CAST")
    return population as List<T>
}
