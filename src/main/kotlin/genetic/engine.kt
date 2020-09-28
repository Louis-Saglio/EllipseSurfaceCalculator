package genetic

interface Individual<T : Individual<T>> {
    fun fitness(): Float
    fun clone(mutationProbability: Float): T
}

fun <T : Individual<T>> evolve(
    individuals: List<Individual<T>>,
    generationNumber: Int,
    mutationProbability: Float,
    log: Boolean = false
): List<Individual<T>> {
    var population = individuals
    for (generationIndex in 0 until generationNumber) {
        population = population
            .sortedBy(Individual<T>::fitness)
            .subList(population.size / 2, population.size)
            .flatMap { listOf(it.clone(mutationProbability), it.clone(mutationProbability)) }
        if (log) println(population.map(Individual<T>::fitness).average())
    }
    return population
}
