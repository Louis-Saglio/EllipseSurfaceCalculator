package genetic

interface Individual<G, T : Individual<G, T>> {
    val innerInstance: G
    fun fitness(): Float
    fun clone(mutationProbability: Float): T
}

fun <G, T : Individual<G, T>> evolve(
    individuals: List<Individual<G, T>>,
    generationNumber: Int,
    mutationProbability: Float,
    log: Boolean = false
): List<Individual<G, T>> {
    var population = individuals
    for (generationIndex in 0 until generationNumber) {
        population = population
            .sortedBy(Individual<G, T>::fitness)
            .subList(population.size / 2, population.size)
            .flatMap { listOf(it.clone(mutationProbability), it.clone(mutationProbability)) }
        if (log) println(population.map(Individual<G, T>::fitness).average())
    }
    return population
}
