package genetic

interface Individual<G, T : Individual<G, T>> {
    val innerInstance: G
    fun fitness(): Float
    fun clone(): T
    fun mutate()
}

fun <G, T : Individual<G, T>> evolve(
    individuals: List<Individual<G, T>>,
    generationNumber: Int,
    log: Boolean = false
): List<T> {
    var population = mutableListOf<T>()
    population.addAll(individuals as Collection<T>)
    for (generationIndex in 0 until generationNumber) {
        population = population
            .onEach { it.mutate() }
            .sortedBy(Individual<G, T>::fitness)
            .subList(population.size / 2, population.size)
            .flatMapTo(mutableListOf()) { listOf(it.clone(), it.clone()) }
        if (log) println(population.map(Individual<G, T>::fitness).average())
    }
    return population
}
