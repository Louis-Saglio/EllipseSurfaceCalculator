package genetic

interface Individual {
    fun mutate()
    fun fitness(): Float
    fun clone(): Individual
}

fun evolve(individuals: List<Individual>, generationNumber: Int, log: Boolean = false): List<Individual> {
    var population = individuals
    for (generationIndex in 0 until generationNumber) {
        population = population
            .onEach(Individual::mutate)
            .sortedBy(Individual::fitness)
            .subList(population.size / 2, population.size)
            .flatMap { listOf(it.clone(), it.clone()) }
        if (log) println(population.map(Individual::fitness).average())
    }
    return population
}
