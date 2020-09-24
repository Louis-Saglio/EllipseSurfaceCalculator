package genetic

interface Individual {
    fun fitness(): Float
    fun clone(mutationProbability: Float): Individual
}

fun evolve(individuals: List<Individual>, generationNumber: Int, mutationProbability: Float, log: Boolean = false): List<Individual> {
    var population = individuals
    for (generationIndex in 0 until generationNumber) {
        population = population
            .sortedBy(Individual::fitness)
            .subList(population.size / 2, population.size)
            .flatMap { listOf(it.clone(mutationProbability), it.clone(mutationProbability)) }
        if (log) println(population.map(Individual::fitness).average())
    }
    return population
}
