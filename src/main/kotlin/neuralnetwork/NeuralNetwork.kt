package neuralnetwork

class InputNode(var output: Float) : Inputable<Float> {
    override fun getOutput(): Float {
        return output
    }
}

class NeuralNetwork(
    private val inputNodes: List<InputNode> = listOf(),
    private val links: Map<Neuron, List<Inputable<Float>>>,
    private val neurons: List<Neuron>,
    private val outputNeurons: List<Neuron>
) {
    fun run(inputs: List<Float>): List<Float> {
        (inputNodes zip inputs).forEach { it.first.output = it.second }
        neurons.forEach {
            it.compute(links[it]?.map(Inputable<Float>::getOutput) ?: (0 until it.getExpectedInputSize()).map { 0f })
        }
        neurons.forEach(Neuron::update)
        return outputNeurons.map(Neuron::getOutput)
    }
}
