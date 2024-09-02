class NeuralNetwork {
    constructor(inputSize, hiddenSize, outputSize, learningRate, activationFunction = 'sigmoid', goal = 'classification') {
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        this.outputSize = outputSize;
        this.learningRate = learningRate;
        this.activationFunction = activationFunction;
        this.goal = goal;

        this.initializeWeightsAndBiases();
    }

    initializeWeightsAndBiases() {
        this.weightsInputHidden = this.initializeWeights(this.inputSize, this.hiddenSize);
        this.weightsHiddenOutput = this.initializeWeights(this.hiddenSize, this.outputSize);
        this.biasHidden = this.initializeBiases(this.hiddenSize);
        this.biasOutput = this.initializeBiases(this.outputSize);
    }

    initializeWeights(inputSize, outputSize) {
        return new Array(inputSize).fill(0).map(() => new Array(outputSize).fill(0).map(() => Math.random() * 0.01 - 0.005));
    }

    initializeBiases(size) {
        return new Array(size).fill(0).map(() => Math.random() * 0.01 - 0.005);
    }

    activation(x) {
        switch (this.activationFunction) {
            case 'relu':
                return x > 0 ? x : 0;
            case 'tanh':
                return Math.tanh(x);
            case 'sigmoid':
            default:
                x = Math.max(Math.min(x, 30), -30);
                return 1 / (1 + Math.exp(-x));
        }
    }

    activationDerivative(x) {
        switch (this.activationFunction) {
            case 'relu':
                return x > 0 ? 1 : 0;
            case 'tanh':
                return 1 - Math.pow(Math.tanh(x), 2);
            case 'sigmoid':
            default:
                x = Math.max(Math.min(x, 30), -30);
                const sigmoid = 1 / (1 + Math.exp(-x));
                return sigmoid * (1 - sigmoid);
        }
    }

    normalize(dataset) {
        const normalizedData = dataset.map(item => {
            const inputs = item.inputs.map(value => value / 100);
            return { inputs, label: item.label };
        });
        return normalizedData;
    }

    train(trainData, epochs) {
        for (let epoch = 0; epoch < epochs; epoch++) {
            for (const { inputs, label } of trainData) {
                if (inputs.includes(NaN) || isNaN(label)) {
                    continue;
                }

                const hiddenLayerInput = this.dotProduct(inputs, this.weightsInputHidden).map((sum, index) => sum + this.biasHidden[index]);
                const hiddenLayerOutput = hiddenLayerInput.map(this.activation.bind(this));

                const outputLayerInput = this.dotProduct(hiddenLayerOutput, this.weightsHiddenOutput).map((sum, index) => sum + this.biasOutput[index]);
                const output = this.activation(outputLayerInput[0]);

                if (isNaN(output) || !isFinite(output)) {
                    continue;
                }

                const outputError = label - output;
                const outputDelta = outputError * this.activationDerivative(outputLayerInput[0]);

                const hiddenError = this.weightsHiddenOutput.map(weight => weight[0] * outputDelta);
                const hiddenDelta = hiddenError.map((error, index) => error * this.activationDerivative(hiddenLayerInput[index]));

                if (hiddenDelta.includes(NaN) || hiddenDelta.includes(Infinity) || isNaN(outputDelta) || !isFinite(outputDelta)) {
                    continue;
                }

                const clipGradient = (value, threshold = 1) => Math.max(Math.min(value, threshold), -threshold);

                this.weightsHiddenOutput = this.updateWeights(this.weightsHiddenOutput, hiddenLayerOutput, [outputDelta], clipGradient);
                this.biasOutput = this.biasOutput.map(bias => bias + clipGradient(this.learningRate * outputDelta));

                this.weightsInputHidden = this.updateWeights(this.weightsInputHidden, inputs, hiddenDelta, clipGradient);
                this.biasHidden = this.biasHidden.map((bias, index) => bias + clipGradient(this.learningRate * hiddenDelta[index]));
            }
        }
    }

    updateWeights(weights, inputs, deltas, clipFunction) {
        return weights.map((weights, i) =>
            weights.map((weight, j) => {
                const updateValue = clipFunction(this.learningRate * deltas[j] * inputs[i]);
                if (isNaN(updateValue) || !isFinite(updateValue)) {
                    return weight;
                }
                return weight + updateValue;
            })
        );
    }

    dotProduct(inputs, weights) {
        return weights[0].map((_, colIndex) => inputs.reduce((sum, input, rowIndex) => sum + input * weights[rowIndex][colIndex], 0));
    }

    predict(inputs, threshold = 0.5) {
        const hiddenLayerInput = this.dotProduct(inputs, this.weightsInputHidden).map((sum, index) => sum + this.biasHidden[index]);
        const hiddenLayerOutput = hiddenLayerInput.map(this.activation.bind(this));

        const outputLayerInput = this.dotProduct(hiddenLayerOutput, this.weightsHiddenOutput).map((sum, index) => sum + this.biasOutput[index]);
        const output = this.activation(outputLayerInput[0]);

        if (this.goal === 'classification') {
            return output >= threshold ? 1 : 0;
        } else {
            return output;
        }
    }
}

function evaluateModel(model, data) {
    let correct = 0;
    data.forEach(({ inputs, label }) => {
        const prediction = model.predict(inputs.map(val => val / 100));
        if (prediction === label) {
            correct++;
        }
    });
    console.log(`Accuracy: ${(correct / data.length) * 100}%`);
}

function confusionMatrix(model, data) {
    let TP = 0, TN = 0, FP = 0, FN = 0;
    data.forEach(({ inputs, label }) => {
        const prediction = model.predict(inputs.map(val => val / 100));
        if (prediction === 1 && label === 1) TP++;
        if (prediction === 0 && label === 0) TN++;
        if (prediction === 1 && label === 0) FP++;
        if (prediction === 0 && label === 1) FN++;
    });
    console.log(`Confusion Matrix:
    TP: ${TP}, FP: ${FP}
    FN: ${FN}, TN: ${TN}`);
}

let neuralNetwork;
let loadedDataset;
let isModelTrained = false;

function handleFileUpload(event) {
    const file = event.target.files[0];
    const reader = new FileReader();
    reader.onload = function(e) {
        const content = e.target.result;
        const extension = file.name.split('.').pop().toLowerCase();
        let dataset;

        if (extension === 'json') {
            try {
                dataset = JSON.parse(content).map(item => ({
                    inputs: [item.math, item.science, item.english],
                    label: item.label
                }));
            } catch (error) {
                alert('Invalid JSON format.');
                return;
            }
        } else if (extension === 'csv' || extension === 'txt') {
            dataset = parseCSV(content);
        } else {
            alert('Unsupported file format.');
            return;
        }

        loadedDataset = dataset;

        if (!neuralNetwork) {
            neuralNetwork = new NeuralNetwork(3, 4, 1, 0.01);
        }

        neuralNetwork.dataset = neuralNetwork.normalize(dataset);
    };
    reader.readAsText(file);
}

function parseCSV(content) {
    const lines = content.split('\n');
    return lines.map(line => {
        const [math, science, english, label] = line.split(',').map(Number);
        return { inputs: [math, science, english], label };
    });
}

function trainModel() {
    const learningRate = parseFloat(document.getElementById('learning-rate').value);
    const epochs = parseInt(document.getElementById('epochs').value);
    const goal = document.getElementById('goal').value;
    const activationFunction = document.getElementById('activation-function').value;
    const hiddenNeurons = parseInt(document.getElementById('hidden-neurons').value);

    if (!loadedDataset) {
        alert('Please upload a dataset first.');
        return;
    }

    neuralNetwork = new NeuralNetwork(3, hiddenNeurons, 1, learningRate, activationFunction, goal);

    neuralNetwork.dataset = neuralNetwork.normalize(loadedDataset);
    neuralNetwork.trainData = neuralNetwork.dataset;

    neuralNetwork.train(neuralNetwork.trainData, epochs);

    isModelTrained = true;
    document.getElementById('performance').innerText = 'Model trained successfully. You can test data now.';
    evaluateModel(neuralNetwork, neuralNetwork.trainData);
    confusionMatrix(neuralNetwork, neuralNetwork.trainData);
}

function testModel() {
    if (!isModelTrained) {
        alert('Please train the model first.');
        return;
    }

    const math = parseFloat(document.getElementById('test-math').value) / 100;
    const science = parseFloat(document.getElementById('test-science').value) / 100;
    const english = parseFloat(document.getElementById('test-english').value) / 100;

    const normalizedInputs = [math, science, english];

    const result = neuralNetwork.predict(normalizedInputs);

    if (neuralNetwork.goal === 'classification') {
        document.getElementById('result').innerText = result === 1 ? 'Pass' : 'Fail';
    } else {
        document.getElementById('result').innerText = `Predicted Score: ${result}`;
    }
    // Evaluate and print the accuracy after the result
    const correct = neuralNetwork.trainData.filter(({ inputs, label }) => {
        const prediction = neuralNetwork.predict(inputs);
        return prediction === label;
    }).length;

    const accuracy = (correct / neuralNetwork.trainData.length) * 100;
    document.getElementById('accuracy').innerText = `Accuracy: ${accuracy.toFixed(2)}%`;
}

function clearForm() {
    document.getElementById('learning-rate').value = '';
    document.getElementById('epochs').value = '';
    document.getElementById('goal').value = 'classification';
    document.getElementById('activation-function').value = 'sigmoid';
    document.getElementById('hidden-neurons').value = '';
    document.getElementById('test-math').value = '';
    document.getElementById('test-science').value = '';
    document.getElementById('test-english').value = '';
    document.getElementById('file-upload').value = '';
    document.getElementById('performance').innerText = '';
    document.getElementById('result').innerText = '';
    document.getElementById('accuracy').innerText = '';
    isModelTrained = false;
}

document.getElementById('file-upload').addEventListener('change', handleFileUpload);
