/**
 * Created by tpayton1 on 11/7/2014.
 */
(function (global) {
    var is_worker = !this.document;

    var script_path = is_worker ? null : (function () {
        // append random number and time to ID
        var id = (Math.random() + '' + (+new Date)).substring(2);
        document.write('<script id="wts' + id + '"></script>');
        return document.getElementById('wts' + id).
            previousSibling.src;
    })();

    function msgFromParent(e) {
        // event handler for parent -> worker messages
        switch (e.data.command) {
            case "Train":
                workerIntellect = new Intellect(e.data.options);
                workerIntellect.Weights = e.data.weights;
                workerIntellect.Biases = e.data.biases;
                var result = workerIntellect.Train(e.data.data, e.data.trainingOptions);
                self.postMessage({
                    command: "Trained",
                    result: result,
                    weights: workerIntellect.Weights,
                    biases: workerIntellect.Biases
                });
                self.close();
                break;
            case "Genetic":
                workerIntellect = new Intellect(e.data.options);
                var result = workerIntellect.Train(e.data.data, e.data.trainingOptions, function (result) {
                    self.postMessage({
                        command: "Trained",
                        result: result,
                        weights: workerIntellect.Weights,
                        biases: workerIntellect.Biases
                    });
                    self.close();
                });
                break;

            case "Stop":
                self.postMessage({
                    command: "Trained",
                    result: result,
                    weights: workerIntellect.Weights,
                    biases: workerIntellect.Biases
                });
                self.close();
                break;
            default:
                console.log("Worker received", e.data);
                break;
        }
    }

    function msgFromWorker(e) {
        // event handler for worker -> parent messages
        switch (e.data.command) {
            case "Trained":
                this.prototype._this.Weights = e.data.weights;
                this.prototype._this.Biases = e.data.biases;
                this.prototype._this.Trained = true;
                this.prototype.callback(e.data.result);
                break;
            case "TrainingCallback":
                this.prototype.trainingCallback(e.data.result);
                break;
            default:
                console.log("Message from Worker", e.data);

        }
    }

    function new_worker() {
        var w = new Worker(script_path);
        w.addEventListener('message', msgFromWorker, false);
        return w;
    }

    if (is_worker) {
        var workerIntellect; // Holds the network for worker threads
        global.addEventListener('message', msgFromParent, false);
    }
    // Main Intellect Library Starts Here.
    var Intellect = function (options) {
        var defaultOptions = {
            Layers: [2, 2, 1],

            // Possible Options:
            // "BackPropagate" = Gradient Descent
            // "Genetic" = Evolutionary Genetic Algorithm
            // or provide your own function
            TrainingMethod: "BackPropagate",

            // Possible Options:
            // "ElliotSigmoid" = .5*x/(1+|x|)+.5
            // "ElliotSymmetrical" = x/(1+|x|)
            // "Sigmoid" = 1/(1+e^-x)
            // "HyperTan" = tanh(x)
            // or provide your own function
            ActivationFunction: "ElliotSigmoid",
            ActivationFunctionDerivative: "Sigmoid",

            // Possible Options:
            // "MeanSquaredError"
            // "QuadradicCost"
            // "CrossEntropy"
            // or provide your own function
            ErrorFunction: "MeanSquaredError",

            // Disable WebWorkers (only used for training)
            SingleThreaded: false,

            // Create a hidden layer if one is not provided
            AutomaticHiddenLayer: true
        };

        if (!is_worker) {
            if (window == this) {
                // Prevent Cluttering the Window Scope;
                return new Intellect(options);
            }

        }


        var extend = function () {
            for (var i = 1; i < arguments.length; i++)
                for (var key in arguments[i])
                    if (arguments[i].hasOwnProperty(key))
                        arguments[0][key] = arguments[i][key];
            return arguments[0];
        };

        // Merge defaultOptions with any provided options
        this.options = extend({}, defaultOptions, options);
        this.SingleThreaded = this.options.SingleThreaded;


        this.Initialize(this.options.Layers);


    };

    Intellect.prototype = {

        // Helper Functions

        // returns a random number from a standard distribution (used for initializing Biases for optimal learning speed)
        // Quicker than Box-Muller transform, credit goes to Protonfish http://www.protonfish.com/random.html
        rand_normal: function () {
            return (Math.random() * 2 - 1) + (Math.random() * 2 - 1) + (Math.random() * 2 - 1);
        },


        // In place Fisher-Yates shuffle.
        shuffle: function (array) {
            var m = array.length, t, i;
            // While there remain elements to shuffle...
            while (m) {
                // Pick a remaining element...
                i = Math.floor(Math.random() * m--);
                // And swap it with the current element.
                t = array[m];
                array[m] = array[i];
                array[i] = t;
            }
            return array;
        },


        randWeights: function (size) {
            var weights = new Array(size);
            for (var i = 0; i < size; i++) {
                weights[i] = this.rand_normal() / Math.sqrt(size);
            }
            return weights;
        },

        randBias: function (size) {
            var biases = new Array(size);
            for (var i = 0; i < size; i++) {
                biases[i] = this.rand_normal();
            }
            return biases;
        },

        zeros: function (size) {
            var arr = new Array(size);
            for (var i = 0; i < size; i++) {
                arr[i] = 0.0;
            }
            return arr;
        },


        // Activation Functions http://jsperf.com/neural-net-activation-functions/2
        ActivationFunctions: {
            Sigmoid: function (x) {
                return 1.0 / (1.0 + Math.exp(-x));
            },
            HyperTan: function (x) {
                return Math.tanh(x);
            },
            ElliotSigmoid: function (x) {
                return (.5 * x / (1 + Math.abs(x))) + .5;
            },
            ElliotSymmetrical: function (x) {
                return x / (1 + Math.abs(x));
            }

        },

        // Partial Derivatives of Activation Functions for BackPropagation Training Method
        ActivationFunctionDerivatives: {
            Sigmoid: function (output, input) {
                return (1 - output) * output;
            },
            HyperTan: function (output, input) {
                return (1 - output) * (1 + output);
            },
            ElliotSigmoid: function (output, input) {
                return 1 / ((1 + Math.abs(input)) * (1 + Math.abs(input)));
            },
            ElliotSymmetrical: function (output, input) {
                return (.5 / (1 + Math.abs(input) * (1 + Math.abs(input))));
            }
        },

        ErrorFunctions: {
            MeanSquaredError: {
                error: function (expected, actual) {
                    var sum = 0;
                    for (var i = 0; i < expected.length; i++) {
                        var error = actual[i] - expected[i]
                        sum += (error * error);
                    }
                    return sum / expected.length;
                },
                delta: function (expected, actual, gradient) {
                    return (expected - actual);
                }
            },
            Quadratic: {
                error: function (expected, actual) {
                    // mean squared error
                    var sum = 0;
                    for (var i = 0; i < expected.length; i++) {
                        var error = actual[i] - expected[i]
                        sum += (error * error)
                    }
                    return .5 * sum;
                },
                delta: function (expected, actual, gradient) {
                    return (expected - actual) * gradient;
                }

            },
            CrossEntropy: {
                error: function (expected, actual) {
                    var sum = 0;
                    for (var i = 0; i < expected.length; i++) {
                        var error = -expected[i] * Math.log(actual[i]) - (1 - expected[i]) * Math.log(1 - actual[i]);
                        sum += isFinite(error) ? error : 0;
                    }
                    return sum / expected.length;
                },
                delta: function (expected, actual) {
                    return (expected - actual);
                }
            }

        },

        // Training Methods
        // Some require the Activation Functions so this is initialized after they are pointing correctly.
        TrainingMethods: {
            // BackPropagate
            // This updates the weights of the NN using the back propagation technique
            // Takes an array of expected outputs, eta: amount to change, alpha:momentum based on previous weight delta
            BackPropagate: function (data, options, callback) {
                var epochs = options.epochs || 20000;
                var shuffle = options.shuffle || false;
                var errorThresh = options.errorThresh || 0.04;
                var log = options.log || false;
                var logPeriod = options.logPeriod || 500;
                options.learningRate = options.learningRate || .45;
                options.momentum = options.momentum || .3;
                var callbackPeriod = options.callbackPeriod || 0;
                var dataLength = data.length;

                // Create a WebWorker to handle the training
                if (!is_worker && !this.SingleThreaded) {
                    this.worker = new_worker();
                    var IntellectThis = this;
                    this.worker.prototype = {
                        _this: IntellectThis,
                        Result: false,
                        trainingCallback: options.callback,
                        callback: callback
                    };
                    options.callback = true;
                    this.worker.postMessage({
                        command: "Train",
                        options: this.options,
                        weights: this.Weights,
                        biases: this.Biases,
                        data: data,
                        trainingOptions: options
                    });
                }
                else {
                    var error = 9999;
                    var preverror = 99999999;
                    for (var epoch = 0; epoch < epochs && error > errorThresh; epoch++) {
                        var sum = 0;
                        if (shuffle) {
                            this.shuffle(data);
                        }
                        for (var n = 0; n < dataLength; n++) {
                            // Feed Forward
                            var actual = this.ComputeOutputs(data[n].input);
                            // Compute the Deltas & Errors for this batch
                            var expected = data[n].output;
                            sum += this.ErrorFunction.error(expected, actual, dataLength);
                            this.CalculateDeltas(expected);
                            // Update Weights and Biases
                            this.UpdateWeights(options);
                        }

                        error = sum;
                        if (error > preverror) {
                            options.learningRate *= .05;
                        }
                        preverror = error;

                        if (log && (epoch % logPeriod == 0)) {
                            console.log("epoch:", epoch, "training error:", error);
                        }

                        if (callbackPeriod && epoch % callbackPeriod == 0) {
                            if (is_worker) {
                                self.postMessage({
                                    command: "TrainingCallback",
                                    result: {epoch: epoch, error: error, learningRate: options.learningRate}
                                });
                            }
                            else {
                                options.callback({epoch: epoch, error: error})
                            }
                        }
                    }
                    var result = {
                        error: error,
                        epoch: epoch,
                        iterations: dataLength * epoch
                    };
                    callback(result);
                    return result;
                }
            },

            Genetic: function (data, options, callback) {
                // performs a single epoch
                var popSize = options.popSize || 30;
                var epochs = options.epochs || 12500;
                var log = options.log || false;
                var logPeriod = options.logPeriod || 500;
                this.options.learningRate = options.learningRate || .7;
                var trainingcallback = options.callback;
                var callbackPeriod = options.callbackPeriod || 15;
                var fitnessFunction = options.testFitnessFunction || this.testFitnessFunction;
                var grabNBest = options.grabNBest || 2;
                var percentageToMate = options.percentageToMate || .10;
                var matablePopulation = options.mateablePopulation || .25;
                this.options.crossOverRate = options.crossOverRate || .85;
                this.options.mutationRate = options.mutationRate || .50;

                // Create a WebWorker to handle the training
                if (!is_worker && !this.SingleThreaded) {
                    this.worker = new_worker();
                    var IntellectThis = this;
                    this.worker.prototype = {
                        _this: IntellectThis,
                        Result: false,
                        trainingCallback: options.callback,
                        callback: callback
                    };
                    options.callback = true;
                    this.worker.postMessage({
                        command: "Genetic",
                        options: this.options,
                        data: data,
                        trainingOptions: options,
                        path: script_path // used for spawning new workers
                    });
                }
                else { // Running SingleThreaded or as a Worker!

                    // Create the population for the genetic algorithm
                    var population = new Array(popSize);
                    for (var intellect = 0; intellect < popSize; intellect++) {
                        population[intellect] = new Intellect(this.options);
                    }

                    // Holds the indexes and finteses of the population, and the populations overal fitness statistics.
                    var fitnesses;
                    var generationInfo = {};
                    // Sorts Population in Descending Order based on Fitness
                    var FitnessSort = function (a, b) {
                        return b.Fitness - a.Fitness;
                    };
                    var CalculateFitnesses = function () {
                        var totalFitness = 0;
                        fitnesses.sort(FitnessSort);
                        for (var i = 0, popSize = fitnesses.length; i < popSize; i++) {
                            totalFitness += fitnesses[i].Fitness;
                        }
                        generationInfo = {
                            totalFitness: totalFitness,
                            Fittest: fitnesses[0].Fitness,
                            Worst: fitnesses[popSize - 1].Fitness,
                            Average: totalFitness / fitnesses.length
                        }
                    };

                    var GetChromoRoulette = function (mom) {
                        var mateable = mom ? percentageToMate : matablePopulation;
                        var theChosenOne = Math.random() * mateable * popSize
                        return Math.floor(theChosenOne);
                    };


                    var _this = this;
                    var epoch = 0;
                    while (epoch < epochs) {
                        // Perform a Genetic Algorithm Iteration!

                        fitnessFunction(data, population, function (f) {
                            fitnesses = f;

                            CalculateFitnesses();
                            var newPop = [];
                            // Add some eliteism by keeping the best
                            for (var i = 0; i < grabNBest; i++) {
                                newPop[i] = population[fitnesses[i].index];
                                //newPop[i].TrainingMethod = newPop[i].TrainingMethods.BackPropagate;
                                //newPop[i].Train(data);
                            }
                            while (newPop.length < population.length) {
                                var momIndex = fitnesses[GetChromoRoulette(true)].index;
                                var dadIndex = fitnesses[GetChromoRoulette()].index;

                                var mom = population[momIndex];
                                var dad = population[dadIndex];

                                var baby1, baby2;
                                if (momIndex == dadIndex) {
                                    baby1 = mom;
                                    baby2 = dad;
                                }
                                else {
                                    baby1 = new Intellect(_this.options);
                                    baby2 = new Intellect(_this.options);
                                    _this.Crossover(mom, dad, baby1, baby2);
                                }
                                // Add a little mutation
                                if (Math.random() < _this.options.mutationRate) {
                                    _this.Mutate(baby1);
                                }
                                if (Math.random() < _this.options.mutationRate) {
                                    _this.Mutate(baby2);
                                }

                                // Add the babies to the new population.
                                newPop.push(baby1);
                                newPop.push(baby2);


                            }
                            population = newPop;


                            if (log && (epoch % logPeriod == 0)) {
                                console.log("epoch:", epoch, generationInfo);
                            }

                            if (trainingcallback && epoch % callbackPeriod == 0) {
                                if (is_worker) {
                                    self.postMessage({
                                        command: "TrainingCallback",
                                        result: {epoch: epoch, generationInfo: generationInfo}
                                    });
                                }
                                else {
                                    options.callback({epoch: epoch, generationInfo: generationInfo})
                                }
                            }
                        });
                        epoch++;
                    }

                    _this.Weights = population[fitnesses[0].index].Weights;
                    _this.Biases = population[fitnesses[0].index].Biases;


                    // perform the genetic algorithm!
                    callback(generationInfo);
                    return generationInfo;

                }
            }

        },

        CalculateDeltas: function (tValues) {
            // Calculate Deltas, must be done backwards
            for (var layer = this.outputLayer; layer >= 0; layer--) {
                for (var node = 0; node < this.Layers[layer]; node++) {
                    var output = this.Outputs[layer][node];
                    var input = this.Sums[layer][node];
                    var error = 0;
                    if (layer == this.outputLayer) {
                        error = output - tValues[node];
                    }
                    else {
                        var deltas = this.Deltas[layer + 1];
                        for (var k = 0; k < deltas.length; k++) {
                            error += deltas[k] * this.Weights[layer + 1][k][node];
                        }
                    }
                    this.Deltas[layer][node] = error * this.ActivationFunctionDerivative(output, input);
                }
            }
        },
        UpdateWeights: function (options) {
            // Adjust Weights
            for (var layer = 1; layer <= this.outputLayer; layer++) {
                var incoming = this.Outputs[layer - 1];
                for (var node = 0, nodes = this.Layers[layer]; node < nodes; node++) {
                    var delta = this.Deltas[layer][node];
                    for (var k = 0, length = incoming.length; k < length; k++) {
                        var prevWeightDelta = (options.learningRate * delta * incoming[k]) + (options.momentum * this.prevWeightsDelta[layer][node][k]);
                        this.prevWeightsDelta[layer][node][k] = prevWeightDelta;
                        this.Weights[layer][node][k] -= prevWeightDelta;
                    }
                    this.Biases[layer][node] -= options.learningRate * delta;
                }
            }

        },

        testFitnessFunction: function (data, population, callback) {
            // Use the training data to determine fitness
            var errorFunction = population[0].ErrorFunction.error;
            var fitnesses = new Array(population.length);
            for (var i = 0; i < population.length; i++) {
                var fitness = 0;
                for (var j = 0, length = data.length; j < length; j++) {
                    fitness -= errorFunction(data[j].output, population[i].ComputeOutputs(data[j].input))

                }
                fitnesses[i] = {index: i, Fitness: fitness};
            }
            callback(fitnesses);
        },


        Crossover: function (mom, dad, baby1, baby2) {
            if ((Math.random() > this.options.crossOverRate)) {
                baby1 = mom;
                baby2 = dad;

            }
            else {
                // determine crossover point
                var cpW = Math.floor(Math.random() * this.numWeights);
                var cpB = Math.floor(Math.random() * this.numNeurons);
                var w = 0;
                var b = 0;

                // SUPER SLOW -- Would be better if Weights and Biases were flattened!
                for (var layer = 1; layer < mom.Layers; layer++) {
                    for (var neuron = 0; neuron < mom.Layers[layer].length; neuron++) {
                        baby1.Biases[layer][neuron] = (cpB <= b) ? mom.Biases[layer][neuron] : dad.Biases[layer][neuron];
                        baby2.Biases[layer][neuron] = (cpB <= b++) ? dad.Weights[layer][neuron] : mom.Weights[layer][neuron];
                        for (var weight = 0, weights = mom.Weights[layer][neuron].length; weight < weights; weight++) {
                            baby1.Biases[layer][neuron][weight] = (cpW <= w) ? mom.Weights[layer][neuron][weight] : dad.Weights[layer][neuron][weight];
                            baby2.Biases[layer][neuron][weight] = (cpW <= w++) ? dad.Weights[layer][neuron][weight] : mom.Weights[layer][neuron][weight];
                        }
                    }
                }

            }
        },

        Mutate: function (intellect) {
            for (var layer = 0; layer < intellect.Layers; layer++) {
                for (var neuron = 0; neuron < intellect.Layers[layer]; neuron++) {
                    if (Math.random() < this.options.mutationRate) {
                        intellect.Biases[layer][neuron] += (tthis.options.learningRate * Math.random() - this.options.learningRate)
                    }
                    for (var k = 0; k < intellect.Weights[layer][neuron]; k++) {
                        if (Math.random() < this.options.mutationRate) {
                            intellect.Weights[layer][neuron][k] += (Math.random() * this.options.learningRate - this.options.learningRate)
                        }
                    }
                }
            }
        },

        Trained: true,

        stop: function () {
            if (this.worker) {
                this.worker.postMessage({command: "Stop"});
            }
        },

        Train: function (data, options, callback) {
            options = options || {};
            options.callback = options.callback || function () {
            };
            callback = callback || function () {
            };
            this.Trained = false; // Used to know when a worker thread is done training
            var result = this.TrainingMethod(data, options, callback);
            return result;
        },


        // Functions
        ComputeOutputs: function (input) {

            this.Outputs[0] = input;
            // Compute the Outputs of each Layer
            for (var layer = 1; layer <= this.outputLayer; layer++) {
                for (var node = 0, nodes = this.Layers[layer]; node < nodes; node++) {
                    var weights = this.Weights[layer][node];
                    var sum = this.Biases[layer][node];
                    for (var k = 0, length = weights.length; k < length; k++) {
                        sum += weights[k] * input[k];
                    }
                    this.Sums[layer][node] = sum;
                    this.Outputs[layer][node] = this.ActivationFunction(sum);
                }
                var output = input = this.Outputs[layer];
            }

            return output;

        },


        Initialize: function (NetworkConfiguration) {
            if (this.options.Layers.length == 2 && this.options.AutomaticHiddenLayer) {
                // Add a hidden layer using the rule that it should 2/3 thei size of the input plus the size of the output
                Math.floor((2 / 3) * this.options.Layers[0] + this.options.Layers[1])

            }
            // Check to see if Workers are supported
            if (!is_worker && !this.SingleThreaded) {
                if (!window.Worker) {
                    this.SingleThreaded = true;
                }
                else {
                    this.cores = navigator.hardwareConcurrency;
                    this.cores = this.cores || 2; // can be improved with a polyfill to estimate
                }
            }
            // Check to see if a function was provided, if not update the string to the actual function
            if (typeof(this.options.ActivationFunction) !== 'function') {
                this.ActivationFunction = this.ActivationFunctions[this.options.ActivationFunction];
            }
            else {
                this.SingleThreaded = true;
                this.ActivationFunction = this.options.ActivationFunction;
            }
            this.options.ActivationFunctionDerivative = this.options.ActivationFunctionDerivative || this.options.ActivationFunction;
            if (typeof(this.options.ActivationFunctionDerivative) !== 'function') {
                this.ActivationFunctionDerivative = this.ActivationFunctionDerivatives[this.options.ActivationFunctionDerivative];
            }
            else {
                this.SingleThreaded = true;
                this.ActivationFunctionDerivative = this.options.ActivationFunctionDerivative;
            }


            if (typeof(this.options.TrainingMethod) !== 'function') {
                this.TrainingMethod = this.TrainingMethods[this.options.TrainingMethod];
            }
            else {
                this.TrainingMethod = this.options.TrainingMethod;
            }

            if (typeof(this.options.ErrorFunction) !== 'function') {
                this.ErrorFunction = this.ErrorFunctions[this.options.ErrorFunction];
            }
            else {
                this.SingleThreaded = true;
                this.ErrorFunction = this.options.ErrorFunction;
            }


            for (var layer = 1; layer <= this.outputLayer; layer++) {
                this.numNeurons += this.Layers[layer];
                this.numWeights += this.Layers[layer] * this.Layers[layer - 1];
            }

            // Setup Network Matrices
            // Sums, Biases, Weights, and Outputs, Deltas, Deltas
            this.Fitness = 0;
            this.Layers = NetworkConfiguration;
            this.outputLayer = this.Layers.length - 1;
            this.Sums = []; // input to node output function
            this.Biases = [];
            this.Weights = [];
            this.Outputs = [];
            this.prevWeightsDelta = [];
            this.prevBiasesDelta = [];
            this.Deltas = [];

            for (var layer = 0; layer <= this.outputLayer; layer++) {
                var size = this.Layers[layer];
                this.prevBiasesDelta[layer] = this.zeros(size);
                this.Outputs[layer] = this.zeros(size);
                this.Deltas[layer] = this.zeros(size);
                this.Sums[layer] = this.zeros(size);

                if (layer > 0) {
                    this.Biases[layer] = this.randBias(size);
                    this.Weights[layer] = new Array(size);
                    this.prevWeightsDelta[layer] = new Array(size);

                    for (var node = 0; node < size; node++) {
                        var prevSize = this.Layers[layer - 1];
                        this.Weights[layer][node] = this.randWeights(prevSize);
                        this.prevWeightsDelta[layer][node] = this.zeros(prevSize);
                    }
                }
            }
        },

        saveNetwork: function () {
            var data = {options: this.options, weights: this.Weights, biases: this.Biases};
            var fileName = "Network - " + new Date() + ".json";
            var a = document.createElement("a");
            document.body.appendChild(a);
            a.style = "display: none";

            var json = JSON.stringify(data),
                blob = new Blob([json], {type: "octet/stream"}),
                url = window.URL.createObjectURL(blob);
            a.href = url;
            a.download = fileName;
            a.click();
            window.URL.revokeObjectURL(url);

        },

        loadNetwork: function (data) {
            data = JSON.parse(data);
            this.options = data.options;
            this.Initialize(this.options.Layers);
            this.Biases = data.biases;
            this.Weights = data.weights;
        }


    }; // end Intellect prototype declaration

    if (!is_worker) {
        window.Intellect = Intellect;
    }
})(this);

