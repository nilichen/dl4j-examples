package org.deeplearning4j.examples.Caltech101;

import org.apache.commons.io.FilenameUtils;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.VGG16;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


public class Caltech101Classification {
    private static final Logger log = LoggerFactory.getLogger(Caltech101Classification.class);

    private static final int numLabels = 101;
    private static final int batchSize = 16;
    private static final String featureExtractionLayer = "fc2";

    private static final long seed = 42;
    private static final int iterations = 1;
    private static final int epochs = 10;
    private static final int trainPerc = 85;
    private static final boolean save = true;


    public void run(String[] args) throws Exception {


        log.info("Load data....");
        Caltech101Iterator.setup(batchSize,trainPerc);
        DataSetIterator trainIter = Caltech101Iterator.trainIterator();
        DataSetIterator testIter = Caltech101Iterator.testIterator();


        log.info("Build model....");
        ZooModel zooModel = new VGG16();
        ComputationGraph vgg16 = (ComputationGraph) zooModel.initPretrained();
        log.info(vgg16.summary());

        FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder()
            .learningRate(5e-5)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .updater(Updater.NESTEROVS)
            .seed(seed)
            .build();

        //Construct a new model with the intended architecture and print summary
        ComputationGraph vgg16Transfer = new TransferLearning.GraphBuilder(vgg16)
            .setWorkspaceMode(WorkspaceMode.SEPARATE)
            .fineTuneConfiguration(fineTuneConf)
            .setFeatureExtractor(featureExtractionLayer) //the specified layer and below are "frozen"
            .removeVertexKeepConnections("predictions") //replace the functionality of the final vertex
            .addLayer("predictions",
                new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                    .nIn(4096).nOut(numLabels)
                    .weightInit(WeightInit.DISTRIBUTION)
                    .dist(new NormalDistribution(0,0.2*(2.0/(4096+numLabels)))) //This weight init dist gave better results than Xavier
                    .activation(Activation.SOFTMAX).build(),
                "fc2")
            .build();
        log.info(vgg16Transfer.summary());


        StatsStorage statsStorage = new InMemoryStatsStorage();
        vgg16Transfer.setListeners(new StatsListener( statsStorage),new ScoreIterationListener(iterations));

        log.info("Train model....");
        for( int i=0; i<epochs; i++ ) {
            vgg16Transfer.fit(trainIter);
            log.info("*** Completed epoch {} ***", i);

            log.info("Evaluate model....");
            Evaluation eval = vgg16Transfer.evaluate(testIter);
            log.info(eval.stats());
            testIter.reset();
        }
        log.info("Model build complete");

        if (save) {
            log.info("Save model....");
            String basePath = FilenameUtils.concat(System.getProperty("user.dir"), "dl4j-examples/src/main/resources/");
            ModelSerializer.writeModel(vgg16Transfer, basePath + "caltech101_vgg16Transfer.zip", false);
        }
        log.info("***************Caltech101 finished********************");
    }

    public static void main(String[] args) throws Exception {
        new Caltech101Classification().run(args);
    }

}
