package org.deeplearning4j.examples.Caltech101;

import org.apache.commons.io.FilenameUtils;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.VGG16;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Animal Classification
 *
 * Example classification of photos from 4 different animals (bear, duck, deer, turtle).
 *
 * References:
 *  - U.S. Fish and Wildlife Service (animal sample dataset): http://digitalmedia.fws.gov/cdm/
 *  - Tiny ImageNet Classification with CNN: http://cs231n.stanford.edu/reports/2015/pdfs/leonyao_final.pdf
 *
 * CHALLENGE: Current setup gets low score results. Can you improve the scores? Some approaches:
 *  - Add additional images to the dataset
 *  - Apply more transforms to dataset
 *  - Increase epochs
 *  - Try different model configurations
 *  - Tune by adjusting learning rate, updaters, activation & loss functions, regularization, ...
 */

public class Caltech101Classification {
    private static final Logger log = LoggerFactory.getLogger(Caltech101Classification.class);

    private static int numLabels = 101;
    private static int batchSize = 64;
    private static final String featureExtractionLayer = "fc2";

    private static long seed = 42;
    private static int listenerFreq = 1;
    private static int iterations = 1;
    private static int epochs = 50;
    private static int trainPerc = 80;
    private static boolean save = true;


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

       // network.setListeners(new ScoreIterationListener(listenerFreq));
//        UIServer uiServer = UIServer.getInstance();
//        StatsStorage statsStorage = new InMemoryStatsStorage();
//        uiServer.attach(statsStorage);
//        vgg16Transfer.setListeners(new StatsListener( statsStorage),new ScoreIterationListener(iterations));

        log.info("Train model....");
        Evaluation eval;
        int iter = 0;
        while(trainIter.hasNext()) {
            vgg16Transfer.fit(trainIter.next());
            if (iter % 40 == 0) {
                log.info("Evaluate model at iter "+iter +" ....");
                eval = vgg16Transfer.evaluate(testIter);
                log.info(eval.stats());
                testIter.reset();
            }
            iter++;
        }
        log.info("Model build complete");

        if (save) {
            log.info("Save model....");
            String basePath = FilenameUtils.concat(System.getProperty("user.dir"), "src/main/resources/");
            ModelSerializer.writeModel(vgg16Transfer, basePath + "caltech101_vgg16Transfer.zip", false);
        }
        log.info("***************Caltech101 finished********************");
    }

    public static void main(String[] args) throws Exception {
        new Caltech101Classification().run(args);
    }

}
