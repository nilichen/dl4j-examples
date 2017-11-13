package org.deeplearning4j.examples.Caltech101;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.FilenameUtils;
import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.api.util.ArchiveUtils;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.VGG16ImagePreProcessor;
import org.slf4j.Logger;

import java.io.File;
import java.io.IOException;
import java.net.URL;
import java.util.Random;

/**
 * Automatically downloads the dataset from
 * http://download.tensorflow.org/example_images/flower_photos.tgz
 * and untar's it to the users home directory
 * @author susaneraly on 3/9/17.
 */
public class Caltech101Iterator {
    private static final Logger log = org.slf4j.LoggerFactory.getLogger(Caltech101Iterator.class);

    private static final String DATA_DIR = FilenameUtils.concat(System.getProperty("user.dir"), "src/main/resources/");
    private static final String DATA_URL = "http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz";
    private static final String CALTECH101_DIR = DATA_DIR + "/101_ObjectCategories";

    private static final String [] allowedExtensions = BaseImageLoader.ALLOWED_FORMATS;
    private static final Random rng  = new Random(13);

    private static int height = 224;
    private static int width = 224;
    private static int channels = 3;
    private static int numExamples = 9146;
    private static int numLabels = 101;

    private static ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
    private static InputSplit trainData,testData;
    private static int batchSize;

    public static DataSetIterator trainIterator() throws IOException {
        return makeIterator(trainData);

    }

    public static DataSetIterator testIterator() throws IOException {
        return makeIterator(testData);

    }

    public static void setup(int batchSizeArg, int trainPerc) throws IOException {
        try {
            downloadAndUntar();
        } catch (IOException e) {
            e.printStackTrace();
            log.error("IOException : ", e);
        }

        batchSize = batchSizeArg;
        File parentDir = new File(CALTECH101_DIR);
        FileSplit filesInDir = new FileSplit(parentDir, allowedExtensions, rng);
        BalancedPathFilter pathFilter = new BalancedPathFilter(rng, labelMaker, numExamples, numLabels, batchSize);
        if (trainPerc >= 100) {
            throw new IllegalArgumentException("Percentage of data set aside for training has to be less than 100%. Test percentage = 100 - training percentage, has to be greater than 0");
        }
        InputSplit[] filesInDirSplit = filesInDir.sample(pathFilter, trainPerc, 100-trainPerc);
        trainData = filesInDirSplit[0];
        testData = filesInDirSplit[1];
    }

    private static DataSetIterator makeIterator(InputSplit split) throws IOException {
        ImageRecordReader recordReader = new ImageRecordReader(height,width,channels,labelMaker);
        recordReader.initialize(split);
        DataSetIterator iter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numLabels);
        iter.setPreProcessor( new VGG16ImagePreProcessor());
        return iter;
    }

    private static void downloadAndUntar() throws IOException {
         File rootFile = new File(DATA_DIR);
         if (!rootFile.exists()) {
             rootFile.mkdir();
         }
         File tarFile = new File(DATA_DIR, "101_ObjectCategories.tar.gz");
         if (!tarFile.isFile()) {
             log.info("Downloading the caltech101 dataset from "+DATA_URL+ "...");
             FileUtils.copyURLToFile(
                     new URL(DATA_URL),
                     tarFile);
         }
         ArchiveUtils.unzipFileTo(tarFile.getAbsolutePath(), rootFile.getAbsolutePath());
    }
}
