
import org.apache.log4j.BasicConfigurator;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import java.io.File;
import java.io.IOException;
import java.util.Scanner;



public class Classification {
    public static void main(String[] args) throws InterruptedException {
        BasicConfigurator.configure();
        initialize();
}

    public static void initialize() throws InterruptedException {
        boolean running = true;
        while (running) {
            //Default data
            Scanner scanner = new Scanner(System.in);
            SourceFile exampleSource = new SourceFile("project.csv", 10127, 19, 2, 0.75, 16);

            //Menu
            System.out.println("Verder gaan met voorbeeldbestand - (V)");
            System.out.println("Pad naar bestand zelf invoeren - (I)");
            System.out.println("Programma beëindigen - (Q)");
            String result = scanner.nextLine().toUpperCase();
            switch (result) {
                case ("V"):
                    System.out.println("Voorbeeldbestand \"" + exampleSource.path + "\" wordt gebruikt");
                    Thread.sleep(1000);
                    loadData(exampleSource);
                    break;
                case ("I"):
                    try {
                        SourceFile newSource = new SourceFile();
                        System.out.println("Pad naar bronbestand: ");
                        newSource.path = scanner.nextLine();
                        File f = new ClassPathResource(newSource.path).getFile();
                        if (f.isFile()) {
                            System.out.println("Aantal records: ");
                            newSource.batchSize = scanner.nextInt();
                            System.out.println("Aantal kolommen: ");
                            newSource.numberOfLabels = scanner.nextInt();
                            System.out.println("Aantal mogelijke uitkomsten: ");
                            newSource.numberOfResults = scanner.nextInt();
                            System.out.println("Train/test verdeling: ");
                            newSource.trainTestDivision = scanner.nextFloat();
                            newSource.nodes = newSource.numberOfLabels / 4 * 3;
                            System.out.println(newSource.path + " wordt gebruikt");
                            Thread.sleep(1000);
                            loadData(newSource);
                        } else {
                            System.out.println("Bestand niet gevonden");
                        }
                    } catch (Exception ex) {
                        System.out.println("bestand niet gevonden");
                        Thread.sleep(1000);
                    }
                    break;

                case ("Q"):
                    System.out.println("Programma wordt afgesloten");
                    Thread.sleep(1000);
                    running = false;
                    break;
                default:
                    System.out.println("Ongeldige invoer");
                    Thread.sleep(1000);
            }
        }
    }




    public static void loadData(SourceFile source){
        try(RecordReader rr = new CSVRecordReader(0,',')){
            rr.initialize(new FileSplit
                    (new ClassPathResource(source.path)
                            .getFile()));
            DataSetIterator iterator = new RecordReaderDataSetIterator(rr, source.batchSize, source.numberOfLabels, source.numberOfResults);
            DataSet allData = iterator.next();
            allData.shuffle(297);
            SplitTestAndTrain testAndTrain = allData.splitTestAndTrain(source.trainTestDivision);
            DataSet trainingData = testAndTrain.getTrain();
            DataSet testingData = testAndTrain.getTest();
            DataNormalization normalizer = new NormalizerStandardize();
            normalizer.fit(trainingData);
            normalizer.transform(trainingData);
            normalizer.transform(testingData);
            trainAndTestNN(trainingData, testingData, source);
        }catch (Exception e){
            e.printStackTrace();
        }
    }

    public static void trainAndTestNN(DataSet trainingData, DataSet testData, SourceFile source) {
        MultiLayerConfiguration configuration = new NeuralNetConfiguration.Builder()
                .seed(6)
                .activation(Activation.TANH)
                .weightInit(WeightInit.XAVIER)
                .updater(new Sgd(0.1))
                .list()
                .layer(0, new DenseLayer.Builder().nIn(source.numberOfLabels).nOut(source.nodes)
                        .build())
                .layer(1, new DenseLayer.Builder().nIn(source.nodes).nOut(source.nodes)
                        .build())
                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .activation(Activation.SOFTMAX)
                        .nIn(source.nodes).nOut(source.numberOfResults).build())
                .build();
        MultiLayerNetwork model = new MultiLayerNetwork(configuration);
        model.init();
        System.out.println("Trainen...");
        for (int i = 0; i < 1000; i++) {
            model.fit(trainingData);
        }

        Evaluation eval = new Evaluation(source.numberOfResults);
        INDArray output = model.output(testData.getFeatures());
        eval.eval(testData.getLabels(), output);
        System.out.println(eval.stats(true) );
        System.out.println("Het netwerk bevat 2 hidden layers met elk " + source.nodes + " nodes");
        System.out.println("Neuraal netwerk werd getraind op basis van " + source.batchSize + " records.");
        System.out.println("Na trainen werd " + (int)(eval.accuracy()*100) + "% van de controledata correct voorspeld");

        System.out.println("\n");

    }








}

