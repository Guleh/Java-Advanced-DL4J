public class SourceFile {

    String path;
    int batchSize ;
    int numberOfLabels;
    int numberOfResults;
    double trainTestDivision;
    int nodes;

    public SourceFile(){};
    public SourceFile(String path, int batchSize, int numberOfLabels, int numberOfResults, double trainTestDivision, int nodes){
        this.path = path;
        this.batchSize = batchSize;
        this.numberOfLabels = numberOfLabels;
        this.numberOfResults = numberOfResults;
        this.trainTestDivision = trainTestDivision;
        this.nodes = nodes;

    }


}
