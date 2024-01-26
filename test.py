import jpype
from jpype import JClass, JArray, getDefaultJVMPath

def initialize_jvm():
    if not jpype.isJVMStarted():
        jpype.startJVM(getDefaultJVMPath(), "-ea", convertStrings=False)

def stop_jvm():
    jpype.shutdownJVM()

def load_weka_classes():
    # Load necessary Weka classes
    Instances = JClass("weka.core.Instances")
    DataSource = JClass("weka.core.converters.ConverterUtils$DataSource")
    NaiveBayes = JClass("weka.classifiers.bayes.NaiveBayes")
    Evaluation = JClass("weka.classifiers.Evaluation")

    return Instances, DataSource, NaiveBayes, Evaluation

def main():
    initialize_jvm()
    try:
        Instances, DataSource, NaiveBayes, Evaluation = load_weka_classes()

        # Load dataset
        source = DataSource("your_dataset.arff")  # Replace with your dataset path
        data = source.getDataSet()

        # Set class index (assuming the class attribute is the last one)
        if data.classIndex() == -1:
            data.setClassIndex(data.numAttributes() - 1)

        # Create Naive Bayes classifier
        classifier = NaiveBayes()
        classifier.buildClassifier(data)

        # Evaluate the classifier
        eval = Evaluation(data)
        eval.crossValidateModel(classifier, data, 10, JArray(JArray(JClass.float, 1))(1))

        # Print evaluation metrics
        print("Accuracy:", eval.pctCorrect())
        print("Precision:", eval.weightedPrecision())
        print("Recall:", eval.weightedRecall())

    except Exception as e:
        print("Error:", e)
    finally:
        stop_jvm()

if __name__ == "__main__":
    main()
