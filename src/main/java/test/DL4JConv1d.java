package test;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.FeedForwardLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.convolution.Convolution;
import org.nd4j.linalg.exception.ND4JException;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

/**
 * Created by jackzhang on 7/20/17.
 */
public class DL4JConv1d {
    private int inChannels;
    private int outChannels;
    private int kernelSize;
    private int stride;
    private int padding;

    public DL4JConv1d(int inChannels, int outChannels, int kernelSize, int stride, int padding) {
        this.inChannels = inChannels;
        this.outChannels = outChannels;
        this.kernelSize = kernelSize;
        this.stride = stride;
        this.padding = padding;
    }

    public INDArray forward(INDArray input, INDArray filter) {

        ConvolutionLayer.Builder builder = new ConvolutionLayer.Builder(this.kernelSize, this.stride, this.padding);
        //FeedForwardLayer conv = new ConvolutionLayer(builder);
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .list()
                .layer(0, new ConvolutionLayer.Builder(input.size(0), this.kernelSize)
                        .nIn(this.inChannels)
                        .stride(this.stride, this.stride)
                        .nOut(outChannels)
                        .padding(0, this.padding)
                        .activation(Activation.TANH)
                        .build())
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        int[] shape = {1, 1, input.size(0), input.size(1)};
        INDArray i = Nd4j.createUninitialized(shape);
        for (int j = 0; j < input.size(0); j++) {
            INDArrayIndex[] index = {NDArrayIndex.point(0), NDArrayIndex.point(0), NDArrayIndex.point(j), NDArrayIndex.all()};
            INDArrayIndex[] inputIndex = {NDArrayIndex.point(j), NDArrayIndex.all()};
            i.put(index, input.get(inputIndex));
        }
        INDArray output = model.output(i, false);
        output = output.reshape(output.size(1), output.size(3));
        return output;
    }
}
