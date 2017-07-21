package test; /**
 * Created by jackzhang on 7/20/17.
 */
//package com.example.app;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import static org.junit.Assert.assertEquals;

public class ConvTest {
    @Test
    public void testConv1dSimple() {
        double[] rawInput = {10, 50, 60, 10, 20, 40, 30};
        INDArray input = Nd4j.create(rawInput);

        double[] rawFilter = {1.0 / 3, 1.0 / 3, 1.0 / 3};
        INDArray filter = Nd4j.create(rawFilter);

        double[] rawExpected = {40, 40, 30, 70.0 / 3, 30};
        INDArray expectedOutput = Nd4j.create(rawExpected);

        Nd4jConv1d conv = new Nd4jConv1d(1, 1, filter.shape()[1], 1, 0);
        INDArray actualOutput = conv.forward(input, filter);

        assertEquals(expectedOutput, actualOutput);
    }

    @Test
    public void testConv1dWithPadding() {
        double[] rawInput = {10, 50, 60, 10, 20, 40, 30};
        INDArray input = Nd4j.create(rawInput);

        double[] rawFilter = {1.0 / 3, 1.0 / 3, 1.0 / 3};
        INDArray filter = Nd4j.create(rawFilter);

        double[] rawExpected = {20, 40, 40, 30, 70.0 / 3, 30, 70.0 / 3};
        INDArray expectedOutput = Nd4j.create(rawExpected);

        DL4JConv1d conv = new DL4JConv1d(1, 1, filter.shape()[1], 1, 1);
        INDArray actualOutput = conv.forward(input, filter);

        assertEquals(expectedOutput, actualOutput);
    }
}