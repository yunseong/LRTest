package edu.snu.utils;

import org.apache.mahout.math.SequentialAccessSparseVector;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.Arrays;

public class TestMahout {

  private static int dimension;

  public static void main(final String[] args) throws Exception {
    dimension = Integer.valueOf(args[1]);
    File dataFile = new File(args[0]);
    BufferedReader bufferedReader = null;
    ArrayList<MahoutRow> rows = new ArrayList<MahoutRow>();
    try {
      bufferedReader = new BufferedReader(new FileReader(dataFile));
      String s = bufferedReader.readLine();
      while (s != null) {
        final String[] split = s.split("\\s+");
        final MahoutRow row = new MahoutRow(Integer.valueOf(split[0]), parseFeatureVector(Arrays.copyOfRange(split, 1, split.length)));
        rows.add(row);
        s = bufferedReader.readLine();
      }
    } catch (Exception e) {
      throw e;
    } finally {
      if (bufferedReader != null) {
        bufferedReader.close();
      }
    }

    org.apache.mahout.math.Vector model = new SequentialAccessSparseVector(dimension + 1);

    for (int i = 0; i < 5; i++) {
      long iterationStart = System.currentTimeMillis();
      for (final MahoutRow row : rows) {
        final double output = row.getOutput();
        final org.apache.mahout.math.Vector input = row.getFeature();
        final org.apache.mahout.math.Vector gradient = MahoutLogisticLoss.gradient(input, model.dot(input), output).plus(model.times(0.1));
        model = model.minus(gradient.times(0.00001));
      }
      System.out.println("Iteration " + i + " took " + (System.currentTimeMillis() - iterationStart) + "ms");

      long accuracyStart = System.currentTimeMillis();
      int posNum = 0;
      int negNum = 0;
      for (final MahoutRow row : rows) {
        final double output = row.getOutput();
        final double predict = model.dot(row.getFeature());
        if (output * predict > 0) {
          posNum++;
        } else {
          negNum++;
        }
      }
      System.out.println("Accuracy: " + (double) posNum / (posNum + negNum));
      System.out.println("Measuring accuracy took " + (System.currentTimeMillis() - accuracyStart) + "ms");
    }
  }

  private static AbstractMap.SimpleEntry<Integer, Double> parseElement(final String elemString) {
    try {
      final String[] split = elemString.split(":");
      if (split.length != 2) {
        throw new RuntimeException("Parse failed: the format of each element of a sparse vector must be [index]:[value]");
      }
      return new AbstractMap.SimpleEntry<Integer, Double>(Integer.valueOf(split[0]) - 1, Double.valueOf(split[1]));
    } catch (final NumberFormatException e) {
      throw new RuntimeException("Parse failed: invalid number format " + e);
    }
  }

  private static org.apache.mahout.math.Vector parseFeatureVector(final String[] split) {
    final org.apache.mahout.math.Vector ret = new SequentialAccessSparseVector(dimension + 1, split.length + 1); // +1 for a constant term
    for (final String elementString : split) {
      final AbstractMap.SimpleEntry<Integer, Double> elementPair = parseElement(elementString);
      ret.set(elementPair.getKey(), elementPair.getValue());
    }
    ret.set(dimension, 1); // a constant term
    return ret;
  }
}

final class MahoutRow {
  private final double output;
  private final org.apache.mahout.math.Vector feature;

  public MahoutRow(final double output, final org.apache.mahout.math.Vector feature) {
    this.output = output;
    this.feature = feature;
  }

  public double getOutput() {
    return output;
  }

  public org.apache.mahout.math.Vector getFeature() {
    return feature;
  }
}

final class MahoutLogisticLoss {

  public static org.apache.mahout.math.Vector gradient(final org.apache.mahout.math.Vector feature, final double predict, final double output) {

    // http://lingpipe-blog.com/2012/02/16/howprevent-overflow-underflow-logistic-regression/
    final double exponent = -predict * output;
    final double maxExponent = Math.max(exponent, 0);
    final double logSumExp = maxExponent + Math.log(Math.exp(-maxExponent) + Math.exp(exponent - maxExponent));
    return feature.times(output * (Math.exp(-logSumExp) - 1));
  }
}