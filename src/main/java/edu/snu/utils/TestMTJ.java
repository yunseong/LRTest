package edu.snu.utils;

import no.uib.cipr.matrix.DenseVector;
import no.uib.cipr.matrix.sparse.SparseVector;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.Arrays;

public class TestMTJ {

  private static int dimension;

  public static void main(final String[] args) throws Exception {
    dimension = Integer.valueOf(args[1]);
    File dataFile = new File(args[0]);
    BufferedReader bufferedReader = null;
    ArrayList<MTJRow> rows = new ArrayList<MTJRow>();
    try {
      bufferedReader = new BufferedReader(new FileReader(dataFile));
      String s = bufferedReader.readLine();
      while (s != null) {
        final String[] split = s.split("\\s+");
        final MTJRow row = new MTJRow(Integer.valueOf(split[0]), parseFeatureVector(Arrays.copyOfRange(split, 1, split.length)));
        rows.add(row);
        s = bufferedReader.readLine();
      }
    } finally {
      if (bufferedReader != null) {
        bufferedReader.close();
      }
    }

    DenseVector model = new DenseVector(dimension + 1);

    for (int i = 0; i < 5; i++) {
      long iterationStart = System.currentTimeMillis();
      for (final MTJRow row : rows) {
        MTJLogisticLoss.gradientDescent(row.getFeature(), row.getOutput(), model);
      }
      System.out.println("Iteration " + i + " took " + (System.currentTimeMillis() - iterationStart) + "ms");

      long accuracyStart = System.currentTimeMillis();
      int posNum = 0;
      int negNum = 0;
      for (final MTJRow row : rows) {
        final double output = row.getOutput();
        final double predict = row.getFeature().dot(model);
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

  private static SparseVector parseFeatureVector(final String[] split) {
    final SparseVector ret = new SparseVector(dimension + 1, split.length + 1); // +1 for a constant term
    for (final String elementString : split) {
      final AbstractMap.SimpleEntry<Integer, Double> elementPair = parseElement(elementString);
      ret.set(elementPair.getKey(), elementPair.getValue());
    }
    ret.set(dimension, 1.0); // a constant term
    return ret;
  }
}

final class MTJRow {
  private final double output;
  private final SparseVector feature;

  public MTJRow(final double output, final SparseVector feature) {
    this.output = output;
    this.feature = feature;
  }

  public double getOutput() {
    return output;
  }

  public SparseVector getFeature() {
    return feature;
  }
}

final class MTJLogisticLoss {
  private static final double stepSize = 0.00001;
  private static final double lambda = 0.1;

  public static void gradientDescent(final SparseVector feature, final double output, final DenseVector model) {

    // http://lingpipe-blog.com/2012/02/16/howprevent-overflow-underflow-logistic-regression/
    final double exponent = -feature.dot(model) * output;
    final double maxExponent = Math.max(exponent, 0);
    final double logSumExp = maxExponent + Math.log(Math.exp(-maxExponent) + Math.exp(exponent - maxExponent));
    final double multiplier = output * (Math.exp(-logSumExp) - 1);
    model.scale(1.0 - stepSize * lambda);
    model.add(-stepSize * multiplier, feature);
  }
}