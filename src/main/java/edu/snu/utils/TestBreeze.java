package edu.snu.utils;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.Arrays;

public class TestBreeze {

  private static int dimension;

  public static void main(final String[] args) throws Exception {
    dimension = Integer.valueOf(args[1]);
    File dataFile = new File(args[0]);
    BufferedReader bufferedReader = null;
    ArrayList<Row> rows = new ArrayList<Row>();
    try {
      bufferedReader = new BufferedReader(new FileReader(dataFile));
      String s = bufferedReader.readLine();
      while (s != null) {
        final String[] split = s.split("\\s+");
        final Row row = new Row(Integer.valueOf(split[0]), parseFeatureVector(Arrays.copyOfRange(split, 1, split.length)));
        rows.add(row);
        s = bufferedReader.readLine();
      }
    } finally {
      if (bufferedReader != null) {
        bufferedReader.close();
      }
    }

    DVector model = new DVector(new double[dimension + 1]);

    for (int i = 0; i < 5; i++) {
      long iterationStart = System.currentTimeMillis();
      for (final Row row : rows) {
        LogisticLoss.gradientDescent(row.getFeature(), row.getOutput(), model);
      }
      System.out.println("Iteration " + i + " took " + (System.currentTimeMillis() - iterationStart) + "ms");

      long accuracyStart = System.currentTimeMillis();
      int posNum = 0;
      int negNum = 0;
      for (final Row row : rows) {
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

  private static SVector parseFeatureVector(final String[] split) {
    final SVector ret = new SVector(new int[]{}, new double[]{}, dimension + 1); // +1 for a constant term
    for (final String elementString : split) {
      final AbstractMap.SimpleEntry<Integer, Double> elementPair = parseElement(elementString);
      ret.update(elementPair.getKey(), elementPair.getValue());
    }
    ret.update(dimension, 1); // a constant term
    return ret;
  }
}

final class Row {
  private final double output;
  private final SVector feature;

  public Row(final double output, final SVector feature) {
    this.output = output;
    this.feature = feature;
  }

  public double getOutput() {
    return output;
  }

  public SVector getFeature() {
    return feature;
  }
}

final class LogisticLoss {
  private static final double stepSize = 0.00001;
  private static final double lambda = 0.1;

  public static void gradientDescent(final SVector feature, final double output, final DVector model) {

    // http://lingpipe-blog.com/2012/02/16/howprevent-overflow-underflow-logistic-regression/
    final double exponent = -model.dot(feature) * output;
    final double maxExponent = Math.max(exponent, 0);
    final double logSumExp = maxExponent + Math.log(Math.exp(-maxExponent) + Math.exp(exponent - maxExponent));
    final double multiplier = output * (Math.exp(-logSumExp) - 1);
//    model.scale(1.0 - stepSize * lambda);
    model.add(-stepSize * multiplier, feature);
  }
}
