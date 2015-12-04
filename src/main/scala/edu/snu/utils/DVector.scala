package edu.snu.utils

import breeze.linalg.DenseVector

class DVector(val vector: DenseVector[Double]) {
  def this(data: Array[Double]) = this(new DenseVector[Double](data))
  def plus(that: DVector): DVector = new DVector(this.vector + that.vector)
  def minus(that: DVector): DVector = new DVector(this.vector - that.vector)
  def add(that: DVector): Unit = this.vector += that.vector
  def add(a: Double, that: SVector): Unit = {
    val data = that.vector.array.data
    val index = that.vector.array.index
    val size = that.activeSize
    val thisData = this.vector.data

    if (a == 1.0) {
      var i = 0
      while (i < size) {
        thisData(index(i)) += data(i)
        i += 1
      }
    } else {
      var i = 0
      while (i < size) {
        thisData(index(i)) += a * data(i)
        i += 1
      }
    }
  }
  def subtract(that: DVector): Unit = this.vector -= that.vector
  def scale(i: Double): Unit = this.vector *= i
  def dot(that: DVector): Double = this.vector.dot(that.vector)
  def dot(that: SVector): Double = {
    val data = that.vector.array.data
    val index = that.vector.array.index
    val size = that.activeSize
    val thisData = this.vector.data

    var sum = 0.0
    var i = 0
    while (i < size) {
      sum += data(i) * thisData(index(i))
      i += 1
    }
    sum
  }
  override def toString = this.vector.toString()
  def activeSize: Int = this.vector.activeSize
  def length: Int = this.vector.length
  def update(i: Int, v: Double): Unit = this.vector.update(i, v)
}
