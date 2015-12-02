import breeze.linalg.SparseVector

class SVector(val vector: SparseVector[Double]) {
  def this(index: Array[Int], data: Array[Double], length: Int) = this(new SparseVector[Double](index, data, length))
  def plus(that: SVector): SVector = new SVector(this.vector + that.vector)
  def minus(that: SVector): SVector = new SVector(this.vector - that.vector)
  def times(i: Double): SVector = new SVector(this.vector * i)
  def add(that: SVector): Unit = this.vector += that.vector
  def subtract(that: SVector): Unit = this.vector -= that.vector
  def scale(i: Double): Unit = this.vector *= i
  def dot(that: SVector): Double = this.vector.dot(that.vector)
  override def toString = this.vector.toString()
  def activeSize: Int = this.vector.activeSize
  def length: Int = this.vector.length
  def update(i: Int, v: Double): Unit = this.vector.update(i, v)
}
