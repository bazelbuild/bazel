package com.github.luben.zstd

import org.scalatest.FlatSpec

import scala.io._
import java.io._
import java.lang.management.ManagementFactory
import java.nio.ByteBuffer

import com.sun.management.ThreadMXBean

class ZstdPerfSpec extends FlatSpec  {

  class AllocTracker {
    val tmx: ThreadMXBean = ManagementFactory.getThreadMXBean().asInstanceOf[ThreadMXBean]
    val use_thread_alloc = try {
      tmx.setThreadAllocatedMemoryEnabled(true)
      tmx.isThreadAllocatedMemoryEnabled()
    } catch {
      case ex: Exception =>
        false
    }

    var allocs = 0L
    var nanos = 0L

    def timeAndAlloc[T](x: => T): T = {
      val t1 = System.nanoTime
      val a1 = tmx.getThreadAllocatedBytes(Thread.currentThread().getId)
      try {
        x
      } finally {
        val a2 = tmx.getThreadAllocatedBytes(Thread.currentThread().getId)
        val t2 = System.nanoTime
        allocs += a2 - a1
        nanos += t2 - t1
      }
    }
  }


  def report(name: String, compressed: Int, size: Int, cycles: Int, nsc: AllocTracker, nsd: AllocTracker ): Unit = {
    val ns = 1000L * 1000 * 1000
    val mb = 1024 * 1024

    val ratio = size.toDouble / compressed
    val seconds_c = nsc.nanos.toDouble / ns
    val seconds_d = nsd.nanos.toDouble / ns
    val alloc_c = nsc.allocs.toDouble / cycles
    val alloc_d = nsd.allocs.toDouble / cycles
    val total_mb  = cycles.toDouble * size / mb
    val speed_c   = total_mb / seconds_c
    val speed_d   = total_mb / seconds_d

    println(s"""
      $name
      --
      Compression:        ${speed_c.toLong} MB/s  alloc/cycle: ${alloc_c.toInt} B
      Decompression:      ${speed_d.toLong} MB/s  alloc/cycle: ${alloc_d.toInt} B
      Compression Ratio:  $ratio
    """)

  }

  def bench(name: String, input: Array[Byte], level: Int = 1): Unit = {
    var nsc = new AllocTracker
    var nsd = new AllocTracker
    var ratio = 0.0
    val output: Array[Byte] = Array.fill[Byte](input.size)(0)
    var compressedSize = 0
    for (i <- 1 to cycles) {
      val compressed = nsc.timeAndAlloc {
        Zstd.compress(input, level)
      }
      compressedSize  = compressed.size
      val size        = nsd.timeAndAlloc {
        Zstd.decompress(output, compressed)
      }
    }
    report(name, compressedSize, input.size, cycles, nsc, nsd)
    assert (input.toSeq == output.toSeq)
  }

  def benchDirectByteBuffer(name: String, input: Array[Byte], level: Int = 1): Unit = {
    var nsc = new AllocTracker
    var nsd = new AllocTracker
    var ratio = 0.0
    val outputBuffer = ByteBuffer.allocateDirect(input.size)
    val inputBuffer = ByteBuffer.allocateDirect(input.size)
    inputBuffer.put(input)
    var compressedSize = 0
    for (i <- 1 to cycles) {
      inputBuffer.rewind()
      val compressedBuffer = nsc.timeAndAlloc {
        Zstd.compress(inputBuffer, level)
      }
      compressedSize  = compressedBuffer.limit()
      outputBuffer.clear()
      val size = nsd.timeAndAlloc {
        Zstd.decompress(outputBuffer, compressedBuffer)
      }
    }
    report(name, compressedSize, input.size, cycles, nsc, nsd)
    assert (inputBuffer.compareTo(outputBuffer) == 0)
  }

  def benchDirectByteBufferWithDict(name: String, input: Array[Byte], level: Int = 1): Unit = {
    var nsc = new AllocTracker
    var nsd = new AllocTracker
    var ratio = 0.0
    val c_ctx = new ZstdCompressCtx()
    c_ctx.setLevel(level)
    val d_ctx = new ZstdDecompressCtx()
    val DICT_SIZE = 65536
    val dictBuf = new Array[Byte](DICT_SIZE)
    val samples = new Array[Array[Byte]](8)
    Range(0, samples.length).foreach { i =>
      val l = input.length / samples.length
      samples(i) = new Array[Byte](l)
      System.arraycopy(input, i * l, samples(i), 0, l)
    }
    val dictSize = Zstd.trainFromBuffer(samples, dictBuf)
    if (Zstd.isError(dictSize)) {
      throw new RuntimeException(Zstd.getErrorName(dictSize));
    }
    val c_zdict = new ZstdDictCompress(dictBuf, 0, dictSize.toInt, level)
    val d_zdict = new ZstdDictDecompress(dictBuf, 0, dictSize.toInt)

    c_ctx.loadDict(c_zdict);
    d_ctx.loadDict(d_zdict);

    val outputBuffer = ByteBuffer.allocateDirect(input.size)
    val inputBuffer = ByteBuffer.allocateDirect(input.size)
    val compressedBuffer = ByteBuffer.allocateDirect(Zstd.compressBound(input.size).toInt)
    inputBuffer.put(input)
    var compressedSize = 0
    for (i <- 1 to cycles) {
      inputBuffer.rewind()
      compressedBuffer.clear()
      nsc.timeAndAlloc {
        val l = c_ctx.compressDirectByteBuffer(compressedBuffer, 0, compressedBuffer.capacity(), inputBuffer, 0, inputBuffer.limit())
        if (Zstd.isError(l)) {
          throw new RuntimeException(Zstd.getErrorName(l));
        }
        compressedSize = l.toInt
      }
      compressedBuffer.position(0)
      compressedBuffer.limit(compressedSize)
      outputBuffer.clear()
      nsd.timeAndAlloc {
        val l = d_ctx.decompressDirectByteBuffer(outputBuffer, 0, outputBuffer.capacity(), compressedBuffer, 0, compressedBuffer.limit())
        if (Zstd.isError(l)) {
          throw new RuntimeException(Zstd.getErrorName(l));
        }
        assert(l == input.size)
      }
    }
    report(name, compressedSize, input.size, cycles, nsc, nsd)
    assert (inputBuffer.compareTo(outputBuffer) == 0)
  }


  def benchStream(name: String, input: Array[Byte], level: Int = 1): Unit = {
    val cycles = 50
    val size  = input.length
    var compressed: Array[Byte] = null

    val nsc = new AllocTracker
    nsc.timeAndAlloc {
      for (i <- 1 to cycles) {
        val os = new ByteArrayOutputStream(Zstd.compressBound(size.toLong).toInt)
        val zos = new ZstdOutputStream(os, level)
        zos.write(input)
        zos.close
        if (compressed == null)
          compressed = os.toByteArray
      }
    }

    val output= Array.fill[Byte](size)(0)
    val nsd = new AllocTracker
    nsd.timeAndAlloc {
      for (i <- 1 to cycles) {

        // now decompress
        val is = new ByteArrayInputStream(compressed)
        val zis = new ZstdInputStream(is)
        var ptr = 0

        while (ptr < size) {
          ptr += zis.read(output, ptr, size - ptr)
        }
        zis.close

      }
    }
    report(name, compressed.size, size, cycles, nsc, nsd)
    assert(input.toSeq == output.toSeq)
  }

  def benchStreamMT(name: String, input: Array[Byte], level: Int = 1): Unit = {
    val cycles = 50
    val size  = input.length
    var compressed: Array[Byte] = null

    val nsc = new AllocTracker
    nsc.timeAndAlloc {
      for (i <- 1 to cycles) {
        val os = new ByteArrayOutputStream(Zstd.compressBound(size.toLong).toInt)
        val zos = new ZstdOutputStream(os, level)
        zos.setWorkers(2)
        zos.write(input)
        zos.close()
        if (compressed == null)
          compressed = os.toByteArray
      }
    }

    val output= Array.fill[Byte](size)(0)
    val nsd = new AllocTracker
    nsd.timeAndAlloc {
      for (i <- 1 to cycles) {

        // now decompress
        val is = new ByteArrayInputStream(compressed)
        val zis = new ZstdInputStream(is)
        var ptr = 0

        while (ptr < size) {
          ptr += zis.read(output, ptr, size - ptr)
        }
        zis.close
      }
    }

    report(name, compressed.size, size, cycles, nsc, nsd)
    assert(input.toSeq == output.toSeq)
  }


  def benchDirectBufferStream(name: String, input: Array[Byte], level: Int = 1): Unit = {
    val cycles = 50

    val compressedBuffer = ByteBuffer.allocateDirect(Zstd.compressBound(input.length.toLong).toInt);
    val inputBuffer = ByteBuffer.allocateDirect(input.length)
    inputBuffer.put(input)
    inputBuffer.flip

    val target = ByteBuffer.allocateDirect(Zstd.compressBound(input.length.toLong).toInt)
    val nsc = new AllocTracker
    nsc.timeAndAlloc {
      for (i <- 1 to cycles) {
        target.clear()
        val compressor = new ZstdDirectBufferCompressingStream(target, level);
        inputBuffer.rewind()
        compressor.compress(inputBuffer)
        compressor.close()
      }
    }
    target.flip
    compressedBuffer.put(target)
    compressedBuffer.flip

    val decompressed = ByteBuffer.allocateDirect(input.length)
    val nsd = new AllocTracker
    nsd.timeAndAlloc {
      for (i <- 1 to cycles) {
        compressedBuffer.rewind
        decompressed.clear
        val zstr = new ZstdDirectBufferDecompressingStream(compressedBuffer)
        while (decompressed.hasRemaining) {
          zstr.read(decompressed)
        }
        zstr.close()
      }
    }

    val output = new Array[Byte](input.length)
    decompressed.flip()
    decompressed.get(output)

    report(name, compressedBuffer.limit(), input.length, cycles, nsc, nsd)
    assert(input.toSeq == output.toSeq)
  }

  val cycles = 200

  val levels = List(-3, -1, 1, 3, 6, 9)
  val buff = Source.fromFile("src/test/resources/xml")(Codec.ISO8859).map{_.toByte }.take(1024 * 1024).toArray
  for (level <- levels) {
    it should s"be fast for compressable data at level $level" in {
      bench(s"Compressable data at $level", buff, level)
      benchDirectByteBuffer(s"Compressable data at $level in a direct ByteBuffer", buff, level)
      benchDirectByteBufferWithDict(s"Compressable data at $level in a direct ByteBuffer and pre-allocated contexts", buff, level)
    }
  }

  val buff1 = Source.fromFile("src/test/resources/xmlx2")(Codec.ISO8859).map{_.toByte }.take(10 * 1024 * 1024).toArray
  for (level <- levels) {
    it should s"be fast with streaming at level $level" in {
        benchStream(s"Streaming at $level", buff1, level)
        benchStreamMT(s"Streaming (multi-threaded) at $level", buff1, level)
        benchDirectBufferStream(s"Streaming at $level to direct ByteBuffers", buff1, level)
    }
  }
}
