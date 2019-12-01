package com.github.luben.zstd

import org.scalatest.FlatSpec
import org.scalatest.prop.Checkers
import org.scalacheck.Arbitrary._
import org.scalacheck.Prop._
import java.io._
import java.nio._
import java.nio.channels.FileChannel
import java.nio.channels.FileChannel.MapMode
import java.nio.file.StandardOpenOption

import scala.io._
import scala.collection.mutable.WrappedArray

class ZstdSpec extends FlatSpec with Checkers {

  implicit override val generatorDrivenConfig =
    PropertyCheckConfiguration(minSize = 0, sizeRange = 130 * 1024)

  val levels = List(1, 3, 6, 9)

  for (level <- levels) {
    "Zstd" should s"should round-trip compression/decompression at level $level" in {
      check { input: Array[Byte] =>
        {
          val size        = input.length
          val compressed  = if (level != 3) Zstd.compress(input, level) else Zstd.compress(input)
          val decompressed= Zstd.decompress(compressed, size)
          input.toSeq == decompressed.toSeq
        }
      }
    }

    it should s"should round-trip compression/decompression with manual array buffers at level $level" in {
      check { input: Array[Byte] =>
        {
          val size        = input.length
          val decompressed= new Array[Byte](size)
          val dstSize     = Zstd.compressBound(size).toInt

          val compressed  = new Array[Byte](dstSize)
          val csize       = Zstd.compressByteArray(compressed, 0, dstSize, input, 0, size, level)
          if (Zstd.isError(csize)) sys.error(Zstd.getErrorName(csize))
          val dsize       = Zstd.decompressByteArray(decompressed, 0, size, compressed, 0, csize.toInt)
          if (Zstd.isError(dsize)) sys.error(Zstd.getErrorName(dsize))
          size == dsize && input.toSeq == decompressed.toSeq
        }
      }
    }

    it should s"round-trip compression/decompression with ByteBuffers at level $level" in {
      check { input: Array[Byte] =>
        {
          val size        = input.length
          val inputBuffer = ByteBuffer.allocateDirect(size)
          inputBuffer.put(input)
          inputBuffer.flip()
          val compressedBuffer = ByteBuffer.allocateDirect(Zstd.compressBound(size).toInt)
          val compressedSize = if (level != 3)
            Zstd.compress(compressedBuffer, inputBuffer, level)
          else
            Zstd.compress(compressedBuffer, inputBuffer)
          compressedBuffer.flip()
          val decompressedBuffer = ByteBuffer.allocateDirect(size)
          val decompressedSize = Zstd.decompress(decompressedBuffer, compressedBuffer)
          assert(decompressedSize == input.length)

          inputBuffer.rewind()
          compressedBuffer.rewind()
          decompressedBuffer.flip()

          val comparison = inputBuffer.compareTo(decompressedBuffer)
          val result = comparison == 0 && Zstd.decompressedSize(compressedBuffer) == decompressedSize
          result
        }
      }
    }

    it should s"compress with a byte[] and uncompress with a ByteBuffer $level" in {
      check { input: Array[Byte] =>
        val size = input.length
        val compressed = Zstd.compress(input, level)

        val compressedBuffer = ByteBuffer.allocateDirect(Zstd.compressBound(size.toLong).toInt)
        compressedBuffer.put(compressed)
        compressedBuffer.limit(compressedBuffer.position())
        compressedBuffer.flip()

        val decompressedBuffer = Zstd.decompress(compressedBuffer, size)
        val decompressed = new Array[Byte](size)
        decompressedBuffer.get(decompressed)
        input.toSeq == decompressed.toSeq
      }
    }

    it should s"compress with a ByteBuffer and uncompress with a byte[] $level" in {
      check { input: Array[Byte] =>
        val size        = input.length
        val inputBuffer = ByteBuffer.allocateDirect(size)
        inputBuffer.put(input)
        inputBuffer.flip()
        val compressedBuffer  = Zstd.compress(inputBuffer, level)
        val compressed = new Array[Byte](compressedBuffer.limit() - compressedBuffer.position())
        compressedBuffer.get(compressed)

        val decompressed = Zstd.decompress(compressed, size)
        input.toSeq == decompressed.toSeq
      }
    }
  }

  it should s"honor non-zero position and limit values in ByteBuffers" in {
    check { input: Array[Byte] =>
      val size = input.length

      //The test here is to compress the input at each of the designated levels, with each new compressed version
      //being added to the same buffer as the one before it, one after the other.  Then decompress the same way.
      //This verifies that the ByteBuffer-based versions behave the way one expects, honoring and updating position
      //and limit as they go
      val inputBuffer = ByteBuffer.allocateDirect(size)
      inputBuffer.put(input)
      inputBuffer.flip()
      val compressedBuffer = ByteBuffer.allocateDirect(Zstd.compressBound(size).toInt * levels.size)
      val decompressedBuffer = ByteBuffer.allocateDirect(size * levels.size)

      val compressedSizes = for (level <- levels) yield {
        inputBuffer.rewind()

        val oldCompressedPosition = compressedBuffer.position()
        val oldInputPosition = inputBuffer.position()

        val compressedSize = Zstd.compress(compressedBuffer, inputBuffer, level)

        assert(inputBuffer.position() == oldInputPosition + size)
        assert(compressedBuffer.position() == oldCompressedPosition + compressedSize)

        compressedSize
      }

      compressedBuffer.flip()
      for ((level, compresedSize) <- levels.zip(compressedSizes)) {
        val oldCompressedPosition = compressedBuffer.position()
        val oldDecompressedPosition = decompressedBuffer.position()

        //This isn't using the streaming mode, so zstd expects the entire contents of the buffer to be one
        //zstd compressed output.  For this test we've stacked the compressed outputs one after the other.
        //Use limit to mark where the end of this particular compresssed output can be found.
        //Note that the duplicate() call doesn't copy memory; it just makes a duplicate ByteBuffer pointing to the same
        //location in memory
        val thisChunkBuffer = compressedBuffer.duplicate()
        thisChunkBuffer.limit(thisChunkBuffer.position() + compresedSize)
        val decompressedSize = Zstd.decompress(decompressedBuffer, thisChunkBuffer)

        assert(decompressedSize == input.length)
        compressedBuffer.position(thisChunkBuffer.position)
        assert(compressedBuffer.position() == oldCompressedPosition + compresedSize)
        assert(decompressedBuffer.position() == oldDecompressedPosition + size)
      }

      //At this point the decompressedBuffer's position should equal it's limit.
      //flip it and verify it has one copy of the input for each of the levels that were compressed
      assert(decompressedBuffer.hasRemaining == false)
      decompressedBuffer.flip()

      for (level <- levels) {
        val slice = decompressedBuffer.slice()
        slice.limit(size)
        inputBuffer.rewind()

        val expected = new Array[Byte](size)
        inputBuffer.get(expected)
        val actual = new Array[Byte](size)
        slice.get(actual)

        assert(actual.toSeq == expected.toSeq)
        decompressedBuffer.position(decompressedBuffer.position() + size)
      }

      assert(decompressedBuffer.position() == levels.size * size)
      assert(!decompressedBuffer.hasRemaining)
      true
    }
  }

  it should "fail to compress when the destination buffer is too small" in {
    check{ input: Array[Byte] =>
      val size = input.length
      val compressedSize = Zstd.compress(input, 3).length
      val compressedBuffer = ByteBuffer.allocateDirect(compressedSize.toInt - 1)
      val inputBuffer = ByteBuffer.allocateDirect(size)
      inputBuffer.put(input)
      inputBuffer.flip()

      val e = intercept[ZstdException] {
        Zstd.compress(compressedBuffer, inputBuffer, 3)
      }

      e.getErrorCode() == Zstd.errDstSizeTooSmall() && e.getMessage().contains("Destination buffer is too small")
    }
  }

  it should "fail to decompress when the destination buffer is too small" in {
    check { input: Array[Byte] =>
      (input.length > 0) ==> {
        val size = input.length
        val compressedSize = Zstd.compressBound(size.toLong)
        val inputBuffer = ByteBuffer.allocateDirect(size)
        inputBuffer.put(input)
        inputBuffer.flip()
        val compressedBuffer = Zstd.compress(inputBuffer, 1)
        val decompressedBuffer = ByteBuffer.allocateDirect(size - 1)

        val e = intercept[ZstdException] {
          Zstd.decompress(decompressedBuffer, compressedBuffer)
        }

        e.getErrorCode() == Zstd.errDstSizeTooSmall() && e.getMessage().contains("Destination buffer is too small")
      }
    }
  }

  for (level <- levels) {
    "ZstdInputStream" should s"should round-trip compression/decompression at level $level" in {
      check { input: Array[Byte] =>
        val size  = input.length
        val os    = new ByteArrayOutputStream(Zstd.compressBound(size.toLong).toInt)
        val zos   = new ZstdOutputStream(os, level)
        val block = 128 * 1024
        var ptr   = 0
        while (ptr < size) {
          val chunk = if (size - ptr > block) block else size - ptr
          zos.write(input, ptr, chunk)
          ptr += chunk
        }
        zos.close
        val compressed = os.toByteArray
        // now decompress
        val is    = new ByteArrayInputStream(compressed)
        val zis   = new ZstdInputStream(is)
        val output= Array.fill[Byte](size)(0)
        ptr       = 0

        while (ptr < size) {
          val chunk = if (size - ptr > block) block else size - ptr
          zis.read(output, ptr, chunk)
          ptr += chunk
        }
        zis.close
        if (input.toSeq != output.toSeq) {
          println(s"AT SIZE $size")
          println(input.toSeq + "!=" + output.toSeq)
          println("COMPRESSED: " + compressed.toSeq)
        }
        input.toSeq == output.toSeq
      }
    }
  }

  for (level <- levels) {
    "ZstdInputStreamMT" should s"should round-trip compression/decompression at level $level" in {
      check { input: Array[Byte] =>
        val size  = input.length
        val os    = new ByteArrayOutputStream(Zstd.compressBound(size.toLong).toInt)
        val zos   = new ZstdOutputStream(os)
        zos.setLevel(level)
        zos.setWorkers(4)
        val block = 128 * 1024
        var ptr   = 0
        while (ptr < size) {
          val chunk = if (size - ptr > block) block else size - ptr
          zos.write(input, ptr, chunk)
          ptr += chunk
        }
        zos.close
        val compressed = os.toByteArray
        // now decompress
        val is    = new ByteArrayInputStream(compressed)
        val zis   = new ZstdInputStream(is)
        val output= Array.fill[Byte](size)(0)
        ptr       = 0

        while (ptr < size) {
          val chunk = if (size - ptr > block) block else size - ptr
          zis.read(output, ptr, chunk)
          ptr += chunk
        }
        zis.close
        if (input.toSeq != output.toSeq) {
          println(s"AT SIZE $size")
          println(input.toSeq + "!=" + output.toSeq)
          println("COMPRESSED: " + compressed.toSeq)
        }
        input.toSeq == output.toSeq
      }
    }
  }

  for (level <- levels) {
    "ZstdDirectBufferDecompressingStream" should s"should round-trip compression/decompression at level $level" in {
      check { input: Array[Byte] =>
        val size  = input.length
        val os    = ByteBuffer.allocateDirect(Zstd.compressBound(size.toLong).toInt)

        // compress
        val ib    = ByteBuffer.allocateDirect(size)
        ib.put(input)
        val osw = new ZstdDirectBufferCompressingStream(os, level)
        ib.flip
        osw.compress(ib)
        osw.close
        os.flip

        // for debugging
        val bytes = new Array[Byte](os.limit())
        os.get(bytes)
        os.rewind()


        // now decompress
        val zis   = new ZstdDirectBufferDecompressingStream(os)
        val output= Array.fill[Byte](size)(0)
        val block = ByteBuffer.allocateDirect(128 * 1024)
        var offset = 0
        while (zis.hasRemaining) {
          block.clear()
          val read = zis.read(block)
          block.flip()
          block.get(output, offset, read)
          offset += read
        }
        zis.close
        if (input.toSeq != output.toSeq) {
          println(s"AT SIZE $size")
          println(input.toSeq + "!=" + output.toSeq)
          println("COMPRESSED: " + bytes.toSeq)
        }
        input.toSeq == output.toSeq
      }
    }
  }

  for (level <- levels) {
    "ZstdInputStream in continuous mode" should s"should round-trip using streaming API with unfinished chunks at level $level" in {
      check { input: Array[Byte] =>
        val size  = input.length
        val os    = new ByteArrayOutputStream(Zstd.compressBound(size.toLong).toInt)
        val zos   = new ZstdOutputStream(os, level)
        val block = 128 * 1024
        var ptr   = 0
        while (ptr < size) {
          val chunk = if (size - ptr > block) block else size - ptr
          zos.write(input, ptr, chunk)
          ptr += chunk
        }
        zos.close
        val compressed = os.toByteArray
        // now decompress
        val is    = new ByteArrayInputStream(compressed)
        val zis   = new ZstdInputStream(is).setContinuous(true);
        val output= Array.fill[Byte](size)(0)
        ptr       = 0

        while (ptr < size) {
          val chunk = if (size - ptr > block) block else size - ptr
          zis.read(output, ptr, chunk)
          ptr += chunk
        }
        zis.close
        if (input.toSeq != output.toSeq) {
          println(s"AT SIZE $size")
          println(input.toSeq + "!=" + output.toSeq)
          println("COMPRESSED: " + compressed.toSeq)
        }
        input.toSeq == output.toSeq
      }
    }
  }

  for (level <- levels)
    "ZstdInputStream" should s"be able to consume files compressed by the zstd binary at level $level" in {
      val orig = new File("src/test/resources/xml")
      val file = new File(s"src/test/resources/xml-$level.zst")
      val fis  = new FileInputStream(file)
      val zis  = new ZstdInputStream(fis)
      assert(!zis.markSupported)
      assert(zis.available > 0)
      assert(zis.skip(0) == 0)
      val length = orig.length.toInt
      val buff = Array.fill[Byte](length)(0)
      var pos  = 0;
      while (pos < length) {
        pos += zis.read(buff, pos, length - pos)
      }

      val original = Source.fromFile(orig)(Codec.ISO8859).map{char => char.toByte}.to[WrappedArray]
      if(original != buff.toSeq)
        sys.error(s"Failed")
      assert(zis.available == 0)
    }

  for (level <- levels) // the worst case
    it should s"be able to consume 1 byte at a time files compressed by the zstd binary at level $level" in {
      val orig = new File("src/test/resources/xml")
      val file = new File(s"src/test/resources/xml-$level.zst")
      val fis  = new FileInputStream(file)
      val zis  = new ZstdInputStream(fis)
      val length = orig.length.toInt
      val buff = Array.fill[Byte](length)(0)
      var pos  = 0
      val block = 1
      while (pos < length) {
        val remain = length - pos;
        val toRead = if (remain > block) block else remain
        pos += zis.read(buff, pos, toRead)
      }

      val original = Source.fromFile(orig)(Codec.ISO8859).map{char => char.toByte}.to[WrappedArray]
      if(original != buff.toSeq)
        sys.error(s"Failed")
    }

  it should s"be able to consume 2 frames in a file compressed by the zstd binary" in {
    val orig = new File("src/test/resources/xmlx2")
    val file = new File(s"src/test/resources/xml-1x2.zst")
    val fis  = new FileInputStream(file)
    val zis  = new ZstdInputStream(fis)
    assert(!zis.markSupported)
    assert(zis.available > 0)
    assert(zis.skip(0) == 0)
    val length = orig.length.toInt
    val buff = Array.fill[Byte](length)(0)
    var pos  = 0;
    while (pos < length) {
      pos += zis.read(buff, pos, length - pos)
    }

    val original = Source.fromFile(orig)(Codec.ISO8859).map{char => char.toByte}.to[WrappedArray]
    if(original != buff.toSeq)
      sys.error(s"Failed")
    assert(zis.available == 0)
  }

  for (level <- levels) // the worst case
    "ZstdDirectBufferDecompressingStream" should s"be able to produce 1 byte at a time files compressed by the zstd binary at level $level" in {
      val orig = new File("src/test/resources/xml")
      val file = new File(s"src/test/resources/xml-$level.zst")
      val channel = FileChannel.open(file.toPath, StandardOpenOption.READ)
      val zis  = new ZstdDirectBufferDecompressingStream(channel.map(MapMode.READ_ONLY, 0, channel.size))
      val length = orig.length.toInt
      val buff = Array.fill[Byte](length)(0)
      var pos  = 0
      val block = ByteBuffer.allocateDirect(1)
      while (pos < length && zis.hasRemaining) {
        block.clear
        val read = zis.read(block)
        if (read != 1) {
          sys.error(s"Failed reading compressed file before end")
        }
        block.flip()
        buff.update(pos, block.get())
        pos += 1
      }

      val original = Source.fromFile(orig)(Codec.ISO8859).map{char => char.toByte}.to[WrappedArray]
      if(original != buff.toSeq)
        sys.error(s"Failed")
    }

  for (level <- levels) // the worst case
    it should s"be able to read 1 byte at a time files compressed by the zstd binary at level $level" in {
      val orig = new File("src/test/resources/xml")
      val file = new File(s"src/test/resources/xml-$level.zst")
      val channel = FileChannel.open(file.toPath, StandardOpenOption.READ)
      val actualBuffer  = channel.map(MapMode.READ_ONLY, 0, channel.size)
      val smallBuffer = ByteBuffer.allocateDirect(1)
      while (smallBuffer.hasRemaining) {
        smallBuffer.put(actualBuffer.get())
      }
      smallBuffer.flip()
      val zis  = new ZstdDirectBufferDecompressingStream(smallBuffer) {
        override protected def refill(toRefill: ByteBuffer) : ByteBuffer = {
          if (actualBuffer.hasRemaining) {
            toRefill.clear()
            while (toRefill.hasRemaining && actualBuffer.hasRemaining) {
              toRefill.put(actualBuffer.get())
            }
            toRefill.flip()
          }
          return toRefill
        }
      }
      val length = orig.length.toInt
      val buff = Array.fill[Byte](length)(0)
      val block = ByteBuffer.allocateDirect(length)
      while (zis.hasRemaining) {
        zis.read(block)
      }
      block.flip()
      block.get(buff)

      val original = Source.fromFile(orig)(Codec.ISO8859).map{char => char.toByte}.to[WrappedArray]
      if(original != buff.toSeq)
        sys.error(s"Failed")
    }


  it should s"be able to consume 2 frames in a file compressed by the zstd binary" in {
    val orig = new File("src/test/resources/xmlx2")
    val file = new File(s"src/test/resources/xml-1x2.zst")
    val fis  = new FileInputStream(file)

    val channel = FileChannel.open(file.toPath, StandardOpenOption.READ)
    val zis  = new ZstdDirectBufferDecompressingStream(channel.map(MapMode.READ_ONLY, 0, channel.size))
    val length = orig.length.toInt

    val window = ByteBuffer.allocateDirect(length)
    while (zis.hasRemaining) {
      zis.read(window)
    }

    val buff = new Array[Byte](length)
    window.flip()
    window.get(buff)

    val original = Source.fromFile(orig)(Codec.ISO8859).map{char => char.toByte}.to[WrappedArray]
    if(original != buff.toSeq)
      sys.error(s"Failed")
  }

  it should s"be able to consume 2 frames even when they are exactly at the buffers limit" in {
    val orig = new File("src/test/resources/xmlx2")
    val file = new File(s"src/test/resources/xml-1x2.zst")
    val length = orig.length.toInt

    val channel = FileChannel.open(file.toPath, StandardOpenOption.READ)
    val readBuffer = channel.map(MapMode.READ_ONLY, 0, channel.size)
    val zis  = new ZstdDirectBufferDecompressingStream(readBuffer)

    val window = ByteBuffer.allocateDirect(length)
    zis.read(window)
    val firstFrameEnd = readBuffer.position()
    zis.read(window)
    if (zis.hasRemaining) {
      sys.error(s"Failed for one big buffer, two reads should be enough")
    }
    zis.close()

    val readBuffer2 = channel.map(MapMode.READ_ONLY, 0, firstFrameEnd);
    val zis2  = new ZstdDirectBufferDecompressingStream(readBuffer2) {
      override protected def refill(toRefill: ByteBuffer) : ByteBuffer = {
        if (toRefill eq readBuffer2) {
          val offset = firstFrameEnd - toRefill.remaining()
          return channel.map(MapMode.READ_ONLY, offset, channel.size - offset)
        }
        return toRefill
      }
    }

    window.clear()

    zis2.read(window)
    zis2.read(window)
    if (zis2.hasRemaining) {
      sys.error(s"Failed, two reads should be enough")
    }
    zis2.close()


    val buff = new Array[Byte](length)
    window.flip()
    window.get(buff)

    val original = Source.fromFile(orig)(Codec.ISO8859).map{char => char.toByte}.to[WrappedArray]
    if(original != buff.toSeq)
      sys.error(s"Failed")
  }

  it should s"be able to consume 2 frames from channel" in {
    val orig = new File("src/test/resources/xmlx2")
    val file = new File("src/test/resources/xml-1x2.zst")
    val length = orig.length.toInt

    val channel = FileChannel.open(file.toPath, StandardOpenOption.READ)
    val readBuffer = ByteBuffer.allocateDirect(4096)
    readBuffer.clear().flip()
    val zis  = new ZstdDirectBufferDecompressingStream(readBuffer) {
      override protected def refill(toRefill: ByteBuffer) : ByteBuffer = {
        toRefill.clear()
        channel.read(toRefill)
        toRefill.flip()
        return toRefill
      }
    }

    val result = ByteBuffer.allocateDirect(length)
    val outChan = new java.nio.channels.WritableByteChannel {
      override def close(): Unit = result.flip()
      override def isOpen() = true
      override def write(src: ByteBuffer): Int = {
        val pos = src.position()
        result.put(src)
        src.position() - pos
      }
    }

    val outBuffer = ByteBuffer.allocateDirect(4096)

    while (zis.hasRemaining) {
      outBuffer.clear()
      zis.read(outBuffer)
      outBuffer.flip()
      while(outBuffer.remaining() > 0) {
        outChan.write(outBuffer)
      }
    }
    outChan.close()

    val buff = new Array[Byte](result.remaining())
    result.get(buff)

    val original = Source.fromFile(orig)(Codec.ISO8859).map{char => char.toByte}.to[WrappedArray]
    if(original != buff.toSeq)
      sys.error(s"Failed")
  }

  "ZstdInputStream" should s"do nothing on double close but throw on reading of closed stream" in {
    val file = new File(s"src/test/resources/xml-1x2.zst")
    val fis  = new FileInputStream(file)
    val zis  = new ZstdInputStream(fis)
    zis.close()
    zis.close()
    assertThrows[IOException] {
      val buff = Array.fill[Byte](100)(0)
      zis.read(buff, 0, 100)
    }
  }

  "ZstdOutputStream" should s"do nothing on double close but throw on writing on closed stream" in {
    val os  = new ByteArrayOutputStream(100)
    val zos = new ZstdOutputStream(os, 1)
    zos.close()
    zos.close()
    assertThrows[IOException] {
      zos.write("foo".toArray.map(_.toByte))
    }
  }

  "ZstdDirectBufferCompressingStream" should s"do nothing on double close but throw on writing on closed stream" in {
    val os  = ByteBuffer.allocateDirect(100)
    val zos = new ZstdDirectBufferCompressingStream(os, 1)
    zos.close()
    zos.close()
    assertThrows[IOException] {
      zos.compress(ByteBuffer.allocateDirect(3))
    }
  }

  for (level <- levels)
    "ZstdOutputStream" should s"produce the same compressed file as zstd binary at level $level" in {
      val file = new File("src/test/resources/xml")
      val length = file.length.toInt
      val fis  = new FileInputStream(file)
      val buff = Array.fill[Byte](length)(0)
      var pos  = 0
      while( pos < length) {
        pos += fis.read(buff, pos, length - pos)
      }

      val os  = new ByteArrayOutputStream(Zstd.compressBound(file.length).toInt)
      val zos = new ZstdOutputStream(os, level)
      zos.write(buff(0).toInt)
      zos.write(buff, 1, length - 1)
      zos.close()

      val compressed = os.toByteArray.toSeq
      val zst = Source.fromFile(s"src/test/resources/xml-$level.zst")(Codec.ISO8859).map{char => char.toByte}.to[WrappedArray]

     if (zst != compressed) {
        sys.error(s"Failed original ${zst.length} != ${compressed.length} result")
      }
    }

  for (level <- levels)
    "ZstdDirectBufferCompressingStream" should s"produce the same compressed file as zstd binary at level $level" in {
      val file = new File("src/test/resources/xml")
      val length = file.length.toInt
      val channel = FileChannel.open(file.toPath, StandardOpenOption.READ)
      val target = ByteBuffer.allocateDirect(Zstd.compressBound(length).toInt)

      val zos = new ZstdDirectBufferCompressingStream(target, level)
      zos.compress(channel.map(MapMode.READ_ONLY, 0, length));
      zos.close()
      channel.close()

      target.flip()

      val compressed = new Array[Byte](target.limit())
      target.get(compressed)
      val zst = Source.fromFile(s"src/test/resources/xml-$level.zst")(Codec.ISO8859).map{char => char.toByte}.to[WrappedArray]

      if (zst != compressed.toSeq) {
        sys.error(s"Failed original ${zst.length} != ${compressed.length} result")
      }
    }

  for (level <- levels)
    "ZstdDirectBufferCompressingStream" should s" even when writing one byte at a time produce the same compressed file as zstd binary at level $level" in {
      val file = new File("src/test/resources/xml")
      val length = file.length.toInt
      val channel = FileChannel.open(file.toPath, StandardOpenOption.READ)
      val target = ByteBuffer.allocateDirect(Zstd.compressBound(length).toInt)

      val zos = new ZstdDirectBufferCompressingStream(ByteBuffer.allocateDirect(1), level) {
        override protected def flushBuffer(toFlush: ByteBuffer): ByteBuffer = {
          toFlush.flip()
          target.put(toFlush)
          toFlush.clear()
          return toFlush
        }

      }
      val source = channel.map(MapMode.READ_ONLY, 0, length)
      while (source.hasRemaining) {
        zos.compress(source);
      }
      zos.close()
      channel.close()

      target.flip()

      val compressed = new Array[Byte](target.limit())
      target.get(compressed)
      val zst = Source.fromFile(s"src/test/resources/xml-$level.zst")(Codec.ISO8859).map{char => char.toByte}.to[WrappedArray]

      if (zst != compressed.toSeq) {
        sys.error(s"Failed original ${zst.length} != ${compressed.length} result")
      }
    }


  for (version <- List("04", "05", "06", "07"))
    "ZstdInputStream" should s"be able to consume files compressed by the zstd binary version $version" in {
      val orig = new File("src/test/resources/xml")
      val file = new File(s"src/test/resources/xml_v$version.zst")
      val fis  = new FileInputStream(file)
      val zis  = new ZstdInputStream(fis)
      assert(!zis.markSupported)
      assert(zis.available > 0)
      assert(zis.skip(0) == 0)
      val length = orig.length.toInt
      val buff = Array.fill[Byte](length)(0)
      var pos  = 0;
      while (pos < length) {
        pos += zis.read(buff, pos, length - pos)
      }

      val original = Source.fromFile(orig)(Codec.ISO8859).map{char => char.toByte}.to[WrappedArray]
      if(original != buff.toSeq)
        sys.error(s"Failed")
      assert(zis.available == 0)
    }

  for (version <- List("04", "05", "06", "07"))
    "ZstdInputStream in continuous mode" should s"be able to consume files compressed by the zstd binary version $version" in {
      val orig = new File("src/test/resources/xml")
      val file = new File(s"src/test/resources/xml_v$version.zst")
      val fis  = new FileInputStream(file)
      val zis  = new ZstdInputStream(fis).setContinuous(true);
      assert(!zis.markSupported)
      assert(zis.available > 0)
      assert(zis.skip(0) == 0)
      val length = orig.length.toInt
      val buff = Array.fill[Byte](length)(0)
      var pos  = 0;
      while (pos < length) {
        pos += zis.read(buff, pos, length - pos)
      }

      val original = Source.fromFile(orig)(Codec.ISO8859).map{char => char.toByte}.to[WrappedArray]
      if(original != buff.toSeq)
        sys.error(s"Failed")
      assert(zis.available == 0)
    }

  "ZstdInputStream.read() of empty frame" should "return -1" in {
    val compressed = Zstd.compress(Array.empty[Byte])
    val zis = new ZstdInputStream(new ByteArrayInputStream(compressed))
    assert(zis.read == -1)
  }

  "ZstdInputStream.read(buf, offset, len) of empty frame" should "return -1" in {
    val compressed = Zstd.compress(Array.empty[Byte])
    val zis = new ZstdInputStream(new ByteArrayInputStream(compressed))
    val buf = new Array[Byte](100)
    assert(zis.read(buf, 0, 1) == -1)
  }
}
