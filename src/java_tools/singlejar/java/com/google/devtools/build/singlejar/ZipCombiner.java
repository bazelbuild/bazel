// Copyright 2014 Google Inc. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.google.devtools.build.singlejar;

import static java.nio.charset.StandardCharsets.ISO_8859_1;
import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.devtools.build.singlejar.ZipEntryFilter.CustomMergeStrategy;
import com.google.devtools.build.singlejar.ZipEntryFilter.StrategyCallback;

import java.io.BufferedOutputStream;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.EOFException;
import java.io.FilterOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Calendar;
import java.util.Date;
import java.util.GregorianCalendar;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.zip.CRC32;
import java.util.zip.DataFormatException;
import java.util.zip.Deflater;
import java.util.zip.Inflater;

import javax.annotation.Nullable;
import javax.annotation.concurrent.NotThreadSafe;

/**
 * An object that combines multiple ZIP files into a single file. It only
 * supports a subset of the ZIP format, specifically:
 * <ul>
 *   <li>It only supports STORE and DEFLATE storage methods.</li>
 *   <li>There may be no data before the first file or between files.</li>
 *   <li>It ignores any data after the last file.</li>
 * </ul>
 *
 * <p>These restrictions are also present in the JDK implementations
 * {@link java.util.jar.JarInputStream}, {@link java.util.zip.ZipInputStream},
 * though they are not documented there.
 *
 * <p>IMPORTANT NOTE: Callers must call {@link #finish()} or {@link #close()}
 * at the end of processing to ensure that the output buffers are flushed and
 * the ZIP file is complete.
 *
 * <p>This class performs only rudimentary data checking. If the input files
 * are damaged, the output will likely also be damaged.
 *
 * <p>Also see:
 * <a href="http://www.pkware.com/documents/casestudies/APPNOTE.TXT">ZIP format</a>
 */
@NotThreadSafe
public final class ZipCombiner implements AutoCloseable {

  /**
   * A Date set to the 1/1/1980, 00:00:00, the minimum value that can be stored
   * in a ZIP file.
   */
  public static final Date DOS_EPOCH = new GregorianCalendar(1980, 0, 1, 0, 0, 0).getTime();

  private static final int DEFAULT_CENTRAL_DIRECTORY_BLOCK_SIZE = 1048576; // 1 MB for each block

  // The following constants are ZIP-specific.
  private static final int LOCAL_FILE_HEADER_MARKER = 0x04034b50;
  private static final int DATA_DESCRIPTOR_MARKER = 0x08074b50;
  private static final int CENTRAL_DIRECTORY_MARKER = 0x02014b50;
  private static final int END_OF_CENTRAL_DIRECTORY_MARKER = 0x06054b50;

  private static final int FILE_HEADER_BUFFER_SIZE = 30;

  private static final int VERSION_TO_EXTRACT_OFFSET = 4;
  private static final int GENERAL_PURPOSE_FLAGS_OFFSET = 6;
  private static final int COMPRESSION_METHOD_OFFSET = 8;
  private static final int MTIME_OFFSET = 10;
  private static final int MDATE_OFFSET = 12;
  private static final int CRC32_OFFSET = 14;
  private static final int COMPRESSED_SIZE_OFFSET = 18;
  private static final int UNCOMPRESSED_SIZE_OFFSET = 22;
  private static final int FILENAME_LENGTH_OFFSET = 26;
  private static final int EXTRA_LENGTH_OFFSET = 28;

  private static final int DIRECTORY_ENTRY_BUFFER_SIZE = 46;

  // Set if the size, compressed size and CRC are set to zero, and present in
  // the data descriptor after the data.
  private static final int SIZE_MASKED_FLAG = 1 << 3;

  private static final int STORED_METHOD = 0;
  private static final int DEFLATE_METHOD = 8;

  private static final int VERSION_STORED = 10; // Version 1.0
  private static final int VERSION_DEFLATE = 20; // Version 2.0

  private static final long MAXIMUM_DATA_SIZE = 0xffffffffL;

  // This class relies on the buffer to have sufficient space for a complete
  // file name. 2^16 is the maximum number of bytes in a file name.
  private static final int BUFFER_SIZE = 65536;

  /** An empty entry used to skip files that have already been copied (or skipped). */
  private static final FileEntry COPIED_FILE_ENTRY = new FileEntry(null, null, 0);

  /** An empty entry used to mark files that have already been renamed. */
  private static final FileEntry RENAMED_FILE_ENTRY = new FileEntry(null, null, 0);

  /** A zero length array of ExtraData. */
  public static final ExtraData[] NO_EXTRA_ENTRIES = new ExtraData[0];

  /**
   * Whether to compress or decompress entries.
   */
  public enum OutputMode {

    /**
     * Output entries using any method.
     */
    DONT_CARE,

    /**
     * Output all entries using DEFLATE method, except directory entries. It is
     * always more efficient to store directory entries uncompressed.
     */
    FORCE_DEFLATE,

    /**
     * Output all entries using STORED method.
     */
    FORCE_STORED;
  }

  // A two-element enum for copyOrSkip type methods.
  private static enum SkipMode {

    /**
     * Copy the read data to the output stream.
     */
    COPY,

    /**
     * Do not write anything to the output stream.
     */
    SKIP;
  }

  /**
   * Stores internal information about merges or skips.
   */
  private static final class FileEntry {

    /** If null, the file should be skipped. Otherwise, it should be merged. */
    private final CustomMergeStrategy mergeStrategy;
    private final ByteArrayOutputStream outputBuffer;
    private final int dosTime;

    private FileEntry(CustomMergeStrategy mergeStrategy, ByteArrayOutputStream outputBuffer,
        int dosTime) {
      this.mergeStrategy = mergeStrategy;
      this.outputBuffer = outputBuffer;
      this.dosTime = dosTime;
    }
  }

  /**
   * The directory entry info used for files whose extra directory entry info is not given
   * explicitly. It uses {@code -1} for {@link DirectoryEntryInfo#withMadeByVersion(short)}, which
   * indicates it will be set to the same version as "needed to extract."
   *
   * <p>The {@link DirectoryEntryInfo#withExternalFileAttribute(int)} value is set to {@code 0},
   * whose meaning depends on the value of {@code madeByVersion}, but is usually a reasonable
   * default.
   */
  public static final DirectoryEntryInfo DEFAULT_DIRECTORY_ENTRY_INFO =
      new DirectoryEntryInfo((short) -1, 0);

  /**
   * Contains information related to a zip entry that is stored in the central directory record.
   * This does not contain all the information stored in the central directory record, only the
   * information that can be customized and is not automatically calculated or detected.
   */
  public static final class DirectoryEntryInfo {
    private final short madeByVersion;
    private final int externalFileAttribute;

    private DirectoryEntryInfo(short madeByVersion, int externalFileAttribute) {
      this.madeByVersion = madeByVersion;
      this.externalFileAttribute = externalFileAttribute;
    }

    /**
     * This will be written as "made by" version in the central directory.
     * If -1 (default) then "made by" will be the same to version "needed to extract".
     */
    public DirectoryEntryInfo withMadeByVersion(short madeByVersion) {
      return new DirectoryEntryInfo(madeByVersion, externalFileAttribute);
    }

    /**
     * This will be written as external file attribute. The meaning of this depends upon the value
     * set with {@link #withMadeByVersion(short)}. If that value indicates a Unix source, then this
     * value has the file mode and permission bits in the upper two bytes (e.g. possibly
     * {@code 0100644} for a regular file).
     */
    public DirectoryEntryInfo withExternalFileAttribute(int externalFileAttribute) {
      return new DirectoryEntryInfo(madeByVersion, externalFileAttribute);
    }
  }

  /**
   * The central directory, which is grown as required; instead of using a single large buffer, we
   * store a sequence of smaller buffers. With a single large buffer, whenever we grow the buffer by
   * 2x, we end up requiring 3x the memory temporarily, which can lead to OOM problems even if there
   * would still be enough memory.
   *
   * <p>The invariants for the fields are as follows:
   * <ul>
   *   <li>All blocks must have the same size.
   *   <li>The list of blocks must contain all blocks, including the current block (even if empty).
   *   <li>The current block offset must apply to the last block in the list, which is
   *       simultaneously the current block.
   *   <li>The current block may only be {@code null} if the list is empty.
   * </ul>
   */
  private static final class CentralDirectory {
    private final int blockSize; // We allow this to be overridden for testing.
    private List<byte[]> blockList = new ArrayList<>();
    private byte[] currentBlock;
    private int currentBlockOffset = 0;
    private int size = 0;

    CentralDirectory(int centralDirectoryBlockSize) {
      this.blockSize = centralDirectoryBlockSize;
    }

    /**
     * Appends the given data to the central directory and returns the start
     * offset within the central directory to allow back-patching.
     */
    int writeToCentralDirectory(byte[] b, int off, int len) {
      checkArgument(len >= 0);
      int offsetStarted = size;
      while (len > 0) {
        if (currentBlock == null
            || currentBlockOffset >= currentBlock.length) {
          currentBlock = new byte[blockSize];
          currentBlockOffset = 0;
          blockList.add(currentBlock);
        }
        int maxCopy = Math.min(blockSize - currentBlockOffset, len);
        System.arraycopy(b, off, currentBlock, currentBlockOffset, maxCopy);
        off += maxCopy;
        len -= maxCopy;
        size += maxCopy;
        currentBlockOffset += maxCopy;
      }
      return offsetStarted;
    }

    /** Calls through to {@link #writeToCentralDirectory(byte[], int, int)}. */
    int writeToCentralDirectory(byte[] b) {
      return writeToCentralDirectory(b, 0, b.length);
    }

    /**
     * Writes an unsigned int in little-endian byte order to the central directory at the
     * given offset. Does not perform range checking.
     */
    void setUnsignedInt(int offset, int value) {
      blockList.get(cdIndex(offset + 0))[cdOffset(offset + 0)] = (byte) (value & 0xff);
      blockList.get(cdIndex(offset + 1))[cdOffset(offset + 1)] = (byte) ((value >> 8) & 0xff);
      blockList.get(cdIndex(offset + 2))[cdOffset(offset + 2)] = (byte) ((value >> 16) & 0xff);
      blockList.get(cdIndex(offset + 3))[cdOffset(offset + 3)] = (byte) ((value >> 24) & 0xff);
    }

    private int cdIndex(int offset) {
      return offset / blockSize;
    }

    private int cdOffset(int offset) {
      return offset % blockSize;
    }

    /**
     * Writes the central directory to the given output stream and returns the size, i.e., the
     * number of bytes written.
     */
    int writeTo(OutputStream out) throws IOException {
      for (int i = 0; i < blockList.size() - 1; i++) {
        out.write(blockList.get(i));
      }
      if (currentBlock != null) {
        out.write(currentBlock, 0, currentBlockOffset);
      }
      return size;
    }
  }

  /**
   * An output stream that counts how many bytes were written.
   */
  private static final class ByteCountingOutputStream extends FilterOutputStream {
    private long bytesWritten = 0L;

    ByteCountingOutputStream(OutputStream out) {
      super(out);
    }

    @Override
    public void write(byte[] b, int off, int len) throws IOException {
      out.write(b, off, len);
      bytesWritten += len;
    }

    @Override
    public void write(int b) throws IOException {
      out.write(b);
      bytesWritten++;
    }
  }

  private final OutputMode mode;
  private final ZipEntryFilter entryFilter;

  private final ByteCountingOutputStream out;

  // An input buffer to allow reading blocks of data. Keeping it here avoids
  // another copy operation that would be required by the BufferedInputStream.
  // The valid data is between bufferOffset and bufferOffset+bufferLength (exclusive).
  private final byte[] buffer = new byte[BUFFER_SIZE];
  private int bufferOffset = 0;
  private int bufferLength = 0;

  private String currentInputFile;

  // An intermediate buffer for the file header data. Keeping it here avoids
  // creating a new buffer for every entry.
  private final byte[] headerBuffer = new byte[FILE_HEADER_BUFFER_SIZE];

  // An intermediate buffer for a central directory entry. Keeping it here
  // avoids creating a new buffer for every entry.
  private final byte[] directoryEntryBuffer = new byte[DIRECTORY_ENTRY_BUFFER_SIZE];

  // The Inflater is a class member to avoid creating a new instance for every
  // entry in the ZIP file.
  private final Inflater inflater = new Inflater(true);

  // The contents of this buffer are never read. The Inflater is only used to
  // determine the length of the compressed data, and the buffer is a throw-
  // away buffer for the decompressed data.
  private final byte[] inflaterBuffer = new byte[BUFFER_SIZE];

  private final Map<String, FileEntry> fileNames = new HashMap<>();

  private final CentralDirectory centralDirectory;
  private int fileCount = 0;

  private boolean finished = false;

  // Package private for testing.
  ZipCombiner(OutputMode mode, ZipEntryFilter entryFilter, OutputStream out,
      int centralDirectoryBlockSize) {
    this.mode = mode;
    this.entryFilter = entryFilter;
    this.out = new ByteCountingOutputStream(new BufferedOutputStream(out));
    this.centralDirectory = new CentralDirectory(centralDirectoryBlockSize);
  }

  /**
   * Creates a new instance with the given parameters. The {@code entryFilter}
   * is called for every entry in the ZIP files and the combined ZIP file is
   * written to {@code out}. The output mode determines whether entries must be
   * written in compressed or decompressed form. Note that the result is
   * invalid if an exception is thrown from any of the methods in this class,
   * and before a call to {@link #close} or {@link #finish}.
   */
  public ZipCombiner(OutputMode mode, ZipEntryFilter entryFilter, OutputStream out) {
    this(mode, entryFilter, out, DEFAULT_CENTRAL_DIRECTORY_BLOCK_SIZE);
  }

  /**
   * Creates a new instance with the given parameters and the DONT_CARE mode.
   */
  public ZipCombiner(ZipEntryFilter entryFilter, OutputStream out) {
    this(OutputMode.DONT_CARE, entryFilter, out);
  }

  /**
   * Creates a new instance with the {@link CopyEntryFilter} as the filter and
   * the given mode and output stream.
   */
  public ZipCombiner(OutputMode mode, OutputStream out) {
    this(mode, new CopyEntryFilter(), out);
  }

  /**
   * Creates a new instance with the {@link CopyEntryFilter} as the filter, the
   * DONT_CARE mode and the given output stream.
   */
  public ZipCombiner(OutputStream out) {
    this(OutputMode.DONT_CARE, new CopyEntryFilter(), out);
  }

  /**
   * Returns whether the output zip already contains a file or directory with
   * the given name.
   */
  public boolean containsFile(String filename) {
    return fileNames.containsKey(filename);
  }

  /**
   * Makes a write call to the output stream, and updates the current offset.
   */
  private void write(byte[] b, int off, int len) throws IOException {
    out.write(b, off, len);
  }

  /** Calls through to {@link #write(byte[], int, int)}. */
  private void write(byte[] b) throws IOException {
    write(b, 0, b.length);
  }

  /**
   * Reads at least one more byte into the internal buffer. This method must
   * only be called when more data is necessary to correctly decode the ZIP
   * format.
   *
   * <p>This method automatically compacts the existing data in the buffer by
   * moving it to the beginning of the buffer.
   *
   * @throws EOFException if no more data is available from the input stream
   * @throws IOException if the underlying stream throws one
   */
  private void readMoreData(InputStream in) throws IOException {
    if ((bufferLength > 0) && (bufferOffset > 0)) {
      System.arraycopy(buffer, bufferOffset, buffer, 0, bufferLength);
    }
    if (bufferLength >= buffer.length) {
      // The buffer size is specifically chosen to avoid this situation.
      throw new AssertionError("Internal error: buffer overrun.");
    }
    bufferOffset = 0;
    int bytesRead = in.read(buffer, bufferLength, buffer.length - bufferLength);
    if (bytesRead <= 0) {
      throw new EOFException();
    }
    bufferLength += bytesRead;
  }

  /**
   * Reads data until the buffer is filled with at least {@code length} bytes.
   *
   * @throws IllegalArgumentException if not 0 <= length <= buffer.length
   * @throws IOException if the underlying input stream throws one or the end
   *                     of the input stream is reached before the required
   *                     number of bytes is read
   */
  private void readFully(InputStream in, int length) throws IOException {
    checkArgument(length >= 0, "length too small: %s", length);
    checkArgument(length <= buffer.length, "length too large: %s", length);
    while (bufferLength < length) {
      readMoreData(in);
    }
  }

  /**
   * Reads an unsigned short in little-endian byte order from the buffer at the
   * given offset. Does not perform range checking.
   */
  private int getUnsignedShort(byte[] source, int offset) {
    int a = source[offset + 0] & 0xff;
    int b = source[offset + 1] & 0xff;
    return (b << 8) | a;
  }

  /**
   * Reads an unsigned int in little-endian byte order from the buffer at the
   * given offset. Does not perform range checking.
   */
  private long getUnsignedInt(byte[] source, int offset) {
    int a = source[offset + 0] & 0xff;
    int b = source[offset + 1] & 0xff;
    int c = source[offset + 2] & 0xff;
    int d = source[offset + 3] & 0xff;
    return ((d << 24) | (c << 16) | (b << 8) | a) & 0xffffffffL;
  }

  /**
   * Writes an unsigned short in little-endian byte order to the buffer at the
   * given offset. Does not perform range checking.
   */
  private void setUnsignedShort(byte[] target, int offset, short value) {
    target[offset + 0] = (byte) (value & 0xff);
    target[offset + 1] = (byte) ((value >> 8) & 0xff);
  }

  /**
   * Writes an unsigned int in little-endian byte order to the buffer at the
   * given offset. Does not perform range checking.
   */
  private void setUnsignedInt(byte[] target, int offset, int value) {
    target[offset + 0] = (byte) (value & 0xff);
    target[offset + 1] = (byte) ((value >> 8) & 0xff);
    target[offset + 2] = (byte) ((value >> 16) & 0xff);
    target[offset + 3] = (byte) ((value >> 24) & 0xff);
  }

  /**
   * Copies or skips {@code length} amount of bytes from the input stream to the
   * output stream. If the internal buffer is not empty, those bytes are copied
   * first. When the method returns, there may be more bytes remaining in the
   * buffer.
   *
   * @throws IOException if the underlying stream throws one
   */
  private void copyOrSkipData(InputStream in, long length, SkipMode skip) throws IOException {
    checkArgument(length >= 0);
    while (length > 0) {
      if (bufferLength == 0) {
        readMoreData(in);
      }
      int bytesToWrite = (length < bufferLength) ? (int) length : bufferLength;
      if (skip == SkipMode.COPY) {
        write(buffer, bufferOffset, bytesToWrite);
      }
      bufferOffset += bytesToWrite;
      bufferLength -= bytesToWrite;
      length -= bytesToWrite;
    }
  }

  /**
   * Copies or skips {@code length} amount of bytes from the input stream to the
   * output stream. If the internal buffer is not empty, those bytes are copied
   * first. When the method returns, there may be more bytes remaining in the
   * buffer. In addition to writing to the output stream, it also writes to the
   * central directory.
   *
   * @throws IOException if the underlying stream throws one
   */
  private void forkOrSkipData(InputStream in, long length, SkipMode skip) throws IOException {
    checkArgument(length >= 0);
    while (length > 0) {
      if (bufferLength == 0) {
        readMoreData(in);
      }
      int bytesToWrite = (length < bufferLength) ? (int) length : bufferLength;
      if (skip == SkipMode.COPY) {
        write(buffer, bufferOffset, bytesToWrite);
        centralDirectory.writeToCentralDirectory(buffer, bufferOffset, bytesToWrite);
      }
      bufferOffset += bytesToWrite;
      bufferLength -= bytesToWrite;
      length -= bytesToWrite;
    }
  }

  /**
   * A mutable integer reference value to allow returning two values from a
   * method.
   */
  private static class MutableInt {

    private int value;

    MutableInt(int initialValue) {
      this.value = initialValue;
    }

    public void setValue(int value) {
      this.value = value;
    }

    public int getValue() {
      return value;
    }
  }

  /**
   * Uses the inflater to decompress some data into the given buffer. This
   * method performs no error checking on the input parameters and also does
   * not update the buffer parameters of the input buffer (such as bufferOffset
   * and bufferLength). It's only here to avoid code duplication.
   *
   * <p>The Inflater may not be in the finished state when this method is
   * called.
   *
   * <p>This method returns 0 if it read data and reached the end of the
   * DEFLATE stream without producing output. In that case, {@link
   * Inflater#finished} is guaranteed to return true.
   *
   * @throws IOException if the underlying stream throws an IOException or if
   *                     illegal data is encountered
   */
  private int inflateData(InputStream in, byte[] dest, int off, int len, MutableInt consumed)
      throws IOException {
    // Defend against Inflater.finished() returning true.
    consumed.setValue(0);
    int bytesProduced = 0;
    int bytesConsumed = 0;
    while ((bytesProduced == 0) && !inflater.finished()) {
      inflater.setInput(buffer, bufferOffset + bytesConsumed, bufferLength - bytesConsumed);
      int remainingBefore = inflater.getRemaining();
      try {
        bytesProduced = inflater.inflate(dest, off, len);
      } catch (DataFormatException e) {
        throw new IOException("Invalid deflate stream in ZIP file.", e);
      }
      bytesConsumed += remainingBefore - inflater.getRemaining();
      consumed.setValue(bytesConsumed);
      if (bytesProduced == 0) {
        if (inflater.needsDictionary()) {
          // The DEFLATE algorithm as used in the ZIP file format does not
          // require an additional dictionary.
          throw new AssertionError("Inflater unexpectedly requires a dictionary.");
        } else if (inflater.needsInput()) {
          readMoreData(in);
        } else if (inflater.finished()) {
          return 0;
        } else {
          // According to the Inflater specification, this cannot happen.
          throw new AssertionError("Inflater unexpectedly produced no output.");
        }
      }
    }
    return bytesProduced;
  }

  /**
   * Copies or skips data from the input stream to the output stream. To
   * determine the length of the data, the data is decompressed with the
   * DEFLATE algorithm, which stores the length implicitly as part of the
   * compressed data, using a combination of end markers and length indicators.
   *
   * @see <a href="http://www.ietf.org/rfc/rfc1951.txt">RFC 1951</a>
   *
   * @throws IOException if the underlying stream throws an IOException
   */
  private long copyOrSkipDeflateData(InputStream in, SkipMode skip) throws IOException {
    long bytesCopied = 0;
    inflater.reset();
    MutableInt consumedBytes = new MutableInt(0);
    while (!inflater.finished()) {
      // Neither the uncompressed data nor the length of it is used. The
      // decompression is only required to determine the correct length of the
      // compressed data to copy.
      inflateData(in, inflaterBuffer, 0, inflaterBuffer.length, consumedBytes);
      int bytesRead = consumedBytes.getValue();
      if (skip == SkipMode.COPY) {
        write(buffer, bufferOffset, bytesRead);
      }
      bufferOffset += bytesRead;
      bufferLength -= bytesRead;
      bytesCopied += bytesRead;
    }
    return bytesCopied;
  }

  /**
   * Returns a 32-bit integer containing a ZIP-compatible encoding of the given
   * date. Only dates between 1980 and 2107 (inclusive) are supported.
   *
   * <p>The upper 16 bits contain the year, month, and day. The lower 16 bits
   * contain the hour, minute, and second. The resolution of the second field
   * is only 4 bits, which means that the only even second values can be
   * stored - this method rounds down to the nearest even value.
   *
   * @throws IllegalArgumentException if the given date is outside the
   *                                  supported range
   */
  // Only visible for testing.
  static int dateToDosTime(Date date) {
    Calendar calendar = new GregorianCalendar();
    calendar.setTime(date);
    int year = calendar.get(Calendar.YEAR);
    if (year < 1980) {
      throw new IllegalArgumentException("date must be in or after 1980");
    }
    // The ZIP format only provides 7 bits for the year.
    if (year > 2107) {
      throw new IllegalArgumentException("date must before 2107");
    }
    int month = calendar.get(Calendar.MONTH) + 1; // Months from Calendar are zero-based.
    int day = calendar.get(Calendar.DAY_OF_MONTH);
    int hour = calendar.get(Calendar.HOUR_OF_DAY);
    int minute = calendar.get(Calendar.MINUTE);
    int second = calendar.get(Calendar.SECOND);
    return ((year - 1980) << 25) | (month << 21) | (day << 16)
        | (hour << 11) | (minute << 5) | (second >> 1);
  }

  /**
   * Fills the directory entry, using the information from the header buffer,
   * and writes it to the central directory. It returns the offset into the
   * central directory that can be used for patching the entry. Requires that
   * the entire entry header is present in {@link #headerBuffer}. It also uses
   * the {@link ByteCountingOutputStream#bytesWritten}, so it must be called
   * just before the header is written to the output stream.
   *
   * @throws IOException if the current offset is too large for the ZIP format
   */
  private int fillDirectoryEntryBuffer(
      DirectoryEntryInfo directoryEntryInfo) throws IOException {
    // central file header signature
    setUnsignedInt(directoryEntryBuffer, 0, CENTRAL_DIRECTORY_MARKER);
    short version = (short) getUnsignedShort(headerBuffer, VERSION_TO_EXTRACT_OFFSET);
    short curMadeMyVersion = (directoryEntryInfo.madeByVersion == -1)
        ? version : directoryEntryInfo.madeByVersion;
    setUnsignedShort(directoryEntryBuffer, 4, curMadeMyVersion); // version made by
    // version needed to extract
    setUnsignedShort(directoryEntryBuffer, 6, version);
    // general purpose bit flag
    setUnsignedShort(directoryEntryBuffer, 8,
        (short) getUnsignedShort(headerBuffer, GENERAL_PURPOSE_FLAGS_OFFSET));
    // compression method
    setUnsignedShort(directoryEntryBuffer, 10,
        (short) getUnsignedShort(headerBuffer, COMPRESSION_METHOD_OFFSET));
    // last mod file time, last mod file date
    setUnsignedShort(directoryEntryBuffer, 12,
        (short) getUnsignedShort(headerBuffer, MTIME_OFFSET));
    setUnsignedShort(directoryEntryBuffer, 14,
        (short) getUnsignedShort(headerBuffer, MDATE_OFFSET));
    // crc-32
    setUnsignedInt(directoryEntryBuffer, 16, (int) getUnsignedInt(headerBuffer, CRC32_OFFSET));
    // compressed size
    setUnsignedInt(directoryEntryBuffer, 20,
        (int) getUnsignedInt(headerBuffer, COMPRESSED_SIZE_OFFSET));
    // uncompressed size
    setUnsignedInt(directoryEntryBuffer, 24,
        (int) getUnsignedInt(headerBuffer, UNCOMPRESSED_SIZE_OFFSET));
    // file name length
    setUnsignedShort(directoryEntryBuffer, 28,
        (short) getUnsignedShort(headerBuffer, FILENAME_LENGTH_OFFSET));
    // extra field length
    setUnsignedShort(directoryEntryBuffer, 30,
        (short) getUnsignedShort(headerBuffer, EXTRA_LENGTH_OFFSET));
    setUnsignedShort(directoryEntryBuffer, 32, (short) 0); // file comment length
    setUnsignedShort(directoryEntryBuffer, 34, (short) 0); // disk number start
    setUnsignedShort(directoryEntryBuffer, 36, (short) 0); // internal file attributes
    setUnsignedInt(directoryEntryBuffer, 38, directoryEntryInfo.externalFileAttribute);
    if (out.bytesWritten >= MAXIMUM_DATA_SIZE) {
      throw new IOException("Unable to handle files bigger than 2^32 bytes.");
    }
    // relative offset of local header
    setUnsignedInt(directoryEntryBuffer, 42, (int) out.bytesWritten);
    fileCount++;
    return centralDirectory.writeToCentralDirectory(directoryEntryBuffer);
  }

  /**
   * Fix the directory entry with the correct crc32, compressed size, and
   * uncompressed size.
   */
  private void fixDirectoryEntry(int offset, long crc32, long compressedSize,
      long uncompressedSize) {
    // The constants from the top don't apply here, because this is the central directory entry.
    centralDirectory.setUnsignedInt(offset + 16, (int) crc32); // crc-32
    centralDirectory.setUnsignedInt(offset + 20, (int) compressedSize); // compressed size
    centralDirectory.setUnsignedInt(offset + 24, (int) uncompressedSize); // uncompressed size
  }

  /**
   * (Un)Compresses and copies the current ZIP file entry. Requires that the
   * entire entry header is present in {@link #headerBuffer}. It currently
   * drops the extra data in the process.
   *
   * @throws IOException if the underlying stream throws an IOException
   */
  private void modifyAndCopyEntry(String filename, InputStream in, int dosTime)
      throws IOException {
    final int method = getUnsignedShort(headerBuffer, COMPRESSION_METHOD_OFFSET);
    final int flags = getUnsignedShort(headerBuffer, GENERAL_PURPOSE_FLAGS_OFFSET);
    final int fileNameLength = getUnsignedShort(headerBuffer, FILENAME_LENGTH_OFFSET);
    final int extraFieldLength = getUnsignedShort(headerBuffer, EXTRA_LENGTH_OFFSET);
    // TODO(bazel-team): Read and copy the extra data if present.

    forkOrSkipData(in, fileNameLength, SkipMode.SKIP);
    forkOrSkipData(in, extraFieldLength, SkipMode.SKIP);
    if (method == STORED_METHOD) {
      long compressedSize = getUnsignedInt(headerBuffer, COMPRESSED_SIZE_OFFSET);
      copyStreamToEntry(filename, new FixedLengthInputStream(in, compressedSize), dosTime,
          NO_EXTRA_ENTRIES, true, DEFAULT_DIRECTORY_ENTRY_INFO);
    } else if (method == DEFLATE_METHOD) {
      inflater.reset();
      copyStreamToEntry(filename, new DeflateInputStream(in), dosTime, NO_EXTRA_ENTRIES, false,
          DEFAULT_DIRECTORY_ENTRY_INFO);
      if ((flags & SIZE_MASKED_FLAG) != 0) {
        copyOrSkipData(in, 16, SkipMode.SKIP);
      }
    } else {
      throw new AssertionError("This should have been checked in validateHeader().");
    }
  }

  /**
   * Copies or skips the current ZIP file entry. Requires that the entire entry
   * header is present in {@link #headerBuffer}. It uses the current mode to
   * decide whether to compress or decompress the entry.
   *
   * @throws IOException if the underlying stream throws an IOException
   */
  private void copyOrSkipEntry(String filename, InputStream in, SkipMode skip, Date date,
      DirectoryEntryInfo directoryEntryInfo) throws IOException {
    copyOrSkipEntry(filename, in, skip, date, directoryEntryInfo, false);
  }

  /**
   * Renames and otherwise copies the current ZIP file entry. Requires that the entire
   * entry header is present in {@link #headerBuffer}. It uses the current mode to
   * decide whether to compress or decompress the entry.
   *
   * @throws IOException if the underlying stream throws an IOException
   */
  private void renameEntry(String filename, InputStream in, Date date,
      DirectoryEntryInfo directoryEntryInfo) throws IOException {
    copyOrSkipEntry(filename, in, SkipMode.COPY, date, directoryEntryInfo, true);
  }

  /**
   * Copies or skips the current ZIP file entry. Requires that the entire entry
   * header is present in {@link #headerBuffer}. It uses the current mode to
   * decide whether to compress or decompress the entry.
   *
   * @throws IOException if the underlying stream throws an IOException
   */
  private void copyOrSkipEntry(String filename, InputStream in, SkipMode skip, Date date,
      DirectoryEntryInfo directoryEntryInfo, boolean rename) throws IOException {
    final int method = getUnsignedShort(headerBuffer, COMPRESSION_METHOD_OFFSET);

    // We can cast here, because the result is only treated as a bitmask.
    int dosTime = date == null ? (int) getUnsignedInt(headerBuffer, MTIME_OFFSET)
        : dateToDosTime(date);
    if (skip == SkipMode.COPY) {
      if ((mode == OutputMode.FORCE_DEFLATE) && (method == STORED_METHOD)
          && !filename.endsWith("/")) {
        modifyAndCopyEntry(filename, in, dosTime);
        return;
      } else if ((mode == OutputMode.FORCE_STORED) && (method == DEFLATE_METHOD)) {
        modifyAndCopyEntry(filename, in, dosTime);
        return;
      }
    }

    int directoryOffset = copyOrSkipEntryHeader(filename, in, date, directoryEntryInfo,
        skip, rename);

    copyOrSkipEntryData(filename, in, skip, directoryOffset);
  }

  /**
   * Copies or skips the header of an entry, including filename and extra data.
   * Requires that the entire entry header is present in {@link #headerBuffer}.
   *
   * @returns the enrty offset in the central directory
   * @throws IOException if the underlying stream throws an IOException
   */
  private int copyOrSkipEntryHeader(String filename, InputStream in, Date date,
      DirectoryEntryInfo directoryEntryInfo, SkipMode skip, boolean rename)
      throws IOException {
    final int fileNameLength = getUnsignedShort(headerBuffer, FILENAME_LENGTH_OFFSET);
    final int extraFieldLength = getUnsignedShort(headerBuffer, EXTRA_LENGTH_OFFSET);

    byte[] fileNameAsBytes = null;
    if (rename) {
      // If the entry is renamed, we patch the filename length in the buffer
      // before it's copied, and before writing to the central directory.
      fileNameAsBytes = filename.getBytes(UTF_8);
      checkArgument(fileNameAsBytes.length <= 65535,
          "File name too long: %s bytes (max. 65535)", fileNameAsBytes.length);
      setUnsignedShort(headerBuffer, FILENAME_LENGTH_OFFSET, (short) fileNameAsBytes.length);
    }

    int directoryOffset = 0;
    if (skip == SkipMode.COPY) {
      if (date != null) {
        int dosTime = dateToDosTime(date);
        setUnsignedShort(headerBuffer, MTIME_OFFSET, (short) dosTime); // lower 16 bits
        setUnsignedShort(headerBuffer, MDATE_OFFSET, (short) (dosTime >> 16)); // upper 16 bits
      }
      // Call this before writing the data out, so that we get the correct offset.
      directoryOffset = fillDirectoryEntryBuffer(directoryEntryInfo);
      write(headerBuffer, 0, FILE_HEADER_BUFFER_SIZE);
    }
    if (!rename) {
      forkOrSkipData(in, fileNameLength, skip);
    } else {
      forkOrSkipData(in, fileNameLength, SkipMode.SKIP);
      write(fileNameAsBytes);
      centralDirectory.writeToCentralDirectory(fileNameAsBytes);
    }
    forkOrSkipData(in, extraFieldLength, skip);
    return directoryOffset;
  }

  /**
   * Copy or skip the data of an entry. Requires that the
   * entire entry header is present in {@link #headerBuffer}.
   *
   * @throws IOException if the underlying stream throws an IOException
   */
  private void copyOrSkipEntryData(String filename, InputStream in, SkipMode skip,
      int directoryOffset) throws IOException {
    final int flags = getUnsignedShort(headerBuffer, GENERAL_PURPOSE_FLAGS_OFFSET);
    final int method = getUnsignedShort(headerBuffer, COMPRESSION_METHOD_OFFSET);
    if ((flags & SIZE_MASKED_FLAG) != 0) {
      // The compressed data size is unknown.
      if (method != DEFLATE_METHOD) {
        throw new AssertionError("This should have been checked in validateHeader().");
      }
      copyOrSkipDeflateData(in, skip);
      // The flags indicate that a data descriptor must follow the data.
      readFully(in, 16);
      if (getUnsignedInt(buffer, bufferOffset) != DATA_DESCRIPTOR_MARKER) {
        throw new IOException("Missing data descriptor for " + filename + " in " + currentInputFile
            + ".");
      }
      long crc32 = getUnsignedInt(buffer, bufferOffset + 4);
      long compressedSize = getUnsignedInt(buffer, bufferOffset + 8);
      long uncompressedSize = getUnsignedInt(buffer, bufferOffset + 12);
      if (skip == SkipMode.COPY) {
        fixDirectoryEntry(directoryOffset, crc32, compressedSize, uncompressedSize);
      }
      copyOrSkipData(in, 16, skip);
    } else {
      // The size value is present in the header, so just copy that amount.
      long compressedSize = getUnsignedInt(headerBuffer, COMPRESSED_SIZE_OFFSET);
      copyOrSkipData(in, compressedSize, skip);
    }
  }

  /**
   * An input stream that reads a fixed number of bytes from the given input
   * stream before it returns end-of-input. It uses the local buffer, so it
   * can't be static.
   */
  private class FixedLengthInputStream extends InputStream {

    private final InputStream in;
    private long remainingBytes;
    private final byte[] singleByteBuffer = new byte[1];

    FixedLengthInputStream(InputStream in, long remainingBytes) {
      this.in = in;
      this.remainingBytes = remainingBytes;
    }

    @Override
    public int read() throws IOException {
      int bytesRead = read(singleByteBuffer, 0, 1);
      return (bytesRead == -1) ? -1 : singleByteBuffer[0];
    }

    @Override
    public int read(byte b[], int off, int len) throws IOException {
      checkArgument(len >= 0);
      checkArgument(off >= 0);
      checkArgument(off + len <= b.length);
      if (remainingBytes == 0) {
        return -1;
      }
      if (bufferLength == 0) {
        readMoreData(in);
      }
      int bytesToCopy = len;
      if (remainingBytes < bytesToCopy) {
        bytesToCopy = (int) remainingBytes;
      }
      if (bufferLength < bytesToCopy) {
        bytesToCopy = bufferLength;
      }
      System.arraycopy(buffer, bufferOffset, b, off, bytesToCopy);
      bufferOffset += bytesToCopy;
      bufferLength -= bytesToCopy;
      remainingBytes -= bytesToCopy;
      return bytesToCopy;
    }
  }

  /**
   * An input stream that reads from a given input stream, decoding that data
   * according to the DEFLATE algorithm. The DEFLATE data stream implicitly
   * contains its own end-of-input marker. It uses the local buffer, so it
   * can't be static.
   */
  private class DeflateInputStream extends InputStream {

    private final InputStream in;
    private final byte[] singleByteBuffer = new byte[1];
    private final MutableInt consumedBytes = new MutableInt(0);

    DeflateInputStream(InputStream in) {
      this.in = in;
    }

    @Override
    public int read() throws IOException {
      int bytesRead = read(singleByteBuffer, 0, 1);
      // Do an unsigned cast on the byte from the buffer if it exists.
      return (bytesRead == -1) ? -1 : (singleByteBuffer[0] & 0xff);
    }

    @Override
    public int read(byte b[], int off, int len) throws IOException {
      if (inflater.finished()) {
        return -1;
      }
      int length = inflateData(in, b, off, len, consumedBytes);
      int bytesRead = consumedBytes.getValue();
      bufferOffset += bytesRead;
      bufferLength -= bytesRead;
      return length == 0 ? -1 : length;
    }
  }

  /**
   * Handles a custom merge operation with the given strategy. This method
   * creates an appropriate input stream and hands it to the strategy for
   * processing. Requires that the entire entry header is present in {@link
   * #headerBuffer}.
   *
   * @throws IOException if one of the underlying stream throws an IOException,
   *                     if the ZIP entry data is inconsistent, or if the
   *                     implementation cannot handle the compression method
   *                     given in the ZIP entry
   */
  private void handleCustomMerge(final InputStream in, CustomMergeStrategy mergeStrategy,
      ByteArrayOutputStream outputBuffer) throws IOException {
    final int flags = getUnsignedShort(headerBuffer, GENERAL_PURPOSE_FLAGS_OFFSET);
    final int method = getUnsignedShort(headerBuffer, COMPRESSION_METHOD_OFFSET);
    final long compressedSize = getUnsignedInt(headerBuffer, COMPRESSED_SIZE_OFFSET);

    final int fileNameLength = getUnsignedShort(headerBuffer, FILENAME_LENGTH_OFFSET);
    final int extraFieldLength = getUnsignedShort(headerBuffer, EXTRA_LENGTH_OFFSET);

    copyOrSkipData(in, fileNameLength, SkipMode.SKIP);
    copyOrSkipData(in, extraFieldLength, SkipMode.SKIP);
    if (method == STORED_METHOD) {
      mergeStrategy.merge(new FixedLengthInputStream(in, compressedSize), outputBuffer);
    } else if (method == DEFLATE_METHOD) {
      inflater.reset();
      // TODO(bazel-team): Defend against the mergeStrategy not reading the complete input.
      mergeStrategy.merge(new DeflateInputStream(in), outputBuffer);
      if ((flags & SIZE_MASKED_FLAG) != 0) {
        copyOrSkipData(in, 16, SkipMode.SKIP);
      }
    } else {
      throw new AssertionError("This should have been checked in validateHeader().");
    }
  }

  /**
   * Implementation of the strategy callback.
   */
  private class TheStrategyCallback implements StrategyCallback {

    private String filename;
    private final InputStream in;

    // Use an atomic boolean to make sure that only a single call goes
    // through, even if there are multiple concurrent calls. Paranoid
    // defensive programming.
    private final AtomicBoolean callDone = new AtomicBoolean();

    TheStrategyCallback(String filename, InputStream in) {
      this.filename = filename;
      this.in = in;
    }

    // Verify that this is the first call and throw an exception if not.
    private void checkCall() {
      checkState(callDone.compareAndSet(false, true), "The callback was already called once.");
    }

    @Override
    public void copy(Date date) throws IOException {
      checkCall();
      if (!containsFile(filename)) {
        fileNames.put(filename, COPIED_FILE_ENTRY);
        copyOrSkipEntry(filename, in, SkipMode.COPY, date, DEFAULT_DIRECTORY_ENTRY_INFO);
      } else { // can't copy, name already used for renamed entry
        copyOrSkipEntry(filename, in, SkipMode.SKIP, null, DEFAULT_DIRECTORY_ENTRY_INFO);
      }
    }

    @Override
    public void rename(String newName, Date date) throws IOException {
      checkCall();
      if (!containsFile(newName)) {
        fileNames.put(newName, RENAMED_FILE_ENTRY);
        renameEntry(newName, in, date, DEFAULT_DIRECTORY_ENTRY_INFO);
      } else {
        copyOrSkipEntry(filename, in, SkipMode.SKIP, null, DEFAULT_DIRECTORY_ENTRY_INFO);
      }
      filename = newName;
    }

    @Override
    public void skip() throws IOException {
      checkCall();
      if (!containsFile(filename)) {// don't overwrite possible RENAMED_FILE_ENTRY value
        fileNames.put(filename, COPIED_FILE_ENTRY);
      }
      copyOrSkipEntry(filename, in, SkipMode.SKIP, null, DEFAULT_DIRECTORY_ENTRY_INFO);
    }

    @Override
    public void customMerge(Date date, CustomMergeStrategy strategy) throws IOException {
      checkCall();
      ByteArrayOutputStream outputBuffer = new ByteArrayOutputStream();
      fileNames.put(filename, new FileEntry(strategy, outputBuffer, dateToDosTime(date)));
      handleCustomMerge(in, strategy, outputBuffer);
    }
  }

  /**
   * Validates that the current entry obeys all the restrictions of this implementation.
   *
   * @throws IOException if the current entry doesn't obey the restrictions
   */
  private void validateHeader() throws IOException {
    // We only handle DEFLATE and STORED, like java.util.zip.
    final int method = getUnsignedShort(headerBuffer, COMPRESSION_METHOD_OFFSET);
    if ((method != DEFLATE_METHOD) && (method != STORED_METHOD)) {
      throw new IOException("Unable to handle compression methods other than DEFLATE!");
    }

    // If the method is STORED, then the size must be available in the header.
    final int flags = getUnsignedShort(headerBuffer, GENERAL_PURPOSE_FLAGS_OFFSET);
    if ((method == STORED_METHOD) && ((flags & SIZE_MASKED_FLAG) != 0)) {
      throw new IOException("If the method is STORED, then the size must be available in the"
          + " header!");
    }

    // If the method is STORED, the compressed and uncompressed sizes must be equal.
    final long compressedSize = getUnsignedInt(headerBuffer, COMPRESSED_SIZE_OFFSET);
    final long uncompressedSize = getUnsignedInt(headerBuffer, UNCOMPRESSED_SIZE_OFFSET);
    if ((method == STORED_METHOD) && (compressedSize != uncompressedSize)) {
      throw new IOException("Compressed and uncompressed sizes for STORED entry differ!");
    }

    // The compressed or uncompressed size being set to 0xffffffff is a strong indicator that the
    // ZIP file is in ZIP64 mode, which supports files larger than 2^32.
    // TODO(bazel-team): Support the ZIP64 extension.
    if ((compressedSize == MAXIMUM_DATA_SIZE) || (uncompressedSize == MAXIMUM_DATA_SIZE)) {
      throw new IOException("Unable to handle ZIP64 compressed files.");
    }
  }

  /**
   * Reads a file entry from the input stream, calls the entryFilter to
   * determine what to do with the entry, and performs the requested operation.
   * Returns true if the input stream contained another entry.
   *
   * @throws IOException if one of the underlying stream throws an IOException,
   *                     if the ZIP contains unsupported, inconsistent or
   *                     incomplete data or if the filter throws an IOException
   */
  private boolean handleNextEntry(final InputStream in) throws IOException {
    // Just try to read the complete header and fail if it didn't work.
    try {
      readFully(in, FILE_HEADER_BUFFER_SIZE);
    } catch (EOFException e) {
      return false;
    }

    System.arraycopy(buffer, bufferOffset, headerBuffer, 0, FILE_HEADER_BUFFER_SIZE);
    bufferOffset += FILE_HEADER_BUFFER_SIZE;
    bufferLength -= FILE_HEADER_BUFFER_SIZE;
    if (getUnsignedInt(headerBuffer, 0) != LOCAL_FILE_HEADER_MARKER) {
      return false;
    }
    validateHeader();

    final int fileNameLength = getUnsignedShort(headerBuffer, FILENAME_LENGTH_OFFSET);
    readFully(in, fileNameLength);
    // TODO(bazel-team): If I read the spec correctly, this should be UTF-8 rather than ISO-8859-1.
    final String filename = new String(buffer, bufferOffset, fileNameLength, ISO_8859_1);

    FileEntry handler = fileNames.get(filename);
    // The handler is null if this is the first time we see an entry with this filename,
    // or if all previous entries with this name were renamed by the filter (and we can
    // pretend we didn't encounter the name yet).
    // If the handler is RENAMED_FILE_ENTRY, a previous entry was renamed as filename,
    // in which case the filter should now be invoked for this name for the first time,
    // giving the filter a chance to choose an unique name.
    if (handler == null || handler == RENAMED_FILE_ENTRY) {
      TheStrategyCallback callback = new TheStrategyCallback(filename, in);
      entryFilter.accept(filename, callback);
      if (fileNames.get(callback.filename) == null && fileNames.get(filename) == null) {
        throw new IllegalStateException();
      }
    } else if (handler.mergeStrategy == null) {
      copyOrSkipEntry(filename, in, SkipMode.SKIP, null, DEFAULT_DIRECTORY_ENTRY_INFO);
    } else {
      handleCustomMerge(in, handler.mergeStrategy, handler.outputBuffer);
    }
    return true;
  }

  /**
   * Clears the internal buffer.
   */
  private void clearBuffer() {
    bufferOffset = 0;
    bufferLength = 0;
  }

  /**
   * Copies another ZIP file into the output. If multiple entries with the same
   * name are present, the first such entry is copied, but the others are
   * ignored. This is also true for multiple invocations of this method. The
   * {@code inputName} parameter is used to provide better error messages in the
   * case of a failure to decode the ZIP file.
   *
   * @throws IOException if one of the underlying stream throws an IOException,
   *                     if the ZIP contains unsupported, inconsistent or
   *                     incomplete data or if the filter throws an IOException
   */
  public void addZip(String inputName, InputStream in) throws IOException {
    if (finished) {
      throw new IllegalStateException();
    }
    if (in == null) {
      throw new NullPointerException();
    }
    clearBuffer();
    currentInputFile = inputName;
    while (handleNextEntry(in)) {/*handleNextEntry has side-effect.*/}
  }

  public void addZip(InputStream in) throws IOException {
    addZip(null, in);
  }

  private void copyStreamToEntry(String filename, InputStream in, int dosTime,
      ExtraData[] extraDataEntries, boolean compress, DirectoryEntryInfo directoryEntryInfo)
      throws IOException {
    fileNames.put(filename, COPIED_FILE_ENTRY);

    byte[] fileNameAsBytes = filename.getBytes(UTF_8);
    checkArgument(fileNameAsBytes.length <= 65535,
        "File name too long: %s bytes (max. 65535)", fileNameAsBytes.length);

    // Note: This method can be called with an input stream that uses the buffer field of this
    // class. We use a local buffer here to avoid conflicts.
    byte[] localBuffer = new byte[4096];

    byte[] uncompressedData = null;
    if (!compress) {
      ByteArrayOutputStream temp = new ByteArrayOutputStream();
      int bytesRead;
      while ((bytesRead = in.read(localBuffer)) != -1) {
        temp.write(localBuffer, 0, bytesRead);
      }
      uncompressedData = temp.toByteArray();
    }
    byte[] extraData = null;
    if (extraDataEntries.length != 0) {
      int totalLength = 0;
      for (ExtraData extra : extraDataEntries) {
        int length = extra.getData().length;
        if (totalLength > 0xffff - 4 - length) {
          throw new IOException("Total length of extra data too big.");
        }
        totalLength += length + 4;
      }
      extraData = new byte[totalLength];
      int position = 0;
      for (ExtraData extra : extraDataEntries) {
        byte[] data = extra.getData();
        setUnsignedShort(extraData, position + 0, extra.getId());
        setUnsignedShort(extraData, position + 2, (short) data.length);
        System.arraycopy(data, 0, extraData, position + 4, data.length);
        position += data.length + 4;
      }
    }

    // write header
    Arrays.fill(headerBuffer, (byte) 0);
    setUnsignedInt(headerBuffer, 0, LOCAL_FILE_HEADER_MARKER); // file header signature
    if (compress) {
      setUnsignedShort(headerBuffer, 4, (short) VERSION_DEFLATE); // version to extract
      setUnsignedShort(headerBuffer, 6, (short) SIZE_MASKED_FLAG); // general purpose bit flag
      setUnsignedShort(headerBuffer, 8, (short) DEFLATE_METHOD); // compression method
    } else {
      setUnsignedShort(headerBuffer, 4, (short) VERSION_STORED); // version to extract
      setUnsignedShort(headerBuffer, 6, (short) 0); // general purpose bit flag
      setUnsignedShort(headerBuffer, 8, (short) STORED_METHOD); // compression method
    }
    setUnsignedShort(headerBuffer, 10, (short) dosTime); // mtime
    setUnsignedShort(headerBuffer, 12, (short) (dosTime >> 16)); // mdate
    if (uncompressedData != null) {
      CRC32 crc = new CRC32();
      crc.update(uncompressedData);
      setUnsignedInt(headerBuffer, 14, (int) crc.getValue()); // crc32
      setUnsignedInt(headerBuffer, 18, uncompressedData.length); // compressed size
      setUnsignedInt(headerBuffer, 22, uncompressedData.length); // uncompressed size
    } else {
      setUnsignedInt(headerBuffer, 14, 0); // crc32
      setUnsignedInt(headerBuffer, 18, 0); // compressed size
      setUnsignedInt(headerBuffer, 22, 0); // uncompressed size
    }
    setUnsignedShort(headerBuffer, 26, (short) fileNameAsBytes.length); // file name length
    if (extraData != null) {
      setUnsignedShort(headerBuffer, 28, (short) extraData.length); // extra field length
    } else {
      setUnsignedShort(headerBuffer, 28, (short) 0); // extra field length
    }

    // This call works for both compressed or uncompressed entries.
    int directoryOffset = fillDirectoryEntryBuffer(directoryEntryInfo);
    write(headerBuffer);
    write(fileNameAsBytes);
    centralDirectory.writeToCentralDirectory(fileNameAsBytes);
    if (extraData != null) {
      write(extraData);
      centralDirectory.writeToCentralDirectory(extraData);
    }

    // write data
    if (uncompressedData != null) {
      write(uncompressedData);
    } else {
      try (DeflaterOutputStream deflaterStream = new DeflaterOutputStream()) {
        int bytesRead;
        while ((bytesRead = in.read(localBuffer)) != -1) {
          deflaterStream.write(localBuffer, 0, bytesRead);
        }
        deflaterStream.finish();

        // write data descriptor
        Arrays.fill(headerBuffer, (byte) 0);
        setUnsignedInt(headerBuffer, 0, DATA_DESCRIPTOR_MARKER);
        setUnsignedInt(headerBuffer, 4, deflaterStream.getCRC()); // crc32
        setUnsignedInt(headerBuffer, 8, deflaterStream.getCompressedSize()); // compressed size
        setUnsignedInt(headerBuffer, 12, deflaterStream.getUncompressedSize()); // uncompressed size
        write(headerBuffer, 0, 16);
        fixDirectoryEntry(directoryOffset, deflaterStream.getCRC(),
            deflaterStream.getCompressedSize(), deflaterStream.getUncompressedSize());
      }
    }
  }

  /**
   * Adds a new entry into the output, by reading the input stream until it
   * returns end of stream. Equivalent to
   * {@link #addFile(String, Date, InputStream, DirectoryEntryInfo)}, but uses
   * {@link #DEFAULT_DIRECTORY_ENTRY_INFO} for the file's directory entry.
   */
  public void addFile(String filename, Date date, InputStream in) throws IOException {
    addFile(filename, date, in, DEFAULT_DIRECTORY_ENTRY_INFO);
  }

  /**
   * Adds a new entry into the output, by reading the input stream until it
   * returns end of stream. This method does not call {@link
   * ZipEntryFilter#accept}.
   *
   * @throws IOException if one of the underlying streams throws an IOException
   *                     or if the input stream returns more data than
   *                     supported by the ZIP format
   * @throws IllegalStateException if an entry with the given name already
   *                               exists
   * @throws IllegalArgumentException if the given file name is longer than
   *                                  supported by the ZIP format
   */
  public void addFile(String filename, Date date, InputStream in,
      DirectoryEntryInfo directoryEntryInfo) throws IOException {
    checkNotFinished();
    if (in == null) {
      throw new NullPointerException();
    }
    if (filename == null) {
      throw new NullPointerException();
    }
    checkState(!fileNames.containsKey(filename),
        "jar already contains a file named %s", filename);
    int dosTime = dateToDosTime(date != null ? date : new Date());
    copyStreamToEntry(filename, in, dosTime, NO_EXTRA_ENTRIES,
        mode != OutputMode.FORCE_STORED, // Always compress if we're allowed to.
        directoryEntryInfo);
  }

  /**
   * Adds a new directory entry into the output. This method does not call
   * {@link ZipEntryFilter#accept}. Uses {@link #DEFAULT_DIRECTORY_ENTRY_INFO} for the added
   * directory entry.
   *
   * @throws IOException if one of the underlying streams throws an IOException
   * @throws IllegalStateException if an entry with the given name already
   *                               exists
   * @throws IllegalArgumentException if the given file name is longer than
   *                                  supported by the ZIP format
   */
  public void addDirectory(String filename, Date date, ExtraData[] extraDataEntries)
      throws IOException {
    checkNotFinished();
    checkArgument(filename.endsWith("/")); // Can also throw NPE.
    checkState(!fileNames.containsKey(filename),
        "jar already contains a directory named %s", filename);
    int dosTime = dateToDosTime(date != null ? date : new Date());
    copyStreamToEntry(filename, new ByteArrayInputStream(new byte[0]), dosTime, extraDataEntries,
        false, // Never compress directory entries.
        DEFAULT_DIRECTORY_ENTRY_INFO);
  }

  /**
   * Adds a new directory entry into the output. This method does not call
   * {@link ZipEntryFilter#accept}.
   *
   * @throws IOException if one of the underlying streams throws an IOException
   * @throws IllegalStateException if an entry with the given name already
   *                               exists
   * @throws IllegalArgumentException if the given file name is longer than
   *                                  supported by the ZIP format
   */
  public void addDirectory(String filename, Date date)
      throws IOException {
    addDirectory(filename, date, NO_EXTRA_ENTRIES);
  }

  /**
   * A deflater output stream that also counts uncompressed and compressed
   * numbers of bytes and computes the CRC so that the data descriptor marker
   * is written correctly.
   *
   * <p>Not static, so it can access the write() methods.
   */
  private class DeflaterOutputStream extends OutputStream {

    private final Deflater deflater = new Deflater(Deflater.DEFAULT_COMPRESSION, true);
    private final CRC32 crc = new CRC32();
    private final byte[] outputBuffer = new byte[4096];
    private long uncompressedBytes = 0;
    private long compressedBytes = 0;

    @Override
    public void write(int b) throws IOException {
      byte[] buf = new byte[] { (byte) (b & 0xff) };
      write(buf, 0, buf.length);
    }

    @Override
    public void write(byte b[], int off, int len) throws IOException {
      checkNotFinished();
      uncompressedBytes += len;
      crc.update(b, off, len);
      deflater.setInput(b, off, len);
      while (!deflater.needsInput()) {
        deflate();
      }
    }

    @Override
    public void close() throws IOException {
      super.close();
      deflater.end();
    }

    /**
     * Writes out the remaining buffered data without closing the output
     * stream.
     */
    public void finish() throws IOException {
      checkNotFinished();
      deflater.finish();
      while (!deflater.finished()) {
        deflate();
      }
      if ((compressedBytes >= MAXIMUM_DATA_SIZE) || (uncompressedBytes >= MAXIMUM_DATA_SIZE)) {
        throw new IOException("Too much data for ZIP entry.");
      }
    }

    private void deflate() throws IOException {
      int length = deflater.deflate(outputBuffer);
      ZipCombiner.this.write(outputBuffer, 0, length);
      compressedBytes += length;
    }

    public int getCRC() {
      return (int) crc.getValue();
    }

    public int getCompressedSize() {
      return (int) compressedBytes;
    }

    public int getUncompressedSize() {
      return (int) uncompressedBytes;
    }

    private void checkNotFinished() {
      if (deflater.finished()) {
        throw new IllegalStateException();
      }
    }
  }

  /**
   * Writes any remaining output data to the output stream and also creates the
   * merged entries by calling the {@link CustomMergeStrategy} implementations
   * given back from the ZIP entry filter.
   *
   * @throws IOException if the output stream or the filter throws an
   *                     IOException
   * @throws IllegalStateException if this method was already called earlier
   */
  public void finish() throws IOException {
    checkNotFinished();
    finished = true;
    for (Map.Entry<String, FileEntry> entry : fileNames.entrySet()) {
      String filename = entry.getKey();
      CustomMergeStrategy mergeStrategy = entry.getValue().mergeStrategy;
      ByteArrayOutputStream outputBuffer = entry.getValue().outputBuffer;
      int dosTime = entry.getValue().dosTime;
      if (mergeStrategy == null) {
        // Do nothing.
      } else {
        mergeStrategy.finish(outputBuffer);
        copyStreamToEntry(filename, new ByteArrayInputStream(outputBuffer.toByteArray()), dosTime,
            NO_EXTRA_ENTRIES, true, DEFAULT_DIRECTORY_ENTRY_INFO);
      }
    }

    // Write central directory.
    if (out.bytesWritten >= MAXIMUM_DATA_SIZE) {
      throw new IOException("Unable to handle files bigger than 2^32 bytes.");
    }
    int startOfCentralDirectory = (int) out.bytesWritten;
    int centralDirectorySize = centralDirectory.writeTo(out);

    // end of central directory signature
    setUnsignedInt(directoryEntryBuffer, 0, END_OF_CENTRAL_DIRECTORY_MARKER);
    // number of this disk
    setUnsignedShort(directoryEntryBuffer, 4, (short) 0);
    // number of the disk with the start of the central directory
    setUnsignedShort(directoryEntryBuffer, 6, (short) 0);
    // total number of entries in the central directory on this disk
    setUnsignedShort(directoryEntryBuffer, 8, (short) fileCount);
    // total number of entries in the central directory
    setUnsignedShort(directoryEntryBuffer, 10, (short) fileCount);
    // size of the central directory
    setUnsignedInt(directoryEntryBuffer, 12, centralDirectorySize);
    // offset of start of central directory with respect to the starting disk number
    setUnsignedInt(directoryEntryBuffer, 16, startOfCentralDirectory);
    // .ZIP file comment length
    setUnsignedShort(directoryEntryBuffer, 20, (short) 0);
    write(directoryEntryBuffer, 0, 22);

    out.flush();
  }

  private void checkNotFinished() {
    if (finished) {
      throw new IllegalStateException();
    }
  }

  /**
   * Writes any remaining output data to the output stream and closes it.
   *
   * @throws IOException if the output stream or the filter throws an
   *                     IOException
   */
  @Override
  public void close() throws IOException {
    if (!finished) {
      finish();
    }
    out.close();
  }

  /**
   * Turns this JAR file into an executable JAR by prepending an executable.
   * JAR files are placed at the end of a file, and executables are placed
   * at the beginning, so a file can be both, if desired.
   *
   * @param launcherIn   The InputStream, from which the launcher is read.
   * @throws NullPointerException if launcherIn is null
   * @throws IOException if reading from launcherIn or writing to the output
   *                     stream throws an IOException.
   */
  public void prependExecutable(InputStream launcherIn) throws IOException {
    if (launcherIn == null) {
      throw new NullPointerException("No launcher specified");
    }
    byte[] buf = new byte[BUFFER_SIZE];
    int bytesRead;
    while ((bytesRead = launcherIn.read(buf)) > 0) {
      out.write(buf, 0, bytesRead);
    }
  }

  /**
   * Ensures the truth of an expression involving one or more parameters to the calling method.
   */
  private static void checkArgument(boolean expression,
      @Nullable String errorMessageTemplate,
      @Nullable Object... errorMessageArgs) {
    if (!expression) {
      throw new IllegalArgumentException(String.format(errorMessageTemplate, errorMessageArgs));
    }
  }

  /**
   * Ensures the truth of an expression involving one or more parameters to the calling method.
   */
  private static void checkArgument(boolean expression) {
    if (!expression) {
      throw new IllegalArgumentException();
    }
  }

  /**
   * Ensures the truth of an expression involving state.
   */
  private static void checkState(boolean expression,
      @Nullable String errorMessageTemplate,
      @Nullable Object... errorMessageArgs) {
    if (!expression) {
      throw new IllegalStateException(String.format(errorMessageTemplate, errorMessageArgs));
    }
  }
}
