// Copyright 2015 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.android.ziputils;

import static com.google.devtools.build.android.ziputils.DataDescriptor.EXTLEN;
import static com.google.devtools.build.android.ziputils.DataDescriptor.EXTSIZ;
import static com.google.devtools.build.android.ziputils.DirectoryEntry.CENLEN;
import static com.google.devtools.build.android.ziputils.DirectoryEntry.CENOFF;
import static com.google.devtools.build.android.ziputils.DirectoryEntry.CENSIZ;
import static com.google.devtools.build.android.ziputils.EndOfCentralDirectory.ENDOFF;
import static com.google.devtools.build.android.ziputils.EndOfCentralDirectory.ENDSUB;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.util.Map.Entry;

/**
 * API for reading a zip file. This does not perform decompression of entry data, but provides
 * a raw view of the content of a zip archive.
 */
public class ZipIn {

  private static final byte[] EOCD_SIG = {0x50, 0x4b, 0x05, 0x06};
  private static final byte[] HEADER_SIG = {0x50, 0x4b, 0x03, 0x04};
  private static final byte[] DATA_DESC_SIG = {0x50, 0x4b, 0x07, 0x08};


  /**
   * Max end-of-central-directory size, including variable length file comment..
   */
  private static final int MAX_EOCD_SIZE = 1024;

  /**
   * Max local file header size, including long filename.
   */
  private static final int MAX_HEADER_SIZE = 64 * 1024;

  /**
   * Default size of direct byte buffer used for reading content. Actual allocation will not
   * exceed the archive content size, and may be at least as big as the largest entry.
   */
  private static final int READ_BLOCK_SIZE = 20 * 1024 * 1024;

  private final String filename; // filename or nickname.
  private final FileChannel fileChannel; // input file.
  private BufferedFile bufferedFile;
  private CentralDirectory cdir = null;
  private EndOfCentralDirectory eocd = null;
  private final boolean useDirectory;
  private final boolean ignoreDeleted;
  private final boolean verbose = false;

  /**
   * Creates a {@code ZipIn} view of a file, with a (nick)name.
   *
   * @param channel File channel open for reading.
   * @param filename filename or nickname.
   */
  public ZipIn(FileChannel channel, String filename) {
    this.fileChannel = channel;
    this.filename = filename;
    this.useDirectory = true;
    this.ignoreDeleted = useDirectory;
  }

  /**
   * Gets the file name for this zip input file.
   * @return the filename set at time of construction.
   */
  public String getFilename() {
    return filename;
  }

  /**
   * Returns a view of the "end of central directory" record expected at (or towards) the end of a
   * zip file.
   *
   * @return A read-only, {@link EndOfCentralDirectory}.
   * @throws IOException
   */
  public EndOfCentralDirectory endOfCentralDirectory() throws IOException {
    if (eocd == null) {
      loadEndOfCentralDirectory();
    }
    return eocd;
  }

  /**
   * Returns a memory mapped view of the central directory.
   *
   * @return A read-only, {@link CentralDirectory} of the central directory.
   * @throws IOException
   */
  public CentralDirectory centralDirectory() throws IOException {
    if (cdir == null) {
      loadCentralDirectory();
    }
    return cdir;
  }

  /**
   * Scans all entries in the zip file and invokes the given {@link EntryHandler} on each.
   *
   * @param handler handler to invoke for each file entry.
   * @throws IOException
   */
  public void scanEntries(EntryHandler handler) throws IOException {
    centralDirectory();
    ZipEntry zipEntry = nextFrom(null);
    while (zipEntry.getCode() != ZipEntry.Status.ENTRY_NOT_FOUND) {
      if (zipEntry.getCode() != ZipEntry.Status.ENTRY_OK) {
        throw new IOException(zipEntry.getCode().toString());
      }
      handler.handle(this, zipEntry.getHeader(), zipEntry.getDirEntry(), zipEntry.getContent());
      if (useDirectory && ignoreDeleted) {
        zipEntry = ZipIn.this.nextFrom(zipEntry.getDirEntry());
      } else {
        zipEntry = nextFrom(zipEntry.limit());
      }
    }
  }

  /**
   * Finds the next header, by scanning for a local header signature starting
   * at {@code offset}. This method will find headers for deleted or updated entries that
   * are not listed in the central directory, and may pickup false positive (e.g. entries
   * of an embedded zip file stored without compression). This method is primarily intended
   * for applications trying to recover data from corrupt archives.
   *
   * @param offset offset where to start the search.
   * @return the next local header at or beyond {@code offset}, or {@code null} if no
   * header is found.
   * @throws IOException
   */
  public LocalFileHeader nextHeaderFrom(long offset) throws IOException {
    int skipped = 0;
    for (ByteBuffer buffer = getData(offset + skipped, MAX_HEADER_SIZE);
        buffer.limit() >= LocalFileHeader.SIZE;
        buffer = getData(offset + skipped, MAX_HEADER_SIZE)) {
      int markerOffset = ScanUtil.scanTo(HEADER_SIG, buffer);
      if (markerOffset < 0) {
        skipped += buffer.limit() - 3;
      } else {
        skipped += markerOffset;
        LocalFileHeader header =  markerOffset == 0 ? localHeaderIn(buffer, offset + skipped)
            : localHeaderAt(offset + skipped);
        if (header != null) {
          if (skipped > 0) {
            System.out.println("Warning: local header search: skipped " + skipped + " bytes");
          }
          return header;
        }
        // If localHeaderIn or localHeaderAt decided it is not a header location,
        // we continue the search.
        skipped += 4;
      }
    }
    return null;
  }

  /**
   * Finds the header at the next higher offset listed in the central directory as containing
   * a local file header, starting from the offset of the given {@code dirEntry}. This method will
   * bypass any deleted or updated entries not listed in the directory, and also any entries from
   * embedded zip files, or random instance of the header signature. This is the preferred method
   * for sequentially reading the entries of a valid zip file.
   *
   * @param dirEntry directory entry for the "current entry", providing the start point
   * for searching the central directory for the entry with the next higher offset.
   * @return the next header according to the central directory, or {@code null} if there are no
   * more headers.
   * @throws IOException
   */
  public LocalFileHeader nextHeaderFrom(DirectoryEntry dirEntry) throws IOException {
    Integer nextOffset = dirEntry == null ? -1 : dirEntry.get(CENOFF);
    while ((nextOffset = cdir.mapByOffset().higherKey(nextOffset)) != null) {
      LocalFileHeader header = localHeaderAt(nextOffset);
      if (header != null) {
        return header;
      }
      System.out.println("Warning: no header for file listed in directory "
          + dirEntry.getFilename());
      // The file is corrupt! Continue to see how bad it is.
    }
    return null;
  }

  /**
   * Provides a {@code LocalFileHeader} view of a local header located at the offset indicated
   * by the given {@code dirEntry}.
   *
   * @param dirEntry the directory entry referring to the headers location.
   * @return the requested header, or {@code null} if the given location can't possibly contain a
   * valid file header (e.g. missing header signature), or if {@code dirEntry} is {@code null}.
   * @throws IOException
   */
  public LocalFileHeader localHeaderFor(DirectoryEntry dirEntry) throws IOException {
    return dirEntry == null ? null : localHeaderAt(dirEntry.get(CENOFF));
  }

  /**
   * Provides a {@code LocalFileHeader} view of a local header located at the offset indicated
   * by the given {@code dirEntry}.
   *
   * @param offset offset a which the a header is presumed to exist.
   * @return the requested header, or {@code null} if the given location can't possibly contain a
   * valid file header (e.g. missing header signature).
   * @throws IOException
   */
  public LocalFileHeader localHeaderAt(long offset) throws IOException {
    return localHeaderIn(getData(offset, MAX_HEADER_SIZE), offset);
  }

  /**
   * Finds the next zip file entry, by scanning for a local header using the
   * {@link #nextHeaderFrom(long) }method.
   *
   * @param offset offset where to start the search.
   * @return a {@code ZipEntry} object with the result of the search.
   * @throws IOException
   */
  public ZipEntry nextFrom(long offset) throws IOException {
    LocalFileHeader header = ZipIn.this.nextHeaderFrom(offset);
    return entryWith(header);
  }

  /**
   * Finds the next zip file entry, by first invoking
   * {@link #nextHeaderFrom(com.google.devtools.build.android.ziputils.DirectoryEntry) }
   * to find its header.
   *
   * @param entry the directory entry for the "current" zip entry, or {@code null} to get
   * the first entry.
   * @return a {@code ZipEntry} object with the result of the search.
   * @throws IOException
   */
  public ZipEntry nextFrom(DirectoryEntry entry) throws IOException {
    int offset = entry == null ? -1 : entry.get(CENOFF);
    Entry<Integer, DirectoryEntry> mapEntry = cdir.mapByOffset().higherEntry(offset);
    if (mapEntry == null) {
      return entryWith(null);
    }
    LocalFileHeader header = localHeaderAt(mapEntry.getKey());
    return entryWith(header, mapEntry.getValue());
  }
  
  /**
   * Finds the zip file entry, for a given directory entry.
   *
   * @param entry the directory entry for which a zip entry is requested.
   * @return a {@code ZipEntry} object with the result of the search.
   * @throws IOException
   */
  public ZipEntry entryFor(DirectoryEntry entry) throws IOException {
    return entryWith(localHeaderFor(entry), entry);
  }

  /**
   * Returns the zip file entry at the given offset.
   *
   * @param offset presumed location of local file header.
   * @return a {@link ZipEntry} for the given location.
   * @throws IOException
   */
  public ZipEntry entryAt(long offset) throws IOException {
    LocalFileHeader header = localHeaderAt(offset);
    return entryWith(header);
  }

  /**
   * Constructs a {@link ZipEntry} view of the entry at the location of the given header.
   *
   * @param header a previously located header. If (@code useDirectory} is set, this will
   * attempt to lookup a corresponding directory entry. If there is none, and {@code ignoreDeleted}
   * is also set, the return value will flag this entry with a
   * {@code ZipEntry.Status.ENTRY_NOT_FOUND} status code.
   *
   * @return  {@link ZipEntry} for the given location.
   * @throws IOException
   */
  public ZipEntry entryWith(LocalFileHeader header) throws IOException {
    if (header == null) {
      return new ZipEntry().withCode(ZipEntry.Status.ENTRY_NOT_FOUND);
    }
    // header != null
    long offset = header.fileOffset();
    DirectoryEntry dirEntry = null;
    if (useDirectory) {
      dirEntry = cdir.mapByOffset().get((int) offset);
      if (dirEntry == null && ignoreDeleted) {
        return new ZipEntry().withCode(ZipEntry.Status.ENTRY_DELETED);
      }
    }
    return entryWith(header, dirEntry);
  }

  /**
   * Scans for a data descriptor from a given offset.
   *
   * @param offset position where to start the search.
   * @param dirEntry directory entry for validation, or {@code null}.
   * @return A data descriptor view for the next position containing the data descriptor signature.
   * @throws IOException
   */
  public DataDescriptor descriptorFrom(final long offset, final DirectoryEntry dirEntry)
      throws IOException {
    int skipped = 0;
    for (ByteBuffer buffer = getData(offset + skipped, MAX_HEADER_SIZE);
        buffer.limit() >= 16; buffer = getData(offset + skipped, MAX_HEADER_SIZE)) {
      int markerOffset = ScanUtil.scanTo(DATA_DESC_SIG, buffer);
      if (markerOffset < 0) {
        skipped += buffer.limit() - 3;
      } else {
        skipped += markerOffset;
        return markerOffset == 0 ? descriptorIn(buffer, offset + skipped, dirEntry)
            : descriptorAt(offset + skipped, dirEntry);
      }
    }
    return null;
  }

  /**
   * Creates a data descriptor view at a given offset.
   *
   * @param offset presumed location of data descriptor.
   * @param dirEntry directory entry to use for validation, or {@code null}.
   * @return a data descriptor view over the given file offset.
   * @throws IOException
   */
  public DataDescriptor descriptorAt(long offset, DirectoryEntry dirEntry) throws IOException {
    return descriptorIn(getData(offset, 16), offset, dirEntry);
  }

  /**
   * Constructs a zip entry object for the location of the given header, with the corresponding
   * directory entry.
   *
   * @param header local file header for the entry.
   * @param dirEntry corresponding directory entry, or {@code null} if not available.
   * @return a zip entry with the given header and directory entry.
   * @throws IOException
   */
  private ZipEntry entryWith(LocalFileHeader header, DirectoryEntry dirEntry) throws IOException {
    ZipEntry zipEntry = new ZipEntry().withHeader(header).withEntry(dirEntry);
    int offset = (int) (header.fileOffset() + header.getSize());
    // !useDirectory || dirEntry != null || !ignoreDeleted
    String entryName = header.getFilename();
    if (dirEntry != null && !entryName.equals(dirEntry.getFilename())) {
      return zipEntry.withEntry(dirEntry).withCode(ZipEntry.Status.FILENAME_ERROR);
    }
    int sizeByHeader = header.dataSize();
    int sizeByDir = dirEntry != null ? dirEntry.dataSize() : -1;
    ByteBuffer content;
    if (sizeByDir == sizeByHeader && sizeByDir >= 0) {
      // Ideal case, header and directory in agreement
      content = getData(offset, sizeByHeader);
      if (content.limit() == sizeByHeader) {
        return zipEntry.withContent(content).withCode(ZipEntry.Status.ENTRY_OK);
      } else {
        return zipEntry.withContent(content).withCode(ZipEntry.Status.NOT_ENOUGH_DATA);
      }
    }
    if (sizeByDir >= 0) {
      // If file is correct, we get here because of a 0x8 flag, and we expect
      // data to be followed by a data descriptor.
      content = getData(offset, sizeByDir);
      DataDescriptor dataDesc = descriptorAt(offset + sizeByDir, dirEntry);
      if (dataDesc != null) {
        return zipEntry.withContent(content).withDescriptor(dataDesc).withCode(
            ZipEntry.Status.ENTRY_OK);
      }
      return zipEntry.withContent(content).withCode(ZipEntry.Status.NO_DATA_DESC);
    }
    if (!ignoreDeleted) {
      if (sizeByHeader >= 0) {
        content = getData(offset, sizeByHeader);
        if (content.limit() == sizeByHeader) {
          return zipEntry.withContent(content).withCode(ZipEntry.Status.ENTRY_OK);
        }
        return zipEntry.withContent(content).withCode(ZipEntry.Status.NOT_ENOUGH_DATA);
      } else {

        DataDescriptor dataDesc = descriptorFrom(offset, dirEntry);
        if (dataDesc == null) {
          // Only way now would be to decompress
          return zipEntry.withCode(ZipEntry.Status.UNKNOWN_SIZE);
        }
        int sizeByDesc = dataDesc.get(EXTSIZ);
        if (sizeByDesc != dataDesc.fileOffset() - offset) {
          // That just can't be the right
          return zipEntry.withDescriptor(dataDesc).withCode(ZipEntry.Status.UNKNOWN_SIZE);
        }
        content = getData(offset, sizeByDesc);
        return zipEntry.withContent(content).withDescriptor(dataDesc).withCode(
            ZipEntry.Status.ENTRY_OK);
      }
    }
    return zipEntry.withCode(ZipEntry.Status.UNKNOWN_SIZE);
  }

  /**
   * Constructs a local header view over a give byte buffer.
   *
   * @param buffer byte buffer with local header data.
   * @param offset file offset at which the buffer is based.
   * @return a local header view.
   */
  private LocalFileHeader localHeaderIn(ByteBuffer buffer, long offset) {
    return buffer.limit() < LocalFileHeader.SIZE
        || buffer.getInt(0) != LocalFileHeader.SIGNATURE
        ? null : LocalFileHeader.viewOf(buffer).at(offset);
  }

  /**
   * Constructs a data descriptor view over a given byte buffer.
   *
   * @param buf byte buffer with data descriptor data.
   * @param offset file offset at which the buffer is based.
   * @param dirEntry directory entry with presumed reliable content size information.
   * @return a data descriptor
   */
  private DataDescriptor descriptorIn(ByteBuffer buf, long offset, DirectoryEntry dirEntry) {
    if (buf.limit() < 12) {
      return null;
    }
    DataDescriptor desc = DataDescriptor.viewOf(buf).at(offset);
    if (desc.hasMarker() || (dirEntry != null
        && desc.get(EXTSIZ) == dirEntry.get(CENSIZ)
        && desc.get(EXTLEN) == dirEntry.get(CENLEN))) {
      return desc;
    }
    return null;
  }

  /**
   * Obtains a byte buffer at a given offset.
   */
  private ByteBuffer getData(long offset, int size) throws IOException {
    return bufferedFile.getBuffer(offset, size).order(ByteOrder.LITTLE_ENDIAN);
  }

  /**
   * Locates the "end of central directory" record, expected located at the end of the file, and
   * reads it into a byte buffer. Called on the first invocation of
   * {@link #endOfCentralDirectory() }.
   *
   * @throws IOException
   */
  protected void loadEndOfCentralDirectory() throws IOException {
    cdir = null;
    long size = fileChannel.size();
    verbose("Loading ZipIn: " + filename);
    verbose("-- size: " + size);
    int cap = (int) Math.min(size, MAX_EOCD_SIZE);
    ByteBuffer buffer = ByteBuffer.allocate(cap).order(ByteOrder.LITTLE_ENDIAN);
    long offset = size - cap;
    while (true) {
      fileChannel.position(offset);
      while (buffer.hasRemaining()) {
        fileChannel.read(buffer, offset);
      }
      // scan to find it...
      int endOfDirOffset = ScanUtil.scanBackwardsTo(EOCD_SIG, buffer);
      if (endOfDirOffset < 0) {
        if (offset == 0) {
          if (useDirectory) {
            throw new IllegalStateException("No end of central directory marker");
          } else {
            break;
          }
        }
        offset = Math.max(offset - 1000, 0);
        buffer.clear();
        continue;
      }
      long eocdFileOffset = offset + endOfDirOffset;
      verbose("-- EOCD: " + eocdFileOffset + " size: " + (size - eocdFileOffset));
      buffer.position(endOfDirOffset);
      eocd = EndOfCentralDirectory.viewOf(buffer).at(offset + endOfDirOffset);
      // TODO (bazel-team): check that the end of central directory, points to a valid
      // first directory entry. If not, assume we happened to find the signature inside
      // a file comment, and resume the search.
      break;
    }

    if (eocd != null) {
      bufferedFile = new BufferedFile(fileChannel, 0, eocd.get(ENDOFF),
          READ_BLOCK_SIZE);
    } else {
      bufferedFile = new BufferedFile(fileChannel, READ_BLOCK_SIZE);
    }
  }

  /**
   * Maps the central directory to memory. Called on the first invocation of
   * {@link #centralDirectory() }.
   *
   * @throws IOException
   */
  protected void loadCentralDirectory() throws IOException {
    if (eocd == null) {
      loadEndOfCentralDirectory();
    }
    if (eocd == null) {
      return;
    }
    long cdOffset = eocd.get(ENDOFF);
    long len = eocd.fileOffset() - cdOffset;
    verbose("-- CDIR: " + cdOffset + " size: " + len + " count: " + eocd.get(ENDSUB));
    // Read directory to buffer.
    // TODO(bazel-team): we currently assume the directory fits in memory (and int).
    ByteBuffer buffer = ByteBuffer.allocateDirect((int) len);
    while (len > 0) {
      int read = fileChannel.read(buffer, cdOffset);
      len -= read;
      cdOffset += read;
    }
    buffer.rewind();
    cdir = CentralDirectory.viewOf(buffer).at(cdOffset).parse();
    cdir.buffer.flip();
  }

  /**
   * Zip file entry container class, for use with the low-level scanning operations of this
   * API, supporting zip file scanner construction.
   */
  public static class ZipEntry {

    private LocalFileHeader header;
    private DataDescriptor descriptor;
    private ByteBuffer content;
    private DirectoryEntry entry;
    private Status code;

    /**
     * Creates a zip entry, setting the initial status to not found.
     */
    public ZipEntry() {
      code = Status.ENTRY_NOT_FOUND;
    }

    /**
     * Gets the header of this zip entry.
     */
    public LocalFileHeader getHeader() {
      return header;
    }

    /**
     * Sets the header of this zip entry.
     * @return this object.
     */
    public ZipEntry withHeader(LocalFileHeader header) {
      this.header = header;
      return this;
    }

    /**
     * Gets the data descriptor of this zip entry, if any.
     */
    public DataDescriptor getDescriptor() {
      return descriptor;
    }

    /**
     * Sets the data descriptor of this zip entry.
     * @return this object.
     */
    public ZipEntry withDescriptor(DataDescriptor descriptor) {
      this.descriptor = descriptor;
      return this;
    }

    /**
     * Gets a byte buffer for accessing the raw content of this zip entry.
     */
    public ByteBuffer getContent() {
      return content;
    }

    /**
     * Sets the byte buffer providing access to the raw content of this zip entry.
     * @return this object
     */
    public ZipEntry withContent(ByteBuffer content) {
      this.content = content;
      return this;
    }

    /**
     * Gets the central directory entry for this zip entry, if any.
     */
    public DirectoryEntry getDirEntry() {
      return entry;
    }

    /**
     * Sets the central directory entry for this zip entry.
     * @return this object.
     */
    public ZipEntry withEntry(DirectoryEntry entry) {
      this.entry = entry;
      return this;
    }

    /**
     * Gets the status code for parsing this zip entry.
     */
    public Status getCode() {
      return code;
    }

    /**
     * Sets the status code for this zip entry.
     * @return this object.
     */
    public ZipEntry withCode(Status code) {
      this.code = code;
      return this;
    }

    /**
     * Calculates, best-effort, the file offset just past this zip entry.
     */
    public long limit() {
      if (header == null) {
        return 0;
      }
      if (descriptor != null) {
        return descriptor.fileOffset() + descriptor.getSize();
      }
      long offset = header.fileOffset() + header.dataSize();
      if (content != null) {
        offset += content.limit();
      }
      return offset;
    }

    /**
     * Zip entry parsing status codes.
     */
    public enum Status {
      /**
       * This zip entry contains valid header and data
       */
      ENTRY_OK,
      /**
       * No header at the given location
       */
      ENTRY_NOT_FOUND,
      /**
       * The given location contains a header that is not listed in the central directory
       */
      ENTRY_DELETED,
      /**
       * The header in the given location has a different filename than the
       * directory entry for this location.
       */
      FILENAME_ERROR,
      /**
       * The given location has the header signature, but the remaining data is insufficient
       * to constitute a complete entry.
       */
      NOT_ENOUGH_DATA,
      /**
       * The entry appears to be missing an expected data descriptor.
       */
      NO_DATA_DESC,
      /**
       * The implementation was unable to determine the size of the content of the entry.
       * The client will have to either parse using the central directory, or if all else
       * fails, attempt to decompress the entry.
       */
      UNKNOWN_SIZE,
    }
  }

  private void verbose(String msg) {
    if (verbose) {
      System.out.println(msg);
    }
  }
}
