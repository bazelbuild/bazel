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

package com.google.devtools.build.zip;

import com.google.devtools.build.zip.ZipFileEntry.Feature;

import java.nio.charset.Charset;
import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.zip.ZipException;

import javax.annotation.Nullable;

/**
 * A representation of a ZIP file. Contains the file comment, encoding, and entries. Also contains
 * internal information about the structure and location of ZIP file parts. 
 */
class ZipFileData {
  private final Charset charset;
  private String comment;

  private long centralDirectorySize;
  private long centralDirectoryOffset;
  private long expectedEntries;
  private long numEntries;
  private final Map<String, ZipFileEntry> entries;

  private boolean maybeZip64;
  private boolean isZip64;
  private long zip64EndOfCentralDirectoryOffset;

  /**
   * Creates a new ZIP file with the specified charset encoding.
   */
  public ZipFileData(Charset charset) {
    if (charset == null) {
      throw new NullPointerException();
    }
    this.charset = charset;
    comment = "";
    entries = new LinkedHashMap<>();
  }

  /**
   * Returns the encoding of the file.
   */
  public Charset getCharset() {
    return charset;
  }

  /**
   * Returns the file comment.
   */
  public String getComment() {
    return comment;
  }

  /**
   * Sets the file comment from the raw byte data in the file. Converts the bytes to a string using
   * the file's charset encoding.
   *
   * @throws ZipException if the comment is longer than allowed by the ZIP format
   */
  public void setComment(byte[] comment) throws ZipException {
    if (comment == null) {
      throw new NullPointerException();
    }
    if (comment.length > 0xffff) {
      throw new ZipException(String.format("File comment too long. Is %d; max %d.",
          comment.length, 0xffff));
    }
    this.comment = fromBytes(comment);
  }

  /**
   * Sets the file comment.
   *
   * @throws ZipException if the comment will be longer than allowed by the ZIP format when encoded
   *    using the file's charset encoding
   */
  public void setComment(String comment) throws ZipException {
    setComment(getBytes(comment));
  }

  /**
   * Returns the size of the central directory in bytes.
   */
  public long getCentralDirectorySize() {
    return centralDirectorySize;
  }

  /**
   * Sets the size of the central directory in bytes. If the size is larger than 0xffffffff, the
   * file is set to Zip64 mode.
   *
   * <p>See <a href="http://www.pkware.com/documents/casestudies/APPNOTE.TXT">ZIP Format</a>
   * section 4.4.23
   */
  public void setCentralDirectorySize(long centralDirectorySize) {
    this.centralDirectorySize = centralDirectorySize;
    if (centralDirectorySize > 0xffffffffL) {
      setZip64(true);
    }
  }

  /**
   * Returns the file offset of the start of the central directory.
   */
  public long getCentralDirectoryOffset() {
    return centralDirectoryOffset;
  }

  /**
   * Sets the file offset of the start of the central directory. If the offset is larger than
   * 0xffffffff, the file is set to Zip64 mode.
   *
   * <p>See <a href="http://www.pkware.com/documents/casestudies/APPNOTE.TXT">ZIP Format</a>
   * section 4.4.24
   */
  public void setCentralDirectoryOffset(long offset) {
    this.centralDirectoryOffset = offset;
    if (centralDirectoryOffset > 0xffffffffL) {
      setZip64(true);
    }
  }

  /**
   * Returns the number of entries expected to be in the ZIP file. This value is determined from the
   * end of central directory record.
   */
  public long getExpectedEntries() {
    return expectedEntries;
  }

  /**
   * Sets the number of entries expected to be in the ZIP file. This value should be set by reading
   * the end of central directory record.
   *
   * <p>See <a href="http://www.pkware.com/documents/casestudies/APPNOTE.TXT">ZIP Format</a>
   * section 4.4.22
   */
  public void setExpectedEntries(long count) {
    this.expectedEntries = count;
    if (expectedEntries > 0xffff) {
      setZip64(true);
    }
  }

  /**
   * Returns the number of entries actually in the ZIP file. This value is derived from the number
   * of times {@link #addEntry(ZipFileEntry)} was called.
   *
   * <p><em>NOTE:</em> This value should be used rather than getting the size from the 
   * {@link Collection} returned from {@link #getEntries()}, because the value may be too large to
   * be properly represented by an int.
   */
  public long getNumEntries() {
    return numEntries;
  }

  /**
   * Sets the number of entries actually in the ZIP file. If the value is larger than 0xffff, the
   * file is set to Zip64 mode.
   */
  private void setNumEntries(long numEntries) {
    this.numEntries = numEntries;
    if (numEntries > 0xffff) {
      setZip64(true);
    }
  }

  /**
   * Returns a collection of all entries in the ZIP file.
   */
  public Collection<ZipFileEntry> getEntries() {
    return entries.values();
  }

  /**
   * Returns the entry with the given name, or null if it does not exist.
   */
  public ZipFileEntry getEntry(@Nullable String name) {
    return entries.get(name);
  }

  /**
   * Adds an entry to the ZIP file. If this causes the actual number of entries to exceed
   * 0xffffffff, or if the file requires Zip64 features, the file is set to Zip64 mode.
   */
  public void addEntry(ZipFileEntry entry) {
    entries.put(entry.getName(), entry);
    setNumEntries(numEntries + 1);
    if (entry.getFeatureSet().contains(Feature.ZIP64_SIZE)
        || entry.getFeatureSet().contains(Feature.ZIP64_CSIZE)
        || entry.getFeatureSet().contains(Feature.ZIP64_OFFSET)) {
      setZip64(true);
    }
  }

  /**
   * Returns if the file may be in Zip64 mode. This is true if any of the values in the end of
   * central directory record are -1.
   *
   * <p>See <a href="http://www.pkware.com/documents/casestudies/APPNOTE.TXT">ZIP Format</a>
   * section 4.4.19 - 4.4.24
   */
  public boolean isMaybeZip64() {
    return maybeZip64;
  }

  /**
   * Set if the file may be in Zip64 mode. This is true if any of the values in the end of
   * central directory record are -1.
   *
   * <p>See <a href="http://www.pkware.com/documents/casestudies/APPNOTE.TXT">ZIP Format</a>
   * section 4.4.19 - 4.4.24
   */
  public void setMaybeZip64(boolean maybeZip64) {
    this.maybeZip64 = maybeZip64;
  }

  /**
   * Returns if the file is in Zip64 mode. This is true if any of a number of fields exceed the
   * maximum value.
   *
   * <p>See <a href="http://www.pkware.com/documents/casestudies/APPNOTE.TXT">ZIP Format</a> for
   * details
   */
  public boolean isZip64() {
    return isZip64;
  }

  /**
   * Set if the file is in Zip64 mode. This is true if any of a number of fields exceed the maximum
   * value.
   *
   * <p>See <a href="http://www.pkware.com/documents/casestudies/APPNOTE.TXT">ZIP Format</a> for
   * details
   */
  public void setZip64(boolean isZip64) {
    this.isZip64 = isZip64;
    setMaybeZip64(true);
  }

  /**
   * Returns the file offset of the Zip64 end of central directory record. The record is only
   * present if {@link #isZip64()} returns true.
   */
  public long getZip64EndOfCentralDirectoryOffset() {
    return zip64EndOfCentralDirectoryOffset;
  }

  /**
   * Sets the file offset of the Zip64 end of central directory record and sets the file to Zip64
   * mode.
   */
  public void setZip64EndOfCentralDirectoryOffset(long offset) {
    this.zip64EndOfCentralDirectoryOffset = offset;
    setZip64(true);
  }

  /**
   * Returns the byte representation of the specified string using the file's charset encoding.
   */
  public byte[] getBytes(String string) {
    return string.getBytes(charset);
  }

  /**
   * Returns the string represented by the specified byte array using the file's charset encoding.
   */
  public String fromBytes(byte[] bytes) {
    return new String(bytes, charset);
  }
}
