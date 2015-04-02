// Copyright 2015 Google Inc. All rights reserved.
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

import javax.annotation.Nullable;

/**
 * A full representation of a ZIP file entry.
 *
 * <p>See <a href="http://www.pkware.com/documents/casestudies/APPNOTE.TXT">ZIP Format</a> for
 * a description of the entry fields. (Section 4.3.7 and 4.4)
 */
public final class ZipFileEntry {

  /** Compression method for ZIP entries. */
  public enum Compression {
    STORED((short) 0, (short) 0x0a),
    DEFLATED((short) 8, (short) 0x14);

    public static Compression fromValue(int value) {
      for (Compression c : Compression.values()) {
        if (c.getValue() == value) {
          return c;
        }
      }
      return null;
    }

    private short value;
    private short minVersion;

    private Compression(short value, short minVersion) {
      this.value = value;
      this.minVersion = minVersion;
    }

    public short getValue() {
      return value;
    }

    public short getMinVersion() {
      return minVersion;
    }
  }

  /** General purpose bit flag for ZIP entries. */
  public enum Flag {
    DATA_DESCRIPTOR(3);

    private int bit;

    private Flag(int bit) {
      this.bit = bit;
    }

    public int getBit() {
      return bit;
    }
  }

  private String name;
  private long time = -1;
  private long crc = -1;
  private long size = -1;
  private long csize = -1;
  private Compression method;
  private short version = -1;
  private short versionNeeded = -1;
  private short flags;
  private short internalAttributes;
  private int externalAttributes;
  private long localHeaderOffset = -1;
  @Nullable private byte[] extra;
  @Nullable private String comment;

  /**
   * Creates a new zip entry with the specified name.
   *
   * @param name the entry name
   * @throws NullPointerException if the entry name is null
   */
  public ZipFileEntry(String name) {
    setName(name);
  }

  /**
   * Creates a new zip entry with fields taken from the specified zip entry.
   *
   * @param e a zip entry object
   */
  public ZipFileEntry(ZipFileEntry e) {
    if (e == null) {
      throw new NullPointerException();
    }
    this.name = e.getName();
    this.time = e.getTime();
    this.crc = e.getCrc();
    this.size = e.getSize();
    this.csize = e.getCompressedSize();
    this.method = e.getMethod();
    this.version = e.getVersion();
    this.versionNeeded = e.getVersionNeeded();
    this.flags = e.getFlags();
    this.internalAttributes = e.getInternalAttributes();
    this.externalAttributes = e.getExternalAttributes();
    this.localHeaderOffset = e.getLocalHeaderOffset();
    this.extra = e.getExtra();
    this.comment = e.getComment();
  }

  /**
   * Sets the name of the entry.
   *
   * @param name the name
   */
  public void setName(String name) {
    if (name == null) {
      throw new NullPointerException();
    }
    this.name = name;
  }

  /**
   * Returns the name of the entry.
   *
   * @return the name of the entry
   */
  public String getName() {
    return name;
  }

  /**
   * Sets the modification time of the entry.
   *
   * @param time the entry modification time in number of milliseconds since the epoch
   */
  public void setTime(long time) {
    this.time = time;
  }

  /**
   * Returns the modification time of the entry, or -1 if not specified.
   *
   * @return the modification time of the entry, or -1 if not specified
   */
  public long getTime() {
    return time;
  }

  /**
   * Sets the CRC-32 checksum of the uncompressed entry data.
   *
   * @param crc the CRC-32 value
   * @throws IllegalArgumentException if the specified CRC-32 value is less than 0 or greater than
   *     0xFFFFFFFF
   */
  public void setCrc(long crc) {
    if (crc < 0 || crc > 0xffffffffL) {
      throw new IllegalArgumentException("invalid entry crc-32");
    }
    this.crc = crc;
  }

  /**
   * Returns the CRC-32 checksum of the uncompressed entry data, or -1 if not known.
   *
   * @return the CRC-32 checksum of the uncompressed entry data, or -1 if not known
   */
  public long getCrc() {
    return crc;
  }

  /**
   * Sets the uncompressed size of the entry data.
   *
   * @param size the uncompressed size in bytes
   * @throws IllegalArgumentException if the specified size is less than 0, is greater than
   *     0xFFFFFFFF
   */
  public void setSize(long size) {
    if (size < 0 || size > 0xffffffffL) {
      throw new IllegalArgumentException("invalid entry size");
    }
    this.size = size;
  }

  /**
   * Returns the uncompressed size of the entry data, or -1 if not known.
   *
   * @return the uncompressed size of the entry data, or -1 if not known
   */
  public long getSize() {
    return size;
  }

  /**
   * Sets the size of the compressed entry data.
   *
   * @param csize the compressed size in bytes
   * @throws IllegalArgumentException if the specified size is less than 0, is greater than
   *     0xFFFFFFFF
   */
  public void setCompressedSize(long csize) {
    if (csize < 0 || csize > 0xffffffffL) {
      throw new IllegalArgumentException("invalid entry size");
    }
    this.csize = csize;
  }

  /**
   * Returns the size of the compressed entry data, or -1 if not known. In the case of a stored
   * entry, the compressed size will be the same as the uncompressed size of the entry.
   *
   * @return the size of the compressed entry data, or -1 if not known
   */
  public long getCompressedSize() {
    return csize;
  }

  /**
   * Sets the compression method for the entry. Increases the version and version needed if the new
   * compression method requires a higher version.
   *
   * @param method the compression method, either STORED or DEFLATED
   */
  public void setMethod(Compression method) {
    if (method == null) {
      throw new NullPointerException();
    }
    this.method = method;
    short minVersion = method.getMinVersion();
    version = (short) Math.max(version, minVersion);
    versionNeeded = (short) Math.max(versionNeeded, minVersion);
  }

  /**
   * Returns the compression method of the entry.
   *
   * @return the compression method of the entry
   */
  public Compression getMethod() {
    return method;
  }

  /**
   * Sets the made by version for the entry.
   *
   * @param version the made by version to set
   * @throws IllegalArgumentException if the specified version is less than the required version for
   *     the specified compression method
   */
  public void setVersion(short version) {
    if (method != null && version < method.getMinVersion()) {
      throw new IllegalArgumentException(String.format(
          "The minimum allowable version for method %s is 0x%02x.",
          method.name(), method.getMinVersion()));
    }
    this.version = version;
  }

  /**
   * Returns the made by version of the entry.
   *
   * @return the made by version of the entry
   */
  public short getVersion() {
    return version;
  }

  /**
   * Sets the version needed to extract the entry.
   *
   * @param versionNeeded the version needed to extract to set
   * @throws IllegalArgumentException if the specified version is less than the required version for
   *     the specified compression method
   */
  public void setVersionNeeded(short versionNeeded) {
    if (method != null && versionNeeded < method.getMinVersion()) {
      throw new IllegalArgumentException(String.format(
          "The minimum allowable version for method %s is 0x%02x.",
          method.name(), method.getMinVersion()));
    }
    this.versionNeeded = versionNeeded;
  }

  /**
   * Returns the version needed to extract the entry.
   *
   * @return the version needed to extract the entry
   */
  public short getVersionNeeded() {
    return versionNeeded;
  }

  /**
   * Sets the general purpose bit flags for the entry.
   *
   * @param flags the general purpose bit flags to set
   */
  public void setFlags(short flags) {
    this.flags = flags;
  }

  /**
   * Sets or clears the specified bit of the general purpose bit flags.
   *
   * @param flag the flag to set or clear
   * @param set whether the flag is to be set or cleared
   */
  public void setFlag(Flag flag, boolean set) {
    short mask = 0x0000;
    mask |= 1 << flag.getBit();
    if (set) {
      flags |= mask;
    } else {
      flags &= ~mask;
    }
  }

  /**
   * Returns the general purpose bit flags of the entry.
   *
   * <p>See <a href="http://www.pkware.com/documents/casestudies/APPNOTE.TXT">ZIP Format</a>
   * section 4.4.4.
   *
   * @return the general purpose bit flags of the entry
   */
  public short getFlags() {
    return flags;
  }

  /**
   * Sets the internal file attributes of the entry.
   *
   * @param internalAttributes the internal file attributes to set
   */
  public void setInternalAttributes(short internalAttributes) {
    this.internalAttributes = internalAttributes;
  }

  /**
   * Returns the internal file attributes of the entry.
   *
   * @return the internal file attributes of the entry
   */
  public short getInternalAttributes() {
    return internalAttributes;
  }

  /**
   * Sets the external file attributes of the entry.
   *
   * @param externalAttributes the external file attributes to set
   */
  public void setExternalAttributes(int externalAttributes) {
    this.externalAttributes = externalAttributes;
  }

  /**
   * Returns the external file attributes of the entry.
   *
   * @return the external file attributes of the entry
   */
  public int getExternalAttributes() {
    return externalAttributes;
  }

  /**
   * Sets the file offset, in bytes, of the location of the local file header for the entry.
   *
   * <p>See <a href="http://www.pkware.com/documents/casestudies/APPNOTE.TXT">ZIP Format</a>
   * section 4.4.16
   *
   * @param localHeaderOffset the file offset of the local header to set
   * @throws IllegalArgumentException if the specified local header offset is less than 0 or greater
   *     than 0xFFFFFFFF
   */
  void setLocalHeaderOffset(long localHeaderOffset) {
    if (localHeaderOffset < 0 || localHeaderOffset > 0xffffffffL) {
      throw new IllegalArgumentException("invalid local header offset");
    }
    this.localHeaderOffset = localHeaderOffset;
  }

  /**
   * Returns the file offset of the local header of the entry.
   *
   * @return the file offset of the local header of the entry
   */
  public long getLocalHeaderOffset() {
    return localHeaderOffset;
  }

  /**
   * Sets the optional extra field data for the entry.
   *
   * <p><em>NOTE:</em> This sets the extra field exactly as specified. Use
   * {@link #setExtra(ExtraData[])} to guarantee well formed extra field entries.
   *
   * @param extra the extra field data bytes
   * @throws IllegalArgumentException if the length of the specified extra field data is greater
   *    than 0xFFFF bytes
   */
  public void setExtra(@Nullable byte[] extra) {
    if (extra != null && extra.length > 0xffff) {
      throw new IllegalArgumentException("invalid extra field length");
    }
    this.extra = extra;
  }

  /**
   * Sets the optional extra field data from the provided {@link ExtraData} array. Performs the
   * necessary conversion to the raw byte array.
   *
   * <p><em>NOTE:</em> This will guarantee well formed extra field entries, but cannot guarantee
   * usable data if Id or Data is specified incorrectly in {@link ExtraData}.
   *
   * @param extra the extra field data
   * @throws IllegalArgumentException if the length of the specified extra field data is greater
   *    than 0xFFFF bytes
   */
  public void setExtra(ExtraData[] extra) {
    int extraDataLength = 0;
    for (ExtraData e : extra) {
      extraDataLength += 4 + e.getData().length;
    }

    byte[] rawExtra = new byte[extraDataLength];

    int index = 0;
    for (ExtraData e : extra) {
      ZipUtil.shortToLittleEndian(rawExtra, index, e.getId());
      ZipUtil.shortToLittleEndian(rawExtra, index + 2, (short) (e.getData().length & 0xffff));
      System.arraycopy(e.getData(), 0, rawExtra, index + 4, e.getData().length);
      index += 4 + e.getData().length;
    }
    setExtra(rawExtra);
  }

  /**
   * Returns the extra field data for the entry, or null if none.
   *
   * @return the extra field data for the entry, or null if none
   */
  public byte[] getExtra() {
    return extra;
  }

  /**
   * Sets the optional comment string for the entry.
   *
   * @param comment the comment string
   */
  public void setComment(@Nullable String comment) {
    this.comment = comment;
  }

  /**
   * Returns the comment string for the entry, or null if none.
   *
   * @return the comment string for the entry, or null if none
   */
  public String getComment() {
    return comment;
  }
}
