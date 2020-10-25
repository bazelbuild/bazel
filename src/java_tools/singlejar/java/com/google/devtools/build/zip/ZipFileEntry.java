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

import java.util.EnumSet;
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
    STORED((short) 0, Feature.STORED),
    DEFLATED((short) 8, Feature.DEFLATED);

    public static Compression fromValue(int value) {
      for (Compression c : Compression.values()) {
        if (c.getValue() == value) {
          return c;
        }
      }
      return null;
    }

    private short value;
    private Feature feature;

    private Compression(short value, Feature feature) {
      this.value = value;
      this.feature = feature;
    }

    public short getValue() {
      return value;
    }

    public short getMinVersion() {
      return feature.getMinVersion();
    }

    Feature getFeature() {
      return feature;
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

  /** Zip file features that entries may use. */
  enum Feature {
    DEFAULT((short) 0x0a),
    STORED((short) 0x0a),
    DEFLATED((short) 0x14),
    ZIP64_SIZE((short) 0x2d),
    ZIP64_CSIZE((short) 0x2d),
    ZIP64_OFFSET((short) 0x2d);

    private short minVersion;

    private Feature(short minVersion) {
      this.minVersion = minVersion;
    }

    public short getMinVersion() {
      return minVersion;
    }

    static short getMinRequiredVersion(EnumSet<Feature> featureSet) {
      short minVersion = Feature.DEFAULT.getMinVersion();
      for (Feature feature : featureSet) {
        minVersion = (short) Math.max(minVersion, feature.getMinVersion());
      }
      return minVersion;
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
  private ExtraDataList extra;
  @Nullable private String comment;

  private EnumSet<Feature> featureSet;

  /**
   * Creates a new ZIP entry with the specified name.
   *
   * @throws NullPointerException if the entry name is null
   */
  public ZipFileEntry(String name) {
    this.featureSet = EnumSet.of(Feature.DEFAULT);
    setName(name);
    setMethod(Compression.STORED);
    setExtra(new ExtraDataList());
  }

  /**
   * Creates a new ZIP entry with fields taken from the specified ZIP entry.
   */
  public ZipFileEntry(ZipFileEntry e) {
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
    this.extra = new ExtraDataList(e.getExtra());
    this.comment = e.getComment();
    this.featureSet = EnumSet.copyOf(e.getFeatureSet());
  }

  /**
   * Sets the name of the entry.
   */
  public void setName(String name) {
    if (name == null) {
      throw new NullPointerException();
    }
    this.name = name;
  }

  /**
   * Returns the name of the entry.
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
   */
  public long getTime() {
    return time;
  }

  /**
   * Sets the CRC-32 checksum of the uncompressed entry data.
   *
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
   */
  public long getCrc() {
    return crc;
  }

  /**
   * Sets the uncompressed size of the entry data in bytes.
   *
   * @throws IllegalArgumentException if the specified size is less than 0
   */
  public void setSize(long size) {
    if (size < 0) {
      throw new IllegalArgumentException("invalid entry size");
    }
    if (size > 0xffffffffL) {
      featureSet.add(Feature.ZIP64_SIZE);
    } else {
      featureSet.remove(Feature.ZIP64_SIZE);
    }
    this.size = size;
  }

  /**
   * Returns the uncompressed size of the entry data, or -1 if not known.
   */
  public long getSize() {
    return size;
  }

  /**
   * Sets the size of the compressed entry data in bytes.
   *
   * @throws IllegalArgumentException if the specified size is less than 0
   */
  public void setCompressedSize(long csize) {
    if (csize < 0) {
      throw new IllegalArgumentException("invalid entry size");
    }
    if (csize > 0xffffffffL) {
      featureSet.add(Feature.ZIP64_CSIZE);
    } else {
      featureSet.remove(Feature.ZIP64_CSIZE);
    }
    this.csize = csize;
  }

  /**
   * Returns the size of the compressed entry data, or -1 if not known. In the case of a stored
   * entry, the compressed size will be the same as the uncompressed size of the entry.
   */
  public long getCompressedSize() {
    return csize;
  }

  /**
   * Sets the compression method for the entry.
   */
  public void setMethod(Compression method) {
    if (method == null) {
      throw new NullPointerException();
    }
    if (this.method != null) {
      featureSet.remove(this.method.getFeature());
    }
    this.method = method;
    featureSet.add(this.method.getFeature());
  }

  /**
   * Returns the compression method of the entry.
   */
  public Compression getMethod() {
    return method;
  }

  /**
   * Sets the made by version for the entry.
   */
  public void setVersion(short version) {
    this.version = version;
  }

  /**
   * Returns the made by version of the entry, accounting for assigned version and feature set.
   */
  public short getVersion() {
    return (short) Math.max(version, Feature.getMinRequiredVersion(featureSet));
  }

  /**
   * Sets the version needed to extract the entry.
   */
  public void setVersionNeeded(short versionNeeded) {
    this.versionNeeded = versionNeeded;
  }

  /**
   * Returns the version needed to extract the entry, accounting for assigned version and feature
   * set.
   */
  public short getVersionNeeded() {
    return (short) Math.max(versionNeeded, Feature.getMinRequiredVersion(featureSet));
  }

  /**
   * Sets the general purpose bit flags for the entry.
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
    short mask = (short) (1 << flag.getBit());
    if (set) {
      flags |= mask;
    } else {
      flags = (short) (flags & ~mask);
    }
  }

  /**
   * Returns the general purpose bit flags of the entry.
   *
   * <p>See <a href="http://www.pkware.com/documents/casestudies/APPNOTE.TXT">ZIP Format</a>
   * section 4.4.4.
   */
  public short getFlags() {
    return flags;
  }

  /**
   * Sets the internal file attributes of the entry.
   */
  public void setInternalAttributes(short internalAttributes) {
    this.internalAttributes = internalAttributes;
  }

  /**
   * Returns the internal file attributes of the entry.
   */
  public short getInternalAttributes() {
    return internalAttributes;
  }

  /**
   * Sets the external file attributes of the entry.
   */
  public void setExternalAttributes(int externalAttributes) {
    this.externalAttributes = externalAttributes;
  }

  /**
   * Returns the external file attributes of the entry.
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
   * @throws IllegalArgumentException if the specified local header offset is less than 0
   */
  void setLocalHeaderOffset(long localHeaderOffset) {
    if (localHeaderOffset < 0) {
      throw new IllegalArgumentException("invalid local header offset");
    }
    if (localHeaderOffset > 0xffffffffL) {
      featureSet.add(Feature.ZIP64_OFFSET);
    } else {
      featureSet.remove(Feature.ZIP64_OFFSET);
    }
    this.localHeaderOffset = localHeaderOffset;
  }

  /**
   * Returns the file offset of the local header of the entry.
   */
  public long getLocalHeaderOffset() {
    return localHeaderOffset;
  }

  /**
   * Sets the optional extra field data for the entry.
   *
   * @throws IllegalArgumentException if the length of the specified extra field data is greater
   *    than 0xFFFF bytes
   */
  public void setExtra(ExtraDataList extra) {
    if (extra == null) {
      throw new NullPointerException();
    }
    if (extra.getLength() > 0xffff) {
      throw new IllegalArgumentException("invalid extra field length");
    }
    this.extra = extra;
  }

  /**
   * Returns the extra field data for the entry.
   */
  public ExtraDataList getExtra() {
    return extra;
  }

  /**
   * Sets the optional comment string for the entry.
   */
  public void setComment(@Nullable String comment) {
    this.comment = comment;
  }

  /**
   * Returns the comment string for the entry, or null if none.
   */
  public String getComment() {
    return comment;
  }

  /**
   * Returns the feature set that this entry uses.
   */
  EnumSet<Feature> getFeatureSet() {
    return featureSet;
  }

  @Override
  public String toString() {
    return "ZipFileEntry[" + name + "]";
  }
}
