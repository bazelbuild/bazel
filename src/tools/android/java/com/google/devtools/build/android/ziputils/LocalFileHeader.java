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

import static java.nio.ByteOrder.LITTLE_ENDIAN;
import static java.nio.charset.StandardCharsets.UTF_8;

import java.nio.ByteBuffer;

/**
 * Provides a view of a local file header for a zip file entry.
 */
public class LocalFileHeader extends View<LocalFileHeader> {

  /**
   * Returns a {@code LocalFileHeader} view of the given buffer. The buffer is assumed to contain a
   * valid "central directory entry" record beginning at the buffers current position.
   *
   * @param buffer containing the data of a "central directory entry" record.
   * @return a {@code LocalFileHeader} of the data at the current position of the given byte
   * buffer.
   */
  public static LocalFileHeader viewOf(ByteBuffer buffer) {
    LocalFileHeader view = new LocalFileHeader(buffer.slice());
    view.buffer.limit(view.getSize());
    buffer.position(buffer.position() + view.buffer.remaining());
    return view;
  }

  private LocalFileHeader(ByteBuffer buffer) {
    super(buffer);
  }

  /**
   * Creates a {@code LocalFileHeader} with a heap allocated buffer. Apart from the signature
   * and extra data data (if any), the returned object is uninitialized.
   *
   * @param name entry file name. Cannot be {@code null}.
   * @param extraData extra data, or {@code null}
   * @return a {@code LocalFileHeader} with a heap allocated buffer.
   */
  public static LocalFileHeader allocate(String name, byte[] extraData) {
    byte[] nameData = name.getBytes(UTF_8);
    if (extraData == null) {
      extraData = EMPTY;
    }
    int size = SIZE + nameData.length + extraData.length;
    ByteBuffer buffer = ByteBuffer.allocate(size).order(LITTLE_ENDIAN);
    return new LocalFileHeader(buffer).init(nameData, extraData, size);
  }

  /**
   * Creates a {@code LocalFileHeader} over a writable buffer. The given buffer's position is
   * advanced by the number of bytes consumed by the view. Apart from the signature and extra data
   * (if any), the returned view is uninitialized.
   *
   * @param buffer buffer to hold data for the "central directory entry" record.
   * @param name entry file name. Cannot be {@code null}.
   * @param extraData extra data, or {@code null}
   * @return a {@code DirectoryEntry} with a heap allocated buffer.
   */
  public static LocalFileHeader view(ByteBuffer buffer, String name, byte[] extraData) {
    byte[] nameData = name.getBytes(UTF_8);
    if (extraData == null) {
      extraData = EMPTY;
    }
    int size = SIZE + nameData.length + extraData.length;
    LocalFileHeader view = new LocalFileHeader(buffer.slice()).init(nameData, extraData, size);
    buffer.position(buffer.position() + size);
    return view;
  }

  /**
   * Copies this {@code LocalFileHeader} into a heap allocated buffer, overwriting the current
   * path name and extra data with the given values.
   *
   * @param name entry file name. Cannot be {@code null}.
   * @param extraData extra data, or {@code null}
   * @return a {@code LocalFileHeader} with a heap allocated buffer, initialized with the given
   * name and extra data, and otherwise a copy of this object.
   */
  public LocalFileHeader clone(String name, byte[] extraData) {
    return LocalFileHeader.allocate(name, extraData).copy(this, LOCCRC, LOCSIZ, LOCLEN, LOCFLG,
        LOCHOW, LOCTIM, LOCVER);
  }
  
  /**
   * Copies this {@code LocalFileHeader} into a writable buffer. The copy is made at the
   * buffer's current position, and the position is advanced, by the size of the copy.
   *
   * @param buffer buffer to hold data for the copy.
   * @return a {@code LocalFileHeader} backed by the given buffer.
   */
  public LocalFileHeader copy(ByteBuffer buffer) {
    int size = getSize();
    LocalFileHeader view = new LocalFileHeader(buffer.slice());
    this.buffer.rewind();
    view.buffer.put(this.buffer).flip();
    buffer.position(buffer.position() + size);
    this.buffer.rewind();
    return view;
  }

  private LocalFileHeader init(byte[] name, byte[] extra, int size) {
    buffer.putInt(0, SIGNATURE);
    set(LOCNAM, (short) name.length);
    set(LOCEXT, (short) extra.length);
    buffer.position(SIZE);
    buffer.put(name);
    if (extra.length > 0) {
      buffer.put(extra);
    }
    buffer.position(0).limit(size);
    return this;
  }

  /**
   * Flag used to mark a compressed entry, for which the size is unknown at the time
   * of writing the header.
   */
  public static final short SIZE_MASKED_FLAG = 0x8;

  /**
   * Signature of local file header.
   */
  public static final int SIGNATURE = 0x04034b50; // 67324752L

  /**
   * Size of local file header, not including variable data.
   * Also the offset of the filename.
   */
  public static final int SIZE = 30;

  /**
   * For accessing the local header signature, with the
   * {@link View#get(com.google.devtools.build.android.ziputils.View.IntFieldId)}
   * and {@link View#set(com.google.devtools.build.android.ziputils.View.IntFieldId, int)}
   * methods.
   */
  public static final IntFieldId<LocalFileHeader> LOCSIG = new IntFieldId<>(0);

  /**
   * For accessing the "needed version" local header field, with the
   * {@link View#get(com.google.devtools.build.android.ziputils.View.ShortFieldId)}
   * and {@link View#set(com.google.devtools.build.android.ziputils.View.ShortFieldId, short)}
   * methods.
   */
  public static final ShortFieldId<LocalFileHeader> LOCVER = new ShortFieldId<>(4);

  /**
   * For accessing the "flags" local header field, with the
   * {@link View#get(com.google.devtools.build.android.ziputils.View.ShortFieldId)}
   * and {@link View#set(com.google.devtools.build.android.ziputils.View.ShortFieldId, short)}
   * methods.
   */
  public static final ShortFieldId<LocalFileHeader> LOCFLG = new ShortFieldId<>(6);

  /**
   * For accessing the "method" local header field, with the
   * {@link View#get(com.google.devtools.build.android.ziputils.View.ShortFieldId)}
   * and {@link View#set(com.google.devtools.build.android.ziputils.View.ShortFieldId, short)}
   * methods.
   */
  public static final ShortFieldId<LocalFileHeader> LOCHOW = new ShortFieldId<>(8);

  /**
   * For accessing the "modified time" local header field, with the
   * {@link View#get(com.google.devtools.build.android.ziputils.View.IntFieldId)}
   * and {@link View#set(com.google.devtools.build.android.ziputils.View.IntFieldId, int)}
   * methods.
   */
  public static final IntFieldId<LocalFileHeader> LOCTIM = new IntFieldId<>(10);

  /**
   * For accessing the "crc" local header field, with the
   * {@link View#get(com.google.devtools.build.android.ziputils.View.IntFieldId)}
   * and {@link View#set(com.google.devtools.build.android.ziputils.View.IntFieldId, int)}
   * methods.
   */
  public static final IntFieldId<LocalFileHeader> LOCCRC = new IntFieldId<>(14);

  /**
   * For accessing the "compressed size" local header field, with the
   * {@link View#get(com.google.devtools.build.android.ziputils.View.IntFieldId)}
   * and {@link View#set(com.google.devtools.build.android.ziputils.View.IntFieldId, int)}
   * methods.
   */
  public static final IntFieldId<LocalFileHeader> LOCSIZ = new IntFieldId<>(18);

  /**
   * For accessing the "uncompressed size" local header field, with the
   * {@link View#get(com.google.devtools.build.android.ziputils.View.IntFieldId)}
   * and {@link View#set(com.google.devtools.build.android.ziputils.View.IntFieldId, int)}
   * methods.
   */
  public static final IntFieldId<LocalFileHeader> LOCLEN = new IntFieldId<>(22);

  /**
   * For accessing the "filename length" local header field, with the
   * {@link View#get(com.google.devtools.build.android.ziputils.View.ShortFieldId)}
   * and {@link View#set(com.google.devtools.build.android.ziputils.View.ShortFieldId, short)}
   * methods.
   */
  public static final ShortFieldId<LocalFileHeader> LOCNAM = new ShortFieldId<>(26);

  /**
   * For accessing the "extra data length" local header field, with the
   * {@link View#get(com.google.devtools.build.android.ziputils.View.ShortFieldId)}
   * and {@link View#set(com.google.devtools.build.android.ziputils.View.ShortFieldId, short)}
   * methods.
   */
  public static final ShortFieldId<LocalFileHeader> LOCEXT = new ShortFieldId<>(28);

  /**
   * Returns the filename for this entry.
   */
  public final String getFilename() {
    return getString(SIZE, get(LOCNAM));
  }

  /**
   * Returns the extra data for this entry.
   * @return an array of 0 or more bytes.
   */
  public final byte[] getExtraData() {
    return getBytes(SIZE + get(LOCNAM), get(LOCEXT));
  }

  /**
   * Returns the size of this header, including filename, and extra data (if any).
   */
  public final int getSize() {
    return SIZE + get(LOCNAM) + get(LOCEXT);
  }

  /**
   * Returns entry data size, based on directory entry information. For a valid zip file, this will
   * be the correct size of the entry data, or -1, if the size cannot be determined from the
   * header. Notice, if ths method returns 0, it may  be because the writer of the zip file forgot
   * to set the {@link #SIZE_MASKED_FLAG}.
   *
   * @return if the {@link #LOCHOW} field is 0, returns the value of the
   * {@link #LOCLEN} field (uncompressed size). If {@link #LOCHOW} is not 0, and the
   * {@link #SIZE_MASKED_FLAG} is not set, returns the value of the {@link #LOCSIZ} field
   * (compressed size). Otherwise return -1 (size unknown).
   */
  public int dataSize() {
    return get(LOCHOW) == 0 ? get(LOCLEN)
        : (get(LOCFLG) & SIZE_MASKED_FLAG) == 0 ? get(LOCSIZ) : -1;
  }
}
