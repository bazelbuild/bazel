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
 * Provides a view of a central directory entry.
 */
public class DirectoryEntry extends View<DirectoryEntry> {

  /**
   * Returns a {@code DirectoryEntry of the given buffer. The buffer is assumed to contain a
   * valid "central directory entry" record beginning at the buffers current position.
   *
   * @param buffer containing the data of a "central directory entry" record.
   * @return a {@code DirectoryEntry} of the data at the current position of the given byte
   * buffer.
   */
  public static DirectoryEntry viewOf(ByteBuffer buffer) {
    DirectoryEntry view = new DirectoryEntry(buffer.slice());
    int size = view.getSize();
    buffer.position(buffer.position() + size);
    view.buffer.position(0).limit(size);
    return view;
  }

  private DirectoryEntry(ByteBuffer buffer) {
    super(buffer);
  }

  /**
   * Creates a {@code DirectoryEntry} with a heap allocated buffer. Apart from the signature,
   * and provided extra data and comment data, the returned object is uninitialized.
   *
   * @param name entry file name. Cannot be {@code null}.
   * @param extraData extra data, or {@code null}
   * @param comment zip file comment, or {@code null}.
   * @return a {@code DirectoryEntry} with a heap allocated buffer.
   */
  public static DirectoryEntry allocate(String name, byte[] extraData, String comment) {
    byte[] nameData = name.getBytes(UTF_8);
    byte[] commentData = comment != null ? comment.getBytes(UTF_8) : EMPTY;
    if (extraData == null) {
      extraData = EMPTY;
    }
    int size = SIZE + nameData.length + extraData.length + commentData.length;
    ByteBuffer buffer = ByteBuffer.allocate(size).order(LITTLE_ENDIAN);
    return new DirectoryEntry(buffer).init(nameData, extraData, commentData, size);
  }

  /**
   * Creates a {@code DirectoryEntry} over a writable buffer. The given buffers position is
   * advanced by the number of bytes consumed by the view. Apart from the signature, and
   * provided extra data and comment data, the returned view is uninitialized.
   *
   * @param buffer buffer to hold data for the "central directory entry" record.
   * @param name entry file name. Cannot be {@code null}.
   * @param extraData extra data, or {@code null}
   * @param comment zip file global comment, or {@code null}.
   * @return a {@code DirectoryEntry} with a heap allocated buffer.
   */
  public static DirectoryEntry view(ByteBuffer buffer, String name, byte[] extraData,
      String comment) {
    byte[] nameData = name.getBytes(UTF_8);
    byte[] commentData = comment != null ? comment.getBytes(UTF_8) : EMPTY;
    if (extraData == null) {
      extraData = EMPTY;
    }
    int size = SIZE + nameData.length + extraData.length + commentData.length;
    DirectoryEntry view = new DirectoryEntry(buffer.slice()).init(nameData, extraData,
        commentData, size);
    buffer.position(buffer.position() + size);
    return view;
  }

  /**
   * Copies this {@code DirectoryEntry} into a heap allocated buffer, overwriting path name,
   * extra data and comment with the given values.
   *
   * @param name entry file name. Cannot be {@code null}.
   * @param extraData extra data, or {@code null}
   * @param comment zip file global comment, or {@code null}.
   * @return a {@code DirectoryEntry} with a heap allocated buffer, initialized with the
   * given values, and otherwise as a copy of this object.
   */
  public DirectoryEntry clone(String name, byte[] extraData, String comment) {
    return DirectoryEntry.allocate(name, extraData, comment).copy(this, CENOFF, CENCRC, CENSIZ,
        CENLEN, CENFLG, CENHOW, CENTIM, CENVER, CENVER, CENDSK, CENATX, CENATT);
  }
  
  /**
   * Copies this {@code DirectoryEntry} into a writable buffer. The copy is made at the
   * buffer's current position, and the position is advanced, by the size of the copy.
   *
   * @param buffer buffer to hold data for the copy.
   * @return a {@code DirectoryEntry} backed by the given buffer.
   */
  public DirectoryEntry copy(ByteBuffer buffer) {
    int size = getSize();
    DirectoryEntry view = new DirectoryEntry(buffer.slice());
    this.buffer.rewind();
    view.buffer.put(this.buffer).flip();
    buffer.position(buffer.position() + size);
    this.buffer.rewind().limit(size);
    return view;
  }

  private DirectoryEntry init(byte[] name, byte[] extra, byte[] comment, int size) {
    buffer.putInt(0, SIGNATURE);
    set(CENNAM, (short) name.length);
    set(CENEXT, (short) extra.length);
    set(CENCOM, (short) comment.length);
    buffer.position(SIZE);
    buffer.put(name);
    if (extra.length > 0) {
      buffer.put(extra);
    }
    if (comment.length > 0) {
      buffer.put(comment);
    }
    buffer.position(0).limit(size);
    return this;
  }

  public final int getSize() {
    return SIZE + get(CENNAM) + get(CENEXT) + get(CENCOM);
  }

  /**
   * Directory entry signature.
   */
  public static final int SIGNATURE = 0x02014b50; // 33639248L

  /**
   * Size of directory entry, not including variable data.
   * Also the offset of the entry filename.
   */
  public static final int SIZE = 46;

  /**
   * For accessing the directory entry signature, with the
   * {@link View#get(com.google.devtools.build.android.ziputils.View.IntFieldId)}
   * and {@link View#set(com.google.devtools.build.android.ziputils.View.IntFieldId, int)}
   * methods.
   */
  public static final IntFieldId<DirectoryEntry> CENSIG = new IntFieldId<>(0);

  /**
   * For accessing the "made by version" directory entry field, with the
   * {@link View#get(com.google.devtools.build.android.ziputils.View.ShortFieldId)}
   * and {@link View#set(com.google.devtools.build.android.ziputils.View.ShortFieldId, short)}
   * methods.
   */
  public static final ShortFieldId<DirectoryEntry> CENVEM = new ShortFieldId<>(4);

  /**
   * For accessing the "version needed" directory entry field, with the
   * {@link View#get(com.google.devtools.build.android.ziputils.View.ShortFieldId)}
   * and {@link View#set(com.google.devtools.build.android.ziputils.View.ShortFieldId, short)}
   * methods.
   */
  public static final ShortFieldId<DirectoryEntry> CENVER = new ShortFieldId<>(6);

  /**
   * For accessing the "flags" directory entry field, with the
   * {@link View#get(com.google.devtools.build.android.ziputils.View.ShortFieldId)}
   * and {@link View#set(com.google.devtools.build.android.ziputils.View.ShortFieldId, short)}
   * methods.
   */
  public static final ShortFieldId<DirectoryEntry> CENFLG = new ShortFieldId<>(8);

  /**
   * For accessing the "method" directory entry field, with the
   * {@link View#get(com.google.devtools.build.android.ziputils.View.ShortFieldId)}
   * and {@link View#set(com.google.devtools.build.android.ziputils.View.ShortFieldId, short)}
   * methods.
   */
  public static final ShortFieldId<DirectoryEntry> CENHOW = new ShortFieldId<>(10);

  /**
   * For accessing the "modified time" directory entry field, with the
   * {@link View#get(com.google.devtools.build.android.ziputils.View.IntFieldId)}
   * and {@link View#set(com.google.devtools.build.android.ziputils.View.IntFieldId, int)}
   * methods.
   */
  public static final IntFieldId<DirectoryEntry> CENTIM = new IntFieldId<>(12);

  /**
   * For accessing the "crc" directory entry field, with the
   * {@link View#get(com.google.devtools.build.android.ziputils.View.IntFieldId)}
   * and {@link View#set(com.google.devtools.build.android.ziputils.View.IntFieldId, int)}
   * methods.
   */
  public static final IntFieldId<DirectoryEntry> CENCRC = new IntFieldId<>(16);

  /**
   * For accessing the "compressed size" directory entry field, with the
   * {@link View#get(com.google.devtools.build.android.ziputils.View.IntFieldId)}
   * and {@link View#set(com.google.devtools.build.android.ziputils.View.IntFieldId, int)}
   * methods.
   */
  public static final IntFieldId<DirectoryEntry> CENSIZ = new IntFieldId<>(20);

  /**
   * For accessing the "uncompressed size" directory entry field, with the
   * {@link View#get(com.google.devtools.build.android.ziputils.View.IntFieldId)}
   * and {@link View#set(com.google.devtools.build.android.ziputils.View.IntFieldId, int)}
   * methods.
   */
  public static final IntFieldId<DirectoryEntry> CENLEN = new IntFieldId<>(24);

  /**
   * For accessing the "filename length" directory entry field, with the
   * {@link View#get(com.google.devtools.build.android.ziputils.View.ShortFieldId)}
   * and {@link View#set(com.google.devtools.build.android.ziputils.View.ShortFieldId, short)}
   * methods.
   */
  public static final ShortFieldId<DirectoryEntry> CENNAM = new ShortFieldId<>(28);

  /**
   * For accessing the "extra data length" directory entry field, with the
   * {@link View#get(com.google.devtools.build.android.ziputils.View.ShortFieldId)}
   * and {@link View#set(com.google.devtools.build.android.ziputils.View.ShortFieldId, short)}
   * methods.
   */
  public static final ShortFieldId<DirectoryEntry> CENEXT = new ShortFieldId<>(30);

  /**
   * For accessing the "file comment length" directory entry field, with the
   * {@link View#get(com.google.devtools.build.android.ziputils.View.ShortFieldId)}
   * and {@link View#set(com.google.devtools.build.android.ziputils.View.ShortFieldId, short)}
   * methods.
   */
  public static final ShortFieldId<DirectoryEntry> CENCOM = new ShortFieldId<>(32);

  /**
   * For accessing the "disk number" directory entry field, with the
   * {@link View#get(com.google.devtools.build.android.ziputils.View.ShortFieldId)}
   * and {@link View#set(com.google.devtools.build.android.ziputils.View.ShortFieldId, short)}
   * methods.
   */
  public static final ShortFieldId<DirectoryEntry> CENDSK = new ShortFieldId<>(34);

  /**
   * For accessing the "internal attributes" directory entry field, with the
   * {@link View#get(com.google.devtools.build.android.ziputils.View.ShortFieldId)}
   * and {@link View#set(com.google.devtools.build.android.ziputils.View.ShortFieldId, short)}
   * methods.
   */
  public static final ShortFieldId<DirectoryEntry> CENATT = new ShortFieldId<>(36);

  /**
   * For accessing the "external attributes" directory entry field, with the
   * {@link View#get(com.google.devtools.build.android.ziputils.View.IntFieldId)}
   * and {@link View#set(com.google.devtools.build.android.ziputils.View.IntFieldId, int)}
   * methods.
   */
  public static final IntFieldId<DirectoryEntry> CENATX = new IntFieldId<>(38);

  /**
   * For accessing the "local file header offset" directory entry field, with the
   * {@link View#get(com.google.devtools.build.android.ziputils.View.IntFieldId)}
   * and {@link View#set(com.google.devtools.build.android.ziputils.View.IntFieldId, int)}
   * methods.
   */
  public static final IntFieldId<DirectoryEntry> CENOFF = new IntFieldId<>(42);

  /**
   * Returns the filename of this entry.
   */
  public final String getFilename() {
    return getString(SIZE, get(CENNAM));
  }

  /**
   * Returns the extra data of this entry.
   * @return a byte array with 0 or more bytes.
   */
  public final byte[] getExtraData() {
    return getBytes(SIZE + get(CENNAM), get(CENEXT));
  }

  /**
   * Return the comment of this entry.
   * @return a string with 0 or more characters.
   */
  public final String getComment() {
    return getString(SIZE + get(CENNAM) + get(CENEXT), get(CENCOM));
  }

  /**
   * Returns entry data size, based on directory entry information. For a valid zip file, this will
   * be the correct size of of the entry data.
   * @return if the {@link #CENHOW} field is 0, returns the value of the
   * {@link #CENLEN} field (uncompressed size), otherwise returns the value of
   * the {@link #CENSIZ} field (compressed size).
   */
  public int dataSize() {
    return get(CENHOW) == 0 ? get(CENLEN) : get(CENSIZ);
  }
}
