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
 * Provides a view of a zip files "end of central directory" record.
 */
public class EndOfCentralDirectory extends View<EndOfCentralDirectory> {

  /**
   * Returns a {@code EndOfCentralDirectory} of the given buffer.
   *
   * @param buffer containing the data of a "end of central directory" record.
   * @return a {@code EndOfCentralDirectory} of the data at the current position of the given
   * byte buffer.
   */
  public static EndOfCentralDirectory viewOf(ByteBuffer buffer) {
    EndOfCentralDirectory view = new EndOfCentralDirectory(buffer.slice());
    view.buffer.position(0).limit(view.getSize());
    buffer.position(buffer.position() + view.buffer.remaining());
    return view;
  }

  private EndOfCentralDirectory(ByteBuffer buffer) {
    super(buffer);
  }

  /**
   * Creates a {@code EndOfCentralDirectory} with a heap allocated buffer.
   *
   * @param comment zip file comment, or {@code null}.
   * @return a {@code EndOfCentralDirectory} with a heap allocated buffer.
   */
  public static EndOfCentralDirectory allocate(String comment) {
    byte[] commentData = comment != null ? comment.getBytes(UTF_8) : EMPTY;
    int size = SIZE + commentData.length;
    ByteBuffer buffer = ByteBuffer.allocate(size).order(LITTLE_ENDIAN);
    return new EndOfCentralDirectory(buffer).init(commentData);
  }

  /**
   * Creates a {@code EndOfCentralDirectory} over a writable buffer.
   *
   * @param buffer buffer to hold data for the "end of central directory" record.
   * @param comment zip file global comment, or {@code null}.
   * @return a {@code EndOfCentralDirectory} with a heap allocated buffer.
   */
  public static EndOfCentralDirectory view(ByteBuffer buffer, String comment) {
    byte[] commentData = comment != null ? comment.getBytes(UTF_8) : EMPTY;
    int size = SIZE + commentData.length;
    EndOfCentralDirectory view = new EndOfCentralDirectory(buffer.slice()).init(commentData);
    buffer.position(buffer.position() + size);
    return view;
  }

  /**
   * Copies this {@code EndOfCentralDirectory} over a writable buffer.
   *
   * @param buffer writable byte buffer to hold the data of the copy.
   */
  public EndOfCentralDirectory copy(ByteBuffer buffer) {
    EndOfCentralDirectory view = new EndOfCentralDirectory(buffer.slice());
    this.buffer.rewind();
    view.buffer.put(this.buffer).flip();
    this.buffer.rewind();
    buffer.position(buffer.position() + this.buffer.remaining());
    return view;
  }

  private EndOfCentralDirectory init(byte[] comment) {
    buffer.putInt(0, SIGNATURE);
    set(ENDCOM, (short) comment.length);
    if (comment.length > 0) {
      buffer.position(SIZE);
      buffer.put(comment);
    }
    buffer.position(0).limit(SIZE + comment.length);
    return this;
  }

  /**
   * Signature of end of directory record.
   */
  public static final int SIGNATURE = 0x06054b50; //101010256L

  /**
   * Size of end of directory record, not including variable data.
   * Also the offset of the file comment, if any.
   */
  public static final int SIZE = 22;

  /**
   * For accessing the end of central directory signature, with the
   * {@link View#get(com.google.devtools.build.android.ziputils.View.IntFieldId)}
   * and {@link View#set(com.google.devtools.build.android.ziputils.View.IntFieldId, int)}
   * methods.
   */
  public static final IntFieldId<EndOfCentralDirectory> ENDSIG = new IntFieldId<>(0);

  /**
   * For accessing the "this disk number" end of central directory field, with the
   * {@link View#get(com.google.devtools.build.android.ziputils.View.ShortFieldId)}
   * and {@link View#set(com.google.devtools.build.android.ziputils.View.ShortFieldId, short)}
   * methods.
   */
  public static final ShortFieldId<EndOfCentralDirectory> ENDDSK = new ShortFieldId<>(4);


  /**
   * For accessing the "central directory start disk" end of central directory field, with the
   * {@link View#get(com.google.devtools.build.android.ziputils.View.ShortFieldId)}
   * and {@link View#set(com.google.devtools.build.android.ziputils.View.ShortFieldId, short)}
   * methods.
   */
  public static final ShortFieldId<EndOfCentralDirectory> ENDDCD = new ShortFieldId<>(6);

  /**
   * For accessing the "central directory local records" end of central directory field, with the
   * {@link View#get(com.google.devtools.build.android.ziputils.View.ShortFieldId)}
   * and {@link View#set(com.google.devtools.build.android.ziputils.View.ShortFieldId, short)}
   * methods.
   */
  public static final ShortFieldId<EndOfCentralDirectory> ENDSUB = new ShortFieldId<>(8);

  /**
   * For accessing the "central directory total records" end of central directory field, with the
   * {@link View#get(com.google.devtools.build.android.ziputils.View.ShortFieldId)}
   * and {@link View#set(com.google.devtools.build.android.ziputils.View.ShortFieldId, short)}
   * methods.
   */
  public static final ShortFieldId<EndOfCentralDirectory> ENDTOT = new ShortFieldId<>(10);

  /**
   * For accessing the "central directory size" end of central directory field, with the
   * {@link View#get(com.google.devtools.build.android.ziputils.View.IntFieldId)}
   * and {@link View#set(com.google.devtools.build.android.ziputils.View.IntFieldId, int)}
   * methods.
   */
  public static final IntFieldId<EndOfCentralDirectory> ENDSIZ = new IntFieldId<>(12);

  /**
   * For accessing the "central directory offset" end of central directory field, with the
   * {@link View#get(com.google.devtools.build.android.ziputils.View.IntFieldId)}
   * and {@link View#set(com.google.devtools.build.android.ziputils.View.IntFieldId, int)}
   * methods.
   */
  public static final IntFieldId<EndOfCentralDirectory> ENDOFF = new IntFieldId<>(16);

  /**
   * For accessing the "file comment length" end of central directory field, with the
   * {@link View#get(com.google.devtools.build.android.ziputils.View.ShortFieldId)}
   * and {@link View#set(com.google.devtools.build.android.ziputils.View.ShortFieldId, short)}
   * methods.
   */
  public static final ShortFieldId<EndOfCentralDirectory> ENDCOM = new ShortFieldId<>(20);

  /**
   * Returns the file comment.
   * @return the file comment, or an empty string.
   */
  public final String getComment() {
    return getString(SIZE, get(ENDCOM));
  }

  /**
   * Returns the total size of the end of directory record, including file comment, if any.
   * @return the total size of the end of directory record.
   */
  public int getSize() {
    return SIZE + get(ENDCOM);
  }
}
