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

import java.nio.ByteBuffer;

/**
 * Provides a view of a data descriptor record, for a zip file entry.
 */
public class DataDescriptor extends View<DataDescriptor> {
  private boolean hasMarker;

  /**
   * Returns a {@code DataDescriptor} view of the given buffer. The buffer is assumed to contain a
   * valid "data descriptor" record beginning at the buffers current position.
   *
   * @param buffer containing the data of a data descriptor record.
   * @return a {@code DataDescriptor} of the data at the current position of the given byte
   * buffer.
   */
  public static DataDescriptor viewOf(ByteBuffer buffer) {
    DataDescriptor view = new DataDescriptor(buffer.slice());
    int size = view.getSize();
    view.buffer.position(0).limit(size);
    buffer.position(buffer.position() + size);
    return view;
  }

  private DataDescriptor(ByteBuffer buffer) {
    super(buffer);
    hasMarker = buffer.getInt(0) == SIGNATURE;
  }

  /**
   * Creates a {@code DataDescriptor} with a heap allocated buffer.
   *
   * @return a {@code DataDescriptor} with a heap allocated buffer.
   */
  public static DataDescriptor allocate() {
    ByteBuffer buffer = ByteBuffer.allocate(SIZE).order(LITTLE_ENDIAN);
    return new DataDescriptor(buffer).init();
  }

  /**
   * Creates a {@code DataDescriptor} view over a writable buffer. The given buffers position is
   * advanced by the number of bytes consumed by the view.
   *
   * @param buffer buffer to hold data for the "data descriptor" record.
   * @return a {@code DataDescriptor} with a heap allocated buffer.
   */
  public static DataDescriptor view(ByteBuffer buffer) {
    DataDescriptor view = new DataDescriptor(buffer.slice()).init();
    buffer.position(buffer.position() + SIZE);
    return view;
  }

  /**
   * Copies this {@code DataDescriptor} into a writable buffer. The copy is made at the
   * buffer's current position, and the position is advanced, by the size of the copy.
   *
   * @param buffer buffer to hold data for the copy.
   * @return a {@code DataDescriptor} backed by the given buffer.
   */
  public DataDescriptor copy(ByteBuffer buffer) {
    int size = getSize();
    DataDescriptor view = new DataDescriptor(buffer.slice());
    this.buffer.rewind();
    view.buffer.put(this.buffer).flip();
    buffer.position(buffer.position() + size);
    this.buffer.rewind();
    return view;
  }

  private DataDescriptor init() {
    buffer.putInt(0, SIGNATURE);
    hasMarker = true;
    buffer.limit(SIZE);
    return this;
  }

  /**
   * Data descriptor signature.
   */
  public static final int SIGNATURE = 0x08074b50; // 134695760L

  /**
   * Data descriptor size (including optional signature).
   */
  public static final int SIZE = 16;

  /**
   * For accessing the data descriptor signature, with the
   * {@link View#get(com.google.devtools.build.android.ziputils.View.IntFieldId)}
   * and {@link View#set(com.google.devtools.build.android.ziputils.View.IntFieldId, int)}
   * methods.
   */
  public static final IntFieldId<DataDescriptor> EXTSIG = new IntFieldId<>(0);

  /**
   * For accessing the "crc" data descriptor field, with the
   * {@link View#get(com.google.devtools.build.android.ziputils.View.IntFieldId)}
   * and {@link View#set(com.google.devtools.build.android.ziputils.View.IntFieldId, int)}
   * methods.
   */
  public static final IntFieldId<DataDescriptor> EXTCRC = new IntFieldId<>(4);

  /**
   * For accessing the "compressed size" data descriptor field, with the
   * {@link View#get(com.google.devtools.build.android.ziputils.View.IntFieldId)}
   * and {@link View#set(com.google.devtools.build.android.ziputils.View.IntFieldId, int)}
   * methods.
   */
  public static final IntFieldId<DataDescriptor> EXTSIZ = new IntFieldId<>(8);

  /**
   * For accessing the "uncompressed size" data descriptor field, with the
   * {@link View#get(com.google.devtools.build.android.ziputils.View.IntFieldId)}
   * and {@link View#set(com.google.devtools.build.android.ziputils.View.IntFieldId, int)}
   * methods.
   */
  public static final IntFieldId<DataDescriptor> EXTLEN = new IntFieldId<>(12);


  /**
   * Overrides the generic field getter, to handle optionality of signature.
   * @see View#get(com.google.devtools.build.android.ziputils.View.IntFieldId).
   */
  @Override  @SuppressWarnings("unchecked") // safe by specification (FieldId.type()).
  public int get(IntFieldId<? extends DataDescriptor> item) {
    int address = hasMarker ? item.address() : (item.address() - 4);
    return address < 0 ? -1 : buffer.getInt(address);
  }

  /**
   * Returns whether this data descriptor has the optional signature.
   * @return {@code true} if this data descriptor has a signature, {@code false} otherwise.
   */
  public final boolean hasMarker() {
    return hasMarker;
  }

  /**
   * Returns the size of this data descriptor.
   * @return 12, or 16, depending on whether or not this data descriptor has a signature.
   */
  public final int getSize() {
    return hasMarker ? SIZE : SIZE - 4;
  }
}
