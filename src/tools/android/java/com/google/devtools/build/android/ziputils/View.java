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
 * A {@code View} represents a range of a larger sequence of bytes (e.g. part of a file).
 * It consist of an internal byte buffer providing access to the data range, and an
 * offset of the first byte in the buffer within the larger sequence.
 * Subclasses will typically assign a specific interpretations of the data in a view 
 * (e.g. records of a specific type).
 *
 * <p>Instances of subclasses may generally be "allocated", or created "of" or "over" a
 * byte buffer provided at creation time.
 *
 * <p>An "allocated" view gets a new heap allocated byte buffer. Allocation methods
 * typically defines parameters for variable sized part of the object they create a
 * view of. Once allocated, the variable size parts (e.g. filename) cannot be changed.
 *
 * <p>A view created "of" an existing byte buffer, expects the buffer to contain an
 * appropriate record at its current position. The buffers limit is set at the
 * end of the object being viewed.
 *
 * <p>A view created "over" an existing byte buffer, reserves space in the buffer for the
 * object being viewed. The variable sized parts of the object are initialized, but
 * otherwise the existing buffer data are left as-is, and can be manipulated  through the
 * view.
 *
 * <p> An view can also be copied into an existing byte buffer. This is like creating a
 * view "over" the buffer, but initializing the new view as an exact copy of the one
 * being copied.
 *
 * @param <SUB> to be type-safe, a subclass {@code S} must extend {@code View<S>}. It must not
 * extend {@code View<S2>}, where {@code S2} is another subclass, which is not also a superclass
 * of {@code S}. To maintain this guarantee, this class is declared abstract and package private.
 * Unchecked warnings are suppressed as per this specification constraint.
 */
abstract class View<SUB extends View<?>> {

  /** Zero length byte array */
  protected static final byte[] EMPTY = {};

  /** {@code ByteBuffer} backing this view. */
  protected final ByteBuffer buffer;

  /**
   * Offset of first byte covered by this view. For input views, this is the file offset where the
   * item occur. For output, it's the offset at which we expect to write the item (may be -1, for
   * unknown).
   */
  protected long fileOffset;

  /**
   * Creates a view backed by the given {@code ByteBuffer}. Sets the buffer's byte order to
   * little endian, and sets the file offset to -1 (unknown).
   *
   * @param buffer backing byte buffer.
   */
  protected View(ByteBuffer buffer) {
    buffer.order(LITTLE_ENDIAN);
    this.buffer = buffer;
    this.fileOffset = -1;
  }

  /**
   * Sets the file offset of the data item covered by this view,
   *
   * @param fileOffset
   * @return this object.
   */
  @SuppressWarnings("unchecked") // safe by specification
  public SUB at(long fileOffset) {
    this.fileOffset = fileOffset;
    return (SUB) this;
  }

  /**
   * Gets the fileOffset of this view.
   *
   * @return the location of the viewed object within the underlying file.
   */
  public long fileOffset() {
    return fileOffset;
  }

  /**
   * Returns an array with data copied from the backing byte buffer, at the given offset, relative
   * to the beginning of this view. This method does not perform boundary checks of offset or
   * length. This method may temporarily changes the position of the backing buffer. However, after
   * the call, the position will be unchanged.
   *
   * @param off offset relative to this view.
   * @param len number of bytes to return.
   * @return Newly allocated array with copy of requested data.
   * @throws IndexOutOfBoundsException in case of illegal arguments.
   */
  protected byte[] getBytes(int off, int len) {
    if (len == 0) {
      return EMPTY;
    }
    byte[] bytes;
    try {
      bytes = new byte[len];
      int currPos = buffer.position();
      buffer.position(off);
      buffer.get(bytes);
      buffer.position(currPos);
    } catch (Exception ex) {
      throw new IndexOutOfBoundsException();
    }
    return bytes;
  }

  /**
   * Returns a String representation of {@code len} bytes starting at offset {@code off} in this
   * view. This method may temporarily changes the position of the backing buffer. However, after
   * the call, the position will be unchanged.
   *
   * @param off offset relative to backing buffer.
   * @param len number of bytes to return. This method may throw an
   * @return Newly allocated String created by interpreting the specified bytes as UTF-8 data.
   * @throws IndexOutOfBoundsException in case of illegal arguments.
   */
  protected String getString(int off, int len) {
    if (len == 0) {
      return "";
    }
    if (buffer.hasArray()) {
      return new String(buffer.array(), buffer.arrayOffset() + off, len, UTF_8);
    } else {
      return new String(getBytes(off, len), UTF_8);
    }
  }

  /**
   * Gets the value of an identified integer field.
   *
   * @param id field identifier
   * @return the value of the field identified by {@code id}.
   */
  public int get(IntFieldId<? extends SUB> id) {
    return buffer.getInt(id.address());
  }

  /**
   * Gets the value of an identified short field.
   *
   * @param id field identifier
   * @return the value of the field identified by {@code id}.
   */
  public short get(ShortFieldId<? extends SUB> id) {
    return buffer.getShort(id.address());
  }

  /**
   * Sets the value of an identified integer field.
   *
   * @param id field identifier
   * @param value value to set for the field identified by {@code id}.
   * @return this object.
   */
  @SuppressWarnings("unchecked") // safe by specification
  public SUB set(IntFieldId<? extends SUB> id, int value) {
    buffer.putInt(id.address(), value);
    return (SUB) this;
  }

  /**
   * Sets the value of an identified short field.
   *
   * @param id field identifier
   * @param value value to set for the field identified by {@code id}.
   * @return this object.
   */
  @SuppressWarnings("unchecked") // safe by specification
  public SUB set(ShortFieldId<? extends SUB> id, short value) {
    buffer.putShort(id.address(), value);
    return (SUB) this;
  }

  /**
   * Copies the values of one or more identified fields from another view to this view.
   *
   * @param from The view from which to copy field values.
   * @param ids field identifiers for fields to copy.
   * @return this object.
   */
  @SuppressWarnings("unchecked") // safe by specification
  public SUB copy(View<SUB> from, FieldId<? extends SUB, ?>... ids) {
    for (FieldId<? extends SUB, ?> id : ids) {
      int address = id.address;
      buffer.put(address, from.buffer.get(address++));
      buffer.put(address, from.buffer.get(address++));
      if (id.type() == Integer.TYPE) {
        buffer.put(address, from.buffer.get(address++));
        buffer.put(address, from.buffer.get(address));
      }
    }
    return (SUB) this;
  }

  /**
   * Base class for data field descriptors. Describes a data field's type and address in a view.
   * This base class allows the
   * {@link #copy(View, com.google.devtools.build.android.ziputils.View.FieldId[])} method
   * to operate of fields of mixed types.
   *
   * @param <T> {@code Integer.TYPE} or {@code Short.TYPE}.
   * @param <V> subclass of {@code View} for which a field id is defined.
   */
  protected abstract static class FieldId<V extends View<?>, T> {
    private final int address;
    private final Class<T> type;

    protected FieldId(int address, Class<T> type) {
      this.address = address;
      this.type = type;
    }

    /**
     * Returns the class of the field type, {@code Class<T>}.
     */
    public Class<T> type() {
      return type;
    }

    /**
     * Returns the field address, within a record of type {@code V}
     */
    public int address() {
      return address;
    }
  }

  /**
   * Describes an integer fields for a view.
   *
   * @param <V> subclass of {@code View} for which a field id is defined.
   */
  protected static class IntFieldId<V extends View<?>> extends FieldId<V, Integer> {
    protected IntFieldId(int address) {
      super(address, Integer.TYPE);
    }
  }

  /**
   * Describes a short field for a view.
   *
   * @param <V> subclass of {@code View} for which a field id is defined.
   */
  protected static class ShortFieldId<V extends View<?>> extends FieldId<V, Short> {
    protected ShortFieldId(int address) {
      super(address, Short.TYPE);
    }
  }
}
