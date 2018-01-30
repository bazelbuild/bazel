/* Copyright (c) 2008, Nathan Sweet
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted
 * provided that the following conditions are met:
 *
 * - Redistributions of source code must retain the above copyright notice, this list of conditions
 *   and the following disclaimer.
 * - Redistributions in binary form must reproduce the above copyright notice, this list of
 *   conditions and the following disclaimer in the documentation and/or other materials provided
 *   with the distribution.
 * - Neither the name of Esoteric Software nor the names of its contributors may be used to endorse
 *   or promote products derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 * OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE. */

package com.esotericsoftware.kryo;

/**
 * When references are enabled, this tracks objects that have already been read or written, provides
 * an ID for objects that are written, and looks up by ID objects that have been read.
 *
 * @author Nathan Sweet <misc@n4te.com>
 */
public interface ReferenceResolver {
  /**
   * Sets the Kryo instance that this ClassResolver will be used for. This is called automatically
   * by Kryo.
   */
  public void setKryo(Kryo kryo);

  /** Returns an ID for the object if it has been written previously, otherwise returns -1. */
  public int getWrittenId(Object object);

  /**
   * Returns a new ID for an object that is being written for the first time.
   *
   * @return The ID, which is stored more efficiently if it is positive and must not be -1 or -2.
   */
  public int addWrittenObject(Object object);

  /**
   * Reserves the ID for the next object that will be read. This is called only the first time an
   * object is encountered.
   *
   * @param type The type of object that will be read.
   * @return The ID, which is stored more efficiently if it is positive and must not be -1 or -2.
   */
  public int nextReadId(Class type);

  /**
   * Sets the ID for an object that has been read.
   *
   * @param id The ID from {@link #nextReadId(Class)}.
   */
  public void setReadObject(int id, Object object);

  /**
   * Returns the object for the specified ID. The ID and object are guaranteed to have been
   * previously passed in a call to {@link #setReadObject(int, Object)}.
   */
  public Object getReadObject(Class type, int id);

  /** Called by {@link Kryo#reset()}. */
  public void reset();

  /**
   * Returns true if references will be written for the specified type.
   *
   * @param type Will never be a primitive type, but may be a primitive type wrapper.
   */
  public boolean useReferences(Class type);
}
