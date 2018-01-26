// Copyright 2018 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.skyframe.serialization.serializers;

import com.esotericsoftware.kryo.Kryo;
import com.esotericsoftware.kryo.ReferenceResolver;
import com.esotericsoftware.kryo.util.Util;
import java.util.ArrayList;
import java.util.HashMap;

/**
 * {@link ReferenceResolver} implementation that uses object equality.
 *
 * <p>Provided that underlying objects implement equals correctly, this produces canonical, stable,
 * serialized representations.
 *
 * <p>References must match also on class for stability of serialized representation. For example,
 * {@code ArrayList} and {@code LinkedList} objects might evaluate equals, but Kryo has different
 * serialized representations because each distinct type has a different registration index.
 *
 * <p>TODO(shahan): consider changing ClassResolver to allow multiple classes to share a
 * registration in Kryo so this can be avoided.
 */
public class CanonicalReferenceResolver implements ReferenceResolver {

  private final HashMap<Element, Integer> written;
  private final ArrayList<Object> read;

  CanonicalReferenceResolver() {
    this.written = new HashMap<>();
    this.read = new ArrayList<>();
  }

  @Override
  public void setKryo(Kryo unusedKryo) {}

  @Override
  public int getWrittenId(Object object) {
    return written.getOrDefault(new Element(object), -1);
  }

  @Override
  public int addWrittenObject(Object object) {
    int id = written.size();
    written.put(new Element(object), id);
    return id;
  }

  @Override
  public int nextReadId(Class unusedType) {
    int id = read.size();
    read.add(null);
    return id;
  }

  @Override
  public void setReadObject(int id, Object object) {
    read.set(id, object);
  }

  @Override
  public Object getReadObject(Class unusedType, int id) {
    return read.get(id);
  }

  @Override
  public void reset() {
    written.clear();
    read.clear();
  }

  @Override
  public boolean useReferences(Class type) {
    return !Util.isWrapperClass(type);
  }

  /**
   * Wrapper to ensure types are equal in addition to underlying objects.
   *
   * <p>Despite the overhead introduced by churn, this is empirically more efficient than {@link
   * com.google.common.collect.Table}.
   */
  private static class Element {
    public final Object contents;

    private Element(Object contents) {
      this.contents = contents;
    }

    @Override
    public int hashCode() {
      return contents.hashCode();
    }

    @Override
    public boolean equals(Object obj) {
      Element that = (Element) obj;
      return contents.getClass().equals(that.contents.getClass()) && contents.equals(that.contents);
    }
  }
}
