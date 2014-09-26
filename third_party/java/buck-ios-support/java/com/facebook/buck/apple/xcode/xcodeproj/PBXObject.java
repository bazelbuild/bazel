/*
 * Copyright 2013-present Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may
 * not use this file except in compliance with the License. You may obtain
 * a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 */

package com.facebook.buck.apple.xcode.xcodeproj;

import com.facebook.buck.apple.xcode.GidGenerator;
import com.facebook.buck.apple.xcode.XcodeprojSerializer;

import javax.annotation.Nullable;

public abstract class PBXObject {
  @Nullable
  private String globalID;

  @Nullable
  public String getGlobalID() {
    return globalID;
  }

  public void setGlobalID(String gid) {
    globalID = gid;
  }

  /**
   * @return  Type name of the serialized object.
   */
  public abstract String isa();

  /**
   * Populates the serializer with the fields of this object.
   */
  public void serializeInto(@SuppressWarnings("unused") XcodeprojSerializer serializer) {
  }

  /**
   * This method is used to generate stable GIDs and must be stable for identical contents.
   * Returning a constant value is ok but will make the generated project order-dependent.
   */
  public int stableHash() {
    return 0;
  }

  @Override
  public String toString() {
    return String.format("%s isa=%s gid=%s", super.toString(), isa(), getGlobalID());
  }

  /**
   * Generate a stable GID.
   */
  public final String generateGid(GidGenerator generator) {
    return generator.generateGid(isa(), stableHash());
  }
}
