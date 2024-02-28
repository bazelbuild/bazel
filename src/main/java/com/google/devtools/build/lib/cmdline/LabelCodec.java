// Copyright 2023 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.cmdline;

import static com.google.devtools.build.lib.unsafe.UnsafeProvider.getFieldOffset;
import static com.google.devtools.build.lib.unsafe.UnsafeProvider.unsafe;

import com.google.devtools.build.lib.skyframe.serialization.AsyncDeserializationContext;
import com.google.devtools.build.lib.skyframe.serialization.InterningObjectCodec;
import com.google.devtools.build.lib.skyframe.serialization.SerializationContext;
import com.google.devtools.build.lib.skyframe.serialization.SerializationException;
import com.google.errorprone.annotations.Keep;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.io.IOException;

/** Custom codec needed to correctly handle name interning. */
@Keep // Used reflectively.
final class LabelCodec extends InterningObjectCodec<Label> {
  private static final long PACKAGE_IDENTIFIER_OFFSET;
  private static final long NAME_OFFSET;

  static {
    try {
      PACKAGE_IDENTIFIER_OFFSET = getFieldOffset(Label.class, "packageIdentifier");
      NAME_OFFSET = getFieldOffset(Label.class, "name");
    } catch (NoSuchFieldException e) {
      throw new ExceptionInInitializerError(e);
    }
  }

  @Override
  public Class<Label> getEncodedClass() {
    return Label.class;
  }

  @Override
  public void serialize(SerializationContext context, Label obj, CodedOutputStream codedOut)
      throws SerializationException, IOException {
    context.serialize(obj.getPackageIdentifier(), codedOut);
    context.serialize(obj.getName(), codedOut);
  }

  @Override
  public Label deserializeInterned(AsyncDeserializationContext context, CodedInputStream codedIn)
      throws SerializationException, IOException {
    Label label;
    try {
      label = (Label) unsafe().allocateInstance(Label.class);
    } catch (ReflectiveOperationException e) {
      throw new SerializationException("Could not instantiate Label", e);
    }
    context.deserialize(codedIn, label, PACKAGE_IDENTIFIER_OFFSET);
    context.deserialize(codedIn, label, LabelCodec::setName);
    return label;
  }

  @Override
  public Label intern(Label label) {
    return Label.getLabelInterner().intern(label);
  }

  private static void setName(Label label, Object name) {
    unsafe().putObject(label, NAME_OFFSET, Label.internIfConstantName((String) name));
  }
}
