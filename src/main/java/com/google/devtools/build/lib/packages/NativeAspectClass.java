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
package com.google.devtools.build.lib.packages;

import static com.google.devtools.build.lib.skyframe.serialization.strings.UnsafeStringCodec.stringCodec;

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.skyframe.serialization.LeafDeserializationContext;
import com.google.devtools.build.lib.skyframe.serialization.LeafObjectCodec;
import com.google.devtools.build.lib.skyframe.serialization.LeafSerializationContext;
import com.google.devtools.build.lib.skyframe.serialization.SerializationException;
import com.google.errorprone.annotations.Keep;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.io.IOException;

/**
 * A class of aspects that are implemented natively in Bazel.
 *
 * <p>This class just wraps a {@link java.lang.Class} implementing the
 * aspect factory. All wrappers of the same class are equal.
 */
public abstract class NativeAspectClass implements AspectClass {
  @Override
  public String getName() {
    return getClass().getSimpleName();
  }

  public abstract AspectDefinition getDefinition(AspectParameters aspectParameters);

  @Keep // used reflectively
  private static class Codec extends LeafObjectCodec<NativeAspectClass> {
    @Override
    public Class<NativeAspectClass> getEncodedClass() {
      return NativeAspectClass.class;
    }

    @Override
    public void serialize(
        LeafSerializationContext context, NativeAspectClass obj, CodedOutputStream codedOut)
        throws SerializationException, IOException {
      RuleClassProvider ruleClassProvider = context.getDependency(RuleClassProvider.class);
      NativeAspectClass storedAspect = ruleClassProvider.getNativeAspectClass(obj.getKey());
      Preconditions.checkState(
          obj == storedAspect, "Not stored right: %s %s %s", obj, storedAspect, ruleClassProvider);
      context.serializeLeaf(obj.getKey(), stringCodec(), codedOut);
    }

    @Override
    public NativeAspectClass deserialize(
        LeafDeserializationContext context, CodedInputStream codedIn)
        throws SerializationException, IOException {
      String aspectKey = context.deserializeLeaf(codedIn, stringCodec());
      return Preconditions.checkNotNull(
          context.getDependency(RuleClassProvider.class).getNativeAspectClass(aspectKey),
          aspectKey);
    }
  }
}
