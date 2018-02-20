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

package com.google.devtools.build.lib.skyframe.serialization;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableMap;
import com.google.protobuf.CodedOutputStream;
import java.io.IOException;

/** Stateful class for providing additional context to a single serialization "session". */
// TODO(bazel-team): This class is just a shell, fill in.
public class SerializationContext {

  private final ObjectCodecRegistry registry;
  private final ImmutableMap<Class<?>, Object> dependencies;

  public SerializationContext(
      ObjectCodecRegistry registry, ImmutableMap<Class<?>, Object> dependencies) {
    this.registry = registry;
    this.dependencies = dependencies;
  }

  public SerializationContext(ImmutableMap<Class<?>, Object> dependencies) {
    this(AutoRegistry.get(), dependencies);
  }

  // TODO(shahan): consider making codedOut a member of this class.
  public void serialize(Object object, CodedOutputStream codedOut)
      throws IOException, SerializationException {
    if (object == null) {
      codedOut.writeSInt32NoTag(0);
      return;
    }
    ObjectCodecRegistry.CodecDescriptor descriptor = registry.getCodecDescriptor(object.getClass());
    codedOut.writeSInt32NoTag(descriptor.getTag());
    descriptor.serialize(this, object, codedOut);
  }

  @SuppressWarnings("unchecked")
  public <T> T getDependency(Class<T> type) {
    Preconditions.checkNotNull(type);
    return (T) dependencies.get(type);
  }
}
