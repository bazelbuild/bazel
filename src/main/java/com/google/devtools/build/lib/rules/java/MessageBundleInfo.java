// Copyright 2014 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.rules.java;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.packages.NativeInfo;
import com.google.devtools.build.lib.packages.NativeProvider;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec.VisibleForSerialization;
import com.google.devtools.build.lib.syntax.Environment;
import com.google.devtools.build.lib.syntax.FunctionSignature;
import com.google.devtools.build.lib.syntax.SkylarkList;
import com.google.devtools.build.lib.syntax.SkylarkType;
import javax.annotation.Nullable;

/** Marks configured targets that are able to supply message bundles to their dependents. */
@AutoCodec
@Immutable
public final class MessageBundleInfo extends NativeInfo {

  public static final String SKYLARK_NAME = "MessageBundleInfo";

  private static final SkylarkType LIST_OF_ARTIFACTS =
      SkylarkType.Combination.of(SkylarkType.SEQUENCE, SkylarkType.of(Artifact.class));

  private static final FunctionSignature.WithValues<Object, SkylarkType> SIGNATURE =
      FunctionSignature.WithValues.create(
          FunctionSignature.namedOnly("messages"),
          /*defaultValues=*/ null,
          /*types=*/ ImmutableList.of(LIST_OF_ARTIFACTS));

  public static final NativeProvider<MessageBundleInfo> PROVIDER =
      new NativeProvider<MessageBundleInfo>(MessageBundleInfo.class, SKYLARK_NAME, SIGNATURE) {
        @Override
        @SuppressWarnings("unchecked")
        protected MessageBundleInfo createInstanceFromSkylark(
            Object[] args, Environment env, Location loc) {
          return new MessageBundleInfo(ImmutableList.copyOf((SkylarkList<Artifact>) args[0]), loc);
        }
      };

  private final ImmutableList<Artifact> messages;

  public static MessageBundleInfo create(ImmutableList<Artifact> messages) {
    return new MessageBundleInfo(messages, null);
  }

  @VisibleForSerialization
  @AutoCodec.Instantiator
  MessageBundleInfo(ImmutableList<Artifact> messages, Location location) {
    super(PROVIDER, location);
    this.messages = ImmutableList.copyOf(messages);
  }

  public Location getLocation() {
    return location;
  }

  public ImmutableList<Artifact> getMessages() {
    return messages;
  }

  @Nullable
  public static ImmutableList<Artifact> getMessages(TransitiveInfoCollection info) {
    MessageBundleInfo messageBundleInfo =
        (MessageBundleInfo) info.get(MessageBundleInfo.PROVIDER.getKey());
    if (messageBundleInfo != null) {
      return messageBundleInfo.getMessages();
    }
    return null;
  }
}
