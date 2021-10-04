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

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.BuiltinProvider;
import com.google.devtools.build.lib.packages.NativeInfo;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec.VisibleForSerialization;
import com.google.devtools.build.lib.starlarkbuildapi.core.ProviderApi;
import com.google.devtools.build.lib.starlarkbuildapi.java.MessageBundleInfoApi;
import java.util.List;
import javax.annotation.Nullable;
import net.starlark.java.annot.Param;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Sequence;
import net.starlark.java.eval.StarlarkList;
import net.starlark.java.eval.StarlarkThread;
import net.starlark.java.syntax.Location;

/** Marks configured targets that are able to supply message bundles to their dependents. */
@AutoCodec
@Immutable
public final class MessageBundleInfo extends NativeInfo implements MessageBundleInfoApi<Artifact> {

  public static final String STARLARK_NAME = "MessageBundleInfo";

  /** Provider singleton constant. */
  public static final BuiltinProvider<MessageBundleInfo> PROVIDER = new Provider();

  /** Provider class for {@link MessageBundleInfo} objects. */
  @StarlarkBuiltin(name = "Provider", documented = false, doc = "")
  public static class Provider extends BuiltinProvider<MessageBundleInfo> implements ProviderApi {
    private Provider() {
      super(STARLARK_NAME, MessageBundleInfo.class);
    }

    @StarlarkMethod(
        name = "MessageBundleInfo",
        doc = "The <code>MessageBundleInfo</code> constructor.",
        documented = false,
        parameters = {
          @Param(name = "messages", positional = false, named = true),
        },
        selfCall = true,
        useStarlarkThread = true)
    public MessageBundleInfo messageBundleInfo(Sequence<?> messages, StarlarkThread thread)
        throws EvalException {
      List<Artifact> messagesList = Sequence.cast(messages, Artifact.class, "messages");
      return new MessageBundleInfo(ImmutableList.copyOf(messagesList), thread.getCallerLocation());
    }
  }

  private final ImmutableList<Artifact> messages;

  public static MessageBundleInfo create(ImmutableList<Artifact> messages) {
    return new MessageBundleInfo(messages, null);
  }

  @VisibleForSerialization
  @AutoCodec.Instantiator
  MessageBundleInfo(ImmutableList<Artifact> messages, Location creationLocation) {
    super(creationLocation);
    this.messages = Preconditions.checkNotNull(messages);
  }

  @Override
  public BuiltinProvider<MessageBundleInfo> getProvider() {
    return PROVIDER;
  }

  @Override
  public Sequence<Artifact> getMessageBundles() {
    return StarlarkList.immutableCopyOf(getMessages());
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
