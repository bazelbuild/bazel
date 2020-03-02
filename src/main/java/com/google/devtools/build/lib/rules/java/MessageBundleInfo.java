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
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.packages.BuiltinProvider;
import com.google.devtools.build.lib.packages.NativeInfo;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec.VisibleForSerialization;
import com.google.devtools.build.lib.skylarkbuildapi.core.ProviderApi;
import com.google.devtools.build.lib.skylarkinterface.Param;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.Sequence;
import com.google.devtools.build.lib.syntax.StarlarkThread;
import com.google.devtools.build.lib.syntax.StarlarkValue;
import java.util.List;
import javax.annotation.Nullable;

/** Marks configured targets that are able to supply message bundles to their dependents. */
@AutoCodec
@Immutable
public final class MessageBundleInfo extends NativeInfo implements StarlarkValue {

  public static final String SKYLARK_NAME = "MessageBundleInfo";

  /** Provider singleton constant. */
  public static final BuiltinProvider<MessageBundleInfo> PROVIDER = new Provider();

  /** Provider class for {@link MessageBundleInfo} objects. */
  @SkylarkModule(name = "Provider", documented = false, doc = "")
  public static class Provider extends BuiltinProvider<MessageBundleInfo> implements ProviderApi {
    private Provider() {
      super(SKYLARK_NAME, MessageBundleInfo.class);
    }

    @SkylarkCallable(
        name = "MessageBundleInfo",
        doc = "The <code>MessageBundleInfo</code> constructor.",
        documented = false,
        parameters = {
          @Param(name = "messages", positional = false, named = true, type = Sequence.class),
        },
        selfCall = true,
        useStarlarkThread = true)
    public MessageBundleInfo messageBundleInfo(Sequence<?> messages, StarlarkThread thread)
        throws EvalException {
      List<Artifact> messagesList = Sequence.castList(messages, Artifact.class, "messages");
      return new MessageBundleInfo(ImmutableList.copyOf(messagesList), thread.getCallerLocation());
    }
  }

  private final ImmutableList<Artifact> messages;

  public static MessageBundleInfo create(ImmutableList<Artifact> messages) {
    return new MessageBundleInfo(messages, null);
  }

  @VisibleForSerialization
  @AutoCodec.Instantiator
  MessageBundleInfo(ImmutableList<Artifact> messages, Location location) {
    super(PROVIDER, location);
    this.messages = Preconditions.checkNotNull(messages);
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
