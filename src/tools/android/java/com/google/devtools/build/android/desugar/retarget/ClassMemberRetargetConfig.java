/*
 * Copyright 2020 The Bazel Authors. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.google.devtools.build.android.desugar.retarget;

import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.auto.value.AutoValue;
import com.google.auto.value.extension.memoized.Memoized;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.io.Resources;
import com.google.devtools.build.android.desugar.langmodel.MethodInvocationSite;
import com.google.protobuf.ExtensionRegistry;
import com.google.protobuf.TextFormat;
import java.io.IOError;
import java.io.IOException;
import java.net.URL;
import java.util.Collection;
import java.util.LinkedHashSet;
import java.util.Set;
import javax.annotation.Nullable;

/**
 * The configuration that specifies the class members subject to replacement with a new target class
 * member.
 */
@AutoValue
public abstract class ClassMemberRetargetConfig {

  public static final URL DEFAULT_PROTO_URL =
      ClassMemberRetargetConfig.class.getResource("retarget_config.textproto");

  /** The configuration file that defines class member retargeting. */
  abstract ImmutableList<URL> invocationReplacementConfigUrls();

  abstract ImmutableList<MethodInvocationReplacement> inProcessReplacements();

  /**
   * The enabled configuration flags for invocation replacements. A replacement takes effect if and
   * only if the set intersection of this flag and the {@code range} field of {@code
   * MethodInvocations#replacements} is non-empty.
   */
  abstract ImmutableSet<ReplacementRange> enabledInvocationReplacementRanges();

  public static ClassMemberRetargetConfig.Builder builder() {
    return new AutoValue_ClassMemberRetargetConfig.Builder();
  }

  @Memoized
  MethodInvocations invocationReplacementConfigProto() {
    MethodInvocations.Builder invocationReplacementsBuilder = MethodInvocations.newBuilder();
    try {
      for (URL url : invocationReplacementConfigUrls()) {
        String protoText = Resources.toString(url, UTF_8);
        MethodInvocations methodInvocations =
            TextFormat.parse(
                protoText, ExtensionRegistry.getEmptyRegistry(), MethodInvocations.class);
        invocationReplacementsBuilder.mergeFrom(methodInvocations);
      }
      invocationReplacementsBuilder.addAllReplacements(inProcessReplacements());
      return invocationReplacementsBuilder.build();
    } catch (IOException e) {
      throw new IOError(e);
    }
  }

  /**
   * The parsed invocation replacement configuration from {@link #invocationReplacementConfigUrls}.
   */
  @Memoized
  public ImmutableMap<MethodInvocationSite, MethodInvocationSite> invocationReplacements() {
    ImmutableMap.Builder<MethodInvocationSite, MethodInvocationSite> replacementsBuilder =
        ImmutableMap.builder();
    for (MethodInvocationReplacement replacement :
        invocationReplacementConfigProto().getReplacementsList()) {
      MethodInvocationSite invocationSite = MethodInvocationSite.fromProto(replacement.getSource());
      Set<ReplacementRange> replacementRanges = new LinkedHashSet<>(replacement.getRangeList());
      if (replacementRanges.contains(ReplacementRange.ALL)
          || replacementRanges.stream().anyMatch(enabledInvocationReplacementRanges()::contains)) {
        replacementsBuilder.put(
            invocationSite, MethodInvocationSite.fromProto(replacement.getDestination()));
      }
    }
    return replacementsBuilder.build();
  }

  @Nullable
  public final MethodInvocationSite findReplacementSite(
      MethodInvocationSite verbatimInvocationSite) {
    return invocationReplacements().get(verbatimInvocationSite);
  }

  /** The builder for {@link ClassMemberRetargetConfig}. */
  @AutoValue.Builder
  public abstract static class Builder {

    public abstract ImmutableList.Builder<URL> invocationReplacementConfigUrlsBuilder();

    public abstract ImmutableList.Builder<MethodInvocationReplacement>
        inProcessReplacementsBuilder();

    public abstract ImmutableSet.Builder<ReplacementRange>
        enabledInvocationReplacementRangesBuilder();

    public Builder addInvocationReplacementConfigUrl(URL value) {
      invocationReplacementConfigUrlsBuilder().add(value);
      return this;
    }

    public Builder addInProcessReplacement(MethodInvocationReplacement value) {
      inProcessReplacementsBuilder().add(value);
      return this;
    }

    public Builder addEnabledInvocationReplacementRange(ReplacementRange value) {
      enabledInvocationReplacementRangesBuilder().add(value);
      return this;
    }

    public Builder addAllEnabledInvocationReplacementRange(Collection<ReplacementRange> value) {
      enabledInvocationReplacementRangesBuilder().addAll(value);
      return this;
    }

    public abstract ClassMemberRetargetConfig build();
  }
}
