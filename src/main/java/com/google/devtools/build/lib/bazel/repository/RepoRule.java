// Copyright 2025 The Bazel Authors. All rights reserved.
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
//
package com.google.devtools.build.lib.bazel.repository;

import com.google.auto.value.AutoBuilder;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ImmutableTable;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.bazel.bzlmod.AttributeValues;
import com.google.devtools.build.lib.bazel.bzlmod.ExternalDepsException;
import com.google.devtools.build.lib.bazel.bzlmod.RepoRuleId;
import com.google.devtools.build.lib.bazel.bzlmod.RepoSpec;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.LabelConverter;
import com.google.devtools.build.lib.server.FailureDetails.ExternalDeps;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import com.google.protobuf.ByteString;
import java.util.HashSet;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import net.starlark.java.eval.Dict;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkCallable;
import net.starlark.java.eval.StarlarkThread.CallStackEntry;

/**
 * Represents a fully-loaded repo rule, ready to be run.
 *
 * @param recordedRepoMappingEntries The repo mapping entries recorded during the loading of the
 *     repo rule's impl function.
 * @param environ The predeclared set of environment variables this repo rule depends on. Note that
 *     using {@code repository_ctx.getenv} is preferred.
 */
@AutoCodec
public record RepoRule(
    RepoRuleId id,
    ByteString transitiveBzlDigest,
    ImmutableTable<RepositoryName, String, RepositoryName> recordedRepoMappingEntries,
    StarlarkCallable impl,
    Optional<String> doc,
    ImmutableList<Attribute> attributes,
    ImmutableMap<String, Integer> attributeIndices,
    boolean local,
    boolean configure,
    boolean remotable,
    ImmutableSet<String> environ) {

  /**
   * A list of forbidden attribute names. These used to be present on all repo rules simply because
   * they were built-in attributes for all rules.
   */
  private static final ImmutableSet<String> LEGACY_BUILTIN_ATTRIBUTES =
      ImmutableSet.of("name", "tags", "deprecation", "visibility");

  /** Supplies a {@link RepoRule} instance. */
  public interface Supplier {
    RepoRule getRepoRule();
  }

  /**
   * Instantiates a repo.
   *
   * @param kwargs The attributes supplied to the repo rule invocation. Note that the {@code name}
   *     faux-attribute isn't actually stored as an attribute, but {@code kwargs} can contain it as
   *     per repo rule invocation convention.
   * @param repoMappingWhere See {@link AttributeUtils#typeCheckAttrValues}
   */
  public RepoSpec instantiate(
      Map<String, Object> kwargs,
      ImmutableList<CallStackEntry> callStack,
      LabelConverter labelConverter,
      EventHandler eventHandler,
      String repoMappingWhere)
      throws ExternalDepsException {
    try {
      String maybeName =
          kwargs.get("name") instanceof String name ? " with name '%s'".formatted(name) : "";
      ImmutableList<Object> attrValues =
          AttributeUtils.typeCheckAttrValues(
              attributes,
              attributeIndices,
              Maps.filterKeys(kwargs, k -> !LEGACY_BUILTIN_ATTRIBUTES.contains(k)),
              labelConverter,
              ExternalDeps.Code.EXTENSION_EVAL_ERROR,
              callStack,
              "call to '%s' repo rule%s".formatted(id.ruleName(), maybeName),
              repoMappingWhere);
      var attrDict = Dict.<String, Object>builder();
      for (Map.Entry<String, Object> kwarg : kwargs.entrySet()) {
        // Only store explicitly-specified attributes.
        if (!LEGACY_BUILTIN_ATTRIBUTES.contains(kwarg.getKey())
            && !Starlark.isNullOrNone(kwarg.getValue())) {
          attrDict.put(kwarg.getKey(), attrValues.get(attributeIndices.get(kwarg.getKey())));
        }
      }
      return new RepoSpec(id, AttributeValues.create(attrDict.buildImmutable()));
    } catch (ExternalDepsException e) {
      eventHandler.handle(Event.error(callStack.getLast().location, e.getMessage()));
      throw e;
    }
  }

  public static Builder builder() {
    return new AutoBuilder_RepoRule_Builder();
  }

  /** Builder type for {@link RepoRule}. */
  @AutoBuilder
  public abstract static class Builder {
    Set<String> attrNames = new HashSet<>();

    public abstract RepoRuleId.Builder idBuilder();

    public abstract Builder transitiveBzlDigest(ByteString value);

    public abstract Builder recordedRepoMappingEntries(
        ImmutableTable<RepositoryName, String, RepositoryName> value);

    public abstract Builder impl(StarlarkCallable value);

    public abstract Builder doc(Optional<String> value);

    abstract ImmutableList.Builder<Attribute> attributesBuilder();

    abstract ImmutableMap.Builder<String, Integer> attributeIndicesBuilder();

    @CanIgnoreReturnValue
    public final Builder addAttribute(Attribute attribute) {
      attributesBuilder().add(attribute);
      attributeIndicesBuilder().put(attribute.getPublicName(), attrNames.size());
      attrNames.add(attribute.getPublicName());
      return this;
    }

    public final boolean hasAttribute(String attrName) {
      return attrNames.contains(attrName) || LEGACY_BUILTIN_ATTRIBUTES.contains(attrName);
    }

    public abstract Builder local(boolean value);

    public abstract Builder configure(boolean value);

    public abstract Builder remotable(boolean value);

    public abstract Builder environ(ImmutableSet<String> value);

    public abstract RepoRule build();
  }
}
