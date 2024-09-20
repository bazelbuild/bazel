// Copyright 2024 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.analysis.starlark;

import com.google.common.base.Function;
import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.analysis.DormantDependency;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.packages.Attribute.LateBoundDefault;
import com.google.devtools.build.lib.packages.AttributeMap;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.skyframe.serialization.LeafDeserializationContext;
import com.google.devtools.build.lib.skyframe.serialization.LeafObjectCodec;
import com.google.devtools.build.lib.skyframe.serialization.LeafSerializationContext;
import com.google.devtools.build.lib.skyframe.serialization.SerializationException;
import com.google.errorprone.annotations.Keep;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.io.IOException;
import java.io.Serializable;
import javax.annotation.Nullable;
import net.starlark.java.eval.EvalException;

// TODO(lberki): Make this class not inherit from LateBoundDefault since they do very different
// things.
/** A late-bound default that represents materializers. */
public class StarlarkMaterializingLateBoundDefault<ValueT, PrerequisiteT>
    extends LateBoundDefault<Void, ValueT> {
  private final Type<ValueT> type;
  private final Resolver<ValueT, PrerequisiteT> resolver;
  private final Class<? extends PrerequisiteT> analysisContextClass;

  public StarlarkMaterializingLateBoundDefault(
      Type<ValueT> type,
      Class<? extends PrerequisiteT> analysisContextClass,
      Resolver<ValueT, PrerequisiteT> resolver) {
    super(Void.class, (Function<Rule, ValueT> & Serializable) unused -> null);

    Preconditions.checkArgument(type == BuildType.LABEL || type == BuildType.LABEL_LIST);
    this.type = type;
    this.resolver = resolver;
    this.analysisContextClass = analysisContextClass;
  }

  @Override
  public ValueT getDefault(@Nullable Rule rule) {
    return type.getDefaultValue();
  }

  /**
   * The implementation of the actual resolution of the late-bound default.
   *
   * <p>This is a separate interface because StarlarkMaterializingLateBoundDefault must be known to
   * the loading phase but its implementation necessarily deals with analysis-phase data structures.
   */
  public interface Resolver<ValueT, PrerequisiteT> {

    /**
     * Resolves an attribute with a materializer.
     *
     * <p>param rule the rule whose attribute is to be resolved.
     *
     * @param attributes the attributes of the rule, after resolving {@code select()} and the like
     * @param prerequisiteMap a map from attribute name to the prerequisites on that attribute. Only
     *     those attributes are present that represent dependencies and which are available for
     *     dependency resolution. The value of the map is in fact {@code List<? extends
     *     TransitiveInfoCollection}, but we can't say that because this class needs to be available
     *     in the loading phase.
     * @param eventHandler messages from Starlark should be reported here
     * @return the value of the resolved attribute.
     */
    ValueT resolve(
        Rule rule,
        AttributeMap attributes,
        PrerequisiteT prerequisiteMap,
        EventHandler eventHandler)
        throws InterruptedException, EvalException;
  }

  @Override
  public ValueT resolve(
      Rule rule,
      AttributeMap attributes,
      Void unused,
      Object analysisContext,
      EventHandler eventHandler)
      throws InterruptedException, EvalException {
    return resolver.resolve(
        rule, attributes, analysisContextClass.cast(analysisContext), eventHandler);
  }

  @Keep
  private static final class Codec extends LeafObjectCodec<DormantDependency> {
    @Override
    public Class<DormantDependency> getEncodedClass() {
      return DormantDependency.class;
    }

    @Override
    public void serialize(
        LeafSerializationContext context, DormantDependency obj, CodedOutputStream codedOut)
        throws SerializationException, IOException {
      context.serializeLeaf(obj.label(), Label.labelCodec(), codedOut);
    }

    @Override
    public DormantDependency deserialize(
        LeafDeserializationContext context, CodedInputStream codedIn)
        throws SerializationException, IOException {
      Label label = context.deserializeLeaf(codedIn, Label.labelCodec());
      return new DormantDependency(label);
    }
  }
}
