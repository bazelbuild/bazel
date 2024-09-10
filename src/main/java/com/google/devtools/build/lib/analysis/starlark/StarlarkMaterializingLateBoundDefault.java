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
import com.google.devtools.build.lib.cmdline.StarlarkThreadContext;
import com.google.devtools.build.lib.events.Event;
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
import net.starlark.java.eval.Mutability;
import net.starlark.java.eval.NoneType;
import net.starlark.java.eval.Sequence;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkFunction;
import net.starlark.java.eval.StarlarkSemantics;
import net.starlark.java.eval.StarlarkThread;

// TODO(lberki): Make this class not inherit from LateBoundDefault since they do very different
// things.
/** A late-bound default that represents materializers. */
public class StarlarkMaterializingLateBoundDefault<ValueT> extends LateBoundDefault<Void, ValueT> {
  private final Type<ValueT> type;
  private final StarlarkSemantics semantics;
  private final StarlarkFunction implementation;

  public StarlarkMaterializingLateBoundDefault(
      Type<ValueT> type, StarlarkSemantics semantics, StarlarkFunction implementation) {
    super(Void.class, (Function<Rule, ValueT> & Serializable) unused -> null);

    Preconditions.checkArgument(type == BuildType.LABEL || type == BuildType.LABEL_LIST);
    this.type = type;
    this.semantics = semantics;
    this.implementation = implementation;
  }

  @Override
  public ValueT getDefault(@Nullable Rule rule) {
    return type.getDefaultValue();
  }

  private static class MaterializationContext extends StarlarkThreadContext {
    public MaterializationContext() {
      super(null);
    }
  }

  @Override
  public ValueT resolve(
      Rule rule, AttributeMap attributes, Void unused, Object ctx, EventHandler eventHandler)
      throws InterruptedException, EvalException {
    // First call the Starlark implementation...
    Object starlarkResult = runMaterializer(ctx, eventHandler);

    // Then convert the result to the appropriate type.
    if (type == BuildType.LABEL) {
      return switch (starlarkResult) {
        case NoneType none -> null;
        case DormantDependency d -> type.cast(d.label());
        default -> throw new EvalException("Expected a single dormant dependency or None");
      };
    } else if (type == BuildType.LABEL_LIST) {
      return type.cast(
          Sequence.cast(starlarkResult, Label.class, "return value of materializer")
              .getImmutableList());
    } else {
      throw new IllegalStateException();
    }
  }

  private Object runMaterializer(Object ctx, EventHandler eventHandler)
      throws InterruptedException, EvalException {
    try (Mutability mu = Mutability.create("eval_starlark_materialization")) {
      StarlarkThread thread = StarlarkThread.createTransient(mu, semantics);
      thread.setPrintHandler(Event.makeDebugPrintHandler(eventHandler));

      new MaterializationContext().storeInThread(thread);
      return Starlark.fastcall(thread, implementation, new Object[] {ctx}, new Object[0]);
    }
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
