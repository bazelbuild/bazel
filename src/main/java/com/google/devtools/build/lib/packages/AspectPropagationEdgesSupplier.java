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
package com.google.devtools.build.lib.packages;

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.cmdline.StarlarkThreadContext;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.starlarkbuildapi.StarlarkAspectPropagationContextApi;
import java.util.Objects;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Mutability;
import net.starlark.java.eval.Sequence;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkFunction;
import net.starlark.java.eval.StarlarkList;
import net.starlark.java.eval.StarlarkSemantics;
import net.starlark.java.eval.StarlarkThread;

/**
 * Supplies the list of edges (attribute names or toolchain types) that should be propagated by an
 * aspect. It is extended by 2 classes:
 *
 * <ul>
 *   <li>FixedListSupplier: for the case when the list is fixed and known at the aspect definition
 *       time.
 *   <li>FunctionSupplier: for the case when the list is computed for each target that aspect
 *       visits.
 * </ul>
 *
 * The type <T> is String for {@code attr_aspects} and {@link Label} for {@code toolchains_aspects}.
 */
public sealed interface AspectPropagationEdgesSupplier<T> {

  public static final AspectPropagationEdgesSupplier<String> DEFAULT_ATTR_ASPECTS_SUPPLIER =
      new FixedListSupplier<>(ImmutableSet.of());

  public static final FixedListSupplier<Label> DEFAULT_TOOLCHAINS_ASPECTS_SUPPLIER =
      new FixedListSupplier<>(ImmutableSet.of());

  /** A supplier of the edges that is fixed and known at the aspect definition time. */
  public static final class FixedListSupplier<T> implements AspectPropagationEdgesSupplier<T> {

    private final ImmutableSet<T> edges;

    private FixedListSupplier(ImmutableSet<T> edges) {
      this.edges = edges;
    }

    public ImmutableSet<T> getList() {
      return edges;
    }

    @Override
    public boolean equals(Object o) {
      if (this == o) {
        return true;
      }
      if (o == null || !(o instanceof FixedListSupplier<?> that)) {
        return false;
      }

      return Objects.equals(edges, that.edges);
    }

    @Override
    public int hashCode() {
      return Objects.hashCode(edges);
    }
  }

  /** A supplier of the edges that is computed for each target that aspect visits. */
  public abstract static sealed class FunctionSupplier<T>
      implements AspectPropagationEdgesSupplier<T> {
    private final StarlarkFunction function;
    private final StarlarkSemantics semantics;

    private FunctionSupplier(StarlarkFunction function, StarlarkSemantics semantics) {
      this.function = function;
      this.semantics = semantics;
    }

    public abstract ImmutableSet<T> computeList(
        StarlarkAspectPropagationContextApi context, ExtendedEventHandler eventHandler)
        throws InterruptedException, EvalException;

    @Override
    public boolean equals(Object o) {
      if (this == o) {
        return true;
      }
      if (o == null || !(o instanceof FunctionSupplier<?> that)) {
        return false;
      }

      return Objects.equals(function, that.function) && Objects.equals(semantics, that.semantics);
    }

    @Override
    public int hashCode() {
      return Objects.hash(function, semantics);
    }

    private static final class AspectPropagationEdgesThreadContext extends StarlarkThreadContext {
      private AspectPropagationEdgesThreadContext() {
        super(null);
      }
    }

    protected StarlarkList<?> runFunction(
        StarlarkAspectPropagationContextApi context, ExtendedEventHandler eventHandler)
        throws InterruptedException, EvalException {
      try (Mutability mu = Mutability.create("aspect_propagation_edges")) {
        StarlarkThread thread = StarlarkThread.createTransient(mu, semantics);
        thread.setPrintHandler(Event.makeDebugPrintHandler(eventHandler));
        new AspectPropagationEdgesThreadContext().storeInThread(thread);
        Object starlarkResult = Starlark.positionalOnlyCall(thread, function, context);
        if (starlarkResult instanceof StarlarkList<?> listResult) {
          return listResult;
        }
        throw new EvalException("Expected a list");
      }
    }
  }

  /** A function supplier for {@code attr_aspects}. */
  public static final class AttrAspectsFunctionSupplier extends FunctionSupplier<String> {
    private AttrAspectsFunctionSupplier(StarlarkFunction function, StarlarkSemantics semantics) {
      super(function, semantics);
    }

    @Override
    public ImmutableSet<String> computeList(
        StarlarkAspectPropagationContextApi context, ExtendedEventHandler eventHandler)
        throws InterruptedException, EvalException {
      return parseAttrAspects(runFunction(context, eventHandler), /* allowAll= */ false);
    }
  }

  /** A function supplier for {@code toolchains_aspects}. */
  public static final class ToolchainsAspectsFunctionSupplier extends FunctionSupplier<Label> {
    private final LabelConverter labelConverter;

    private ToolchainsAspectsFunctionSupplier(
        StarlarkFunction function, StarlarkSemantics semantics, LabelConverter labelConverter) {
      super(function, semantics);
      this.labelConverter = labelConverter;
    }

    @Override
    public ImmutableSet<Label> computeList(
        StarlarkAspectPropagationContextApi context, ExtendedEventHandler eventHandler)
        throws InterruptedException, EvalException {
      return parseToolchainsAspects(runFunction(context, eventHandler), labelConverter);
    }
  }

  public static AspectPropagationEdgesSupplier<String> createForAttrAspects(
      Object rawAttrAspects, StarlarkThread thread) throws EvalException {
    if (rawAttrAspects instanceof StarlarkFunction attrAspectsFunction) {
      return new AttrAspectsFunctionSupplier(attrAspectsFunction, thread.getSemantics());
    } else {
      return new FixedListSupplier<>(parseAttrAspects(rawAttrAspects, /* allowAll= */ true));
    }
  }

  public static AspectPropagationEdgesSupplier<Label> createForToolchainsAspects(
      Object rawToolchainsAspects, StarlarkThread thread, LabelConverter labelConverter)
      throws EvalException {
    if (rawToolchainsAspects instanceof StarlarkFunction toolchainsAspectsFunction) {
      return new ToolchainsAspectsFunctionSupplier(
          toolchainsAspectsFunction, thread.getSemantics(), labelConverter);
    } else {
      return new FixedListSupplier<>(parseToolchainsAspects(rawToolchainsAspects, labelConverter));
    }
  }

  private static ImmutableSet<String> parseAttrAspects(Object rawAttrAspects, boolean allowAll)
      throws EvalException {
    Sequence<String> attrAspects = Sequence.cast(rawAttrAspects, String.class, "attr_aspects");

    ImmutableSet.Builder<String> attrAspectsBuilder = ImmutableSet.builder();
    for (String attrName : attrAspects) {
      if (attrName.equals("*")) {
        if (!allowAll) {
          throw new EvalException("'*' is not allowed in 'attr_aspects' list");
        } else if (attrAspects.size() != 1) {
          throw new EvalException("'*' must be the only string in 'attr_aspects' list");
        }
      }
      if (!attrName.startsWith("_")) {
        attrAspectsBuilder.add(attrName);
      } else {
        // Implicit attribute names mean either implicit or late-bound attributes
        // (``$attr`` or ``:attr``). Depend on both.
        attrAspectsBuilder
            .add(AttributeValueSource.COMPUTED_DEFAULT.convertToNativeName(attrName))
            .add(AttributeValueSource.LATE_BOUND.convertToNativeName(attrName));
      }
    }

    return attrAspectsBuilder.build();
  }

  private static ImmutableSet<Label> parseToolchainsAspects(
      Object rawToolchainsAspects, LabelConverter labelConverter) throws EvalException {
    Sequence<String> toolchainsAspects =
        Sequence.cast(rawToolchainsAspects, String.class, "toolchains_aspects");

    ImmutableSet.Builder<Label> parsedLabels = new ImmutableSet.Builder<>();
    for (String input : toolchainsAspects) {
      try {
        Label label = labelConverter.convert(input);
        parsedLabels.add(label);
      } catch (LabelSyntaxException e) {
        throw Starlark.errorf(
            "Unable to parse label '%s' in attribute '%s': %s",
            input, "toolchains_aspects", e.getMessage());
      }
    }
    return parsedLabels.build();
  }
}
