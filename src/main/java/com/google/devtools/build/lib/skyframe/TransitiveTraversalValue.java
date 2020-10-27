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
package com.google.devtools.build.lib.skyframe;

import com.google.common.base.MoreObjects;
import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.packages.AdvertisedProviderSet;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.util.StringCanonicalizer;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import javax.annotation.Nullable;

/**
 * A <i>transitive</i> target reference that, when built in skyframe, loads the entire transitive
 * closure of a target. Retains the first error message found during the transitive traversal, the
 * kind of target, and a set of names of providers if the target is a {@link Rule}.
 *
 * <p>Interns values for error-free traversal nodes that correspond to built-in rules.
 */
@Immutable
@ThreadSafe
@AutoCodec
public abstract class TransitiveTraversalValue implements SkyValue {
  // A quick-lookup cache that allows us to get the value for a given target kind, assuming no error
  // messages for the target. The number of built-in target kinds is limited, so memory bloat is not
  // a concern.
  private static final ConcurrentMap<String, TransitiveTraversalValue> VALUES_BY_TARGET_KIND =
      new ConcurrentHashMap<>();
  /**
   * A strong interner of TransitiveTargetValue objects. Because we only wish to intern values for
   * built-in non-Starlark targets, we need an interner with an additional method to return the
   * canonical representative if it is present without interning our sample. This is only mutated in
   * {@link #forTarget}, and read in {@link #forTarget} and {@link #create}.
   */
  private static final InternerWithPresenceCheck<TransitiveTraversalValue> VALUE_INTERNER =
      new InternerWithPresenceCheck<>();

  private final String kind;

  protected TransitiveTraversalValue(String kind) {
    this.kind = Preconditions.checkNotNull(kind);
  }

  static TransitiveTraversalValue unsuccessfulTransitiveTraversal(
      String errorMessage, Target target) {
    return new TransitiveTraversalValueWithError(
        Preconditions.checkNotNull(errorMessage), target.getTargetKind());
  }

  static TransitiveTraversalValue forTarget(Target target, @Nullable String errorMessage) {
    if (errorMessage == null) {
      if (target instanceof Rule && ((Rule) target).getRuleClassObject().isStarlark()) {
        Rule rule = (Rule) target;
        // Do not intern values for Starlark rules.
        return TransitiveTraversalValue.create(
            rule.getRuleClassObject().getAdvertisedProviders(), rule.getTargetKind(), errorMessage);
      } else {
        TransitiveTraversalValue value = VALUES_BY_TARGET_KIND.get(target.getTargetKind());
        if (value != null) {
          return value;
        }

        AdvertisedProviderSet providers =
            target instanceof Rule
                ? ((Rule) target).getRuleClassObject().getAdvertisedProviders()
                : AdvertisedProviderSet.EMPTY;

        value = new TransitiveTraversalValueWithoutError(providers, target.getTargetKind());
        // May already be there from another target or a concurrent put.
        value = VALUE_INTERNER.intern(value);
        // May already be there from a concurrent put.
        VALUES_BY_TARGET_KIND.putIfAbsent(target.getTargetKind(), value);
        return value;
      }
    } else {
      return new TransitiveTraversalValueWithError(errorMessage, target.getTargetKind());
    }
  }

  @AutoCodec.Instantiator
  public static TransitiveTraversalValue create(
      AdvertisedProviderSet providers, String kind, @Nullable String errorMessage) {
    TransitiveTraversalValue value =
        errorMessage == null
            ? new TransitiveTraversalValueWithoutError(providers, kind)
            : new TransitiveTraversalValueWithError(errorMessage, kind);
    if (errorMessage == null) {
      TransitiveTraversalValue oldValue = VALUE_INTERNER.getCanonical(value);
      return oldValue == null ? value : oldValue;
    }
    return value;
  }

  /** Returns if the associated target can have any provider. True for "alias" rules. */
  public abstract boolean canHaveAnyProvider();

  /**
   * Returns the set of provider names from the target, if the target is a {@link Rule}. Otherwise
   * returns the empty set.
   */
  public abstract AdvertisedProviderSet getProviders();

  /** Returns the target kind. */
  public String getKind() {
    return kind;
  }

  /**
   * Returns a deterministic error message, if any, from loading the target and its transitive
   * dependencies.
   */
  @Nullable
  public abstract String getErrorMessage();

  @Override
  public boolean equals(Object o) {
    if (this == o) {
      return true;
    }
    if (!(o instanceof TransitiveTraversalValue)) {
      return false;
    }
    TransitiveTraversalValue that = (TransitiveTraversalValue) o;
    return Objects.equals(this.getErrorMessage(), that.getErrorMessage())
        && Objects.equals(this.getKind(), that.getKind())
        && this.getProviders().equals(that.getProviders());
  }

  @Override
  public int hashCode() {
    return Objects.hash(getErrorMessage(), getKind(), getProviders());
  }

  @ThreadSafe
  public static SkyKey key(Label label) {
    Preconditions.checkArgument(!label.getRepository().isDefault());
    return label;
  }

  /** A transitive target reference without error. */
  public static final class TransitiveTraversalValueWithoutError extends TransitiveTraversalValue {
    private final AdvertisedProviderSet advertisedProviders;

    private TransitiveTraversalValueWithoutError(
        AdvertisedProviderSet providers, @Nullable String kind) {
      super(kind);
      this.advertisedProviders = Preconditions.checkNotNull(providers);
    }

    @Override
    public boolean canHaveAnyProvider() {
      return advertisedProviders.canHaveAnyProvider();
    }

    @Override
    public AdvertisedProviderSet getProviders() {
      return advertisedProviders;
    }

    @Override
    @Nullable
    public String getErrorMessage() {
      return null;
    }

    @Override
    public String toString() {
      return MoreObjects.toStringHelper(this)
          .add("kind", getKind())
          .add("providers", advertisedProviders)
          .toString();
    }
  }

  /** A transitive target reference with error. */
  public static final class TransitiveTraversalValueWithError extends TransitiveTraversalValue {
    private final String errorMessage;

    private TransitiveTraversalValueWithError(String errorMessage, String kind) {
      super(kind);
      this.errorMessage = StringCanonicalizer.intern(Preconditions.checkNotNull(errorMessage));
    }

    @Override
    public boolean canHaveAnyProvider() {
      return AdvertisedProviderSet.EMPTY.canHaveAnyProvider();
    }

    @Override
    public AdvertisedProviderSet getProviders() {
      return AdvertisedProviderSet.EMPTY;
    }

    @Override
    @Nullable
    public String getErrorMessage() {
      return errorMessage;
    }

    @Override
    public String toString() {
      return MoreObjects.toStringHelper(this)
          .add("error", errorMessage)
          .add("kind", getKind())
          .toString();
    }
  }
}
