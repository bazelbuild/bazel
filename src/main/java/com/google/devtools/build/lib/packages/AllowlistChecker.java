// Copyright 2021 The Bazel Authors. All rights reserved.
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

import com.google.auto.value.AutoValue;
import javax.annotation.Nullable;

/**
 * Set of classes for RuleClass to use to track how an Allowlist must be checked.
 *
 * <p>This class is used at both loading time in packages an thus cannot actually refer to {@link
 * Allowlist}. These are processed in {@link RuleConfiguredTargetBuilder}.
 */
@AutoValue
public abstract class AllowlistChecker {
  /** Track whether checking rule instance or rule definition location */
  public enum LocationCheck {
    INSTANCE, // pass if rule instance in allowlist
    DEFINITION, // pass if rule definition in allowlist
    INSTANCE_OR_DEFINITION // pass if either in allowlist
  }

  /** Return attribute name containing the allowlist to check against. */
  public abstract String allowlistAttr();

  /** Return error message to print if allowlist check fails. */
  public abstract String errorMessage();

  /** Return what rule location to check against allowlist. */
  public abstract LocationCheck locationCheck();

  /** If non-null, check that the attribute is explicitly set before checking allowlist. */
  @Nullable
  public abstract String attributeSetTrigger();

  public static Builder builder() {
    return new AutoValue_AllowlistChecker.Builder();
  }

  /** Standard builder class. */
  @AutoValue.Builder
  public abstract static class Builder {
    public abstract Builder setAllowlistAttr(String allowlistAttr);

    public abstract Builder setErrorMessage(String errorMessage);

    public abstract Builder setAttributeSetTrigger(String attributeSetTrigger);

    public abstract Builder setLocationCheck(LocationCheck locationCheck);

    public abstract AllowlistChecker build();
  }
}
