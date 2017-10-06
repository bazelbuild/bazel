// Copyright 2017 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.syntax;

import com.google.auto.value.AutoValue;

/**
 * Options that affect Skylark semantics.
 *
 * <p>For descriptions of what these options do, see {@link SkylarkSemanticsOptions}.
 */
// TODO(brandjon): User error messages that reference options should maybe be substituted with the
// option name outside of the core Skylark interpreter?
// TODO(brandjon): Eventually these should be documented in full here, and SkylarkSemanticsOptions
// should refer to this class for documentation. But this doesn't play nice with the options
// parser's annotation mechanism.
@AutoValue
public abstract class SkylarkSemantics {

  /**
   * The AutoValue-generated concrete class implementing this one.
   *
   * <p>AutoValue implementation classes are usually package-private. We expose it here for the
   * benefit of code that relies on reflection.
   */
  public static final Class<? extends SkylarkSemantics> IMPL_CLASS =
      AutoValue_SkylarkSemantics.class;

  // <== Add new options here in alphabetic order ==>
  public abstract boolean incompatibleBzlDisallowLoadAfterStatement();
  public abstract boolean incompatibleCheckedArithmetic();
  public abstract boolean incompatibleComprehensionVariablesDoNotLeak();
  public abstract boolean incompatibleDepsetIsNotIterable();
  public abstract boolean incompatibleDescriptiveStringRepresentations();
  public abstract boolean incompatibleDictLiteralHasNoDuplicates();
  public abstract boolean incompatibleDisallowDictPlus();
  public abstract boolean incompatibleDisallowKeywordOnlyArgs();
  public abstract boolean incompatibleDisallowSetConstructor();
  public abstract boolean incompatibleDisallowToplevelIfStatement();
  public abstract boolean incompatibleListPlusEqualsInplace();
  public abstract boolean incompatibleLoadArgumentIsLabel();
  public abstract boolean incompatibleNewActionsApi();
  public abstract boolean incompatibleStringIsNotIterable();
  public abstract boolean internalDoNotExportBuiltins();
  public abstract boolean internalSkylarkFlagTestCanary();

  public static Builder builder() {
    return new AutoValue_SkylarkSemantics.Builder();
  }

  public static final SkylarkSemantics DEFAULT_SEMANTICS = builder()
      // <== Add new options here in alphabetic order ==>
      .incompatibleBzlDisallowLoadAfterStatement(false)
      .incompatibleCheckedArithmetic(true)
      .incompatibleComprehensionVariablesDoNotLeak(true)
      .incompatibleDepsetIsNotIterable(false)
      .incompatibleDescriptiveStringRepresentations(true)
      .incompatibleDictLiteralHasNoDuplicates(true)
      .incompatibleDisallowDictPlus(false)
      .incompatibleDisallowKeywordOnlyArgs(true)
      .incompatibleDisallowSetConstructor(true)
      .incompatibleDisallowToplevelIfStatement(true)
      .incompatibleListPlusEqualsInplace(false)
      .incompatibleLoadArgumentIsLabel(false)
      .incompatibleNewActionsApi(false)
      .incompatibleStringIsNotIterable(false)
      .internalDoNotExportBuiltins(false)
      .internalSkylarkFlagTestCanary(false)
      .build();

  /** Builder for {@link SkylarkSemantics}. All fields are mandatory. */
  @AutoValue.Builder
  public abstract static class Builder {

    // <== Add new options here in alphabetic order ==>
    public abstract Builder incompatibleBzlDisallowLoadAfterStatement(boolean value);
    public abstract Builder incompatibleCheckedArithmetic(boolean value);
    public abstract Builder incompatibleComprehensionVariablesDoNotLeak(boolean value);
    public abstract Builder incompatibleDepsetIsNotIterable(boolean value);
    public abstract Builder incompatibleDescriptiveStringRepresentations(boolean value);
    public abstract Builder incompatibleDictLiteralHasNoDuplicates(boolean value);
    public abstract Builder incompatibleDisallowDictPlus(boolean value);
    public abstract Builder incompatibleDisallowKeywordOnlyArgs(boolean value);
    public abstract Builder incompatibleDisallowSetConstructor(boolean value);
    public abstract Builder incompatibleDisallowToplevelIfStatement(boolean value);
    public abstract Builder incompatibleListPlusEqualsInplace(boolean value);
    public abstract Builder incompatibleLoadArgumentIsLabel(boolean value);
    public abstract Builder incompatibleNewActionsApi(boolean value);
    public abstract Builder incompatibleStringIsNotIterable(boolean value);
    public abstract Builder internalDoNotExportBuiltins(boolean value);
    public abstract Builder internalSkylarkFlagTestCanary(boolean value);

    public abstract SkylarkSemantics build();
  }
}
