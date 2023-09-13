// Copyright 2020 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.analysis.config;

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.util.List;
import javax.annotation.Nullable;
import net.starlark.java.eval.StarlarkValue;

/**
 * An interface for language-specific configurations.
 *
 * <p>Implementations must have a constructor that takes a single {@link BuildOptions} argument. If
 * the constructor reads any {@link FragmentOptions} from this argument, the fragment must declare
 * them via {@link RequiresOptions}.
 *
 * <p>All implementations must be immutable and communicate this as clearly as possible (e.g.
 * declare {@link com.google.common.collect.ImmutableList} signatures on their interfaces vs. {@link
 * List}). This is because fragment instances may be shared across configurations.
 *
 * <p>Fragments are Starlark values, as returned by {@code ctx.fragments.android}, for example.
 */
@Immutable
public abstract class Fragment implements StarlarkValue {

  /**
   * When a fragment doesn't want to be part of the configuration (for example, when its required
   * options are missing and the fragment determines this means the configuration doesn't need it),
   * it should override this method.
   */
  public boolean shouldInclude() {
    return true;
  }

  @Override
  public boolean isImmutable() {
    return true; // immutable and Starlark-hashable
  }

  /**
   * Validates the options for this Fragment. Issues warnings for the use of deprecated options, and
   * warnings or errors for any option settings that conflict.
   */
  @SuppressWarnings("unused")
  public void reportInvalidOptions(EventHandler reporter, BuildOptions buildOptions) {}

  /**
   * Context needed by implementations of {@link Fragment#processForOutputPathMnemonic}.
   *
   * <p>The Fragment constructor should already have sufficient access to targetOptions as per
   * RequiresOption above. So a getTargetOption method should not be necessary.
   */
  public static interface OutputDirectoriesContext {
    /** If available, get the baseline version of some FragmentOption */
    @Nullable
    public <T extends FragmentOptions> T getBaseline(Class<T> optionsClass);

    /**
     * Adds given String to the explicit part of the output path.
     *
     * <p>A null or empty value is not added to the mnemonic. Ideally this function will eventually
     * just error when supplied those values.
     *
     * @throws AddToMnemonicException if given value cannot be put in an output path.
     */
    @CanIgnoreReturnValue
    public OutputDirectoriesContext addToMnemonic(@Nullable String value)
        throws AddToMnemonicException;

    /**
     * Mark the option as explicit in output path so it no longer contributes to hash computation.
     *
     * <p>Options which are marked must be explicitly included in the output path by {@link
     * addToMnemonic} (or indirectly in {@link Fragment.getOutputDirectoryName}) and thus will not
     * be included in the hash of changed options used to generically disambiguate output
     * directories of different configurations. (See {@link OutputPathMnemonicComputer}.)
     *
     * <p>This tag should only be added to options that can guarantee that any change to that option
     * corresponds to a change to {@link OutputPathMnemonicComputer.computeMnemonic}. Put
     * mathematically, given any two BuildOptions instances A and B with respective values for the
     * marked option a and b (where all other options are the same and there is some potentially
     * null baseline): {@code a == b iff computeMnemonic(A, baseline) == computeMnemonic(b,
     * baseline)}
     *
     * <p>As a historical note, this used to be implemented as EXPLICIT_IN_OUTPUT_PATH
     */
    @CanIgnoreReturnValue
    public OutputDirectoriesContext markAsExplicitInOutputPathFor(String optionName);

    /** bubble up error with adding to mnemonic (likely a problematic value supplied) */
    public static final class AddToMnemonicException extends Exception {
      final Exception tunneledException;
      final String badValue;

      AddToMnemonicException(String badValue, Exception e) {
        super("Invalid option value " + badValue, e);
        this.tunneledException = e;
        this.badValue = badValue;
      }
    }
  }

  /**
   * Returns a fragment of the output directory name for this set of options. See {@link
   * BuildConfigurationFunction.computeMnemonic})
   */
  public void processForOutputPathMnemonic(OutputDirectoriesContext ctx)
      throws OutputDirectoriesContext.AddToMnemonicException {}

  /** Returns the option classes needed to create a fragment. */
  public static ImmutableSet<Class<? extends FragmentOptions>> requiredOptions(
      Class<? extends Fragment> fragmentClass) {
    RequiresOptions annotation = fragmentClass.getAnnotation(RequiresOptions.class);
    return annotation == null ? ImmutableSet.of() : ImmutableSet.copyOf(annotation.options());
  }

  /** Returns {@code true} if the given fragment requires access to starlark options. */
  public static boolean requiresStarlarkOptions(Class<? extends Fragment> fragmentClass) {
    RequiresOptions annotation = fragmentClass.getAnnotation(RequiresOptions.class);
    return annotation != null && annotation.starlark();
  }
}
