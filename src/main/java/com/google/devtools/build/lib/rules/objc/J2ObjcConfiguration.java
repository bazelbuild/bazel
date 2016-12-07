// Copyright 2015 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.rules.objc;

import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration.Fragment;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.ConfigurationEnvironment;
import com.google.devtools.build.lib.analysis.config.ConfigurationFragmentFactory;
import com.google.devtools.build.lib.analysis.config.FragmentOptions;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import java.util.Collections;
import java.util.List;

/**
 * A J2ObjC transpiler configuration fragment containing J2ObjC translation flags.
 * This configuration fragment is used by Java rules that can be transpiled
 * (specifically, J2ObjCAspects thereof).
 */
@Immutable
public class J2ObjcConfiguration extends Fragment {
  /**
   * Always-on flags for J2ObjC translation. These flags are always used when invoking the J2ObjC
   * transpiler, and cannot be overriden by user-specified flags in {@link
   * J2ObjcCommandLineOptions}. See http://j2objc.org/docs/j2objc.html for flag documentation.
   */
  private static final List<String> J2OBJC_ALWAYS_ON_TRANSLATION_FLAGS = ImmutableList.of(
      "-encoding",
      "UTF-8",
      "--doc-comments",
      "-XcombineJars");

  /**
   * Default flags for J2ObjC translation. These flags are used by default when invoking the J2ObjC
   * transpiler, but can be overriden by user-specified flags in {@link J2ObjcCommandLineOptions}.
   * See http://j2objc.org/docs/j2objc.html for flag documentation.
   */
  private static final List<String> J2OBJC_DEFAULT_TRANSLATION_FLAGS = ImmutableList.of("-g");

  /**
   * Disallowed flags for J2ObjC translation. See http://j2objc.org/docs/j2objc.html for flag
   * documentation.
   */
  static final List<String> J2OBJC_BLACKLISTED_TRANSLATION_FLAGS =
      ImmutableList.of("--prefixes", "--prefix", "-x");

  static final String INVALID_TRANSLATION_FLAGS_MSG_TEMPLATE =
      "J2Objc translation flags: %s not supported. Unsupported flags are: %s";

  /**
   * Configuration loader for {@link J2ObjcConfiguration}.
   */
  public static class Loader implements ConfigurationFragmentFactory {
    @Override
    public Fragment create(ConfigurationEnvironment env, BuildOptions buildOptions) {
      return new J2ObjcConfiguration(buildOptions.get(J2ObjcCommandLineOptions.class));
    }

    @Override
    public Class<? extends Fragment> creates() {
      return J2ObjcConfiguration.class;
    }

    @Override
    public ImmutableSet<Class<? extends FragmentOptions>> requiredOptions() {
      return ImmutableSet.<Class<? extends FragmentOptions>>of(J2ObjcCommandLineOptions.class);
    }
  }

  private final List<String> translationFlags;
  private final boolean removeDeadCode;
  private final boolean explicitJreDeps;
  private final boolean annotationProcessingEnabled;

  J2ObjcConfiguration(J2ObjcCommandLineOptions j2ObjcOptions) {
    this.removeDeadCode = j2ObjcOptions.removeDeadCode;
    this.explicitJreDeps = j2ObjcOptions.explicitJreDeps;
    this.translationFlags = ImmutableList.<String>builder()
        .addAll(J2OBJC_DEFAULT_TRANSLATION_FLAGS)
        .addAll(j2ObjcOptions.translationFlags)
        .addAll(J2OBJC_ALWAYS_ON_TRANSLATION_FLAGS)
        .build();
    this.annotationProcessingEnabled = j2ObjcOptions.annotationProcessingEnabled;
  }

  /**
   * Returns the translation flags used to invoke the J2ObjC transpiler. The returned flags contain
   * the default flags from {@link #J2OBJC_DEFAULT_TRANSLATION_FLAGS}, user-specified flags from
   * {@link J2ObjcCommandLineOptions}, and always-on flags from {@link
   * #J2OBJC_ALWAYS_ON_TRANSLATION_FLAGS}. The set of disallowed flags can be found at
   * {@link #J2OBJC_BLACKLISTED_TRANSLATION_FLAGS}.
   */
  public Iterable<String> getTranslationFlags() {
    return translationFlags;
  }

  /**
   * Returns whether to perform J2ObjC dead code removal. If true, the list of entry classes will be
   * collected transitively throuh "entry_classes" attribute on j2objc_library and used as entry
   * points to perform dead code analysis. Unused classes will then be removed from the final ObjC
   * app bundle.
   */
  public boolean removeDeadCode() {
    return removeDeadCode;
  }

  /**
   * Returns whether explicit JRE dependencies are required. If true, all j2objc_library rules will
   * implicitly depend on jre_core_lib instead of jre_full_lib.
   */
  public boolean explicitJreDeps() {
    return explicitJreDeps;
  }

  /**
   * Returns whether to enable J2ObjC support for Java annotation processing.
   */
  public boolean annotationProcessingEnabled() {
    return annotationProcessingEnabled;
  }

  @Override
  public void reportInvalidOptions(EventHandler reporter, BuildOptions buildOptions) {
    if (!Collections.disjoint(translationFlags, J2OBJC_BLACKLISTED_TRANSLATION_FLAGS)) {
      String errorMsg = String.format(INVALID_TRANSLATION_FLAGS_MSG_TEMPLATE,
          Joiner.on(",").join(translationFlags),
          Joiner.on(",").join(J2OBJC_BLACKLISTED_TRANSLATION_FLAGS));
      reporter.handle(Event.error(errorMsg));
    }
  }
}
