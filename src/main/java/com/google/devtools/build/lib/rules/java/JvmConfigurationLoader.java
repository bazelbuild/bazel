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

package com.google.devtools.build.lib.rules.java;

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.RedirectChaser;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration.Fragment;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.ConfigurationEnvironment;
import com.google.devtools.build.lib.analysis.config.ConfigurationFragmentFactory;
import com.google.devtools.build.lib.analysis.config.FragmentOptions;
import com.google.devtools.build.lib.analysis.config.InvalidConfigurationException;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.NoSuchTargetException;
import com.google.devtools.build.lib.packages.NoSuchThingException;
import com.google.devtools.build.lib.packages.RawAttributeMapper;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.syntax.Type;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.List;
import java.util.Map;
import javax.annotation.Nullable;

/**
 * A provider to load jvm configurations from the package path.
 *
 * <p>If the given {@code javaHome} is a label, i.e. starts with {@code "//"}, then the loader will
 * look at the {@code java_runtime_suite} it refers to, and select the runtime from the entry in
 * {@code java_runtime_suite.runtimes} for {@code cpu}.
 *
 * <p>The loader also supports legacy mode, where the JVM can be defined with an abolute path.
 */
public final class JvmConfigurationLoader implements ConfigurationFragmentFactory {
  @Override
  public Jvm create(ConfigurationEnvironment env, BuildOptions buildOptions)
      throws InvalidConfigurationException, InterruptedException {
    JavaOptions javaOptions = buildOptions.get(JavaOptions.class);
    if (javaOptions.disableJvm) {
      // TODO(bazel-team): Instead of returning null here, add another method to the interface.
      return null;
    }

    String cpu = buildOptions.get(BuildConfiguration.Options.class).cpu;

    return createFromJavaRuntimeSuite(env, javaOptions.javaBase, cpu);
  }

  @Override
  public Class<? extends Fragment> creates() {
    return Jvm.class;
  }

  @Override
  public ImmutableSet<Class<? extends FragmentOptions>> requiredOptions() {
    return ImmutableSet.<Class<? extends FragmentOptions>>of(JavaOptions.class);
  }

  @Nullable
  private static Jvm createFromJavaRuntimeSuite(
      ConfigurationEnvironment lookup, Label javaBase, String cpu)
      throws InvalidConfigurationException, InterruptedException {
    try {
      javaBase = RedirectChaser.followRedirects(lookup, javaBase, "jdk");
      if (javaBase == null) {
        return null;
      }
      Target javaHomeTarget = lookup.getTarget(javaBase);
      if (javaHomeTarget instanceof Rule) {
        switch (((Rule) javaHomeTarget).getRuleClass()) {
          case "java_runtime_suite":
            return createFromRuntimeSuite(lookup, (Rule) javaHomeTarget, cpu);
          case "java_runtime":
            return createFromRuntime(lookup, javaHomeTarget.getLabel());
          default:
            throw new InvalidConfigurationException(
                "Unexpected javabase rule kind '"
                    + ((Rule) javaHomeTarget).getRuleClass()
                    + "'. Expected java_runtime_suite");
        }
      }
      throw new InvalidConfigurationException(
          "No JVM target found under " + javaBase + " that would work for " + cpu);
    } catch (NoSuchThingException e) {
      lookup.getEventHandler().handle(Event.error(e.getMessage()));
      throw new InvalidConfigurationException(e.getMessage(), e);
    }
  }

  // TODO(b/34175492): eventually the Jvm fragement will containg only the label of a java_runtime
  // rule, and all of the configuration will be accessed using JavaRuntimeInfo.

  private static Jvm createFromRuntimeSuite(
      ConfigurationEnvironment lookup, Rule javaRuntimeSuite, String cpu)
      throws InvalidConfigurationException, InterruptedException, NoSuchTargetException,
          NoSuchPackageException {
    Label javaRuntimeLabel = selectRuntime(javaRuntimeSuite, cpu);
    PathFragment javaHome = getJavaHome(lookup, javaRuntimeLabel);
    return new Jvm(javaHome, javaRuntimeSuite.getLabel());
  }

  private static Jvm createFromRuntime(ConfigurationEnvironment lookup, Label javaRuntimeLabel)
      throws InvalidConfigurationException, InterruptedException, NoSuchTargetException,
          NoSuchPackageException {
    return new Jvm(getJavaHome(lookup, javaRuntimeLabel), javaRuntimeLabel);
  }

  private static PathFragment getJavaHome(ConfigurationEnvironment lookup, Label javaRuntimeLabel)
      throws NoSuchPackageException, NoSuchTargetException, InterruptedException,
          InvalidConfigurationException {
    Target javaRuntimeTarget = lookup.getTarget(javaRuntimeLabel);
    if (javaRuntimeTarget == null) {
      return null;
    }
    if (!(javaRuntimeTarget instanceof Rule)) {
      throw new InvalidConfigurationException(
          String.format("Invalid java_runtime '%s'", javaRuntimeLabel));
    }
    Rule javaRuntimeRule = (Rule) javaRuntimeTarget;
    if (!javaRuntimeRule.getRuleClass().equals("java_runtime")) {
      throw new InvalidConfigurationException(
          String.format("Expected a java_runtime rule, was '%s'", javaRuntimeRule.getRuleClass()));
    }
    RawAttributeMapper attributes = RawAttributeMapper.of(javaRuntimeRule);
    PathFragment javaHomePath = JavaRuntime.defaultJavaHome(javaRuntimeLabel);
    if (attributes.isAttributeValueExplicitlySpecified("java_home")) {
      javaHomePath = javaHomePath.getRelative(attributes.get("java_home", Type.STRING));
      List<Label> srcs = attributes.get("srcs", BuildType.LABEL_LIST);
      if (javaHomePath.isAbsolute() && !srcs.isEmpty()) {
        throw new InvalidConfigurationException(
            String.format(
                "'java_home' with an absolute path requires 'srcs' to be empty. "
                    + "'java_home' was %s, 'srcs' was %s",
                javaHomePath, srcs.toString()));
      }
    }
    return javaHomePath;
  }

  private static Label selectRuntime(Rule javaRuntimeSuite, String cpu)
      throws InvalidConfigurationException {
    RawAttributeMapper suiteAttributes = RawAttributeMapper.of(javaRuntimeSuite);
    Map<String, Label> runtimes = suiteAttributes.get("runtimes", BuildType.LABEL_DICT_UNARY);
    if (runtimes.containsKey(cpu)) {
      return runtimes.get(cpu);
    }
    if (suiteAttributes.isAttributeValueExplicitlySpecified("default")) {
      return suiteAttributes.get("default", BuildType.LABEL);
    }
    throw new InvalidConfigurationException(
        "No JVM target found under " + javaRuntimeSuite + " that would work for " + cpu);
  }
}
