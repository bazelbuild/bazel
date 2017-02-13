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
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
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
 * look at the target it refers to.
 *
 * <ul>
 *   <li>If the target is a {@code java_runtime_suite}, the loader will select the runtime from the
 *       entry in {@code java_runtime_suite.runtimes} for {@code cpu}.
 *   <li>If the target is a filegroup, then the loader will look in it's srcs for a filegroup that
 *       ends with {@code -<cpu>}. It will use that filegroup to construct the actual {@link Jvm}
 *       instance, using the filegroups {@code path} attribute to construct the new {@code javaHome}
 *       path.
 * </ul>
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
    String javaHome = javaOptions.javaBase;
    String cpu = buildOptions.get(BuildConfiguration.Options.class).cpu;

    try {
      return createDefault(env, javaHome, cpu);
    } catch (LabelSyntaxException e) {
      // Try again with legacy
    }

    return createLegacy(javaHome);
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
  private static Jvm createDefault(ConfigurationEnvironment lookup, String javaHome, String cpu)
      throws InvalidConfigurationException, LabelSyntaxException, InterruptedException {
    try {
      Label label = Label.parseAbsolute(javaHome);
      label = RedirectChaser.followRedirects(lookup, label, "jdk");
      if (label == null) {
        return null;
      }
      Target javaHomeTarget = lookup.getTarget(label);
      if (javaHomeTarget instanceof Rule) {
        switch (((Rule) javaHomeTarget).getRuleClass()) {
          case "filegroup":
            return createFromFilegroup(lookup, javaHomeTarget, cpu);
          case "java_runtime_suite":
            return createFromRuntimeSuite(lookup, (Rule) javaHomeTarget, cpu);
          default:
            throw new InvalidConfigurationException(
                "Unexpected javabase rule kind '" + ((Rule) javaHomeTarget).getRuleClass() + "'");
        }
      }
      throw new InvalidConfigurationException(
          "No JVM target found under " + javaHome + " that would work for " + cpu);
    } catch (NoSuchThingException e) {
      lookup.getEventHandler().handle(Event.error(e.getMessage()));
      throw new InvalidConfigurationException(e.getMessage(), e);
    }
  }

  private static Jvm createFromFilegroup(
      ConfigurationEnvironment lookup, Target javaHomeTarget, String cpu)
      throws InvalidConfigurationException, InterruptedException, NoSuchPackageException,
          NoSuchTargetException {

    RawAttributeMapper javaHomeAttributes = RawAttributeMapper.of((Rule) javaHomeTarget);
    if (javaHomeAttributes.isConfigurable("srcs", BuildType.LABEL_LIST)) {
      throw new InvalidConfigurationException(
          String.format(
              "\"srcs\" in %s is configurable. JAVABASE targets don't support configurable"
                  + " attributes",
              javaHomeTarget));
    }
    List<Label> labels = javaHomeAttributes.get("srcs", BuildType.LABEL_LIST);
    Label selectedJvmLabel = null;
    Label defaultJvmLabel = null;
    for (Label jvmLabel : labels) {
      if (jvmLabel.getName().endsWith("-" + cpu)) {
        selectedJvmLabel = jvmLabel;
        break;
      }
      // When we open sourced Bazel, we used the string "default" to look up the Jvm. This is
      // incorrect for cross-platform builds, but works for purely local builds. Since we now
      // need to support cross-platform builds, we need to look up by the CPU, rather than the
      // hard-coded string "default". However, for local builds the Jvm is setup with a
      // mechanism where we don't currently have access to the CPU value (this is different from
      // C++, where we infer the CPU from the local machine). As such, looking up only by CPU
      // breaks builds that currently work, unless we add alias rules for all possible CPU
      // values (but this is problematic if Bazel is ported to more platforms). For now, we're
      // working around this problem by falling back to -default if we can't find a Jvm ending
      // in -<cpu>. This is backwards compatible, but still allows cross-platform builds. In the
      // medium term, we should rewrite Jvm setup to use a Skylark remote repository, and also
      // remove the necessity of having a Jvm defined for all platforms even if there's no Java
      // code.
      if (jvmLabel.getName().endsWith("-default")) {
        defaultJvmLabel = jvmLabel;
      }
    }
    if (selectedJvmLabel == null) {
      selectedJvmLabel = defaultJvmLabel;
    }
    if (selectedJvmLabel != null) {
      selectedJvmLabel =
          RedirectChaser.followRedirects(lookup, selectedJvmLabel, "Architecture-specific JDK");
      if (selectedJvmLabel == null) {
        return null;
      }

      Target jvmTarget = lookup.getTarget(selectedJvmLabel);

      PathFragment javaHomePath = defaultJavaHome(jvmTarget.getLabel());

      if ((jvmTarget instanceof Rule) && "filegroup".equals(((Rule) jvmTarget).getRuleClass())) {
        RawAttributeMapper jvmTargetAttributes = RawAttributeMapper.of((Rule) jvmTarget);
        if (jvmTargetAttributes.isConfigurable("path", Type.STRING)) {
          throw new InvalidConfigurationException(
              String.format(
                  "\"path\" in %s is configurable. JVM targets don't support configurable"
                      + " attributes",
                  jvmTarget));
        }
        String path = jvmTargetAttributes.get("path", Type.STRING);
        if (path != null) {
          javaHomePath = javaHomePath.getRelative(path);
        }
      }
      return new Jvm(javaHomePath, selectedJvmLabel);
    }
    throw new InvalidConfigurationException(
        String.format("No JVM target found under %s that would work for %s", javaHomeTarget, cpu));
  }

  // TODO(b/34175492): this is temporary until support for filegroup-based javabases is removed.
  // Eventually the Jvm fragement will containg only the label of a java_runtime rule, and all of
  // the configuration will be accessed using JavaRuntimeProvider.
  private static Jvm createFromRuntimeSuite(
      ConfigurationEnvironment lookup, Rule javaRuntimeSuite, String cpu)
      throws InvalidConfigurationException, InterruptedException, NoSuchTargetException,
          NoSuchPackageException {
    Label javaRuntimeLabel = selectRuntime(javaRuntimeSuite, cpu);
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
    PathFragment javaHomePath = defaultJavaHome(javaRuntimeLabel);
    if (attributes.isAttributeValueExplicitlySpecified("java_home")) {
      javaHomePath = javaHomePath.getRelative(attributes.get("java_home", Type.STRING));
    }
    return new Jvm(javaHomePath, javaRuntimeLabel);
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

  private static PathFragment defaultJavaHome(Label javaBase) {
    if (javaBase.getPackageIdentifier().getRepository().isDefault()) {
      return javaBase.getPackageFragment();
    }
    return javaBase.getPackageIdentifier().getSourceRoot();
  }

  private static Jvm createLegacy(String javaHome) throws InvalidConfigurationException {
    PathFragment javaHomePathFrag = new PathFragment(javaHome);
    if (!javaHomePathFrag.isAbsolute()) {
      throw new InvalidConfigurationException(
          "Illegal javabase value '" + javaHome + "', javabase must be an absolute path or label");
    }
    return new Jvm(javaHomePathFrag, null);
  }
}
