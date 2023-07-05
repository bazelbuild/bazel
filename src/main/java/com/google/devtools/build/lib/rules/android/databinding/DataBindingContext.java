// Copyright 2018 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.rules.android.databinding;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.rules.android.AndroidDataContext;
import com.google.devtools.build.lib.rules.android.AndroidResources;
import com.google.devtools.build.lib.rules.java.JavaPluginInfo;
import java.util.function.BiConsumer;
import java.util.function.Consumer;

/** Contains Android Databinding configuration and resource generation information. */
public interface DataBindingContext {

  /**
   * Returns the file where data binding's resource processing produces binding xml. For example,
   * given:
   *
   * <pre>{@code
   * <layout>
   *   <data>
   *     <variable name="foo" type="String" />
   *   </data>
   * </layout>
   * <LinearLayout>
   *   ...
   * </LinearLayout>
   * }</pre>
   *
   * <p>data binding strips out and processes this part:
   *
   * <pre>{@code
   * <data>
   *   <variable name="foo" type="String" />
   * </data>
   * }</pre>
   *
   * for each layout file with data binding expressions. Since this may produce multiple files,
   * outputs are zipped up into a single container.
   */
  void supplyLayoutInfo(Consumer<Artifact> consumer);

  /** The javac flags that are needed to configure data binding's annotation processor. */
  void supplyJavaCoptsUsing(
      RuleContext ruleContext,
      boolean isBinary,
      Consumer<Iterable<String>> consumer);

  /**
   * Adds data binding's annotation processor as a plugin to the given Java compilation context.
   *
   * <p>This extends the Java compilation to translate data binding .xml into corresponding classes.
   *
   * <p>The BiConsumer accepts as its first argument the JavaPluginInfoProvider, and the list of
   * outputs of the processor as the second argument.
   */
  void supplyAnnotationProcessor(
      RuleContext ruleContext, BiConsumer<JavaPluginInfo, Iterable<Artifact>> consumer)
      throws RuleErrorException;

  /**
   * Processes deps that also apply data binding.
   *
   * @param ruleContext the current rule
   * @param isBinary whether this rule is a "binary" rule (i.e., top-level android rule)
   * @return the deps' metadata outputs. These need to be staged as compilation inputs to the
   *     current rule.
   */
  ImmutableList<Artifact> processDeps(RuleContext ruleContext, boolean isBinary);

  /**
   * Creates and adds the generated Java source for data binding annotation processor to read and
   * translate layout info xml (from {@link #supplyLayoutInfo(Consumer)} into the classes that end
   * user code consumes.
   *
   * <p>This triggers the annotation processor. Annotation processor settings are configured
   * separately in {@link #supplyJavaCoptsUsing(RuleContext, boolean, Consumer)}.
   */
  ImmutableList<Artifact> getAnnotationSourceFiles(RuleContext ruleContext);

  /**
   * Adds the appropriate {@link UsesDataBindingProvider} for a rule if it should expose one.
   *
   * <p>A rule exposes {@link UsesDataBindingProvider} if either it or its deps set {@code
   * enable_data_binding = 1}.
   */
  void addProvider(RuleConfiguredTargetBuilder builder, RuleContext ruleContext);

  /**
   * Process the given Android Resources for databinding. In databinding v2, this strips out the
   * databinding and generates the layout info file.
   */
  AndroidResources processResources(
      AndroidDataContext dataContext,
      AndroidResources resources,
      String appId);

  /** Returns whether this context supports generating AndroidX dependencies. */
  boolean usesAndroidX();
}
