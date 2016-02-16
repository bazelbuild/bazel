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
package com.google.devtools.build.lib.packages.util;

import com.google.common.base.Predicates;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.packages.ConstantRuleVisibility;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.NoSuchTargetException;
import com.google.devtools.build.lib.packages.NoSuchThingException;
import com.google.devtools.build.lib.packages.PackageFactory;
import com.google.devtools.build.lib.packages.PackageFactory.EnvironmentExtension;
import com.google.devtools.build.lib.packages.Preprocessor;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.RuleVisibility;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.pkgcache.PackageCacheOptions;
import com.google.devtools.build.lib.pkgcache.PackageManager;
import com.google.devtools.build.lib.pkgcache.PathPackageLocator;
import com.google.devtools.build.lib.skyframe.DiffAwareness;
import com.google.devtools.build.lib.skyframe.PrecomputedValue;
import com.google.devtools.build.lib.skyframe.SequencedSkyframeExecutor;
import com.google.devtools.build.lib.skyframe.SkyValueDirtinessChecker;
import com.google.devtools.build.lib.skyframe.SkyframeExecutor;
import com.google.devtools.build.lib.testutil.FoundationTestCase;
import com.google.devtools.build.lib.testutil.TestRuleClassProvider;
import com.google.devtools.build.lib.util.BlazeClock;
import com.google.devtools.build.lib.util.io.TimestampGranularityMonitor;
import com.google.devtools.build.lib.vfs.ModifiedFileSet;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.common.options.OptionsParser;

import org.junit.Before;

import java.io.IOException;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;
import java.util.UUID;

/**
 * This is a specialization of {@link FoundationTestCase} that's useful for
 * implementing tests of the "packages" library.
 */
public abstract class PackageLoadingTestCase extends FoundationTestCase {

  private static final int GLOBBING_THREADS = 7;
  
  protected ConfiguredRuleClassProvider ruleClassProvider;
  protected SkyframeExecutor skyframeExecutor;

  @Before
  public final void initializeSkyframeExecutor() throws Exception {
    ruleClassProvider = TestRuleClassProvider.getRuleClassProvider();
    skyframeExecutor = createSkyframeExecutor(getEnvironmentExtensions(),
        Preprocessor.Factory.Supplier.NullSupplier.INSTANCE, ConstantRuleVisibility.PUBLIC, "");
    setUpSkyframe(parsePackageCacheOptions());
  }

  protected SkyframeExecutor createSkyframeExecutor(
      Iterable<EnvironmentExtension> environmentExtensions,
      Preprocessor.Factory.Supplier preprocessorFactorySupplier,
      RuleVisibility defaultVisibility,
      String defaultsPackageContents) {
    SkyframeExecutor skyframeExecutor =
        SequencedSkyframeExecutor.create(
            new PackageFactory(ruleClassProvider, environmentExtensions),
            new TimestampGranularityMonitor(BlazeClock.instance()),
            new BlazeDirectories(outputBase, outputBase, rootDirectory),
            null, /* BinTools */
            null, /* workspaceStatusActionFactory */
            ruleClassProvider.getBuildInfoFactories(),
            ImmutableList.<DiffAwareness.Factory>of(),
            Predicates.<PathFragment>alwaysFalse(),
            preprocessorFactorySupplier,
            ImmutableMap.<SkyFunctionName, SkyFunction>of(),
            ImmutableList.<PrecomputedValue.Injected>of(),
            ImmutableList.<SkyValueDirtinessChecker>of());
    skyframeExecutor.preparePackageLoading(
        new PathPackageLocator(outputBase, ImmutableList.of(rootDirectory)),
        defaultVisibility, true, GLOBBING_THREADS, defaultsPackageContents,
        UUID.randomUUID());
    return skyframeExecutor;
  }

  protected Iterable<EnvironmentExtension> getEnvironmentExtensions() {
    return ImmutableList.<EnvironmentExtension>of();
  }

  private void setUpSkyframe(PackageCacheOptions packageCacheOptions) {
    PathPackageLocator pkgLocator = PathPackageLocator.create(
        outputBase, packageCacheOptions.packagePath, reporter, rootDirectory, rootDirectory);
    skyframeExecutor.preparePackageLoading(pkgLocator,
        packageCacheOptions.defaultVisibility, true,
        7, ruleClassProvider.getDefaultsPackageContent(),
        UUID.randomUUID());
    skyframeExecutor.setDeletedPackages(ImmutableSet.copyOf(packageCacheOptions.deletedPackages));
  }

  private PackageCacheOptions parsePackageCacheOptions(String... options) throws Exception {
    OptionsParser parser = OptionsParser.newOptionsParser(PackageCacheOptions.class);
    parser.parse(new String[] { "--default_visibility=public" });
    parser.parse(options);
    return parser.getOptions(PackageCacheOptions.class);
  }

  protected void setPackageCacheOptions(String... options) throws Exception {
    setUpSkyframe(parsePackageCacheOptions(options));
  }

  protected Target getTarget(String label)
      throws NoSuchPackageException, NoSuchTargetException,
      LabelSyntaxException, InterruptedException {
    return getTarget(Label.parseAbsolute(label));
  }

  protected Target getTarget(Label label)
      throws NoSuchPackageException, NoSuchTargetException, InterruptedException {
    return getPackageManager().getTarget(reporter, label);
  }

  /**
   * Create and return a scratch rule.
   *
   * @param packageName the package name of the rule.
   * @param ruleName the name of the rule.
   * @param lines the text of the rule.
   * @return the rule instance for the created rule.
   * @throws IOException
   * @throws Exception
   */
  protected Rule scratchRule(String packageName, String ruleName, String... lines)
      throws Exception {
    scratch.file(packageName + "/BUILD", lines);
    return (Rule) getTarget("//" + packageName + ":" + ruleName);
  }

  /**
   * A Utility method that generates build file rules for tests.
   * @param rule the name of the rule class.
   * @param name the name of the rule instance.
   * @param body an array of strings containing the contents of the rule.
   * @return a string containing the build file rule.
   */
  protected String genRule(String rule, String name, String... body) {
    StringBuilder buf = new StringBuilder();
    buf.append(rule);
    buf.append("(name='");
    buf.append(name);
    buf.append("',\n");
    for (String line : body) {
      buf.append(line);
    }
    buf.append(")\n");
    return buf.toString();
  }

  /**
   * A utility function which generates the "deps" clause for a build file
   * rule from a list of targets.
   * @param depTargets the list of targets.
   * @return a string containing the deps clause
   */
  protected static String deps(String... depTargets) {
    StringBuilder buf = new StringBuilder();
    buf.append("    deps=[");
    String sep = "'";
    for (String dep : depTargets) {
      buf.append(sep);
      buf.append(dep);
      buf.append("'");
      sep = ", '";
    }
    buf.append("]");
    return buf.toString();
  }

  /**
   * Utility method for tests. Converts an array of strings into a set of labels.
   *
   * @param strings the set of strings to be converted to labels.
   * @throws LabelSyntaxException if there are any syntax errors in the strings.
   */
  public static Set<Label> asLabelSet(String... strings) throws LabelSyntaxException {
    return asLabelSet(ImmutableList.copyOf(strings));
  }

  /**
   * Utility method for tests. Converts an array of strings into a set of labels.
   *
   * @param strings the set of strings to be converted to labels.
   * @throws LabelSyntaxException if there are any syntax errors in the strings.
   */
  public static Set<Label> asLabelSet(Iterable<String> strings) throws LabelSyntaxException {
    Set<Label> result = Sets.newTreeSet();
    for (String s : strings) {
      result.add(Label.parseAbsolute(s));
    }
    return result;
  }

  protected final Set<Target> asTargetSet(String... strLabels)
      throws LabelSyntaxException, NoSuchThingException, InterruptedException {
    return asTargetSet(Arrays.asList(strLabels));
  }

  protected Set<Target> asTargetSet(Iterable<String> strLabels)
      throws LabelSyntaxException, NoSuchThingException, InterruptedException {
    Set<Target> targets = new HashSet<>();
    for (String strLabel : strLabels) {
      targets.add(getTarget(strLabel));
    }
    return targets;
  }

  protected PackageManager getPackageManager() {
    return skyframeExecutor.getPackageManager();
  }

  protected SkyframeExecutor getSkyframeExecutor() {
    return skyframeExecutor;
  }

  /**
   * Invalidates all existing packages below the usual rootDirectory. Must be called _after_ the
   * files are modified.
   *
   * @throws InterruptedException
   */
  protected void invalidatePackages() throws InterruptedException {
    skyframeExecutor.invalidateFilesUnderPathForTesting(
        reporter, ModifiedFileSet.EVERYTHING_MODIFIED, rootDirectory);
  }

  protected String getErrorMsgNonEmptyList(String attrName, String ruleType, String ruleName) {
    return "non empty attribute '" + attrName + "' in '" + ruleType
        + "' rule '" + ruleName + "' has to have at least one value";
  }
}
