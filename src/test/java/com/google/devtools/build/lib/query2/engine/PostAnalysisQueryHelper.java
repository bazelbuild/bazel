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
package com.google.devtools.build.lib.query2.engine;

import static com.google.devtools.build.lib.testutil.FoundationTestCase.failFastHandler;

import com.google.common.collect.Lists;
import com.google.devtools.build.lib.analysis.AnalysisResult;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.util.AnalysisTestCase;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.packages.util.MockObjcSupport;
import com.google.devtools.build.lib.packages.util.MockProtoSupport;
import com.google.devtools.build.lib.packages.util.MockToolsConfig;
import com.google.devtools.build.lib.pkgcache.PackageManager;
import com.google.devtools.build.lib.query2.PostAnalysisQueryEnvironment;
import com.google.devtools.build.lib.query2.PostAnalysisQueryEnvironment.TopLevelConfigurations;
import com.google.devtools.build.lib.query2.engine.AbstractQueryTest.QueryHelper;
import com.google.devtools.build.lib.query2.engine.QueryUtil.AggregateAllOutputFormatterCallback;
import com.google.devtools.build.lib.skyframe.SkyframeExecutor;
import com.google.devtools.build.lib.skyframe.SkyframeExecutorWrappingWalkableGraph;
import com.google.devtools.build.lib.testutil.Scratch;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.skyframe.WalkableGraph;
import java.io.IOException;
import java.lang.annotation.Annotation;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Set;
import org.junit.After;
import org.junit.Before;

/**
 * {@link QueryHelper} for queries that need the analysis phase. Big warts: uses an {@link
 * AnalysisTestCase} to do analysis before query, but {@link AnalysisTestCase} is meant to be
 * inherited from, not composed. In particular, means that @Before and @After annotations of {@link
 * AnalysisTestCase} must be run manually. @BeforeClass and @AfterClass are completely ignored for
 * now.
 */
public abstract class PostAnalysisQueryHelper<T> extends AbstractQueryHelper<T> {
  protected String parserPrefix;
  protected AnalysisHelper analysisHelper;
  private boolean wholeTestUniverse;

  @Override
  public void setUp() throws Exception {
    super.setUp();
    parserPrefix = "";
    analysisHelper = new AnalysisHelper();
    wholeTestUniverse = false;
    // Reverse the @Before method list, so that superclass is called before subclass.
    for (Method method :
        Lists.reverse(getMethodsAnnotatedWith(AnalysisHelper.class, Before.class))) {
      method.invoke(analysisHelper);
    }
    MockToolsConfig mockToolsConfig = analysisHelper.getMockToolsConfig();
    MockProtoSupport.setup(mockToolsConfig);
    MockObjcSupport.setup(mockToolsConfig);
  }

  void cleanUp() {
    for (Method method : getMethodsAnnotatedWith(AnalysisHelper.class, After.class)) {
      try {
        method.invoke(analysisHelper);
      } catch (ReflectiveOperationException e) {
        throw new IllegalStateException(e);
      }
    }
  }

  MockToolsConfig getMockToolsConfig() {
    return analysisHelper.getMockToolsConfig();
  }

  boolean isWholeTestUniverse() {
    return wholeTestUniverse;
  }

  public List<String> getUniverseScope() {
    return universeScope;
  }

  @Override
  public void setUniverseScope(String universeScope) {
    if (!wholeTestUniverse) {
      this.universeScope = new ArrayList<>(Arrays.asList(universeScope.split(",")));
    }
  }

  public void setWholeTestUniverseScope(String universeScope) {
    this.universeScope = new ArrayList<>(Arrays.asList(universeScope.split(",")));
    wholeTestUniverse = true;
  }

  @Override
  public void setBlockUniverseEvaluationErrors(boolean blockUniverseEvaluationErrors) {}

  @Override
  public Path getRootDirectory() {
    return analysisHelper.getRootDirectory();
  }

  @Override
  public PathFragment getBlacklistedPackagePrefixesFile() {
    return PathFragment.EMPTY_FRAGMENT;
  }

  public SkyframeExecutor getSkyframeExecutor() {
    return analysisHelper.getSkyframeExecutor();
  }

  public PackageManager getPackageManager() {
    return analysisHelper.getPackageManager();
  }

  @Override
  public void clearAllFiles() throws IOException {
    FileSystemUtils.deleteTree(analysisHelper.getRootDirectory());
  }

  @Override
  public void useRuleClassProvider(ConfiguredRuleClassProvider ruleClassProvider) throws Exception {
    analysisHelper.useRuleClassProvider(ruleClassProvider);
  }

  @Override
  public void writeFile(String fileName, String... lines) throws IOException {
    analysisHelper.getScratch().file(fileName, lines);
  }

  Scratch getScratch() {
    return analysisHelper.getScratch();
  }

  void turnOffFailFast() {
    analysisHelper.getReporter().removeHandler(failFastHandler);
  }

  @Override
  public void overwriteFile(String fileName, String... lines) throws IOException {
    analysisHelper.getScratch().overwriteFile(fileName, lines);
  }

  @Override
  public void ensureSymbolicLink(String link, String target) throws IOException {
    Path rootDirectory = getRootDirectory();
    Path linkPath = rootDirectory.getRelative(link);
    Path targetPath = rootDirectory.getRelative(target);
    FileSystemUtils.createDirectoryAndParents(linkPath.getParentDirectory());
    FileSystemUtils.ensureSymbolicLink(linkPath, targetPath);
  }

  @Override
  public QueryEnvironment<T> getQueryEnvironment() {
    throw new UnsupportedOperationException();
  }

  public PostAnalysisQueryEnvironment<T> getPostAnalysisQueryEnvironment(
      Collection<String> universe) throws QueryException {
    if (universe.equals(Collections.singletonList(ConfiguredTargetQueryTest.DEFAULT_UNIVERSE))) {
      throw new QueryException(
          "Tests must set universe scope by either having parsable labels in each query expression "
              + "or setting explicitly through query helper.");
    }
    AnalysisResult analysisResult;
    try {
      analysisResult = analysisHelper.update(universe.toArray(new String[0]));
    } catch (Exception e) {
      throw new QueryException(e.getMessage());
    }
    WalkableGraph walkableGraph =
        SkyframeExecutorWrappingWalkableGraph.of(analysisHelper.getSkyframeExecutor());

    return getPostAnalysisQueryEnvironment(
        walkableGraph, new TopLevelConfigurations(analysisResult.getTopLevelTargetsWithConfigs()));
  }

  protected abstract PostAnalysisQueryEnvironment<T> getPostAnalysisQueryEnvironment(
      WalkableGraph walkableGraph, TopLevelConfigurations topLevelConfigurations);

  @Override
  public ResultAndTargets<T> evaluateQuery(String query)
      throws QueryException, InterruptedException {
    PostAnalysisQueryEnvironment<T> env = getPostAnalysisQueryEnvironment(universeScope);
    AggregateAllOutputFormatterCallback<T, ?> callback =
        QueryUtil.newOrderedAggregateAllOutputFormatterCallback(env);
    QueryEvalResult queryEvalResult;
    try {
      queryEvalResult =
          env.evaluateQuery(env.transformParsedQuery(QueryParser.parse(query, env)), callback);
    } catch (IOException e) {
      // Should be impossible since AggregateAllOutputFormatterCallback doesn't throw IOException.
      throw new IllegalStateException(e);
    }
    Set<T> targets = env.createThreadSafeMutableSet();
    targets.addAll(callback.getResult());
    return new ResultAndTargets<>(queryEvalResult, targets);
  }

  @Override
  public void assertPackageNotLoaded(String packageName) throws Exception {}

  /**
   * Returns all methods with the given annotation for the given class in the entire hierarchy.
   * Methods are returned in hierarchy order: superclass after subclass.
   */
  private static List<Method> getMethodsAnnotatedWith(
      Class<?> type, Class<? extends Annotation> annotation) {
    List<Method> methods = new ArrayList<>();
    Class<?> klass = type;
    // need to iterate through hierarchy in order to retrieve methods from above the current
    // instance.
    while (klass != Object.class) {
      // iterate though the list of methods declared in the class represented by klass variable, and
      // add those annotated with the specified annotation
      final List<Method> allMethods = new ArrayList<>(Arrays.asList(klass.getDeclaredMethods()));
      for (final Method method : allMethods) {
        if (method.isAnnotationPresent(annotation)) {
          methods.add(method);
        }
      }
      // move to the upper class in the hierarchy in search for more methods
      klass = klass.getSuperclass();
    }
    return methods;
  }

  public void useConfiguration(String... args) throws Exception {
    analysisHelper.useConfiguration(args);
  }

  /** Helper class that provides a framework for testing {@code PostAnalysisQueryHelper} */
  public static class AnalysisHelper extends AnalysisTestCase {
    Path getRootDirectory() {
      return rootDirectory;
    }

    @Override
    protected AnalysisResult update(String... labels) throws Exception {
      return super.update(labels);
    }

    protected SkyframeExecutor getSkyframeExecutor() {
      return skyframeExecutor;
    }

    protected PackageManager getPackageManager() {
      return packageManager;
    }

    protected MockToolsConfig getMockToolsConfig() {
      return mockToolsConfig;
    }

    protected Reporter getReporter() {
      return reporter;
    }

    @Override
    protected BuildConfiguration getTargetConfiguration() throws InterruptedException {
      return super.getTargetConfiguration();
    }

    @Override
    protected BuildConfiguration getHostConfiguration() {
      return super.getHostConfiguration();
    }

    @Override
    protected void useRuleClassProvider(ConfiguredRuleClassProvider ruleClassProvider)
        throws Exception {
      super.useRuleClassProvider(ruleClassProvider);
      update();
    }
  }
}
