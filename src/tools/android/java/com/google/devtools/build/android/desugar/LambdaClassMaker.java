// Copyright 2016 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.android.desugar;

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.collect.Iterables.getOnlyElement;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import java.io.IOException;
import java.lang.invoke.MethodHandle;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.Set;
import java.util.stream.Stream;

class LambdaClassMaker {

  static final String LAMBDA_METAFACTORY_DUMPER_PROPERTY = "jdk.internal.lambda.dumpProxyClasses";

  private final Path rootDirectory;
  private final Map<Path, LambdaInfo> generatedClasses = new LinkedHashMap<>();
  private final Set<Path> existingPaths = new LinkedHashSet<>();

  public LambdaClassMaker(Path rootDirectory) {
    checkArgument(
        Files.isDirectory(rootDirectory), "The argument '%s' is not a directory.", rootDirectory);
    this.rootDirectory = rootDirectory;
  }

  public void generateLambdaClass(
      String invokerInternalName,
      LambdaInfo lambdaInfo,
      MethodHandle bootstrapMethod,
      ArrayList<Object> bsmArgs)
      throws IOException {
    // Invoking the bootstrap method will dump the generated class.  Ignore any pre-existing
    // matching files, which can come from desugar's implementation using classes being desugared.
    existingPaths.addAll(findUnprocessed(invokerInternalName + "$$Lambda$"));
    try {
      bootstrapMethod.invokeWithArguments(bsmArgs);
    } catch (Throwable e) {
      throw new IllegalStateException(
          "Failed to generate lambda class for class "
              + invokerInternalName
              + " using "
              + bootstrapMethod
              + " with arguments "
              + bsmArgs,
          e);
    }

    Path generatedClassFile = getOnlyElement(findUnprocessed(invokerInternalName + "$$Lambda$"));
    generatedClasses.put(generatedClassFile, lambdaInfo);
    existingPaths.add(generatedClassFile);
  }

  /**
   * Returns absolute paths to .class files generated since the last call to this method together
   * with a string descriptor of the factory method.
   */
  public ImmutableMap<Path, LambdaInfo> drain() {
    ImmutableMap<Path, LambdaInfo> result = ImmutableMap.copyOf(generatedClasses);
    generatedClasses.clear();
    return result;
  }

  private ImmutableList<Path> findUnprocessed(String pathPrefix) throws IOException {
    // pathPrefix is an internal class name prefix containing '/', but paths obtained on Windows
    // will not contain '/' and searches will fail.  So, construct an absolute path from the given
    // string and use its string representation to find the file we need regardless of host
    // system's file system
    Path rootPathPrefix = rootDirectory.resolve(pathPrefix);
    final String rootPathPrefixStr = rootPathPrefix.toString();

    if (!Files.exists(rootPathPrefix.getParent())) {
      return ImmutableList.of();
    }
    try (Stream<Path> paths = Files.list(rootPathPrefix.getParent())) {
      return paths
          .filter(
              path ->
                  path.toString().startsWith(rootPathPrefixStr) && !existingPaths.contains(path))
          .collect(ImmutableList.toImmutableList());
    }
  }
}
