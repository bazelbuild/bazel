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
//
package com.google.devtools.build.lib.vfs;

import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.collect.ImmutableList.toImmutableList;
import static com.google.common.truth.Truth.assertThat;
import static java.util.Arrays.stream;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoMoreInteractions;
import static org.mockito.Mockito.when;

import com.google.common.collect.ImmutableClassToInstanceMap;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.testing.junit.testparameterinjector.TestParameterInjector;
import com.google.testing.junit.testparameterinjector.TestParameters;
import com.google.testing.junit.testparameterinjector.TestParameters.TestParametersValues;
import com.google.testing.junit.testparameterinjector.TestParameters.TestParametersValuesProvider;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.util.Arrays;
import java.util.stream.Collectors;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;

/** Unit tests for PathTransformingDelegateFileSystem. Make sure all methods rewrite paths. */
@RunWith(TestParameterInjector.class)
public class PathTransformingDelegateFileSystemTest {
  private final FileSystem delegateFileSystem = createMockFileSystem();
  private final TestDelegateFileSystem fileSystem = new TestDelegateFileSystem(delegateFileSystem);

  private static FileSystem createMockFileSystem() {
    FileSystem fileSystem = mock(FileSystem.class);
    when(fileSystem.getDigestFunction()).thenReturn(DigestHashFunction.SHA256);
    when(fileSystem.getPath(any(PathFragment.class))).thenCallRealMethod();
    return fileSystem;
  }

  @Before
  public void verifyGetDigestFunctionCalled() {
    // getDigestFunction gets called in the constructor of PathTransformingDelegateFileSystem, make
    // sure to "consume" that so that tests don't need to account for that.
    verify(delegateFileSystem).getDigestFunction();
  }

  @Test
  @TestParameters(valuesProvider = FileSystemMethodProvider.class)
  public void simplePathMethod_callsDelegateWithRewrittenPath(Method method) throws Exception {
    PathFragment path = PathFragment.create("/original/dir/file");

    method.invoke(fileSystem, pathAndDefaultArgs(method, path));

    method.invoke(
        verify(delegateFileSystem),
        pathAndDefaultArgs(method, PathFragment.create("/transformed/dir/file")));
    verifyNoMoreInteractions(delegateFileSystem);
  }

  @Test
  public void readSymbolicLink_callsDelegateWithRewrittenPathAndTransformsItBack()
      throws Exception {
    PathFragment path = PathFragment.create("/original/dir/file");
    when(delegateFileSystem.readSymbolicLink(PathFragment.create("/transformed/dir/file")))
        .thenReturn(PathFragment.create("/transformed/resolved"));

    PathFragment resolvedPath = fileSystem.readSymbolicLink(path);

    assertThat(resolvedPath).isEqualTo(PathFragment.create("/original/resolved"));
  }

  @Test
  public void resolveSymbolicLinks_callsDelegateWithRewrittenPathAndTransformsItBack()
      throws Exception {
    PathFragment path = PathFragment.create("/original/dir/file");
    when(delegateFileSystem.resolveSymbolicLinks(PathFragment.create("/transformed/dir/file")))
        .thenReturn(Path.create("/transformed/resolved", delegateFileSystem));

    Path resolvedPath = fileSystem.resolveSymbolicLinks(path);

    assertThat(resolvedPath.asFragment()).isEqualTo(PathFragment.create("/original/resolved"));
    assertThat(resolvedPath.getFileSystem()).isSameInstanceAs(fileSystem);
  }

  @Test
  public void createSymbolicLink_callsDelegateWithRewrittenPathNotTarget() throws Exception {
    PathFragment target = PathFragment.create("/original/target");

    fileSystem.createSymbolicLink(PathFragment.create("/original/dir/file"), target);

    verify(delegateFileSystem)
        .createSymbolicLink(PathFragment.create("/transformed/dir/file"), target);
    verifyNoMoreInteractions(delegateFileSystem);
  }

  private static final ImmutableClassToInstanceMap<?> DEFAULT_VALUES =
      ImmutableClassToInstanceMap.builder()
          .put(boolean.class, false)
          .put(int.class, 0)
          .put(long.class, 0L)
          .put(String.class, "")
          .build();

  private static Object[] pathAndDefaultArgs(Method method, PathFragment path) {
    Class<?>[] types = method.getParameterTypes();
    Object[] result = new Object[types.length];
    for (int i = 0; i < types.length; ++i) {
      if (types[i].equals(PathFragment.class)) {
        result[i] = path.replaceName(path.getBaseName() + i);
        continue;
      }
      result[i] =
          checkNotNull(
              DEFAULT_VALUES.get(types[i]), "Missing default value for: %s", types[i].getName());
    }
    return result;
  }

  private static class TestDelegateFileSystem extends PathTransformingDelegateFileSystem {

    private static final PathFragment ORIGINAL = PathFragment.create("/original");
    private static final PathFragment TRANSFORMED = PathFragment.create("/transformed");

    TestDelegateFileSystem(FileSystem fileSystem) {
      super(fileSystem);
    }

    @Override
    protected PathFragment toDelegatePath(PathFragment path) {
      return TRANSFORMED.getRelative(path.relativeTo(ORIGINAL));
    }

    @Override
    protected PathFragment fromDelegatePath(PathFragment delegatePath) {
      return ORIGINAL.getRelative(delegatePath.relativeTo(TRANSFORMED));
    }
  }

  private static class FileSystemMethodProvider implements TestParametersValuesProvider {

    private static final ImmutableSet<Method> IGNORED =
        ImmutableSet.of(
            getFileSystemMethod("getPath", PathFragment.class),
            getFileSystemMethod("readSymbolicLink", PathFragment.class),
            getFileSystemMethod("resolveSymbolicLinks", PathFragment.class),
            getFileSystemMethod("createSymbolicLink", PathFragment.class, PathFragment.class));

    private static Method getFileSystemMethod(String name, Class<?>... parameterTypes) {
      try {
        return FileSystem.class.getDeclaredMethod(name, parameterTypes);
      } catch (NoSuchMethodException e) {
        throw new IllegalArgumentException(e);
      }
    }

    @Override
    public ImmutableList<TestParametersValues> provideValues() {
      return stream(FileSystem.class.getDeclaredMethods())
          .filter(
              m ->
                  !IGNORED.contains(m)
                      && !Modifier.isStatic(m.getModifiers())
                      && !Modifier.isFinal(m.getModifiers())
                      && ImmutableList.copyOf(m.getParameterTypes()).contains(PathFragment.class))
          .map(
              m ->
                  TestParametersValues.builder()
                      .name(m.getName() + parameterString(m.getParameterTypes()))
                      .addParameter("method", m)
                      .build())
          .collect(toImmutableList());
    }

    private static String parameterString(Class<?>[] types) {
      return Arrays.stream(types)
          .map(Class::getSimpleName)
          .collect(Collectors.joining(", ", "(", ")"));
    }
  }
}
