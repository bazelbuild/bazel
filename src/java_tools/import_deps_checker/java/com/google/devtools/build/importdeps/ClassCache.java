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
package com.google.devtools.build.importdeps;

import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.base.Preconditions.checkState;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.MoreObjects;
import com.google.common.base.Predicates;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ImmutableSortedMap;
import com.google.common.io.Closer;
import com.google.devtools.build.importdeps.AbstractClassEntryState.ExistingState;
import com.google.devtools.build.importdeps.AbstractClassEntryState.IncompleteState;
import com.google.devtools.build.importdeps.AbstractClassEntryState.MissingState;
import com.google.devtools.build.importdeps.ClassInfo.MemberInfo;
import java.io.Closeable;
import java.io.IOException;
import java.io.InputStream;
import java.io.PrintStream;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.function.Predicate;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;
import javax.annotation.Nullable;
import org.objectweb.asm.ClassReader;
import org.objectweb.asm.ClassVisitor;
import org.objectweb.asm.FieldVisitor;
import org.objectweb.asm.MethodVisitor;
import org.objectweb.asm.Opcodes;

/** A cache that stores all the accessible classes in a set of JARs. */
public final class ClassCache implements Closeable {

  private final LazyClasspath lazyClasspath;
  private boolean isClosed;

  public ClassCache(
      ImmutableSet<Path> bootclasspath,
      ImmutableSet<Path> directClasspath,
      ImmutableSet<Path> regularClasspath,
      ImmutableSet<Path> inputJars,
      boolean populateMembers)
      throws IOException {
    lazyClasspath =
        new LazyClasspath(
            bootclasspath, directClasspath, regularClasspath, inputJars, populateMembers);
  }

  public AbstractClassEntryState getClassState(String internalName) {
    checkState(!isClosed, "The cache has been closed.");
    LazyClassEntry entry = lazyClasspath.getLazyEntry(internalName);
    if (entry == null) {
      return MissingState.singleton();
    }
    return entry.getState(lazyClasspath);
  }

  public ImmutableMap<Path, Boolean> collectUsedJarsInRegularClasspath() {
    return lazyClasspath.collectUsedJarsInRegularClasspath();
  }

  @Override
  public void close() throws IOException {
    lazyClasspath.close();
    isClosed = true;
  }

  static class LazyClassEntry {
    private final String internalName;
    private final ZipFile zipFile;
    private final Path jarPath;
    private final boolean isDirectDep;

    /**
     * The state of this class entry. If {@literal null}, then this class has not been resolved yet.
     */
    @Nullable private AbstractClassEntryState state = null;

    private LazyClassEntry(
        String internalName, ZipFile zipFile, Path jarPath, boolean isDirectDep) {
      this.internalName = internalName;
      this.zipFile = zipFile;
      this.jarPath = jarPath;
      this.isDirectDep = isDirectDep;
    }

    ZipFile getZipFile() {
      return zipFile;
    }

    @Nullable
    public AbstractClassEntryState getState(LazyClasspath classpath) {
      resolveIfNot(classpath);
      checkState(
          state != null && !state.isMissingState(),
          "The state cannot be null or MISSING. %s",
          state);
      return state;
    }

    @Override
    public String toString() {
      return MoreObjects.toStringHelper(this)
          .add("internalName", internalName)
          .add("state", state)
          .toString();
    }

    boolean isResolved() {
      return state != null;
    }

    private void resolveIfNot(LazyClasspath lazyClasspath) {
      if (isResolved()) {
        return;
      }
      resolveClassEntry(this, lazyClasspath, /* explicitUse= */ true);
      checkNotNull(state, "After resolution, the state cannot be null");
    }

    private static void resolveClassEntry(
        LazyClassEntry classEntry, LazyClasspath lazyClasspath, boolean explicitUse) {
      if (classEntry.state != null) {
        // Already resolved. See if it is the existing state.
        if (classEntry.state instanceof ExistingState) {
          ExistingState state = (ExistingState) classEntry.state;
          if (!state.direct() && explicitUse) {
            // If the state was previously indirect, update now for direct dep
            classEntry.state = ExistingState.create(state.classInfo().get(), explicitUse);
          }
        }
        return;
      }

      String entryName = classEntry.internalName + ".class";
      ZipEntry zipEntry =
          checkNotNull(
              classEntry.zipFile.getEntry(entryName), "The zip entry %s is null.", entryName);
      try (InputStream inputStream = classEntry.zipFile.getInputStream(zipEntry)) {
        ClassReader classReader = new ClassReader(inputStream);
        ImmutableList.Builder<ResolutionFailureChain> resolutionFailureChainsBuilder =
            ImmutableList.builder();
        for (String superName :
            combineWithoutNull(classReader.getSuperName(), classReader.getInterfaces())) {
          Optional<ResolutionFailureChain> failurePath =
              resolveSuperClassEntry(superName, lazyClasspath);
          failurePath.ifPresent(resolutionFailureChainsBuilder::add);
        }
        ClassInfoBuilder classInfoBuilder =
            new ClassInfoBuilder().setJarPath(classEntry.jarPath).setDirect(classEntry.isDirectDep);
        // Only visit the class if we need to extract its list of members.  If we do visit, skip
        // code and debug attributes since we just care about finding declarations here.
        if (lazyClasspath.populateMembers) {
          classReader.accept(
              classInfoBuilder,
              ClassReader.SKIP_CODE | ClassReader.SKIP_DEBUG | ClassReader.SKIP_FRAMES);
        } else {
          classInfoBuilder.setNames(
              classReader.getClassName(), classReader.getSuperName(), classReader.getInterfaces());
        }

        ImmutableList<ResolutionFailureChain> resolutionFailureChains =
            resolutionFailureChainsBuilder.build();
        if (resolutionFailureChains.isEmpty()) {
          classEntry.state =
              ExistingState.create(
                  classInfoBuilder.build(lazyClasspath, /*incomplete=*/ false), explicitUse);
        } else {
          ClassInfo classInfo = classInfoBuilder.build(lazyClasspath, /*incomplete=*/ true);
          classEntry.state =
              IncompleteState.create(
                  classInfo,
                  ResolutionFailureChain.createWithParent(classInfo, resolutionFailureChains));
        }
      } catch (IOException e) {
        throw new RuntimeException("Error when resolving class entry " + entryName, e);
      } catch (RuntimeException e) {
        System.err.println(
            "A runtime exception occurred. The following is the content in the class index. "
                + e.getMessage());
        lazyClasspath.printClasspath(System.err);
        throw e;
      }
    }

    private static Optional<ResolutionFailureChain> resolveSuperClassEntry(
        String superName, LazyClasspath lazyClasspath) {
      LazyClassEntry superClassEntry = lazyClasspath.getLazyEntry(superName);

      if (superClassEntry == null) {
        return Optional.of(ResolutionFailureChain.createMissingClass(superName));
      } else {
        resolveClassEntry(superClassEntry, lazyClasspath, /* explicitUse= */ false);
        AbstractClassEntryState superState = superClassEntry.state;
        if (superState instanceof ExistingState) {
          // Do nothing. Good to proceed.
          return Optional.empty();
        } else if (superState instanceof IncompleteState) {
          return Optional.of(superState.asIncompleteState().resolutionFailureChain());
        } else {
          throw new RuntimeException("Cannot reach here. superState is " + superState);
        }
      }
    }
  }

  private static ImmutableList<String> combineWithoutNull(
      @Nullable String first, @Nullable String[] others) {
    ImmutableList.Builder<String> list = ImmutableList.builder();
    if (first != null) {
      list.add(first);
    }
    if (others != null) {
      list.add(others);
    }
    return list.build();
  }

  /** The classpath, emulating the behavior of the real classpath. */
  @VisibleForTesting
  static final class LazyClasspath implements Closeable {
    private final ClassIndex bootclasspath;
    private final ClassIndex regularClasspath;
    private final ClassIndex inputJars;
    private final ImmutableList<ClassIndex> orderedClasspath;
    final boolean populateMembers; // accessed from other inner classes
    private final Closer closer = Closer.create();

    public LazyClasspath(
        ImmutableSet<Path> bootclasspath,
        ImmutableSet<Path> directClasspath,
        ImmutableSet<Path> regularClasspath,
        ImmutableSet<Path> inputJars,
        boolean populateMembers)
        throws IOException {
      this.populateMembers = populateMembers;
      this.bootclasspath = new ClassIndex("boot classpath", bootclasspath, Predicates.alwaysTrue());
      this.inputJars = new ClassIndex("input jars", inputJars, Predicates.alwaysTrue());
      this.regularClasspath =
          new ClassIndex(
              "regular classpath",
              regularClasspath,
              jar ->
                  bootclasspath.contains(jar)
                      || inputJars.contains(jar)
                      || directClasspath.contains(jar));
      // Reflect runtime resolution order, with input before classpath similar to javac
      this.orderedClasspath =
          ImmutableList.of(this.bootclasspath, this.inputJars, this.regularClasspath);
      this.orderedClasspath.forEach(closer::register);
    }

    public LazyClassEntry getLazyEntry(String internalName) {
      return orderedClasspath
          .stream()
          .map(classIndex -> classIndex.getClassEntry(internalName))
          .filter(Objects::nonNull)
          .findFirst()
          .orElse(null);
    }

    public ImmutableMap<Path, Boolean> collectUsedJarsInRegularClasspath() {
      return regularClasspath.collectUsedJarFiles();
    }

    public void printClasspath(PrintStream stream) {
      orderedClasspath.forEach(c -> c.printClasspath(stream));
    }

    @Override
    public void close() throws IOException {
      closer.close();
    }
  }

  /**
   * Representation of a class path, composed of a list of JARs. It indexes all the class files with
   * the class names.
   */
  private static final class ClassIndex implements Closeable {

    private final String name;
    private final ImmutableMap<String, LazyClassEntry> classIndex;
    private final Closer closer;

    public ClassIndex(String name, ImmutableSet<Path> jarFiles, Predicate<Path> isDirect)
        throws IOException {
      this.name = name;
      this.closer = Closer.create();
      classIndex = buildClassIndex(jarFiles, closer, isDirect);
    }

    @Override
    public void close() throws IOException {
      closer.close();
    }

    public LazyClassEntry getClassEntry(String internalName) {
      return classIndex.get(internalName);
    }

    /** Second argument in the Map is if the jar is used directly (at least once). */
    public ImmutableMap<Path, Boolean> collectUsedJarFiles() {
      Map<Path, Boolean> usedJars = new HashMap<>();
      for (Map.Entry<String, LazyClassEntry> entry : classIndex.entrySet()) {
        LazyClassEntry clazz = entry.getValue();
        if (clazz.isResolved()) {
          if (!usedJars.containsKey(clazz.jarPath) || clazz.state.direct()) {
            usedJars.put(clazz.jarPath, clazz.state.direct());
          }
        }
      }
      return ImmutableSortedMap.copyOf(usedJars);
    }

    private void printClasspath(PrintStream stream) {
      stream.println("Classpath: " + name);
      int counter = 0;
      for (Map.Entry<String, LazyClassEntry> entry : classIndex.entrySet()) {
        stream.printf("%d  %s\n    %s\n\n", ++counter, entry.getKey(), entry.getValue());
      }
    }

    private static ImmutableMap<String, LazyClassEntry> buildClassIndex(
        ImmutableSet<Path> jars, Closer closer, Predicate<Path> isDirect) throws IOException {
      HashMap<String, LazyClassEntry> result = new HashMap<>();
      for (Path jarPath : jars) {
        boolean jarIsDirect = isDirect.test(jarPath);
        try {
          ZipFile zipFile = closer.register(new ZipFile(jarPath.toFile()));
          zipFile
              .stream()
              .forEach(
                  entry -> {
                    String name = entry.getName();
                    if (!name.endsWith(".class")) {
                      return; // Not a class file.
                    }
                    String internalName = name.substring(0, name.lastIndexOf('.'));
                    result.computeIfAbsent(
                        internalName,
                        key -> new LazyClassEntry(key, zipFile, jarPath, jarIsDirect));
                  });
        } catch (Throwable e) {
          throw new RuntimeException("Error in reading zip file " + jarPath, e);
        }
      }
      return ImmutableMap.copyOf(result);
    }
  }

  /** Builder to build a ClassInfo object from the class file. */
  private static class ClassInfoBuilder extends ClassVisitor {

    private String internalName;
    private final ImmutableSet.Builder<MemberInfo> members = ImmutableSet.builder();
    private ImmutableList<String> superClasses;
    private Path jarPath;
    private boolean directDep;

    public ClassInfoBuilder() {
      super(Opcodes.ASM7);
    }

    @Override
    public void visit(
        int version,
        int access,
        String name,
        String signature,
        String superName,
        String[] interfaces) {
      setNames(name, superName, interfaces);
    }

    @Override
    public FieldVisitor visitField(
        int access, String name, String desc, String signature, Object value) {
      members.add(MemberInfo.create(name, desc));
      return null;
    }

    @Override
    public MethodVisitor visitMethod(
        int access, String name, String desc, String signature, String[] exceptions) {
      members.add(MemberInfo.create(name, desc));
      return null;
    }

    void setNames(String name, String superName, String[] interfaces) {
      checkState(internalName == null && superClasses == null, "This visitor is already used.");
      internalName = name;
      superClasses = combineWithoutNull(superName, interfaces);
    }

    public ClassInfoBuilder setJarPath(Path jarPath) {
      this.jarPath = jarPath;
      return this;
    }

    public ClassInfoBuilder setDirect(boolean direct) {
      this.directDep = direct;
      return this;
    }

    public ClassInfo build(LazyClasspath lazyClasspath, boolean incomplete) {
      ImmutableList<ClassInfo> superClassInfos =
          superClasses
              .stream()
              .map(lazyClasspath::getLazyEntry)
              // nulls possible when building ClassInfo for an "incomplete" class
              .filter(entry -> entry != null && entry.state != null)
              .map(entry -> entry.state.classInfo().get())
              .collect(ImmutableList.toImmutableList());
      checkState(
          incomplete || superClassInfos.size() == superClasses.size(),
          "Missing class info for some of %s's super types %s: %s",
          internalName,
          superClasses,
          superClassInfos);
      return ClassInfo.create(
          checkNotNull(internalName),
          checkNotNull(jarPath),
          directDep,
          superClassInfos,
          members.build());
    }
  }
}
