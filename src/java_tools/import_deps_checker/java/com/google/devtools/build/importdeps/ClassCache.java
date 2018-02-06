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

import com.google.common.base.MoreObjects;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.io.Closer;
import com.google.devtools.build.importdeps.AbstractClassEntryState.ExistingState;
import com.google.devtools.build.importdeps.AbstractClassEntryState.IncompleteState;
import com.google.devtools.build.importdeps.AbstractClassEntryState.MissingState;
import com.google.devtools.build.importdeps.ClassInfo.MemberInfo;
import java.io.Closeable;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;
import javax.annotation.Nullable;
import org.objectweb.asm.ClassReader;
import org.objectweb.asm.ClassVisitor;
import org.objectweb.asm.FieldVisitor;
import org.objectweb.asm.MethodVisitor;
import org.objectweb.asm.Opcodes;

/** A cache that stores all the accessible classes. */
public final class ClassCache implements Closeable {

  private final ImmutableMap<String, LazyClassEntry> classIndex;
  /**
   * If the cache is open, then the {@code closer} is nonnull. After the cache is closed, the {@code
   * closer} is set to {@literal null}.
   */
  @Nullable private Closer closer;

  public ClassCache(Path... jars) throws IOException {
    this(ImmutableList.copyOf(jars));
  }

  public ClassCache(ImmutableList<Path> jars) throws IOException {
    closer = Closer.create();
    this.classIndex = buildClassIndex(jars, closer);
  }

  public AbstractClassEntryState getClassState(String internalName) {
    ensureCacheIsOpen();
    LazyClassEntry entry = classIndex.get(internalName);
    if (entry == null) {
      return MissingState.singleton();
    }
    return entry.getState(classIndex);
  }

  @Override
  public void close() throws IOException {
    if (closer == null) {
      return;
    }
    closer.close();
    closer = null;
  }

  private static ImmutableMap<String, LazyClassEntry> buildClassIndex(
      ImmutableList<Path> jars, Closer closer) throws IOException {
    HashMap<String, LazyClassEntry> result = new HashMap<>();
    for (Path jarPath : jars) {
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
                  result.computeIfAbsent(internalName, key -> new LazyClassEntry(key, zipFile));
                });
      } catch (Throwable e) {
        throw new RuntimeException("Error in reading zip file " + jarPath, e);
      }
    }
    return ImmutableMap.copyOf(result);
  }

  private void ensureCacheIsOpen() {
    checkState(closer != null, "The cache should be open!");
  }

  static class LazyClassEntry {
    private final String internalName;
    private final ZipFile zipFile;

    /**
     * The state of this class entry. If {@literal null}, then this class has not been resolved yet.
     */
    @Nullable private AbstractClassEntryState state = null;

    private LazyClassEntry(String internalName, ZipFile zipFile) {
      this.internalName = internalName;
      this.zipFile = zipFile;
    }

    ZipFile getZipFile() {
      return zipFile;
    }

    @Nullable
    public AbstractClassEntryState getState(ImmutableMap<String, LazyClassEntry> classIndex) {
      resolveIfNot(classIndex);
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

    private void resolveIfNot(ImmutableMap<String, LazyClassEntry> classIndex) {
      if (state != null) {
        return;
      }
      resolveClassEntry(this, classIndex);
      checkNotNull(state, "After resolution, the state cannot be null");
    }

    private static void resolveClassEntry(
        LazyClassEntry classEntry, ImmutableMap<String, LazyClassEntry> classIndex) {
      if (classEntry.state != null) {
        // Already resolved. See if it is the existing state.
        return;
      }

      String entryName = classEntry.internalName + ".class";
      ZipEntry zipEntry =
          checkNotNull(
              classEntry.zipFile.getEntry(entryName), "The zip entry %s is null.", entryName);
      try (InputStream inputStream = classEntry.zipFile.getInputStream(zipEntry)) {
        ClassReader classReader = new ClassReader(inputStream);
        ImmutableList<String> resolutionFailurePath = null;
        for (String superName :
            combineWithoutNull(classReader.getSuperName(), classReader.getInterfaces())) {
          LazyClassEntry superClassEntry = classIndex.get(superName);

          if (superClassEntry == null) {
            resolutionFailurePath = ImmutableList.of(superName);
            break;
          } else {
            resolveClassEntry(superClassEntry, classIndex);
            AbstractClassEntryState superState = superClassEntry.state;
            if (superState instanceof ExistingState) {
              // Do nothing. Good to proceed.
              continue;
            } else if (superState instanceof IncompleteState) {
              resolutionFailurePath =
                  ImmutableList.<String>builder()
                      .add(superName)
                      .addAll(((IncompleteState) superState).getResolutionFailurePath())
                      .build();
              break;
            } else {
              throw new RuntimeException("Cannot reach here. superState is " + superState);
            }
          }
        }
        ClassInfoBuilder classInfoBuilder = new ClassInfoBuilder();
        classReader.accept(classInfoBuilder, ClassReader.SKIP_DEBUG | ClassReader.SKIP_FRAMES);
        if (resolutionFailurePath == null) {
          classEntry.state = ExistingState.create(classInfoBuilder.build(classIndex));
        } else {
          classEntry.state =
              IncompleteState.create(classInfoBuilder.build(classIndex), resolutionFailurePath);
        }
      } catch (IOException e) {
        throw new RuntimeException("Error when resolving class entry " + entryName);
      } catch (RuntimeException e) {
        System.err.println(
            "A runtime exception occurred. The following is the content in the class index. "
                + e.getMessage());
        int counter = 0;
        for (Map.Entry<String, LazyClassEntry> entry : classIndex.entrySet()) {
          System.err.printf("%d  %s\n    %s\n\n", ++counter, entry.getKey(), entry.getValue());
        }
        throw e;
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

  /** Builder to build a ClassInfo object from the class file. */
  private static class ClassInfoBuilder extends ClassVisitor {

    private String internalName;
    private final ImmutableSet.Builder<MemberInfo> members = ImmutableSet.builder();
    private ImmutableList<String> superClasses;

    public ClassInfoBuilder() {
      super(Opcodes.ASM6);
    }

    @Override
    public void visit(
        int version,
        int access,
        String name,
        String signature,
        String superName,
        String[] interfaces) {
      checkState(internalName == null && superClasses == null, "This visitor is already used.");
      internalName = name;
      superClasses = combineWithoutNull(superName, interfaces);
    }

    @Override
    public FieldVisitor visitField(
        int access, String name, String desc, String signature, Object value) {
      members.add(MemberInfo.create(internalName, name, desc));
      return null;
    }

    @Override
    public MethodVisitor visitMethod(
        int access, String name, String desc, String signature, String[] exceptions) {
      members.add(MemberInfo.create(internalName, name, desc));
      return null;
    }

    public ClassInfo build(ImmutableMap<String, LazyClassEntry> classIndex) {
      return ClassInfo.create(
          checkNotNull(internalName),
          superClasses
              .stream()
              .map(classIndex::get)
              .filter(Objects::nonNull)
              .map(entry -> entry.state.classInfo().get())
              .collect(ImmutableList.toImmutableList()),
          checkNotNull(members).build());
    }
  }
}
