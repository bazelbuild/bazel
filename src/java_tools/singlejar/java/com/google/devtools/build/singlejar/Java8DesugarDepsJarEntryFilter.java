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

package com.google.devtools.build.singlejar;

import com.google.devtools.build.android.desugar.proto.DesugarDeps.Dependency;
import com.google.devtools.build.android.desugar.proto.DesugarDeps.DesugarDepsInfo;
import com.google.devtools.build.android.desugar.proto.DesugarDeps.InterfaceDetails;
import com.google.devtools.build.android.desugar.proto.DesugarDeps.InterfaceWithCompanion;
import com.google.devtools.build.android.desugar.proto.DesugarDeps.Type;
import com.google.protobuf.ByteString;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * {@link ZipEntryFilter} that implements consistency checking of {@code META-INF/desugar_deps}
 * files emitted by {@link com.google.devtools.build.android.desugar.Desugar}.  This is used to
 * implement singlejar's {@code --check_desugar_deps} flag.
 */
class Java8DesugarDepsJarEntryFilter
    implements ZipEntryFilter, ZipEntryFilter.CustomMergeStrategy {

  private final Map<String, ByteString> neededDeps = new LinkedHashMap<>();
  private final Map<String, ByteString> missingInterfaces = new LinkedHashMap<>();
  private final Map<String, List<String>> extendedInterfaces = new HashMap<>();
  private final Map<String, Boolean> hasDefaultMethods = new HashMap<>();
  private final Set<String> seen = new HashSet<>();

  private final ZipEntryFilter delegate;

  public Java8DesugarDepsJarEntryFilter(ZipEntryFilter delegate) {
    this.delegate = delegate;
  }

  @Override
  public void accept(String filename, StrategyCallback callback) throws IOException {
    if ("META-INF/desugar_deps".equals(filename)) {
      callback.customMerge(null, this);
    } else if (filename.startsWith("j$/")) {
      throw new IOException("Unexpectedly found desugar_jdk_libs file: " + filename);
    } else {
      seen.add(filename);
      delegate.accept(filename, callback);
    }
  }

  @Override
  public void merge(InputStream in, OutputStream out) throws IOException {
    DesugarDepsInfo depsInfo = DesugarDepsInfo.parseFrom(in);
    for (Dependency assumed : depsInfo.getAssumePresentList()) {
      neededDeps.putIfAbsent(
          assumed.getTarget().getBinaryName() + ".class",
          assumed.getOrigin().getBinaryNameBytes());
    }
    for (Dependency missing : depsInfo.getMissingInterfaceList()) {
      missingInterfaces.putIfAbsent(
          missing.getTarget().getBinaryName(),
          missing.getOrigin().getBinaryNameBytes());
    }
    for (InterfaceDetails itf : depsInfo.getInterfaceWithSupertypesList()) {
      if (itf.getExtendedInterfaceCount() > 0
          && !extendedInterfaces.containsKey(itf.getOrigin().getBinaryName())) {
        // Avoid Guava dependency
        ArrayList<String> supertypes = new ArrayList<>(itf.getExtendedInterfaceCount());
        for (Type extended : itf.getExtendedInterfaceList()) {
          supertypes.add(extended.getBinaryName());
        }
        extendedInterfaces.putIfAbsent(itf.getOrigin().getBinaryName(), supertypes);
      }
    }
    for (InterfaceWithCompanion companion : depsInfo.getInterfaceWithCompanionList()) {
      if (companion.getNumDefaultMethods() > 0) {
        // Only remember interfaces that definitely have default methods for now.
        // For all other interfaces we'll transitively check extended interfaces
        // in HasDefaultMethods.
        hasDefaultMethods.putIfAbsent(companion.getOrigin().getBinaryName(), true);
      }
    }
    // Don't write anything to out, we just want to check these files for consistency
  }

  @Override
  public void finish(OutputStream out) throws IOException {
    for (Map.Entry<String, ByteString> need : neededDeps.entrySet()) {
      if (!seen.contains(need.getKey())) {
        throw new IOException(need.getKey() + " referenced by " + need.getValue().toStringUtf8()
            + " but not found.  Is the former defined in a neverlink library?");
      }
    }

    for (Map.Entry<String, ByteString> missing : missingInterfaces.entrySet()) {
      if (hasDefaultMethods(missing.getKey())) {
        throw new IOException(missing.getKey()
            + " needed to desugar "
            + missing.getValue().toStringUtf8()
            + ".  Please add a dependency to the former to the library containing the latter.");
      }
    }
    // Don't write anything to out, we just want to check these files for consistency
  }

  @Override
  public boolean skipEmpty() {
    return true;  // We never want to write these files into the output Jar
  }

  private boolean hasDefaultMethods(String itf) {
    Boolean cached = hasDefaultMethods.putIfAbsent(itf, false);
    if (cached != null) {
      return cached;  // Already in the map
    }

    List<String> extended = extendedInterfaces.get(itf);
    if (extended != null) {
      for (String supertype : extended) {
        if (hasDefaultMethods(supertype)) {
          hasDefaultMethods.put(itf, true);
          return true;
        }
      }
    }
    // We primed with false above in case of cycles so just return that
    return false;
  }
}
