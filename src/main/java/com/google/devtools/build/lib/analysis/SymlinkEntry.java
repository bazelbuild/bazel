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

package com.google.devtools.build.lib.analysis;

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.starlarkbuildapi.SymlinkEntryApi;
import com.google.devtools.build.lib.vfs.PathFragment;
import net.starlark.java.eval.Printer;

/**
 * An entry in the runfiles map.
 *
 * <p>build-runfiles.cc enforces the following constraints: The PathFragment must not be an absolute
 * path, nor contain "..". Overlapping runfiles links are also refused. This is the case where you
 * ask to create a link to "foo" and also "foo/bar.txt". I.e. you're asking it to make "foo" both a
 * file (symlink) and a directory.
 *
 * <p>Links to directories are heavily discouraged.
 */
//
// O intrepid fixer or bugs and implementor of features, dare not to add a .equals() method
// to this class, lest you condemn yourself, or a fellow other developer to spending two
// delightful hours in a fancy hotel on a Chromebook that is utterly unsuitable for Java
// development to figure out what went wrong, just like I just did.
//
// The semantics of the symlinks nested set dictates that later entries overwrite earlier
// ones. However, the semantics of nested sets dictate that if there are duplicate entries, they
// are only returned once in the iterator.
//
// These two things, innocent when taken alone, result in the effect that when there are three
// entries for the same path, the first one and the last one the same, and the middle one
// different, the *middle* one will take effect: the middle one overrides the first one, and the
// first one prevents the last one from appearing on the iterator.
//
// The lack of a .equals() method prevents this by making the first entry in the above case not
// equal to the third one if they are not the same instance (which they almost never are).
//
// Goodnight, prince(ss)?, and sweet dreams.
public final class SymlinkEntry implements SymlinkEntryApi {
  private final PathFragment path;
  private final Artifact artifact;

  SymlinkEntry(PathFragment path, Artifact artifact) {
    this.path = Preconditions.checkNotNull(path);
    this.artifact = Preconditions.checkNotNull(artifact);
  }

  @Override
  public String getPathString() {
    return path.getPathString();
  }

  public PathFragment getPath() {
    return path;
  }

  @Override
  public Artifact getArtifact() {
    return artifact;
  }

  @Override
  public boolean isImmutable() {
    return true;
  }

  @Override
  public void repr(Printer printer) {
    printer.append("SymlinkEntry(path = ");
    printer.repr(getPathString());
    printer.append(", target_file = ");
    artifact.repr(printer);
    printer.append(")");
  }
}
