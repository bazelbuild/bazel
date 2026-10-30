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
package com.google.devtools.build.lib.actions;

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.base.Preconditions.checkState;
import static java.util.stream.Collectors.toList;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.MoreObjects;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.actions.Artifact.FilesetArtifact;
import com.google.devtools.build.lib.actions.Artifact.RunfilesArtifact;
import com.google.devtools.build.lib.actions.Artifact.TreeArtifact;
import com.google.devtools.build.lib.actions.Artifact.TreeFileArtifact;
import com.google.devtools.build.lib.skyframe.TreeArtifactValue;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.function.BiConsumer;
import javax.annotation.Nullable;

/**
 * Helper for {@link InputMetadataProvider} implementations.
 *
 * <p>Allows {@link FileArtifactValue} lookups by exec path or {@link ActionInput}. <i>Also</i>
 * allows {@link ActionInput} to be looked up by exec path.
 *
 * <p>This class implements a closed hash-map with the "links" of each bucket's linked list being
 * stored in a flat array to avoid memory allocations and garbage collection.
 *
 * <p>This class is thread-compatible.
 */
public final class ActionInputMap implements InputMetadataProvider {

  /** The number of elements contained in this map. */
  private int size;

  /**
   * The hash buckets. Values are indexes into the four arrays. The number of buckets is always the
   * smallest power of 2 that is larger than the number of elements.
   */
  private int[] table;

  /** Flat array of the next pointers that make up the linked list behind each hash bucket. */
  private int[] next;

  /** The {@link ActionInput} keys stored in this map. */
  private ActionInput[] keys;

  /**
   * The values stored in this map. Each value is one of {@link FileArtifactValue}, {@link
   * TreeArtifactValue} or {@link RunfilesArtifactValue}.
   */
  private Object[] values;

  /**
   * The tree artifacts stored in this map, indexed by their exec paths. This is used to locate
   * TreeFileArtifacts without storing them and their metadata in the main map, which would be
   * significantly more expensive.
   */
  private final TreeMap<PathFragment, TreeArtifactValue> treeArtifacts =
      Maps.newTreeMap(PathFragment.HIERARCHICAL_COMPARATOR);

  /** The filesets stored in this map. */
  private final Map<Artifact, FilesetOutputTree> filesets = Maps.newTreeMap();

  /** The runfiles trees stored in this map. */
  private final List<RunfilesTree> runfilesTrees = new ArrayList<>();

  public ActionInputMap(int sizeHint) {
    sizeHint = Math.max(1, sizeHint);
    int tableSize = Integer.highestOneBit(sizeHint) << 1;
    size = 0;

    table = new int[tableSize];
    Arrays.fill(table, -1);

    next = new int[sizeHint];
    keys = new ActionInput[sizeHint];
    values = new Object[sizeHint];
  }

  private int getIndex(PathFragment execPath) {
    int hashCode = execPath.hashCode();
    int index = hashCode & (table.length - 1);
    if (table[index] == -1) {
      return -1;
    }
    index = table[index];
    while (index != -1) {
      PathFragment otherExecPath = keys[index].getExecPath();
      if (hashCode == otherExecPath.hashCode() && execPath.equals(otherExecPath)) {
        return index;
      }
      index = next[index];
    }
    return -1;
  }

  @Nullable
  @Override
  public FileArtifactValue getInputMetadata(ActionInput input) {
    return getInputMetadataChecked(input);
  }

  @Nullable
  @Override
  public FileArtifactValue getInputMetadataChecked(ActionInput input) {
    if (input instanceof RunfilesArtifact) {
      RunfilesArtifactValue runfilesMetadata = getRunfilesMetadata(input);
      return runfilesMetadata == null ? null : runfilesMetadata.getMetadata();
    }
    if (input instanceof FilesetArtifact) {
      FilesetOutputTree filesetOutput = getFileset(input);
      return filesetOutput == null ? null : filesetOutput.getMetadata();
    }

    if (input instanceof TreeFileArtifact treeFileArtifact) {
      TreeArtifact parent = treeFileArtifact.getParent();
      int treeIndex = getIndex(parent.getExecPath());
      // If the parent is a subtree artifact, and the subtree is not found, fallback to check the
      // top-level tree artifact.
      // This distinction is necessary to handle two scenarios:
      // 1. When an action is passed a subtree artifact, the subtree and its TreeArtifactValue are
      //    stored in treeArtifacts. In this case, when requesting the metadata for a
      //    TreeFileArtifact under a subtree, we are able to resolve the correct tree artifact
      //    directly.
      // 2. When an action is passed a top-level tree artifact (that itself contains a subtree of
      //    files), the top-level tree is stored with a flattened TreeArtifactValue containing all
      //    children under the top-level tree. In this case, when requesting the metadata for a
      //    TreeFileArtifact under a subtree, we need to need search for the top-level tree artifact
      //    rather than the file's parent (subtree).
      // tree artifact or a subdirectory (tree artifact) as its input.
      if (treeIndex == -1 && parent.isSubTreeArtifact()) {
        treeIndex = getIndex(parent.getParent().getExecPath());
      }
      if (treeIndex != -1) {
        checkArgument(
            values[treeIndex] instanceof TreeArtifactValue,
            "Requested tree file artifact under non-tree/omitted tree artifact: %s",
            input);
        return ((TreeArtifactValue) values[treeIndex]).getChildValues().get(treeFileArtifact);
      }
    }
    int index = getIndex(input.getExecPath());
    if (index != -1) {
      Object value = values[index];
      return value instanceof TreeArtifactValue treeValue
          ? treeValue.getMetadata()
          : (FileArtifactValue) value;
    }
    if (input instanceof Artifact) {
      // Non tree artifacts cannot overlap with tree files, therefore we can skip searching the
      // parents.
      return null;
    }

    // Check the trees in case input is a non-Artifact ActionInput pointing to a tree artifact file.
    // This can happen if both a TreeArtifact and a Fileset containing the TreeArtifact are inputs
    // to the same action.
    return getMetadataFromTreeArtifacts(input.getExecPath());
  }

  @Nullable
  @Override
  public FilesetOutputTree getFileset(ActionInput input) {
    checkArgument(input instanceof FilesetArtifact, input);

    return filesets.get(input);
  }

  @Override
  public Map<Artifact, FilesetOutputTree> getFilesets() {
    return Collections.unmodifiableMap(filesets);
  }

  @Nullable
  @Override
  public RunfilesArtifactValue getRunfilesMetadata(ActionInput input) {
    checkArgument(input instanceof RunfilesArtifact, input);

    int index = getIndex(input.getExecPath());
    if (index == -1) {
      return null;
    }

    return (RunfilesArtifactValue) values[index];
  }

  @Override
  public ImmutableList<RunfilesTree> getRunfilesTrees() {
    return ImmutableList.copyOf(runfilesTrees);
  }

  /**
   * For each tree artifact in this input map, invokes the given callback with its exec path and its
   * metadata.
   */
  public void forEachTreeArtifact(BiConsumer<PathFragment, TreeArtifactValue> consumer) {
    treeArtifacts.forEach(consumer);
  }

  /**
   * Returns metadata for given path.
   *
   * <p>This method is less efficient than {@link #getInputMetadata(ActionInput)}, please use that
   * method instead of this one when looking up {@linkplain ActionInput action inputs}.
   */
  @Nullable
  public FileArtifactValue getMetadata(PathFragment execPath) {
    int index = getIndex(execPath);
    if (index != -1) {
      Object value = values[index];
      return value instanceof TreeArtifactValue treeValue
          ? treeValue.getMetadata()
          : (FileArtifactValue) value;
    }

    // Fall back to searching the tree artifacts.
    return getMetadataFromTreeArtifacts(execPath);
  }

  @Nullable
  private FileArtifactValue getMetadataFromTreeArtifacts(PathFragment execPath) {
    TreeArtifactValue tree = getEnclosingTreeMetadata(execPath);
    if (tree == null) {
      return null;
    }

    Map.Entry<?, FileArtifactValue> entry = tree.findChildEntryByExecPath(execPath);
    return entry != null ? entry.getValue() : null;
  }

  @Nullable
  @Override
  public TreeArtifactValue getTreeMetadata(ActionInput input) {
    checkArgument(input instanceof TreeArtifact, input);
    return getTreeMetadata(input.getExecPath());
  }

  @Nullable
  public TreeArtifactValue getTreeMetadata(PathFragment execPath) {
    int index = getIndex(execPath);
    if (index < 0) {
      return null;
    }
    Object value = values[index];
    return value instanceof TreeArtifactValue treeValue ? treeValue : null;
  }

  @Nullable
  @Override
  public TreeArtifactValue getEnclosingTreeMetadata(PathFragment execPath) {
    Map.Entry<PathFragment, TreeArtifactValue> entry = treeArtifacts.floorEntry(execPath);
    if (entry != null && execPath.startsWith(entry.getKey())) {
      return entry.getValue();
    }
    return null;
  }

  @Nullable
  @Override
  public ActionInput getInput(PathFragment execPath) {
    int index = getIndex(execPath);
    if (index != -1) {
      return keys[index];
    }

    // Search ancestor paths since execPathString may point to a TreeFileArtifact within one of the
    // tree artifacts.
    TreeArtifactValue tree = getEnclosingTreeMetadata(execPath);
    if (tree == null) {
      return null;
    }

    // We must return an entry from the map since a duplicate would not have the generating action
    // key set.
    Map.Entry<TreeFileArtifact, ?> entry = tree.findChildEntryByExecPath(execPath);
    return entry != null ? entry.getKey() : null;
  }

  /**
   * Returns count of unique, top-level {@linkplain ActionInput action inputs} in the map.
   *
   * <p>Top-level means that each tree artifact, counts as 1, irrespective of the number of children
   * it has.
   */
  public int sizeForDebugging() {
    return size;
  }

  public void put(ActionInput input, FileArtifactValue metadata) {
    checkArgument(
        !(input instanceof TreeArtifact),
        "Can't add tree artifact: %s using put -- please use putTreeArtifact for that",
        input);
    checkArgument(
        !(input instanceof RunfilesArtifact),
        "Can't add runfiles tree: %s using put -- please use putRunfilesMetadata for that",
        input);

    int oldIndex = putIfAbsent(input, metadata);
    checkArgument(
        oldIndex == -1 || !(keys[oldIndex] instanceof TreeArtifact),
        "Tried to overwrite tree artifact with a file: '%s' with the same exec path",
        input);
  }

  public void putFileset(Artifact input, FilesetOutputTree outputTree) {
    checkArgument(input.isFileset(), input);

    filesets.put(input, outputTree);
  }

  public void putRunfilesMetadata(Artifact input, RunfilesArtifactValue metadata) {
    checkArgument(input.isRunfilesTree(), input);

    int oldIndex = putIfAbsent(input, metadata);
    checkState(oldIndex == -1);

    runfilesTrees.add(metadata.getRunfilesTree());
  }

  public void putTreeArtifact(Artifact tree, TreeArtifactValue treeArtifactValue) {
    checkArgument(tree.isTreeArtifact(), tree);
    int oldIndex = putIfAbsent(tree, treeArtifactValue);
    if (oldIndex != -1) {
      checkArgument(
          keys[oldIndex] instanceof TreeArtifact,
          "Tried to overwrite file with a tree artifact: '%s' with the same exec path",
          tree);
      return;
    }

    treeArtifacts.put(tree.getExecPath(), treeArtifactValue);
  }

  private int putIfAbsent(ActionInput input, Object metadata) {
    checkNotNull(input);
    if (size >= keys.length) {
      resize();
    }
    PathFragment execPath = input.getExecPath();
    int hashCode = execPath.hashCode();
    int index = hashCode & (table.length - 1);
    int nextIndex = table[index];
    if (nextIndex == -1) {
      table[index] = size;
    } else {
      do {
        index = nextIndex;
        PathFragment otherExecPath = keys[index].getExecPath();
        if (hashCode == otherExecPath.hashCode() && execPath.equals(otherExecPath)) {
          return index;
        }
        nextIndex = next[index];
      } while (nextIndex != -1);
      next[index] = size;
    }
    next[size] = -1;
    keys[size] = input;
    values[size] = metadata;
    size++;
    return -1;
  }

  @VisibleForTesting
  void clear() {
    Arrays.fill(table, -1);
    Arrays.fill(next, -1);
    Arrays.fill(keys, null);
    Arrays.fill(values, null);
    size = 0;
    treeArtifacts.clear();
    filesets.clear();
    runfilesTrees.clear();
  }

  private void resize() {
    // Resize the data containers.
    keys = Arrays.copyOf(keys, size * 2);
    values = Arrays.copyOf(values, size * 2);
    next = Arrays.copyOf(next, size * 2);

    // Resize and recreate the table and links if necessary. We can take shortcuts here as we know
    // there are no duplicate keys.
    if (table.length < size * 2) {
      table = new int[table.length * 2];
      Arrays.fill(table, -1);
      for (int i = 0; i < size; i++) {
        int index = keys[i].getExecPath().hashCode() & (table.length - 1);
        next[i] = table[index];
        table[index] = i;
      }
    }
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this)
        .add("size", size)
        .add("all-files", sizeForDebugging())
        .add("first-fifty-keys", Arrays.stream(keys).limit(50).collect(toList()))
        .add("first-fifty-values", Arrays.stream(values).limit(50).collect(toList()))
        .toString();
  }
}
