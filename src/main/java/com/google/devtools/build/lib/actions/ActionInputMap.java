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
import static java.util.stream.Collectors.toList;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.MoreObjects;
import com.google.common.base.Objects;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifact;
import com.google.devtools.build.lib.actions.Artifact.TreeFileArtifact;
import com.google.devtools.build.lib.bugreport.BugReporter;
import com.google.devtools.build.lib.skyframe.TreeArtifactValue;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
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
public final class ActionInputMap implements InputMetadataProvider, ActionInputMapSink {

  private static final Object PLACEHOLDER = new Object();

  static class TrieArtifact {
    private Map<String, TrieArtifact> subFolders = ImmutableMap.of();
    @Nullable private TreeArtifactValue treeArtifactValue;

    TrieArtifact add(PathFragment treeExecPath, TreeArtifactValue treeArtifactValue) {
      TrieArtifact current = this;
      for (String segment : treeExecPath.segments()) {
        current = current.getOrPutSubFolder(segment);
      }
      current.treeArtifactValue = treeArtifactValue;
      return current;
    }

    TrieArtifact getOrPutSubFolder(String name) {
      TrieArtifact trieArtifact;
      // We special case when we have 0 or 1 child in order to save memory.
      // This way, we do not allocate hash maps for leaf entries and path entries with a single
      // child (prefixes of unbranched paths, e.g. [a/b/c/d]/tree{1..n}).
      // Invariant: subFolders is an immutable map iff subFolders.size() <= 1.
      switch (subFolders.size()) {
        case 0:
          trieArtifact = new TrieArtifact();
          subFolders = ImmutableMap.of(name, trieArtifact);
          break;
        case 1:
          trieArtifact = subFolders.get(name);
          if (trieArtifact == null) {
            trieArtifact = new TrieArtifact();
            subFolders = new HashMap<>(subFolders);
            subFolders.put(name, trieArtifact);
          }
          break;
        default:
          trieArtifact = subFolders.computeIfAbsent(name, ignored -> new TrieArtifact());
      }
      return trieArtifact;
    }

    @Nullable
    TreeArtifactValue findTreeArtifactNodeAtPrefix(PathFragment execPath) {
      TrieArtifact current = this;
      for (String segment : execPath.segments()) {
        current = current.subFolders.get(segment);
        if (current == null) {
          break;
        }
        if (current.treeArtifactValue != null) {
          return current.treeArtifactValue;
        }
      }
      return null;
    }
  }

  private final BugReporter bugReporter;

  /** The number of elements contained in this map. */
  private int size;

  /**
   * The hash buckets. Values are indexes into the four arrays. The number of buckets is always the
   * smallest power of 2 that is larger than the number of elements.
   */
  private int[] table;

  /** Flat array of the next pointers that make up the linked list behind each hash bucket. */
  private int[] next;

  /**
   * The {@link ActionInput} keys stored in this map. For performance reasons, they need to be
   * stored as {@link Object}s as otherwise, the JVM does not seem to be as good optimizing the
   * store operations (maybe it does checks on the type being stored?).
   */
  private Object[] keys;

  /**
   * Extra storage for the execPathStrings of the values in {@link #keys}. This extra storage is
   * necessary for speed as otherwise, we'd need to cast to {@link ActionInput}, which is slow.
   */
  private Object[] paths;

  /**
   * The {@link FileArtifactValue} data stored in this map. Same as the other arrays, this is stored
   * as {@link Object} for performance reasons.
   */
  private Object[] values;

  private TrieArtifact treeArtifactsRoot = new TrieArtifact();

  @Deprecated
  @VisibleForTesting
  public ActionInputMap(int sizeHint) {
    this(BugReporter.defaultInstance(), sizeHint);
  }

  public ActionInputMap(BugReporter bugReporter, int sizeHint) {
    this.bugReporter = bugReporter;
    sizeHint = Math.max(1, sizeHint);
    int tableSize = Integer.highestOneBit(sizeHint) << 1;
    size = 0;

    table = new int[tableSize];
    Arrays.fill(table, -1);

    next = new int[sizeHint];
    keys = new Object[sizeHint];
    paths = new Object[sizeHint];
    values = new Object[sizeHint];
  }

  private int getIndex(String execPathString) {
    int hashCode = execPathString.hashCode();
    int index = hashCode & (table.length - 1);
    if (table[index] == -1) {
      return -1;
    }
    index = table[index];
    while (index != -1) {
      if (hashCode == paths[index].hashCode() && execPathString.equals(paths[index])) {
        return index;
      }
      index = next[index];
    }
    return -1;
  }

  @Nullable
  @Override
  public FileArtifactValue getInputMetadata(ActionInput input) {
    if (input instanceof TreeFileArtifact) {
      TreeFileArtifact treeFileArtifact = (TreeFileArtifact) input;
      int treeIndex = getIndex(treeFileArtifact.getParent().getExecPathString());
      if (treeIndex != -1) {
        checkArgument(
            values[treeIndex] instanceof TrieArtifact,
            "Requested tree file artifact under non-tree/omitted tree artifact: %s",
            input);
        return ((TrieArtifact) values[treeIndex])
            .treeArtifactValue
            .getChildValues()
            .get(treeFileArtifact);
      }
    }
    int index = getIndex(input.getExecPathString());
    if (index != -1) {
      return values[index] instanceof TrieArtifact
          ? ((TrieArtifact) values[index]).treeArtifactValue.getMetadata()
          : (FileArtifactValue) values[index];
    }
    if (input instanceof Artifact) {
      // Non tree-artifacts can overlap with tree files, but need to be registered separately
      // therefore we can skip searching the parents.
      return null;
    }

    // Check the trees in case input is a plain action input pointing to a tree artifact file.
    FileArtifactValue result = getMetadataFromTreeArtifacts(input.getExecPath());

    if (result != null) {
      bugReporter.sendBugReport(
          new IllegalArgumentException(
              String.format(
                  "Tree artifact file: '%s' referred to as an action input", input.getExecPath())));
    }
    return result;
  }

  /**
   * Returns metadata for given path.
   *
   * <p>This method is less efficient than {@link #getMetadata(ActionInput)}, please use it instead
   * of this one when looking up {@linkplain ActionInput action inputs}.
   */
  @Nullable
  public FileArtifactValue getMetadata(PathFragment execPath) {
    int index = getIndex(execPath.getPathString());
    if (index != -1) {
      Object value = values[index];
      return (value instanceof TrieArtifact)
          ? ((TrieArtifact) value).treeArtifactValue.getMetadata()
          : (FileArtifactValue) value;
    }

    // Fall back to searching the tree artifacts.
    return getMetadataFromTreeArtifacts(execPath);
  }

  @Nullable
  private FileArtifactValue getMetadataFromTreeArtifacts(PathFragment execPath) {
    TreeArtifactValue tree = treeArtifactsRoot.findTreeArtifactNodeAtPrefix(execPath);
    if (tree == null) {
      return null;
    }

    Map.Entry<?, FileArtifactValue> entry = tree.findChildEntryByExecPath(execPath);
    return entry != null ? entry.getValue() : null;
  }

  /**
   * Returns the {@link TreeArtifactValue} for the given path, or {@code null} if no such tree
   * artifact exists.
   */
  @Nullable
  public TreeArtifactValue getTreeMetadata(PathFragment execPath) {
    int index = getIndex(execPath.getPathString());
    if (index < 0) {
      return null;
    }
    Object value = values[index];
    if (!(value instanceof TrieArtifact)) {
      return null;
    }
    return ((TrieArtifact) value).treeArtifactValue;
  }

  /**
   * Returns the {@link TreeArtifactValue} for the shortest prefix of the given path, possibly the
   * path itself, that corresponds to a tree artifact; or {@code null} if no such tree artifact
   * exists.
   */
  @Nullable
  public TreeArtifactValue getTreeMetadataForPrefix(PathFragment execPath) {
    return treeArtifactsRoot.findTreeArtifactNodeAtPrefix(execPath);
  }

  @Nullable
  @Override
  public ActionInput getInput(String execPathString) {
    int index = getIndex(execPathString);
    if (index != -1) {
      return (ActionInput) keys[index];
    }

    // Search ancestor paths since execPathString may point to a TreeFileArtifact within one of the
    // tree artifacts.
    PathFragment execPath = PathFragment.create(execPathString);
    TreeArtifactValue tree = treeArtifactsRoot.findTreeArtifactNodeAtPrefix(execPath);
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
   * it has. Same goes for nested trees/files.
   */
  public int sizeForDebugging() {
    return size;
  }

  @Override
  public boolean put(ActionInput input, FileArtifactValue metadata, @Nullable Artifact depOwner) {
    return putWithNoDepOwner(input, metadata);
  }

  @Override
  public void putTreeArtifact(
      SpecialArtifact tree, TreeArtifactValue treeArtifactValue, @Nullable Artifact depOwner) {
    // Use a placeholder value so that we don't have to create a new trie entry if the entry is
    // already in the map.
    int oldIndex = putIfAbsent(tree, PLACEHOLDER);
    if (oldIndex != -1) {
      checkArgument(
          isATreeArtifact((ActionInput) keys[oldIndex]),
          "Tried to overwrite file with a tree artifact: '%s' with the same exec path",
          tree);
      return;
    }

    // Overwrite the placeholder.
    values[size - 1] =
        treeArtifactValue.equals(TreeArtifactValue.OMITTED_TREE_MARKER)
            // No trie entry for omitted tree -- it cannot have any children anyway.
            ? FileArtifactValue.OMITTED_FILE_MARKER
            : treeArtifactsRoot.add(tree.getExecPath(), treeArtifactValue);
  }

  public boolean putWithNoDepOwner(ActionInput input, FileArtifactValue metadata) {
    checkArgument(
        !isATreeArtifact(input),
        "Can't add tree artifact: %s using put -- please use putTreeArtifact for that",
        input);
    int oldIndex = putIfAbsent(input, metadata);
    checkArgument(
        oldIndex == -1 || !isATreeArtifact((ActionInput) keys[oldIndex]),
        "Tried to overwrite tree artifact with a file: '%s' with the same exec path",
        input);
    return oldIndex == -1;
  }

  private int putIfAbsent(ActionInput input, Object metadata) {
    Preconditions.checkNotNull(input);
    if (size >= keys.length) {
      resize();
    }
    String path = input.getExecPathString();
    int hashCode = path.hashCode();
    int index = hashCode & (table.length - 1);
    int nextIndex = table[index];
    if (nextIndex == -1) {
      table[index] = size;
    } else {
      do {
        index = nextIndex;
        if (hashCode == paths[index].hashCode() && Objects.equal(path, paths[index])) {
          return index;
        }
        nextIndex = next[index];
      } while (nextIndex != -1);
      next[index] = size;
    }
    next[size] = -1;
    keys[size] = input;
    paths[size] = input.getExecPathString();
    values[size] = metadata;
    size++;
    return -1;
  }

  @VisibleForTesting
  void clear() {
    Arrays.fill(table, -1);
    Arrays.fill(next, -1);
    Arrays.fill(keys, null);
    Arrays.fill(paths, null);
    Arrays.fill(values, null);
    size = 0;
    treeArtifactsRoot = new TrieArtifact();
  }

  private void resize() {
    // Resize the data containers.
    keys = Arrays.copyOf(keys, size * 2);
    paths = Arrays.copyOf(paths, size * 2);
    values = Arrays.copyOf(values, size * 2);
    next = Arrays.copyOf(next, size * 2);

    // Resize and recreate the table and links if necessary. We can take shortcuts here as we know
    // there are no duplicate keys.
    if (table.length < size * 2) {
      table = new int[table.length * 2];
      next = new int[size * 2];
      Arrays.fill(table, -1);
      for (int i = 0; i < size; i++) {
        int index = paths[i].hashCode() & (table.length - 1);
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
        .add("first-fifty-paths", Arrays.stream(paths).limit(50).collect(toList()))
        .toString();
  }

  private static boolean isATreeArtifact(ActionInput input) {
    return input instanceof SpecialArtifact && ((SpecialArtifact) input).isTreeArtifact();
  }
}
