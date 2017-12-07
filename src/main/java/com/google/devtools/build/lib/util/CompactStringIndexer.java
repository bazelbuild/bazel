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
package com.google.devtools.build.lib.util;

import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.base.Preconditions;
import com.google.common.collect.Lists;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadCompatible;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import java.util.ArrayList;

/**
 * Provides memory-efficient bidirectional mapping String <-> unique integer.
 * Uses byte-wise compressed prefix trie internally.
 * <p>
 * Class allows index retrieval for the given string, addition of the new
 * index and string retrieval for the given index. It also allows efficient
 * serialization of the internal data structures.
 * <p>
 * Internally class stores list of nodes with each node containing byte[]
 * representation of compressed trie node:
 * <pre>
 * varint32 parentIndex;  // index of the parent node
 * varint32 keylen;       // length of the node key
 * byte[keylen] key;      // node key data
 * repeated jumpEntry {   // Zero or more jump entries, referencing child nodes
 *   byte key             // jump key (first byte of the child node key)
 *   varint32 nodeIndex   // child index
 * }
 * <p>
 * Note that jumpEntry key byte is actually duplicated in the child node
 * instance. This is done to improve performance of the index->string
 * lookup (so we can avoid jump table parsing during this lookup).
 * <p>
 * Root node of the trie must have parent id pointing to itself.
 * <p>
 * TODO(bazel-team): (2010) Consider more fine-tuned locking mechanism - e.g.
 * distinguishing between read and write locks.
 */
@ThreadSafe
public class CompactStringIndexer extends AbstractIndexer {

  private static final int NOT_FOUND = -1;

  private ArrayList<byte[]> nodes;  // Compressed prefix trie nodes.
  private int rootId;               // Root node id.

  /*
   * Creates indexer instance.
   */
  public CompactStringIndexer (int expectedCapacity) {
    Preconditions.checkArgument(expectedCapacity > 0);
    nodes = Lists.newArrayListWithExpectedSize(expectedCapacity);
    rootId = NOT_FOUND;
  }

  /**
   * Allocates new node index. Must be called only from
   * synchronized methods.
   */
  private int allocateIndex() {
    nodes.add(null);
    return nodes.size() - 1;
  }

  /**
   * Replaces given node record with the new one. Must be called only from
   * synchronized methods.
   * <p>
   * Subclasses can override this method to be notified when an update actually
   * takes place.
   */
  @ThreadCompatible
  protected void updateNode(int index, byte[] content) {
    nodes.set(index, content);
  }

  /**
   * Returns parent id for the given node content.
   *
   * @return parent node id
   */
  private int getParentId(byte[] content) {
    int[] intHolder = new int[1];
    VarInt.getVarInt(content, 0, intHolder);
    return intHolder[0];
  }

  /**
   * Creates new node using specified key suffix. Must be called from
   * synchronized methods.
   *
   * @param parentNode parent node id
   * @param key original key that is being added to the indexer
   * @param offset node key offset in the original key.
   *
   * @return new node id corresponding to the given key
   */
  private int createNode(int parentNode, byte[] key, int offset) {
    int index = allocateIndex();

    int len = key.length - offset;
    Preconditions.checkState(len >= 0);

    // Content consists of parent id, key length and key. There are no jump records.
    byte[] content = new byte[VarInt.varIntSize(parentNode) + VarInt.varIntSize(len) + len];
    // Add parent id.
    int contentOffset = VarInt.putVarInt(parentNode, content, 0);
    // Add node key length.
    contentOffset = VarInt.putVarInt(len, content, contentOffset);
    // Add node key content.
    System.arraycopy(key, offset, content, contentOffset, len);

    updateNode(index, content);
    return index;
  }

  /**
   * Updates jump entry index in the given node.
   *
   * @param node node id to update
   * @param oldIndex old jump entry index
   * @param newIndex updated jump entry index
   */
  private void updateJumpEntry(int node, int oldIndex, int newIndex) {
    byte[] content = nodes.get(node);
    int[] intHolder = new int[1];
    int offset = VarInt.getVarInt(content, 0, intHolder); // parent id
    offset = VarInt.getVarInt(content, offset, intHolder); // key length
    offset += intHolder[0]; // Offset now points to the first jump entry.
    while (offset < content.length) {
      int next = VarInt.getVarInt(content, offset + 1, intHolder); // jump index
      if (intHolder[0] == oldIndex) {
        // Substitute oldIndex value with newIndex.
        byte[] newContent =
            new byte[content.length + VarInt.varIntSize(newIndex) - VarInt.varIntSize(oldIndex)];
        System.arraycopy(content, 0, newContent, 0, offset + 1);
        offset = VarInt.putVarInt(newIndex, newContent, offset + 1);
        System.arraycopy(content, next, newContent, offset, content.length - next);
        updateNode(node, newContent);
        return;
      } else {
        offset = next;
      }
    }
    StringBuilder builder = new StringBuilder().append("Index ").append(oldIndex)
        .append(" is not present in the node ").append(node).append(", ");
    dumpNodeContent(builder, content);
    throw new IllegalArgumentException(builder.toString());
  }

  /**
   * Creates new branch node content at the predefined location, splitting
   * prefix from the given node and optionally adding another child node
   * jump entry.
   *
   * @param originalNode node that will be split
   * @param newBranchNode new branch node id
   * @param splitOffset offset at which to split original node key
   * @param indexKey optional additional jump key
   * @param childIndex optional additional jump index. Optional jump entry will
   *                   be skipped if this index is set to NOT_FOUND.
   */
  private void createNewBranchNode(int originalNode, int newBranchNode, int splitOffset,
      byte indexKey, int childIndex) {
    byte[] content = nodes.get(originalNode);
    int[] intHolder = new int[1];
    int keyOffset = VarInt.getVarInt(content, 0, intHolder); // parent id

    // If original node is a root node, new branch node will become new root. So set parent id
    // appropriately (for root node it is set to the node's own id).
    int parentIndex = (originalNode == intHolder[0] ? newBranchNode : intHolder[0]);

    keyOffset = VarInt.getVarInt(content, keyOffset, intHolder); // key length
    Preconditions.checkState(intHolder[0] >= splitOffset);
    // Calculate new content size.
    int newSize = VarInt.varIntSize(parentIndex)
        + VarInt.varIntSize(splitOffset) + splitOffset
        + 1 + VarInt.varIntSize(originalNode)
        + (childIndex != NOT_FOUND ? 1 + VarInt.varIntSize(childIndex) : 0);
    // New content consists of parent id, new key length, truncated key and two jump records.
    byte[] newContent = new byte[newSize];
    // Add parent id.
    int contentOffset = VarInt.putVarInt(parentIndex, newContent, 0);
    // Add adjusted key length.
    contentOffset = VarInt.putVarInt(splitOffset, newContent, contentOffset);
    // Add truncated key content and first jump key.
    System.arraycopy(content, keyOffset, newContent, contentOffset, splitOffset + 1);
    // Add index for the first jump key.
    contentOffset = VarInt.putVarInt(originalNode, newContent, contentOffset + splitOffset + 1);
    // If requested, add additional jump entry.
    if (childIndex != NOT_FOUND) {
      // Add second jump key.
      newContent[contentOffset] = indexKey;
      // Add index for the second jump key.
      VarInt.putVarInt(childIndex, newContent, contentOffset + 1);
    }
    updateNode(newBranchNode, newContent);
  }

  /**
   * Inject newly created branch node into the trie data structure. Method
   * will update parent node jump entry to point to the new branch node (or
   * will update root id if branch node becomes new root) and will truncate
   * key prefix from the original node that was split (that prefix now
   * resides in the branch node).
   *
   * @param originalNode node that will be split
   * @param newBranchNode new branch node id
   * @param commonPrefixLength how many bytes should be split into the new branch node.
   */
  private void injectNewBranchNode(int originalNode, int newBranchNode, int commonPrefixLength) {
    byte[] content = nodes.get(originalNode);

    int parentId = getParentId(content);
    if (originalNode == parentId) {
      rootId = newBranchNode; // update root index
    } else {
      updateJumpEntry(parentId, originalNode, newBranchNode);
    }

    // Truncate prefix from the original node and set its parent to the our new branch node.
    int[] intHolder = new int[1];
    int suffixOffset = VarInt.getVarInt(content, 0, intHolder); // parent id
    suffixOffset = VarInt.getVarInt(content, suffixOffset, intHolder); // key length
    int len = intHolder[0] - commonPrefixLength;
    Preconditions.checkState(len >= 0);
    suffixOffset += commonPrefixLength;
    // New content consists of parent id, new key length and duplicated key suffix.
    byte[] newContent = new byte[VarInt.varIntSize(newBranchNode) + VarInt.varIntSize(len) +
        (content.length - suffixOffset)];
    // Add parent id.
    int contentOffset = VarInt.putVarInt(newBranchNode, newContent, 0);
    // Add new key length.
    contentOffset = VarInt.putVarInt(len, newContent, contentOffset);
    // Add key and jump table.
    System.arraycopy(content, suffixOffset, newContent, contentOffset,
        content.length - suffixOffset);
    updateNode(originalNode, newContent);
  }

  /**
   * Adds new child node (that uses specified key suffix) to the given
   * current node.
   * Example:
   * <pre>
   * Had "ab". Adding "abcd".
   *
   *           1:"ab",'c'->2
   * 1:"ab" ->     \
   *              2:"cd"
   * </pre>
   */
  private int addChildNode(int parentNode, byte[] key, int keyOffset) {
    int child = createNode(parentNode, key, keyOffset);

    byte[] content = nodes.get(parentNode);
    // Add jump table entry to the parent node.
    int entryOffset = content.length;
    // New content consists of original content and additional jump record.
    byte[] newContent = new byte[entryOffset + 1 + VarInt.varIntSize(child)];
    // Copy original content.
    System.arraycopy(content, 0, newContent, 0, entryOffset);
    // Add jump key.
    newContent[entryOffset] = key[keyOffset];
    // Add jump index.
    VarInt.putVarInt(child, newContent, entryOffset + 1);

    updateNode(parentNode, newContent);
    return child;
  }

  /**
   * Splits node into two at the specified offset.
   * Example:
   * <pre>
   * Had "abcd". Adding "ab".
   *
   *             2:"ab",'c'->1
   * 1:"abcd" ->     \
   *                1:"cd"
   * </pre>
   */
  private int splitNodeSuffix(int nodeToSplit, int commonPrefixLength) {
    int newBranchNode = allocateIndex();
    // Create new node with truncated key.
    createNewBranchNode(nodeToSplit, newBranchNode, commonPrefixLength, (byte) 0, NOT_FOUND);
    injectNewBranchNode(nodeToSplit, newBranchNode, commonPrefixLength);

    return newBranchNode;
  }

  /**
   * Splits node into two at the specified offset and adds another leaf.
   * Example:
   * <pre>
   * Had "abcd". Adding "abef".
   *
   *                3:"ab",'c'->1,'e'->2
   * 1:"abcd" ->    /     \
   *             1:"cd"   2:"ef"
   * </pre>
   */
  private int addBranch(int nodeToSplit, byte[] key, int offset, int commonPrefixLength) {
    int newBranchNode = allocateIndex();
    int child = createNode(newBranchNode, key, offset + commonPrefixLength);
    // Create new node with the truncated key and reference to the new child node.
    createNewBranchNode(nodeToSplit, newBranchNode, commonPrefixLength,
        key[offset + commonPrefixLength], child);
    injectNewBranchNode(nodeToSplit, newBranchNode, commonPrefixLength);

    return child;
  }

  private int findOrCreateIndexInternal(int node, byte[] key, int offset,
      boolean createIfNotFound) {
    byte[] content = nodes.get(node);
    int[] intHolder = new int[1];
    int contentOffset = VarInt.getVarInt(content, 0, intHolder); // parent id
    contentOffset = VarInt.getVarInt(content, contentOffset, intHolder); // key length
    int skyKeyLen = intHolder[0];
    int remainingKeyLen = key.length - offset;
    int minKeyLen = Math.min(skyKeyLen, remainingKeyLen);

    // Compare given key/offset content with the node key. Skip first key byte for recursive
    // calls - this byte is equal to the byte in the jump entry and was already compared.
    for (int i = (offset > 0 ? 1 : 0); i < minKeyLen; i++) { // compare key
      if (key[offset + i] != content[contentOffset + i]) {
        // Mismatch found somewhere in the middle of the node key. If requested, node
        // should be split and another leaf added for the new key.
        return createIfNotFound ? addBranch(node, key, offset, i) : NOT_FOUND;
      }
    }

    if (remainingKeyLen > minKeyLen) {
      // Node key matched portion of the key - find appropriate jump entry. If found - recursion.
      // If not - mismatch (we will add new child node if requested).
      contentOffset += skyKeyLen;
      while (contentOffset < content.length) {
        if (key[offset + skyKeyLen] == content[contentOffset]) {  // compare index value
          VarInt.getVarInt(content, contentOffset + 1, intHolder);
          // Found matching jump entry - recursively compare the child.
          return findOrCreateIndexInternal(intHolder[0], key, offset + skyKeyLen,
              createIfNotFound);
        } else {
          // Jump entry key does not match. Skip rest of the entry data.
          contentOffset = VarInt.getVarInt(content, contentOffset + 1, intHolder);
        }
      }
      // There are no matching jump entries - report mismatch or create a new leaf if necessary.
      return createIfNotFound ? addChildNode(node, key, offset + skyKeyLen) : NOT_FOUND;
    } else if (skyKeyLen > minKeyLen) {
      // Key suffix is a subset of the node key. Report mismatch or split the node if requested).
      return createIfNotFound ? splitNodeSuffix(node, minKeyLen) : NOT_FOUND;
    } else {
      // Node key exactly matches key suffix - return associated index value.
      return node;
    }
  }

  private synchronized int findOrCreateIndex(byte[] key, boolean createIfNotFound) {
    if (rootId == NOT_FOUND) {
      // Root node does not seem to exist - create it if needed.
      if (createIfNotFound) {
        rootId = createNode(0, key, 0);
        Preconditions.checkState(rootId == 0);
        return 0;
      } else {
        return NOT_FOUND;
      }
    }
    return findOrCreateIndexInternal(rootId, key, 0, createIfNotFound);
  }

  private byte[] reconstructKeyInternal(int node, int suffixSize) {
    byte[] content = nodes.get(node);
    Preconditions.checkNotNull(content);
    int[] intHolder = new int[1];
    int contentOffset = VarInt.getVarInt(content, 0, intHolder); // parent id
    int parentNode = intHolder[0];
    contentOffset = VarInt.getVarInt(content, contentOffset, intHolder); // key length
    int len = intHolder[0];
    byte[] key;
    if (node != parentNode) {
      // We haven't reached root node yet. Make a recursive call, adjusting suffix length.
      key = reconstructKeyInternal(parentNode, suffixSize + len);
    } else {
      // We are in a root node. Finally allocate array for the key. It will be filled up
      // on our way back from recursive call tree.
      key = new byte[suffixSize + len];
    }
    // Fill appropriate portion of the full key with the node key content.
    System.arraycopy(content, contentOffset, key, key.length - suffixSize - len, len);
    return key;
  }

  private byte[] reconstructKey(int node) {
    return reconstructKeyInternal(node, 0);
  }

  /* (non-Javadoc)
   * @see com.google.devtools.build.lib.util.StringIndexer#clear()
   */
  @Override
  public synchronized void clear() {
    nodes.clear();
  }

  /* (non-Javadoc)
   * @see com.google.devtools.build.lib.util.StringIndexer#size()
   */
  @Override
  public synchronized int size() {
    return nodes.size();
  }

  protected int getOrCreateIndexForBytes(byte[] bytes) {
    return findOrCreateIndex(bytes, true);
  }

  protected synchronized boolean addBytes(byte[] bytes) {
    int count = nodes.size();
    int index = getOrCreateIndexForBytes(bytes);
    return index >= count;
  }

  protected int getIndexForBytes(byte[] bytes) {
    return findOrCreateIndex(bytes, false);
  }

  /* (non-Javadoc)
   * @see com.google.devtools.build.lib.util.StringIndexer#getOrCreateIndex(java.lang.String)
   */
  @Override
  public int getOrCreateIndex(String s) {
    return getOrCreateIndexForBytes(string2bytes(s));
  }

  /* (non-Javadoc)
   * @see com.google.devtools.build.lib.util.StringIndexer#getIndex(java.lang.String)
   */
  @Override
  public int getIndex(String s) {
    return getIndexForBytes(string2bytes(s));
  }

  /* (non-Javadoc)
   * @see com.google.devtools.build.lib.util.StringIndexer#addString(java.lang.String)
   */
  @Override
  public boolean addString(String s) {
    return addBytes(string2bytes(s));
  }

  protected synchronized byte[] getBytesForIndex(int i) {
    Preconditions.checkArgument(i >= 0);
    if (i >= nodes.size()) {
      return null;
    }
    return reconstructKey(i);
  }

  /* (non-Javadoc)
   * @see com.google.devtools.build.lib.util.StringIndexer#getStringForIndex(int)
   */
  @Override
  public String getStringForIndex(int i) {
    byte[] bytes = getBytesForIndex(i);
    return bytes != null ? bytes2string(bytes) : null;
  }

  private void dumpNodeContent(StringBuilder builder, byte[] content) {
    int[] intHolder = new int[1];
    int offset = VarInt.getVarInt(content, 0, intHolder);
    builder.append("parent: ").append(intHolder[0]);
    offset = VarInt.getVarInt(content, offset, intHolder);
    int len = intHolder[0];
    builder.append(", len: ").append(len).append(", key: \"")
        .append(new String(content, offset, len, UTF_8)).append('"');
    offset += len;
    while (offset < content.length) {
      builder.append(", '").append(new String(content, offset, 1, UTF_8)).append("': ");
      offset = VarInt.getVarInt(content, offset + 1, intHolder);
      builder.append(intHolder[0]);
    }
    builder.append(", size: ").append(content.length);
  }

  private int dumpContent(StringBuilder builder, int node, int indent, boolean[] seen) {
    for(int i = 0; i < indent; i++) {
      builder.append("  ");
    }
    builder.append(node).append(": ");
    if (node >= nodes.size()) {
      builder.append("OUT_OF_BOUNDS\n");
      return 0;
    } else if (seen[node]) {
      builder.append("ALREADY_SEEN\n");
      return 0;
    }
    seen[node] = true;
    byte[] content = nodes.get(node);
    if (content == null) {
      builder.append("NULL\n");
      return 0;
    }
    dumpNodeContent(builder, content);
    builder.append("\n");
    int contentSize = content.length;

    int[] intHolder = new int[1];
    int contentOffset = VarInt.getVarInt(content, 0, intHolder); // parent id
    contentOffset = VarInt.getVarInt(content, contentOffset, intHolder); // key length
    contentOffset += intHolder[0];
    while (contentOffset < content.length) {
      contentOffset = VarInt.getVarInt(content, contentOffset + 1, intHolder);
      contentSize += dumpContent(builder, intHolder[0], indent + 1, seen);
    }
    return contentSize;
  }

  @Override
  public synchronized String toString() {
    StringBuilder builder = new StringBuilder();
    builder.append("size = ").append(nodes.size()).append("\n");
    if (!nodes.isEmpty()) {
      int contentSize = dumpContent(builder, rootId, 0, new boolean[nodes.size()]);
      builder.append("contentSize = ").append(contentSize).append("\n");
    }
    return builder.toString();
  }

}
