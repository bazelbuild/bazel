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

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

import com.google.common.base.Function;
import com.google.common.base.Strings;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.testutil.TestUtils;

import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.util.List;
import java.util.SortedMap;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Test for the StringIndexer classes.
 */
public abstract class StringIndexerTest {

  private static final int ATTEMPTS = 1000;
  private SortedMap<Integer, String> mappings;
  protected StringIndexer indexer;
  private final Object lock = new Object();

  @Before
  public final void createIndexer() throws Exception  {
    indexer = newIndexer();
    mappings = Maps.newTreeMap();
  }

  protected abstract StringIndexer newIndexer();

  protected void assertSize(int expected) {
    assertEquals(expected, indexer.size());
  }

  protected void assertNoIndex(String s) {
    int size = indexer.size();
    assertEquals(-1, indexer.getIndex(s));
    assertEquals(size, indexer.size());
  }

  protected void assertIndex(int expected, String s) {
    // System.out.println("Adding " + s + ", expecting " + expected);
    int index = indexer.getOrCreateIndex(s);
    // System.out.println(csi);
    assertEquals(expected, index);
    mappings.put(expected, s);
  }

  protected void assertContent() {
    for (int i = 0; i < indexer.size(); i++) {
      assertNotNull(mappings.get(i));
      assertThat(mappings).containsEntry(i, indexer.getStringForIndex(i));
    }
  }

  private void assertConcurrentUpdates(Function<Integer, String> keyGenerator) throws Exception {
    final AtomicInteger safeIndex = new AtomicInteger(-1);
    List<String> keys = Lists.newArrayListWithCapacity(ATTEMPTS);
    ThreadPoolExecutor executor = new ThreadPoolExecutor(3, 3, 5, TimeUnit.SECONDS,
        new ArrayBlockingQueue<Runnable>(ATTEMPTS));
    synchronized(lock) {
      for (int i = 0; i < ATTEMPTS; i++) {
        final String key = keyGenerator.apply(i);
        keys.add(key);
        executor.execute(new Runnable() {
          @Override
          public void run() {
            int index = indexer.getOrCreateIndex(key);
            if (safeIndex.get() < index) { safeIndex.set(index); }
            indexer.addString(key);
          }
        });
      }
    }
    try {
      while(!executor.getQueue().isEmpty()) {
        // Validate that we can execute concurrent queries too.
        if (safeIndex.get() >= 0) {
          int index = safeIndex.get();
          // Retrieve string using random existing index and validate reverse mapping.
          String key = indexer.getStringForIndex(index);
          assertNotNull(key);
          assertEquals(index, indexer.getIndex(key));
        }
      }
    } finally {
      executor.shutdown();
      executor.awaitTermination(TestUtils.WAIT_TIMEOUT_SECONDS, TimeUnit.SECONDS);
    }
    for (String key : keys) {
      // Validate mapping between keys and indices.
      assertEquals(key, indexer.getStringForIndex(indexer.getIndex(key)));
    }
  }

  @Test
  public void concurrentAddChildNode() throws Exception {
    assertConcurrentUpdates(new Function<Integer, String>() {
      @Override
      public String apply(Integer from) { return Strings.repeat("a", from + 1); }
    });
  }

  @Test
  public void concurrentSplitNodeSuffix() throws Exception {
    assertConcurrentUpdates(new Function<Integer, String>() {
      @Override
      public String apply(Integer from) { return Strings.repeat("b", ATTEMPTS - from); }
    });
  }

  @Test
  public void concurrentAddBranch() throws Exception {
    assertConcurrentUpdates(new Function<Integer, String>() {
      @Override
      public String apply(Integer from) { return String.format("%08o", from); }
    });
  }

  @RunWith(JUnit4.class)
  public static class CompactStringIndexerTest extends StringIndexerTest {
    @Override
    protected StringIndexer newIndexer() {
      return new CompactStringIndexer(1);
    }

    @Test
    public void basicOperations() {
      assertSize(0);
      assertNoIndex("abcdef");
      assertIndex(0, "abcdef"); // root node creation
      assertIndex(0, "abcdef"); // root node match
      assertSize(1);
      assertIndex(2, "abddef"); // node branching, index 1 went to "ab" node.
      assertSize(3);
      assertIndex(1, "ab");
      assertSize(3);
      assertIndex(3, "abcdefghik"); // new leaf creation
      assertSize(4);
      assertIndex(4, "abcdefgh");  // node split
      assertSize(5);
      assertNoIndex("a");
      assertNoIndex("abc");
      assertNoIndex("abcdefg");
      assertNoIndex("abcdefghil");
      assertNoIndex("abcdefghikl");
      assertContent();
      indexer.clear();
      assertSize(0);
      assertNull(indexer.getStringForIndex(0));
      assertNull(indexer.getStringForIndex(1000));
    }

    @Test
    public void parentIndexUpdate() {
      assertSize(0);
      assertIndex(0, "abcdefghik");  // Create 3 nodes with single common parent "abcdefgh".
      assertIndex(2, "abcdefghlm");  // Index 1 went to "abcdefgh".
      assertIndex(3, "abcdefghxyz");
      assertSize(4);
      assertIndex(5, "abcdpqr"); // Split parent. Index 4 went to "abcd".
      assertSize(6);
      assertIndex(1, "abcdefgh"); // Check branch node indices.
      assertIndex(4, "abcd");
      assertSize(6);
      assertContent();
    }

    @Test
    public void emptyRootNode() {
      assertSize(0);
      assertIndex(0, "abc");
      assertNoIndex("");
      assertIndex(2, "def");  // root node key is now empty string and has index 1.
      assertSize(3);
      assertIndex(1, "");
      assertSize(3);
      assertContent();
    }

    protected void setupTestContent() {
      assertSize(0);
      assertIndex(0, "abcdefghi");  // Create leafs
      assertIndex(2, "abcdefjkl");
      assertIndex(3, "abcdefmno");
      assertIndex(4, "abcdefjklpr");
      assertIndex(6, "abcdstr");
      assertIndex(8, "012345");
      assertSize(9);
      assertIndex(1, "abcdef");  // Validate inner nodes
      assertIndex(5, "abcd");
      assertIndex(7, "");
      assertSize(9);
      assertContent();
    }

    @Test
    public void dumpContent() {
      indexer = newIndexer();
      indexer.addString("abc");
      String content = indexer.toString();
      assertThat(content).contains("size = 1");
      assertThat(content).contains("contentSize = 5");
      indexer = newIndexer();
      setupTestContent();
      content = indexer.toString();
      assertThat(content).contains("size = 9");
      assertThat(content).contains("contentSize = 60");
      System.out.println(indexer);
    }

    @Test
    public void addStringResult() {
      assertSize(0);
      assertTrue(indexer.addString("abcdef"));
      assertTrue(indexer.addString("abcdgh"));
      assertFalse(indexer.addString("abcd"));
      assertTrue(indexer.addString("ab"));
    }
  }

  @RunWith(JUnit4.class)
  public static class CanonicalStringIndexerTest extends StringIndexerTest{
    @Override
    protected StringIndexer newIndexer() {
      return new CanonicalStringIndexer(new ConcurrentHashMap<String, Integer>(),
          new ConcurrentHashMap<Integer, String>());
    }

    @Test
    public void basicOperations() {
      assertSize(0);
      assertNoIndex("abcdef");
      assertIndex(0, "abcdef");
      assertIndex(0, "abcdef");
      assertSize(1);
      assertIndex(1, "abddef");
      assertSize(2);
      assertIndex(2, "ab");
      assertSize(3);
      assertIndex(3, "abcdefghik");
      assertSize(4);
      assertIndex(4, "abcdefgh");
      assertSize(5);
      assertNoIndex("a");
      assertNoIndex("abc");
      assertNoIndex("abcdefg");
      assertNoIndex("abcdefghil");
      assertNoIndex("abcdefghikl");
      assertContent();
      indexer.clear();
      assertSize(0);
      assertNull(indexer.getStringForIndex(0));
      assertNull(indexer.getStringForIndex(1000));
    }

    @Test
    public void addStringResult() {
      assertSize(0);
      assertTrue(indexer.addString("abcdef"));
      assertTrue(indexer.addString("abcdgh"));
      assertTrue(indexer.addString("abcd"));
      assertTrue(indexer.addString("ab"));
      assertFalse(indexer.addString("ab"));
    }

    protected void setupTestContent() {
      assertSize(0);
      assertIndex(0, "abcdefghi");
      assertIndex(1, "abcdefjkl");
      assertIndex(2, "abcdefmno");
      assertIndex(3, "abcdefjklpr");
      assertIndex(4, "abcdstr");
      assertIndex(5, "012345");
      assertSize(6);
      assertIndex(6, "abcdef");
      assertIndex(7, "abcd");
      assertIndex(8, "");
      assertIndex(2, "abcdefmno");
      assertSize(9);
      assertContent();
    }
  }

}
