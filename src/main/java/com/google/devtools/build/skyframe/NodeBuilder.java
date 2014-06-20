// Copyright 2014 Google Inc. All rights reserved.
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
package com.google.devtools.build.skyframe;

import com.google.common.annotations.VisibleForTesting;
import com.google.devtools.build.lib.events.ErrorEventListener;

import java.util.Map;
import java.util.concurrent.CountDownLatch;

import javax.annotation.Nullable;

/**
 * Machinery to evaluate a single node.
 *
 * <p>The builder is supposed to access only direct dependencies of the node. However, the direct
 * dependencies need not be known in advance. The builder can request arbitrary nodes using
 * {@link Environment#getDep}. If the nodes are not ready, the call will return null; in that case
 * the builder can either try to proceed (and potentially indicate more dependencies by additional
 * {@code getDep} calls), or just return null, in which case the missing dependencies will be
 * computed and the builder will be started again.
 */
public interface NodeBuilder {

  /**
   * The services provided to the node builder by the graph implementation.
   */
  interface Environment {
    /**
     * Returns a direct dependency. If the specified node is not in the set of already evaluated
     * direct dependencies, returns null. Also returns null if the specified node has already been
     * evaluated and found to be in error.
     *
     * <p>On a subsequent build, if any of this node's dependencies have changed they will be
     * re-evaluated in the same order as originally requested by the {@code NodeBuilder} using
     * this {@code getDep} call (see {@link #getDeps} for when preserving the order is not
     * important).
     */
    @Nullable
    Node getDep(NodeKey nodeName);

    /**
     * Returns a direct dependency. If the specified node is not in the set of already evaluated
     * direct dependencies, returns null. If the specified node has already been evaluated and found
     * to be in error, throws the {@link Exception} coming from the error. Node builders may use
     * this method to continue evaluation even if one of their children is in error by catching the
     * thrown exception and proceeding. The caller must specify the exception that might be thrown
     * using the {@code exceptionClass} argument. If the child's exception is not an instance of
     * {@code exceptionClass}, returns null without throwing.
     */
    @Nullable
    <E extends Throwable> Node getDepOrThrow(NodeKey depKey, Class<E> exceptionClass) throws E;

    /**
     * Returns true iff any of the past {@link #getDep}(s) or {@link #getDepOrThrow} method calls
     * for this instance returned null (because the node was not yet present and done in the graph).
     *
     * <p>If this returns true, the {@link NodeBuilder} should return null.
     */
    boolean depsMissing();

    /**
     * Requests {@code depKeys} "in parallel", independent of each others' values. These keys may be
     * thought of as a "dependency group" -- they are requested together by this node.
     *
     * <p>In general, if the result of one getDep call can affect the argument of a later getDep
     * call, the two calls cannot be merged into a single getDeps call, since the result of the
     * first call might change on a later build. Inversely, if the result of one getDep call cannot
     * affect the parameters of the next getDep call, the two keys can form a dependency group and
     * the two getDep calls merged into one getDeps call.
     *
     * <p>This means that on subsequent builds, when checking to see if a node requires rebuilding,
     * all the nodes in this group may be simultaneously checked. A NodeBuilder should request a
     * dependency group if checking the deps serially on a subsequent build would take too long, and
     * if the builder would request all deps anyway as long as no earlier deps had changed.
     * NodeBuilder.Environment implementations may also choose to request these deps in
     * parallel on the first build, potentially speeding up the build.
     *
     * <p>While re-evaluating every node in the group may take longer than re-evaluating just the
     * first one and finding that it has changed, no extra work is done: the contract of the
     * dependency group means that the builder, when called to rebuild this node, will request all
     * nodes in the group again anyway, so they would have to have been built in any case.
     *
     * <p>Example of when to use getDeps: A ListProcessor node is built with key inputListRef. The
     * builder first calls getDep(InputList.key(inputListRef)), and retrieves inputList. It then
     * iterates through inputList, calling getDep on each input. Finally, it processes the whole
     * list and returns. Say inputList is (a, b, c). Since the builder will unconditionally call
     * getDep(a), getDep(b), and getDep(c), the builder can instead just call getDeps({a, b, c}). If
     * the node is later dirtied the evaluator will build a, b, and c in parallel (assuming the
     * inputList node was unchanged), and re-evaluate the ListProcessor node only if at least one of
     * them was changed. On the other hand, if the InputList changes to be (a, b, d), then the
     * evaluator will see that the first dep has changed, and call the builder to rebuild from
     * scratch, without considering the dep group of {a, b, c}.
     *
     * <p>Example of when not to use getDeps: A BestMatch node is built with key
     * &lt;potentialMatchesRef, matchCriterion&gt;. The builder first calls
     * getDep(PotentialMatches.key(potentialMatchesRef) and retrieves potentialMatches. It then
     * iterates through potentialMatches, calling getDep on each potential match until it finds one
     * that satisfies matchCriterion. In this case, if potentialMatches is (a, b, c), it would be
     * <i>incorrect</i> to call getDeps({a, b, c}), because it is not known yet whether requesting b
     * or c will be necessary -- if a matches, then we will never call b or c.
     */
    Map<NodeKey, Node> getDeps(Iterable<NodeKey> depKeys);

    /**
     * The same as {@link #getDeps} but the returned objects may throw when attempting to retrieve
     * their value. Note that even if the requested nodes can throw different kinds of exceptions,
     * only exceptions of type {@code E} will be preserved in the returned objects. All others will
     * be null.
     */
    <E extends Throwable> Map<NodeKey, NodeOrException<E>> getDepsOrThrow(
        Iterable<NodeKey> depKeys, Class<E> exceptionClass);

    /**
     * Returns the {@link ErrorEventListener} that a NodeBuilder should use to print any errors,
     * warnings, or progress messages while building.
     */
    ErrorEventListener getListener();

    /**
     * Gets the latch that is counted down when an exception is thrown in {@code
     * AbstractQueueVisitor}. For use in tests to check if an exception actually was thrown. Calling
     * {@code AbstractQueueVisitor#awaitExceptionForTestingOnly} can throw a spurious {@link
     * InterruptedException} because {@link CountDownLatch#await} checks the interrupted bit before
     * returning, even if the latch is already at 0. See bug "testTwoErrors is flaky".
     */
    @VisibleForTesting
    CountDownLatch getExceptionLatchForTesting();
  }

  /**
   * When a node is requested, this is called with the name of the node and a node building
   * environment.
   *
   * <p>This method should return a constructed node, or null if any dependencies were missing
   * ({@link Environment#depsMissing} was true before returning). In that case the missing
   * dependencies will be computed and the node builder restarted.
   *
   * <p>Implementations must be threadsafe and reentrant.
   *
   * @throws NodeBuilderException on failure
   * @throws InterruptedException when the user interrupts the build
   */
  @Nullable Node build(NodeKey nodeKey, Environment env) throws NodeBuilderException,
      InterruptedException;

  /**
   * Extracts a tag (target label) from a NodeKey if it has one. Otherwise return null.
   *
   * <p>The tag is used for filtering out non-error event messages that do not match --output_filter
   * flag. If a NodeBuilder returns null in this method it means that all the info/warning messages
   * associated with this node will be shown, no matter what --output_filter says.
   */
  @Nullable
  String extractTag(NodeKey nodeKey);
}
