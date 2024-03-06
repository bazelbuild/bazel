// Copyright 2024 The Bazel Authors. All rights reserved.
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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Maps;
import com.google.common.flogger.GoogleLogger;
import com.google.devtools.build.lib.collect.ConcurrentIdentitySet;
import java.lang.reflect.Array;
import java.lang.reflect.Field;
import java.lang.reflect.InaccessibleObjectException;
import java.lang.reflect.Modifier;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Queue;
import javax.annotation.Nullable;

/**
 * Traverses the Java object graph.
 *
 * <p>When given a Java object, it walks the objects reachable from it. Returns each object only
 * once, regardless of the number of edges it is reachable through.
 *
 * <p>For each object and reference edge found, the appropriate method of {@link ObjectReceiver} is
 * called.
 *
 * <p>The traversal is customizable by passing in {@link DomainSpecificTraverser} instances. Each
 * object can choose to handle any given object instance; in that case, it should return the
 * user-friendly "context" the object is encountered in and the outgoing edges in the object graph
 * it has.
 *
 * <p>If an object is not handled by any domain-specific traverser, Java reflection is used to
 * discover its outgoing references. In this case, domain-specific traversers are still consulted to
 * learn whether any of the fields should be ignored.
 *
 * <p>The traversal stops at objects that:
 *
 * <ul>
 *   <li>Are in the {@code seenObjects} set passed into the constructor
 *   <li>For which at least one {@link DomainSpecificTraverser} returns false in {@link
 *       DomainSpecificTraverser#admit(Object)}
 * </ul>
 *
 * <p>A traversal currently operates on a single thread. It's not an inherent limitation, it's just
 * that it was found to be much faster than spawning a new Executor task for each Java object.
 */
public class ObjectGraphTraverser {
  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  /**
   * Cache for traversal data by object type.
   *
   * <p>Not a static field because it depends on what domain-specific traversers there are.
   */
  public static class FieldCache {
    private final Map<Class<?>, List<Field>> fieldCache;
    private final ImmutableList<DomainSpecificTraverser> domainSpecificTraversers;

    public FieldCache(ImmutableList<DomainSpecificTraverser> domainSpecificTraversers) {
      this.fieldCache = Maps.newConcurrentMap();
      this.domainSpecificTraversers = domainSpecificTraversers;
    }
  }

  /** Domain-specific knowledge about classes to traverse. */
  public interface DomainSpecificTraverser {

    /**
     * Called for each object to be traversed.
     *
     * <p>In order for the traversal of an object to be attempted, the {@link #admit(Object)} admit
     * method of all domain-specific traversals must return true.
     *
     * <p>If the implementation knows how to traverse this object, it should return true and call
     * methods on {@link Traversal} accordingly.
     *
     * <p>If not domain-specific traversal handles an object, its fields will be visited by Java
     * reflection.
     *
     * @return true if the object is handled.
     */
    boolean maybeTraverse(Object o, Traversal traversal);

    /**
     * Called on each object to be traversed.
     *
     * <p>If the implementation thinks this instance should <b>not</b> be traversed, it should
     * return false. An implementation may well allow traversing an object and yet decline to handle
     * it in {@link #maybeTraverse(Object, Traversal)}; in that case, the default traversal will be
     * applied to the object.
     *
     * @return false if the implementation wants to prohibit the traversal of this object.
     */
    boolean admit(Object o);

    /**
     * Compute the user-friendly context for an array item.
     *
     * <p>This is used to describe what an object is in a way that's more meaningful to the user
     * than its raw class. Only called of {@code collectContext} is true. If multiple
     * domain-specific traversals provide a context, the first one takes priority.
     *
     * <p>This method is not called for references reported by domain-specific traversers.
     *
     * @param from the array the reference originates from
     * @param fromContext the context of the array the reference originates from
     * @param to the referenced object
     * @return the context of {@code to}, or null, if its class is enough
     */
    @Nullable
    String contextForArrayItem(Object from, String fromContext, Object to);

    /**
     * Compute the user-friendly context for a field.
     *
     * <p>This is used to describe what an object is in a way that's more meaningful to the user
     * than its raw class. Only called of {@code collectContext} is true. If multiple
     * domain-specific traversals provide a context, the first one takes priority.
     *
     * <p>This method is not called for references reported by domain-specific traversers.
     *
     * @param from the object the reference originates from
     * @param fromContext the context of the object the reference originates from
     * @param field the field the reference is through
     * @param to the referenced object
     * @return the context of {@code to}, or null, if its class is enough
     */
    @Nullable
    String contextForField(Object from, String fromContext, Field field, Object to);

    /**
     * Return the set of fields of a class that should be ignored.
     *
     * @return the set of ignored fields or null if the implementation doesn't know about the given
     *     class.
     */
    @Nullable
    ImmutableSet<String> ignoredFields(Class<?> clazz);
  }

  /**
   * Callback through which {@link DomainSpecificTraverser} returns what objects and edges it found.
   */
  public interface Traversal {
    /**
     * Should be called when the domain-specific traverser finds an object. It should be called for
     * every object for which {@link DomainSpecificTraverser#maybeTraverse(Object, Traversal)}
     * returns true.
     *
     * @param o the object found
     * @param context the context the object is in or null of its class is enough
     */
    void objectFound(Object o, String context);

    /**
     * Should be called for each outgoing reference in an object handled by the {@link
     * DomainSpecificTraverser} implementation.
     *
     * <p>Objects reported through this method are subject to two kinds of filtering: each object is
     * only processed once and domain-specific traversers can prohibit the traversal of any object
     * by returning false from {@link DomainSpecificTraverser#admit(Object)}.
     *
     * @param to the object referenced
     * @param context the context of the referenced object or null if its class is enough
     */
    void edgeFound(Object to, String context);
  }

  /** The type of an object graph edge. */
  public enum EdgeType {
    /** An edge to an object discovered during this traversal. */
    CURRENT_TRAVERSAL,

    /** An edge to an object already seen in previous traversals. */
    ALREADY_SEEN
  };

  /** A callback where {@link ObjectGraphTraverser} reports the objects and edges encountered. */
  public interface ObjectReceiver {
    /** Reports an object in a given context. */
    void objectFound(Object o, String context);

    /** Reports an edge in the object graph. */
    void edgeFound(Object from, Object to, String toContext, EdgeType edgeType);
  }

  /** An object to be traversed in the queue. */
  private static class WorkItem {
    private final Object object;
    private final String context;

    private WorkItem(Object object, String context) {
      this.object = object;
      this.context = context;
    }
  }

  private final FieldCache fieldCache;

  private final boolean collectContext;
  private final Traversal traversal;
  private final ObjectReceiver receiver;
  private final Object instanceId;

  private Object currentObject;
  private final Queue<WorkItem> queue = new ArrayDeque<>();

  private final ConcurrentIdentitySet localObjects;
  private final ConcurrentIdentitySet seenObjects;

  /**
   * Creates a new traverser.
   *
   * @param fieldCache the cache for reflection results.
   * @param seenObjects the set of objects already seen. These are not traversed and references to
   *     them are reported as {@link EdgeType#ALREADY_SEEN} .
   * @param collectContext whether to collect context for each object. Costs some CPU.
   * @param receiver the object to report found objects and edges to.
   * @param instanceId an object identifying this traversal.
   */
  public ObjectGraphTraverser(
      FieldCache fieldCache,
      ConcurrentIdentitySet seenObjects,
      boolean collectContext,
      ObjectReceiver receiver,
      Object instanceId) {
    this.fieldCache = fieldCache;
    this.seenObjects = seenObjects;
    this.collectContext = collectContext;
    this.receiver = receiver;
    this.instanceId = instanceId;
    this.traversal =
        new Traversal() {
          @Override
          public void objectFound(Object o, String context) {
            receiver.objectFound(o, context);
          }

          @Override
          public void edgeFound(Object to, String context) {
            enqueueMaybe(to, context);
          }
        };

    this.localObjects = new ConcurrentIdentitySet(64);
  }

  /**
   * Traverses a given object.
   *
   * <p>Can be called multiple times, but no two traversals should be active at the same time in a
   * given {@link ObjectGraphTraverser} instance.
   */
  public void traverse(Object o) {
    queue.offer(new WorkItem(o, null));
    while (!queue.isEmpty()) {
      WorkItem workItem = queue.remove();
      try {
        process(workItem.object, workItem.context);
      } catch (RuntimeException e) {
        logger.atSevere().withCause(e).log("While walking object graph for key %s:", instanceId);
      }
    }
  }

  private void enqueueMaybe(Object to, String toContext) {
    if (to == null) {
      return;
    }

    if (to instanceof Integer
        || to instanceof Long
        || to instanceof Short
        || to instanceof Byte
        || to instanceof Float
        || to instanceof Double
        || to instanceof Character
        || to instanceof Boolean) {
      // Boxed primitive type
      return;
    }

    for (DomainSpecificTraverser traverser : fieldCache.domainSpecificTraversers) {
      if (!traverser.admit(to)) {
        return;
      }
    }

    if (!localObjects.add(to)) {
      // A reference to an object visited during this traversal.
      receiver.edgeFound(currentObject, to, toContext, EdgeType.CURRENT_TRAVERSAL);
      return;
    }

    if (!seenObjects.add(to)) {
      // A reference to an object already seen, but not during this traversal.
      receiver.edgeFound(currentObject, to, toContext, EdgeType.ALREADY_SEEN);
      return;
    }

    // A new object.
    receiver.edgeFound(currentObject, to, toContext, EdgeType.CURRENT_TRAVERSAL);

    queue.offer(new WorkItem(to, toContext));
  }

  @Nullable
  private String contextOrNull(String context, String defaultContext) {
    if (!collectContext) {
      return null;
    }

    if (context != null) {
      return context;
    }

    return defaultContext;
  }

  private void process(Object o, String context) {
    currentObject = o;

    if (o instanceof String) {
      traversal.objectFound(o, contextOrNull(context, "STRING"));
      return;
    }

    for (DomainSpecificTraverser traverser : fieldCache.domainSpecificTraversers) {
      if (traverser.maybeTraverse(o, traversal)) {
        return;
      }
    }

    if (o instanceof Class<?>) {
      traversal.objectFound(o, contextOrNull(context, "CLASS"));
      return;
    }

    Class<?> clazz = o.getClass();

    if (clazz.isArray()) {
      traversal.objectFound(o, contextOrNull(context, "[] " + clazz.getComponentType().getName()));

      // We only care about objects
      if (!clazz.getComponentType().isPrimitive()) {
        for (int i = 0; i < Array.getLength(o); i++) {
          Object to = Array.get(o, i);
          String toContext = null;
          if (collectContext) {
            for (DomainSpecificTraverser traverser : fieldCache.domainSpecificTraversers) {
              String candidate = traverser.contextForArrayItem(o, context, to);
              if (candidate != null) {
                toContext = candidate;
                break;
              }
            }
          }

          enqueueMaybe(to, toContext);
        }
      }
    } else {
      traversal.objectFound(o, context);

      List<Field> fields = fieldCache.fieldCache.computeIfAbsent(clazz, this::getFields);
      for (Field field : fields) {
        try {
          Object to = field.get(o);
          String toContext = null;
          if (collectContext) {
            for (DomainSpecificTraverser traverser : fieldCache.domainSpecificTraversers) {
              String candidate = traverser.contextForField(o, context, field, to);
              if (candidate != null) {
                toContext = candidate;
                break;
              }
            }
          }
          enqueueMaybe(to, toContext);
        } catch (IllegalAccessException e) {
          throw new IllegalStateException(e);
        }
      }
    }
  }

  private ImmutableList<Field> getFields(Class<?> clazz) {
    ImmutableSet<String> ignoreSet = ImmutableSet.of();
    for (DomainSpecificTraverser traverser : fieldCache.domainSpecificTraversers) {
      ImmutableSet<String> candidate = traverser.ignoredFields(clazz);
      if (candidate != null) {
        ignoreSet = candidate;
        break;
      }
    }

    ArrayList<Field> fields = new ArrayList<>();
    for (Class<?> next = clazz; next != null; next = next.getSuperclass()) {
      for (Field field : next.getDeclaredFields()) {
        if ((field.getModifiers() & Modifier.STATIC) != 0) {
          continue; // Skips static or transient fields.
        }
        if (ignoreSet.contains(field.getName())) {
          continue;
        }

        if (field.getType().isPrimitive()) {
          // We only care about the object graph
          continue;
        }

        try {
          field.setAccessible(true);
        } catch (InaccessibleObjectException e) {
          // Ignore this field then, dunno why this happens.
          continue;
        }
        fields.add(field);
      }
    }

    return ImmutableList.copyOf(fields);
  }
}
