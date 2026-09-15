/*
 * ProGuard -- shrinking, optimization, obfuscation, and preverification
 *             of Java bytecode.
 *
 * Copyright (c) 2002-2019 Guardsquare NV
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the Free
 * Software Foundation; either version 2 of the License, or (at your option)
 * any later version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
 * more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 */
package proguard.backport;

import proguard.classfile.*;
import proguard.classfile.instruction.visitor.InstructionVisitor;
import proguard.classfile.util.*;
import proguard.classfile.visitor.*;

/**
 * This ClassVisitor will replace any occurrence of stream related methods / types
 * that have been introduced in Java 8 to the streamsupport library.
 *
 * @author Thomas Neidhart
 */
public class StreamSupportConverter
extends      AbstractAPIConverter
{
    /**
     * Create a new StreamSupportConverter instance.
     */
    public StreamSupportConverter(ClassPool          programClassPool,
                                  ClassPool          libraryClassPool,
                                  WarningPrinter     warningPrinter,
                                  ClassVisitor       modifiedClassVisitor,
                                  InstructionVisitor extraInstructionVisitor)
    {
        super(programClassPool,
              libraryClassPool,
              warningPrinter,
              modifiedClassVisitor,
              extraInstructionVisitor);

        TypeReplacement[] typeReplacements = new TypeReplacement[]
        {
            // j.u.stream package has been added in Java 8
            replace("java/util/stream/**",   "java8/util/stream/<1>"),

            // j.u.function package has been added in Java 8
            replace("java/util/function/**", "java8/util/function/<1>"),

            // j.u classes / interfaces that have been added in Java 8
            replace("java/util/DoubleSummaryStatistics", "java8/util/DoubleSummaryStatistics"),
            replace("java/util/IntSummaryStatistics",    "java8/util/IntSummaryStatistics"),
            replace("java/util/LongSummaryStatistics",   "java8/util/LongSummaryStatistics"),
            replace("java/util/PrimitiveIterator**",     "java8/util/PrimitiveIterator<1>"),
            replace("java/util/Optional**",              "java8/util/Optional<1>"),
            replace("java/util/Spliterator**",           "java8/util/Spliterator<1>"),
            replace("java/util/SplittableRandom",        "java8/util/SplittableRandom"),
            replace("java/util/StringJoiner",            "java8/util/StringJoiner"),

            // j.u.c classes / exceptions that have been added in Java 8 or being updated in Java 8
            replace("java/util/concurrent/CompletionException", "java8/util/concurrent/CompletionException"),
            replace("java/util/concurrent/CountedCompleter",    "java8/util/concurrent/CountedCompleter"),
            replace("java/util/concurrent/ForkJoinPool",        "java8/util/concurrent/ForkJoinPool"),
            replace("java/util/concurrent/ForkJoinTask",        "java8/util/concurrent/ForkJoinTask"),
            replace("java/util/concurrent/ForkJoinWorkerTask",  "java8/util/concurrent/ForkJoinWorkerTask"),
            replace("java/util/concurrent/Phaser",              "java8/util/concurrent/Phaser"),
            replace("java/util/concurrent/RecursiveAction",     "java8/util/concurrent/RecursiveAction"),
            replace("java/util/concurrent/RecursiveTask",       "java8/util/concurrent/RecursiveTask"),
            replace("java/util/concurrent/ThreadLocalRandom",   "java8/util/concurrent/ThreadLocalRandom"),

            // j.l classes / annotations that have been added in Java 8
            replace("java/lang/FunctionalInterface",     "java8/lang/FunctionalInterface"),
        };

        MethodReplacement[] methodReplacements = new MethodReplacement[]
        {
            // default methods in j.u.Collection
            replace("java/util/Collection",            "stream",         "()Ljava/util/stream/Stream;",
                    "java8/util/stream/StreamSupport", "stream",         "(Ljava/util/Collection;)Ljava8/util/stream/Stream;"),
            replace("java/util/Collection",            "parallelStream", "()Ljava/util/stream/Stream;",
                    "java8/util/stream/StreamSupport", "parallelStream", "(Ljava/util/Collection;)Ljava8/util/stream/Stream;"),
            replace("java/util/Collection",            "spliterator",    "()Ljava/util/Spliterator;",
                    "java8/util/Spliterators",         "spliterator",    "(Ljava/util/Collection;)Ljava8/util/Spliterator;"),
            replace("java/util/Collection",            "removeIf",       "(Ljava/util/function/Predicate;)Z",
                    "java8/lang/Iterables",            "removeIf",       "(Ljava/lang/Iterable;Ljava8/util/function/Predicate;)Z"),

            // default methods in j.l.Iterable
            replace("java/lang/Iterable",      "forEach",     "(Ljava/util/function/Consumer;)V",
                    "java8/lang/Iterables",    "forEach",     "(Ljava/lang/Iterable;Ljava8/util/function/Consumer;)V"),
            replace("java/lang/Iterable",      "spliterator", "()Ljava/util/stream/Stream;",
                    "java8/lang/Iterables",    "spliterator", "(Ljava/lang/Iterable;)Ljava8/util/stream/Stream;"),

            // remaining default methods in j.u.List
            replace("java/util/List",   "<default>", "**",
                    "java8/util/Lists", "<1>",       "<1>"),

            // default methods in j.u.Map
            replace("java/util/Map",   "<default>", "**",
                    "java8/util/Maps", "<1>",       "<1>"),

            // static methods in j.u.Map$Entry
            replace("java/util/Map$Entry", "<static>", "**",
                    "java8/util/Maps",     "<1>",      "<1>"),

            // default and static methods in j.u.Comparator
            replace("java/util/Comparator",   "<static>",  "**",
                    "java8/util/Comparators", "<1>",       "<1>"),
            replace("java/util/Comparator",   "<default>", "**",
                    "java8/util/Comparators", "<1>",       "<1>"),

            // all methods of new classes in j.u.
            replace("java/util/DoubleSummaryStatistics",   "**",  "**",
                    "java8/util/DoubleSummaryStatistics",  "<1>", "<1>"),
            replace("java/util/IntSummaryStatistics",      "**",  "**",
                    "java8/util/IntSummaryStatistics",     "<1>", "<1>"),
            replace("java/util/LongSummaryStatistics",     "**",  "**",
                    "java8/util/LongSummaryStatistics",    "<1>", "<1>"),
            replace("java/util/PrimitiveIterator**",       "**",  "**",
                    "java8/util/PrimitiveIterator<1>",     "<1>", "<1>"),
            replace("java/util/Optional**",                "**",  "**",
                    "java8/util/Optional<1>",              "<1>", "<1>"),
            replace("java/util/Spliterator**",             "**",  "**",
                    "java8/util/Spliterator<1>",           "<1>", "<1>"),
            replace("java/util/SplittableRandom",          "**",  "**",
                    "java8/util/SplittableRandom",         "<1>", "<1>"),
            replace("java/util/StringJoiner",              "**",  "**",
                    "java8/util/StringJoiner",             "<1>", "<1>"),

            // default and static methods in new interfaces.
            replace("java/util/function/BiConsumer",        "<default>", "**",
                    "java8/util/function/BiConsumers",      "<1>",       "<1>"),
            replace("java/util/function/BiFunction",        "<default>", "**",
                    "java8/util/function/BiFunctions",      "<1>",       "<1>"),
            replace("java/util/function/BinaryOperator",    "<static>", "**",
                    "java8/util/function/BinaryOperators",  "<1>",       "<1>"),
            replace("java/util/function/BiPredicate",       "<default>", "**",
                    "java8/util/function/BiPredicates",     "<1>",       "<1>"),
            replace("java/util/function/Consumer",          "<default>", "**",
                    "java8/util/function/Consumers",        "<1>",       "<1>"),
            replace("java/util/function/DoubleConsumer",    "<default>", "**",
                    "java8/util/function/DoubleConsumers",  "<1>",       "<1>"),
            replace("java/util/function/DoublePredicate",   "<default>", "**",
                    "java8/util/function/DoublePredicates", "<1>",       "<1>"),

            replace("java/util/function/DoubleUnaryOperator",   "<static>",  "**",
                    "java8/util/function/DoubleUnaryOperators", "<1>",       "<1>"),
            replace("java/util/function/DoubleUnaryOperator",   "<default>", "**",
                    "java8/util/function/DoubleUnaryOperators", "<1>",       "<1>"),

            replace("java/util/function/Function",   "<static>",  "**",
                    "java8/util/function/Functions", "<1>",       "<1>"),
            replace("java/util/function/Function",   "<default>", "**",
                    "java8/util/function/Functions", "<1>",       "<1>"),

            replace("java/util/function/IntConsumer",    "<default>", "**",
                    "java8/util/function/IntConsumers",  "<1>",       "<1>"),
            replace("java/util/function/IntPredicate",   "<default>", "**",
                    "java8/util/function/IntPredicates", "<1>",       "<1>"),

            replace("java/util/function/IntUnaryOperator",   "<static>",  "**",
                    "java8/util/function/IntUnaryOperators", "<1>",       "<1>"),
            replace("java/util/function/IntUnaryOperator",   "<default>", "**",
                    "java8/util/function/IntUnaryOperators", "<1>",       "<1>"),

            replace("java/util/function/LongConsumer",    "<default>", "**",
                    "java8/util/function/LongConsumers",  "<1>",       "<1>"),
            replace("java/util/function/LongPredicate",   "<default>", "**",
                    "java8/util/function/LongPredicates", "<1>",       "<1>"),

            replace("java/util/function/LongUnaryOperator",   "<static>",  "**",
                    "java8/util/function/LongUnaryOperators", "<1>",       "<1>"),
            replace("java/util/function/LongUnaryOperator",   "<default>", "**",
                    "java8/util/function/LongUnaryOperators", "<1>",       "<1>"),

            replace("java/util/function/Predicate",   "<static>",  "**",
                    "java8/util/function/Predicates", "<1>",       "<1>"),
            replace("java/util/function/Predicate",   "<default>", "**",
                    "java8/util/function/Predicates", "<1>",       "<1>"),

            replace("java/util/function/UnaryOperator",   "<static>",  "**",
                    "java8/util/function/UnaryOperators", "<1>",       "<1>"),

            // static methods in new interfaces.
            replace("java/util/stream/DoubleStream",   "<static>", "**",
                    "java8/util/stream/DoubleStreams", "<1>",      "<1>"),
            replace("java/util/stream/IntStream",      "<static>", "**",
                    "java8/util/stream/IntStreams",    "<1>",      "<1>"),
            replace("java/util/stream/LongStream",     "<static>", "**",
                    "java8/util/stream/LongStreams",   "<1>",      "<1>"),
            replace("java/util/stream/Stream",         "<static>", "**",
                    "java8/util/stream/RefStreams",    "<1>",      "<1>"),

            replace("java/util/stream/Collector",   "<static>",  "**",
                    "java8/util/stream/Collectors", "<1>",       "<1>"),

            // remaining methods in new classes.
            replace("java/util/stream/**",     "**",  "**",
                    "java8/util/stream/<1>",   "<1>", "<1>"),
            replace("java/util/function/**",   "**",  "**",
                    "java8/util/function/<1>", "<1>", "<1>"),

            // default methods in Iterator.
            replace("java/util/Iterator",   "forEachRemaining", "(Ljava/util/function/Consumer;)V",
                    "java8/util/Iterators", "forEachRemaining", "(Ljava/lang/Iterable;Ljava8/util/function/Consumer;)V"),

            // new methods in j.u.c.
            replace("java/util/concurrent/ConcurrentMap",   "<default>",    "**",
                    "java8/util/concurrent/ConcurrentMaps", "<1>",          "<1>"),

            replace("java/util/concurrent/CompletionException",  "**",    "**",
                    "java8/util/concurrent/CompletionException", "<1>",   "<1>"),
            replace("java/util/concurrent/CountedCompleter",     "**",    "**",
                    "java8/util/concurrent/CountedCompleter",    "<1>",   "<1>"),
            replace("java/util/concurrent/ForkJoinPool",         "**",    "**",
                    "java8/util/concurrent/ForkJoinPool",        "<1>",   "<1>"),
            replace("java/util/concurrent/ForkJoinTask",         "**",    "**",
                    "java8/util/concurrent/ForkJoinTask",        "<1>",   "<1>"),
            replace("java/util/concurrent/ForkJoinWorkerTask",   "**",    "**",
                    "java8/util/concurrent/ForkJoinWorkerTask",  "<1>",   "<1>"),
            replace("java/util/concurrent/Phaser",               "**",    "**",
                    "java8/util/concurrent/Phaser",              "<1>",   "<1>"),
            replace("java/util/concurrent/RecursiveAction",      "**",    "**",
                    "java8/util/concurrent/RecursiveAction",     "<1>",   "<1>"),
            replace("java/util/concurrent/RecursiveTask",        "**",    "**",
                    "java8/util/concurrent/RecursiveTask",       "<1>",   "<1>"),
            replace("java/util/concurrent/ForkJoinPool",         "**",    "**",
                    "java8/util/concurrent/ForkJoinPool",        "<1>",   "<1>"),

            // static methods
            replace("java/util/concurrent/ThreadLocalRandom",  "ints",    "**",
                    "java8/util/concurrent/ThreadLocalRandom", "ints",    "<1>"),
            replace("java/util/concurrent/ThreadLocalRandom",  "longs",   "**",
                    "java8/util/concurrent/ThreadLocalRandom", "longs",   "<1>"),
            replace("java/util/concurrent/ThreadLocalRandom",  "doubles", "**",
                    "java8/util/concurrent/ThreadLocalRandom", "doubles", "<1>"),
            // remaining
            replace("java/util/concurrent/ThreadLocalRandom",  "**",      "**",
                    "java8/util/concurrent/ThreadLocalRandom", "<1>",     "<1>"),

            // new methods in j.u.Arrays.
            replace("java/util/Arrays",        "spliterator", "**",
                    "java8/util/J8Arrays",     "spliterator", "<1>"),
            replace("java/util/Arrays",        "stream",      "**",
                    "java8/util/J8Arrays",     "stream",      "<1>"),
            replace("java/util/Arrays",        "parallel**",  "**",
                    "java8/util/J8Arrays",     "<1>",         "<1>"),
            replace("java/util/Arrays",        "set**",       "**",
                    "java8/util/J8Arrays",     "<1>",         "<1>"),

            // new methods in j.l.Integer.
            replace("java/lang/Integer",   "min",               "**",
                    "java8/lang/Integers", "min",               "<1>"),
            replace("java/lang/Integer",   "max",               "**",
                    "java8/lang/Integers", "max",               "<1>"),
            replace("java/lang/Integer",   "sum",               "**",
                    "java8/lang/Integers", "sum",               "<1>"),
            replace("java/lang/Integer",   "compare",           "**",
                    "java8/lang/Integers", "compare",           "<1>"),
            replace("java/lang/Integer",   "compareUnsigned",   "**",
                    "java8/lang/Integers", "compareUnsigned",   "<1>"),
            replace("java/lang/Integer",   "remainderUnsigned", "**",
                    "java8/lang/Integers", "remainderUnsigned", "<1>"),
            replace("java/lang/Integer",   "divideUnsigned",    "**",
                    "java8/lang/Integers", "divideUnsigned",    "<1>"),
            replace("java/lang/Integer",   "toUnsignedLong",    "**",
                    "java8/lang/Integers", "toUnsignedLong",    "<1>"),
            replace("java/lang/Integer",   "hashCode",          "(I)I",
                    "java8/lang/Integers", "hashCode",          "(I)I"),

            // new methods in j.l.Long.
            replace("java/lang/Long",   "min",                  "**",
                    "java8/lang/Longs", "min",                  "<1>"),
            replace("java/lang/Long",   "max",                  "**",
                    "java8/lang/Longs", "max",                  "<1>"),
            replace("java/lang/Long",   "sum",                  "**",
                    "java8/lang/Longs", "sum",                  "<1>"),
            replace("java/lang/Long",   "compare",              "**",
                    "java8/lang/Longs", "compare",              "<1>"),
            replace("java/lang/Long",   "compareUnsigned",      "**",
                    "java8/lang/Longs", "compareUnsigned",      "<1>"),
            replace("java/lang/Long",   "remainderUnsigned",    "**",
                    "java8/lang/Longs", "remainderUnsigned",    "<1>"),
            replace("java/lang/Long",   "divideUnsigned",       "**",
                    "java8/lang/Longs", "divideUnsigned",       "<1>"),
            replace("java/lang/Long",   "toUnsignedBigInteger", "**",
                    "java8/lang/Longs", "toUnsignedBigInteger", "<1>"),
            replace("java/lang/Long",   "hashCode",             "(J)I",
                    "java8/lang/Longs", "hashCode",             "(J)I"),

            // new methods in j.l.Double.
            replace("java/lang/Double",   "min",      "**",
                    "java8/lang/Doubles", "min",      "<1>"),
            replace("java/lang/Double",   "max",      "**",
                    "java8/lang/Doubles", "max",      "<1>"),
            replace("java/lang/Double",   "sum",      "**",
                    "java8/lang/Doubles", "sum",      "<1>"),
            replace("java/lang/Double",   "isFinite", "**",
                    "java8/lang/Doubles", "isFinite", "<1>"),
            replace("java/lang/Double",   "hashCode", "(D)I",
                    "java8/lang/Doubles", "hashCode", "(D)I"),

            // missing replacements for methods added to j.u.Random.
            missing("java/util/Random", "ints",    "**"),
            missing("java/util/Random", "longs",   "**"),
            missing("java/util/Random", "doubles", "**")
        };

        setTypeReplacements(typeReplacements);
        setMethodReplacements(methodReplacements);
    }
}
