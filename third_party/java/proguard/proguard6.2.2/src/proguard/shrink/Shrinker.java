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
package proguard.shrink;

import proguard.*;
import proguard.classfile.*;
import proguard.classfile.attribute.visitor.*;
import proguard.classfile.visitor.*;
import proguard.util.PrintWriterUtil;

import java.io.*;

/**
 * This class shrinks class pools according to a given configuration.
 *
 * @author Eric Lafortune
 */
public class Shrinker
{
    private final Configuration configuration;


    /**
     * Creates a new Shrinker.
     */
    public Shrinker(Configuration configuration)
    {
        this.configuration = configuration;
    }


    /**
     * Performs shrinking of the given program class pool.
     */
    public ClassPool execute(ClassPool programClassPool,
                             ClassPool libraryClassPool) throws IOException
    {
        // Check if we have at least some keep commands.
        if (configuration.keep == null)
        {
            throw new IOException("You have to specify '-keep' options for the shrinking step.");
        }

        // We're using the system's default character encoding for writing to
        // the standard output.
        PrintWriter out = new PrintWriter(System.out, true);

        // Clean up any old visitor info.
        programClassPool.classesAccept(new ClassCleaner());
        libraryClassPool.classesAccept(new ClassCleaner());

        // Create a visitor for marking the seeds.
        UsageMarker usageMarker = configuration.whyAreYouKeeping == null ?
            new UsageMarker() :
            new ShortestUsageMarker();

        // Automatically mark the parameterless constructors of seed classes,
        // mainly for convenience and for backward compatibility.
        ClassVisitor classUsageMarker =
            new MultiClassVisitor(
                usageMarker,
                new NamedMethodVisitor(ClassConstants.METHOD_NAME_INIT,
                                       ClassConstants.METHOD_TYPE_INIT,
                                       usageMarker)
            );

        ClassPoolVisitor classPoolvisitor =
            new KeepClassSpecificationVisitorFactory(true, false, false)
                .createClassPoolVisitor(configuration.keep,
                                        classUsageMarker,
                                        usageMarker,
                                        usageMarker,
                                        usageMarker);

        // Mark the seeds.
        programClassPool.accept(classPoolvisitor);
        libraryClassPool.accept(classPoolvisitor);
        libraryClassPool.classesAccept(usageMarker);

        // Mark the inner class and annotation information that has to be kept.
        programClassPool.classesAccept(
            new UsedClassFilter(usageMarker,
            new AllAttributeVisitor(true,
            new MultiAttributeVisitor(
                new InnerUsageMarker(usageMarker),
                new NestUsageMarker(usageMarker),
                new AnnotationUsageMarker(usageMarker),
                new LocalVariableTypeUsageMarker(usageMarker)
            ))));

        // Mark interfaces that have to be kept.
        programClassPool.classesAccept(new InterfaceUsageMarker(usageMarker));

        // Should we explain ourselves?
        if (configuration.whyAreYouKeeping != null)
        {
            out.println();

            // Create a visitor for explaining classes and class members.
            ShortestUsagePrinter shortestUsagePrinter =
                new ShortestUsagePrinter((ShortestUsageMarker)usageMarker,
                                         configuration.verbose,
                                         out);

            ClassPoolVisitor whyClassPoolvisitor =
                new ClassSpecificationVisitorFactory()
                    .createClassPoolVisitor(configuration.whyAreYouKeeping,
                                            shortestUsagePrinter,
                                            shortestUsagePrinter,
                                            shortestUsagePrinter,
                                            null);

            // Mark the seeds.
            programClassPool.accept(whyClassPoolvisitor);
            libraryClassPool.accept(whyClassPoolvisitor);
        }

        if (configuration.printUsage != null)
        {
            PrintWriter usageWriter =
                PrintWriterUtil.createPrintWriterOut(configuration.printUsage);

            try
            {
                // Print out items that will be removed.
                programClassPool.classesAcceptAlphabetically(
                    new UsagePrinter(usageMarker, true, usageWriter));
            }
            finally
            {
                PrintWriterUtil.closePrintWriter(configuration.printUsage,
                                                 usageWriter);
            }
        }

        // Clean up used program classes and discard unused program classes.
        int originalProgramClassPoolSize = programClassPool.size();

        ClassPool newProgramClassPool = new ClassPool();
        programClassPool.classesAccept(
            new UsedClassFilter(usageMarker,
            new MultiClassVisitor(
                new ClassShrinker(usageMarker),
                new ClassPoolFiller(newProgramClassPool)
            )));

        programClassPool.clear();

        // Clean up library classes.
        libraryClassPool.classesAccept(
            new ClassShrinker(usageMarker));

        // Check if we have at least some output classes.
        int newProgramClassPoolSize = newProgramClassPool.size();

        if (configuration.verbose)
        {
            out.println("Removing unused program classes and class elements...");
            out.println("  Original number of program classes: " + originalProgramClassPoolSize);
            out.println("  Final number of program classes:    " + newProgramClassPoolSize);
        }

        if (newProgramClassPoolSize == 0 &&
            (configuration.warn == null || !configuration.warn.isEmpty()))
        {
            if (configuration.ignoreWarnings)
            {
                System.err.println("Warning: the output jar is empty. Did you specify the proper '-keep' options?");
            }
            else
            {
                throw new IOException("The output jar is empty. Did you specify the proper '-keep' options?");
            }
        }

        return newProgramClassPool;
    }
}
