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
package proguard;

import proguard.classfile.ClassPool;
import proguard.classfile.visitor.*;
import proguard.optimize.*;

import java.io.*;

/**
 * This class prints out the seeds specified by keep options.
 *
 * @author Eric Lafortune
 */
public class SeedPrinter
{
    private final PrintWriter printWriter;


    /**
     * Creates a new ConfigurationWriter that prints to the given writer.
     */
    public SeedPrinter(PrintWriter printWriter) throws IOException
    {
        this.printWriter = printWriter;
    }


    /**
     * Prints out the seeds for the classes in the given program class pool.
     * @param configuration the configuration containing the keep options.
     * @throws IOException if an IO error occurs while writing the configuration.
     */
    public void write(Configuration configuration,
                      ClassPool     programClassPool,
                      ClassPool     libraryClassPool) throws IOException
    {
        // Check if we have at least some keep commands.
        if (configuration.keep == null)
        {
            throw new IOException("You have to specify '-keep' options if you want to write out kept elements with '-printseeds'.");
        }

        // Clean up any old visitor info.
        programClassPool.classesAccept(new ClassCleaner());
        libraryClassPool.classesAccept(new ClassCleaner());

        // Create a visitor for printing out the seeds. We're  printing out
        // the program elements that are preserved against shrinking,
        // optimization, or obfuscation.
        KeepMarker keepMarker = new KeepMarker();
        ClassPoolVisitor classPoolvisitor =
            new KeepClassSpecificationVisitorFactory(true, true, true)
                .createClassPoolVisitor(configuration.keep,
                                        keepMarker,
                                        keepMarker,
                                        keepMarker,
                                        null);

        // Mark the seeds.
        programClassPool.accept(classPoolvisitor);
        libraryClassPool.accept(classPoolvisitor);

        // Print out the seeds.
        SimpleClassPrinter printer = new SimpleClassPrinter(false, printWriter);
        programClassPool.classesAcceptAlphabetically(
            new MultiClassVisitor(
                new KeptClassFilter(printer),
                new AllMemberVisitor(new KeptMemberFilter(printer))
            ));
    }
}