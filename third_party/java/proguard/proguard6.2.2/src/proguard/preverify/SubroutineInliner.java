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
package proguard.preverify;

import proguard.Configuration;
import proguard.classfile.*;
import proguard.classfile.attribute.visitor.AllAttributeVisitor;
import proguard.classfile.visitor.*;

/**
 * This class can inline subroutines in methods. This is generally useful (i.e.
 * required) for preverifying code.
 *
 * @author Eric Lafortune
 */
public class SubroutineInliner
{
    private final Configuration configuration;


    /**
     * Creates a new SubroutineInliner.
     */
    public SubroutineInliner(Configuration configuration)
    {
        this.configuration = configuration;
    }


    /**
     * Performs subroutine inlining of the given program class pool.
     */
    public void execute(ClassPool programClassPool)
    {
        // Clean up any old visitor info.
        programClassPool.classesAccept(new ClassCleaner());

        // Inline all subroutines.
        ClassVisitor inliner =
            new AllMethodVisitor(
            new AllAttributeVisitor(
            new CodeSubroutineInliner()));

        // In Java Standard Edition, only class files from Java 6 or higher
        // should be preverified.
        if (!configuration.microEdition &&
            !configuration.android)
        {
            inliner =
                new ClassVersionFilter(ClassConstants.CLASS_VERSION_1_6,
                                       inliner);
        }

        programClassPool.classesAccept(inliner);
    }
}
