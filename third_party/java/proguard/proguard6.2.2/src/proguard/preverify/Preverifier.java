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
 * This class can preverify methods in program class pools, according to a given
 * configuration.
 *
 * @author Eric Lafortune
 */
public class Preverifier
{
    private final Configuration configuration;


    /**
     * Creates a new Preverifier.
     */
    public Preverifier(Configuration configuration)
    {
        this.configuration = configuration;
    }


    /**
     * Performs preverification of the given program class pool.
     */
    public void execute(ClassPool programClassPool)
    {
        // Clean up any old visitor info.
        programClassPool.classesAccept(new ClassCleaner());

        // Preverify all methods.
        // Classes for JME must be preverified.
        // Classes for JSE 6 may optionally be preverified.
        // Classes for JSE 7 or higher must be preverified.
        programClassPool.classesAccept(
            new ClassVersionFilter(configuration.microEdition ?
                                       ClassConstants.CLASS_VERSION_1_0 :
                                       ClassConstants.CLASS_VERSION_1_6,
            new AllMethodVisitor(
            new AllAttributeVisitor(
            new CodePreverifier(configuration.microEdition)))));
    }
}
