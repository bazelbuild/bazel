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

import proguard.classfile.ClassPool;
import proguard.classfile.instruction.visitor.InstructionVisitor;
import proguard.classfile.util.WarningPrinter;
import proguard.classfile.visitor.ClassVisitor;

/**
 * This ClassVisitor will replace any occurrence of java.time.** related methods / types
 * that have been introduced in Java 8 to the threetenbp library.
 *
 * @author Thomas Neidhart
 */
public class JSR310Converter
extends      AbstractAPIConverter
{
    /**
     * Create a new JSR310Converter instance.
     */
    public JSR310Converter(ClassPool          programClassPool,
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
            // java.time package has been added in Java 8
            replace("java/time/**", "org/threeten/bp/<1>"),
        };

        MethodReplacement[] methodReplacements = new MethodReplacement[]
        {
            // all classes in java.time.** are converted to
            // org.threeeten.bp.**.
            replace("java/time/**",        "**",  "**",
                    "org/threeten/bp/<1>", "<1>", "<1>"),
        };

        setTypeReplacements(typeReplacements);
        setMethodReplacements(methodReplacements);
    }
}
