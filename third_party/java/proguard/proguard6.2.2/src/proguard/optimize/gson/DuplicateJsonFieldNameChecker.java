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
package proguard.optimize.gson;

import proguard.classfile.*;
import proguard.classfile.visitor.*;

import java.util.*;

/**
 * This class visitor checks whether the visited class has duplicate field names
 * in its JSON representation.
 *
 * @author Lars Vandenbergh
 */
class      DuplicateJsonFieldNameChecker
implements ClassVisitor
{
    public boolean hasDuplicateJsonFieldNames;


    // Implementations for ClassVisitor.

    @Override
    public void visitProgramClass(ProgramClass programClass)
    {
        for (OptimizedJsonFieldCollector.Mode mode : OptimizedJsonFieldCollector.Mode.values())
        {
            OptimizedJsonInfo optimizedJsonInfo = new OptimizedJsonInfo();
            OptimizedJsonFieldCollector jsonFieldCollector =
                new OptimizedJsonFieldCollector(optimizedJsonInfo,
                                                mode);
            programClass.accept(new MultiClassVisitor(
                jsonFieldCollector,
                new AllFieldVisitor(jsonFieldCollector)));

            OptimizedJsonInfo.ClassJsonInfo classJsonInfo =
                optimizedJsonInfo.classJsonInfos.get(programClass.getName());
            Collection<String[]> jsonFieldNamesCollection =
                classJsonInfo.javaToJsonFieldNames.values();
            Set<String> uniqueFieldNames = new HashSet<String>();
            for (String[] jsonFieldNames : jsonFieldNamesCollection)
            {
                for (String jsonFieldName : jsonFieldNames)
                {
                    if (uniqueFieldNames.contains(jsonFieldName))
                    {
                        hasDuplicateJsonFieldNames = true;
                        return;
                    }
                    else
                    {
                        uniqueFieldNames.add(jsonFieldName);
                    }
                }
            }
        }
    }

    @Override
    public void visitLibraryClass(LibraryClass libraryClass) {}
}
