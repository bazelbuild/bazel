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
package proguard.io;

import proguard.classfile.*;
import proguard.classfile.io.*;
import proguard.classfile.util.*;
import proguard.classfile.visitor.ClassVisitor;

import java.io.*;

/**
 * This DataEntryReader applies a given ClassVisitor to the class
 * definitions that it reads.
 * <p>
 * Class files are read as ProgramClass objects or LibraryClass objects,
 * depending on the <code>isLibrary</code> flag.
 * <p>
 * In case of libraries, only public classes are considered, if the
 * <code>skipNonPublicLibraryClasses</code> flag is set.
 *
 * @author Eric Lafortune
 */
public class ClassReader implements DataEntryReader
{
    private final boolean        isLibrary;
    private final boolean        skipNonPublicLibraryClasses;
    private final boolean        skipNonPublicLibraryClassMembers;
    private final WarningPrinter warningPrinter;
    private final ClassVisitor   classVisitor;


    /**
     * Creates a new DataEntryClassFilter for reading the specified
     * Clazz objects.
     */
    public ClassReader(boolean        isLibrary,
                       boolean        skipNonPublicLibraryClasses,
                       boolean        skipNonPublicLibraryClassMembers,
                       WarningPrinter warningPrinter,
                       ClassVisitor   classVisitor)
    {
        this.isLibrary                        = isLibrary;
        this.skipNonPublicLibraryClasses      = skipNonPublicLibraryClasses;
        this.skipNonPublicLibraryClassMembers = skipNonPublicLibraryClassMembers;
        this.warningPrinter                   = warningPrinter;
        this.classVisitor                     = classVisitor;
    }


    // Implementations for DataEntryReader.

    public void read(DataEntry dataEntry) throws IOException
    {
        try
        {
            // Get the input stream.
            InputStream inputStream = dataEntry.getInputStream();

            // Wrap it into a data input stream.
            DataInputStream dataInputStream = new DataInputStream(inputStream);

            // Create a Clazz representation.
            Clazz clazz;
            if (isLibrary)
            {
                clazz = new LibraryClass();
                clazz.accept(new LibraryClassReader(dataInputStream, skipNonPublicLibraryClasses, skipNonPublicLibraryClassMembers));
            }
            else
            {
                clazz = new ProgramClass();
                clazz.accept(new ProgramClassReader(dataInputStream));
            }

            // Apply the visitor, if we have a real class.
            String className = clazz.getName();
            if (className != null)
            {
                String dataEntryName = dataEntry.getName();
                if (!dataEntryName.equals("module-info.class") &&
                    !dataEntryName.replace(File.pathSeparatorChar, ClassConstants.PACKAGE_SEPARATOR).equals(className + ClassConstants.CLASS_FILE_EXTENSION) &&
                    warningPrinter != null)
                {
                    warningPrinter.print(className,
                                         "Warning: class [" + dataEntry.getName() + "] unexpectedly contains class [" + ClassUtil.externalClassName(className) + "]");
                }

                clazz.accept(classVisitor);
            }

            dataEntry.closeInputStream();
        }
        catch (Exception ex)
        {
            throw (IOException)new IOException("Can't process class ["+dataEntry.getName()+"] ("+ex.getMessage()+")").initCause(ex);
        }
    }
}
