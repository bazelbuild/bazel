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
import proguard.classfile.io.ProgramClassWriter;

import java.io.*;

/**
 * This DataEntryWriter finds received class entries in the given class pool
 * and writes them out to the given data entry writer. For resource entries,
 * it returns valid output streams. For class entries, it returns output
 * streams that must not be used.
 *
 * @see IdleRewriter
 * @author Eric Lafortune
 */
public class ClassDataEntryWriter implements DataEntryWriter
{
    private final ClassPool       classPool;
    private final DataEntryWriter dataEntryWriter;


    /**
     * Creates a new ClassDataEntryWriter.
     * @param classPool       the class pool in which classes are found.
     * @param dataEntryWriter the writer to which the class file is written.
     */
    public ClassDataEntryWriter(ClassPool       classPool,
                                DataEntryWriter dataEntryWriter)
    {
        this.classPool       = classPool;
        this.dataEntryWriter = dataEntryWriter;
    }


    // Implementations for DataEntryWriter.

    public boolean createDirectory(DataEntry dataEntry) throws IOException
    {
        return dataEntryWriter.createDirectory(dataEntry);
    }


    public boolean sameOutputStream(DataEntry dataEntry1,
                                    DataEntry dataEntry2)
    throws IOException
    {
        return dataEntryWriter.sameOutputStream(dataEntry1, dataEntry2);
    }


    public OutputStream createOutputStream(DataEntry dataEntry) throws IOException
    {
        String inputName = dataEntry.getName();

        // Is it a class entry?
        String name = dataEntry.getName();
        if (name.endsWith(ClassConstants.CLASS_FILE_EXTENSION))
        {
            // Does it still have a corresponding class?
            String className = inputName.substring(0, inputName.length() - ClassConstants.CLASS_FILE_EXTENSION.length());
            Clazz clazz = classPool.getClass(className);
            if (clazz != null)
            {
                // Rename the data entry if necessary.
                String newClassName = clazz.getName();
                if (!className.equals(newClassName))
                {
                    dataEntry = new RenamedDataEntry(dataEntry, newClassName + ClassConstants.CLASS_FILE_EXTENSION);
                }

                // Get the output stream for this input entry.
                OutputStream outputStream = dataEntryWriter.createOutputStream(dataEntry);
                if (outputStream != null)
                {
                    // Write the class to the output stream.
                    DataOutputStream classOutputStream = new DataOutputStream(outputStream);
                    try
                    {
                        clazz.accept(new ProgramClassWriter(classOutputStream));
                    }
                    catch (RuntimeException e)
                    {
                        throw (RuntimeException)new RuntimeException("Unexpected error while writing class ["+className+"] ("+e.getMessage()+")").initCause(e);
                    }
                    finally
                    {
                        classOutputStream.close();
                    }
                }
            }

            // Return a dummy, non-null output stream (to work with cascading
            // output writers).
            return new FilterOutputStream(null);
        }

        // Delegate for resource entries.
        return dataEntryWriter.createOutputStream(dataEntry);
    }


    public void close() throws IOException
    {
        // Close the delegate writer.
        dataEntryWriter.close();
    }


    public void println(PrintWriter pw, String prefix)
    {
        pw.println(prefix + "ClassDataEntryWriter");
        dataEntryWriter.println(pw, prefix + "  ");
    }
}