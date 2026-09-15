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
import proguard.classfile.util.SimplifiedVisitor;
import proguard.classfile.visitor.ClassVisitor;

import java.io.*;

/**
 * This ClassVisitor writes out the ProgramClass objects that it visits to the
 * given DataEntry, modified to have the correct name.
 *
 * @author Eric Lafortune
 */
public class DataEntryClassWriter
extends      SimplifiedVisitor
implements   ClassVisitor
{
    private final DataEntryWriter dataEntryWriter;
    private final DataEntry       templateDataEntry;


    /**
     * Creates a new DataEntryClassWriter for writing to the given
     * DataEntryWriter, based on the given template DataEntry.
     */
    public DataEntryClassWriter(DataEntryWriter dataEntryWriter,
                                DataEntry       templateDataEntry)
    {
        this.dataEntryWriter   = dataEntryWriter;
        this.templateDataEntry = templateDataEntry;
    }


    // Implementations for ClassVisitor.

    public void visitProgramClass(ProgramClass programClass)
    {
        // Rename the data entry if necessary.
        String    actualClassName = programClass.getName();
        DataEntry actualDataEntry =
            new RenamedDataEntry(templateDataEntry,
                                 actualClassName + ClassConstants.CLASS_FILE_EXTENSION);

        try
        {
            // Get the output entry corresponding to this input entry.
            OutputStream outputStream = dataEntryWriter.createOutputStream(actualDataEntry);
            if (outputStream != null)
            {
                // Write the class to the output entry.
                DataOutputStream classOutputStream = new DataOutputStream(outputStream);
                try
                {
                    new ProgramClassWriter(classOutputStream).visitProgramClass(programClass);
                }
                finally
                {
                    classOutputStream.close();
                }
            }
        }
        catch (IOException e)
        {
            throw new RuntimeException("Can't write program class ["+actualClassName+"] to ["+actualDataEntry+"] ("+e.getMessage()+")", e);
        }
    }
}
