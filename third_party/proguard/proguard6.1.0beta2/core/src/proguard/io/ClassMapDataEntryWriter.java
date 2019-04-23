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
import proguard.classfile.util.*;
import proguard.classfile.visitor.MemberVisitor;

import java.io.*;
import java.util.Iterator;


/**
 * This DataEntryWriter writes a class mapping to the given data entry, used
 * for debugging of the configuration.
 *
 * Syntax of the mapping file (one line per class):
 *
 * originalClassName,newClassName,hasObfuscatedMethods,hasObfuscatedFields
 *
 * hasObfuscatedMethods and hasObfuscatedFields can either take the value
 * 0 (false) or 1 (true).
 *
 * @author Johan Leys
 */
public class ClassMapDataEntryWriter
extends      SimplifiedVisitor
implements   DataEntryWriter,

             // Implementation interfaces.
             MemberVisitor
{
    private final ClassPool programClassPool;

    private final DataEntryWriter dataEntryWriter;

    private boolean obfuscatedMethods = false;
    private boolean obfuscatedFields  = false;


    public ClassMapDataEntryWriter(ClassPool       programClassPool,
                                   DataEntryWriter dataEntryWriter  )
    {
        this.programClassPool   = programClassPool;
        this.dataEntryWriter    = dataEntryWriter;
    }


    // Implementations for DataEntryWriter.

    public void close() throws IOException
    {
        dataEntryWriter.close();
    }


    public boolean createDirectory(DataEntry dataEntry) throws IOException
    {
        return dataEntryWriter.createDirectory(dataEntry);
    }


    public boolean sameOutputStream(DataEntry dataEntry1, DataEntry dataEntry2) throws IOException
    {
        return dataEntryWriter.sameOutputStream(dataEntry1, dataEntry2);
    }


    public OutputStream createOutputStream(DataEntry dataEntry) throws IOException
    {
        OutputStream os     = dataEntryWriter.createOutputStream(dataEntry);
        PrintWriter  writer = new PrintWriter(new OutputStreamWriter(os));
        writeClassMap(writer, programClassPool);
        writer.close();
        return os;
    }


    public void println(PrintWriter pw, String prefix)
    {
        pw.println(prefix + "ClassMapDataEntryWriter");
        dataEntryWriter.println(pw, prefix + "  ");
    }


    // Private utility methods.

    private void writeClassMap(PrintWriter writer, ClassPool classPool)
    {
        Iterator iterator = classPool.classNames();
        while (iterator.hasNext())
        {
            String className = (String)iterator.next();

            StringBuilder builder   = new StringBuilder();

            builder.append(ClassUtil.externalClassName(className));
            builder.append(",");

            ProgramClass clazz = (ProgramClass)classPool.getClass(className);
            builder.append(ClassUtil.externalClassName(clazz.getName()));
            builder.append(",");

            boolean hasRemovedMethods = (clazz.u2accessFlags & ClassConstants.ACC_REMOVED_METHODS) != 0;
            builder.append(hasRemovedMethods || hasObfuscatedMethods(clazz) ? 1 : 0);
            builder.append(",");

            boolean hasRemovedFields = (clazz.u2accessFlags & ClassConstants.ACC_REMOVED_FIELDS) != 0;
            builder.append(hasRemovedFields || hasObfuscatedFields(clazz) ? 1 : 0);
            writer.println(builder.toString());
        }
    }


    private boolean hasObfuscatedMethods(ProgramClass clazz)
    {
        obfuscatedMethods = false;
        clazz.methodsAccept(this);
        return obfuscatedMethods;
    }


    private boolean hasObfuscatedFields(ProgramClass clazz)
    {
        obfuscatedFields = false;
        clazz.fieldsAccept(this);
        return obfuscatedFields;
    }


    // Implementations for MemberVisitor.

    public void visitAnyMember(Clazz clazz, Member member) {}


    public void visitProgramMethod(ProgramClass programClass, ProgramMethod programMethod)
    {
        obfuscatedMethods |= (programMethod.getAccessFlags() & ClassConstants.ACC_RENAMED) != 0;
    }


    public void visitProgramField(ProgramClass programClass, ProgramField programField)
    {
        obfuscatedFields |= (programField.getAccessFlags() & ClassConstants.ACC_RENAMED) != 0;
    }
}
